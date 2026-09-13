# SPDX-License-Identifier: Apache-2.0
"""Scoped GQA decode routing, verified after the native primitive executes.

The default optimization covers three measured attention geometries, one
request, ordinary FP16/BF16 caches, 16-token kernel pages, and bounded long
contexts. Numerical support for another shape does not authorize its default
use. These tests inspect the actual dispatcher and compare both selected and
fallback paths against attention references; timing is deliberately excluded.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from tools.attention_bench_utils import ref_paged_attn
from vllm_metal.metal import get_ops

NUM_QUERY_HEADS = 32
NUM_KV_HEADS = 8
HEAD_SIZE = 128
BLOCK_SIZE = 16

_TOLERANCES = {
    mx.bfloat16: (3e-2, 2e-2),
    mx.float16: (1.5e-2, 2e-2),
    mx.float32: (2e-4, 2e-4),
}


def _interleaved_table(n_blocks: int) -> list[int]:
    """Run-of-2, skip-1 pattern produced by hybrid GDN block interleave."""
    table: list[int] = []
    b = 3
    while len(table) < n_blocks:
        table += [b, b + 1]
        b += 3
    return table[:n_blocks]


def _assert_close(out: mx.array, ref: mx.array, dtype: mx.Dtype) -> None:
    atol, rtol = _TOLERANCES[dtype]
    np.testing.assert_allclose(
        np.array(out.astype(mx.float32)),
        np.array(ref.astype(mx.float32)),
        atol=atol,
        rtol=rtol,
    )
    # Long-context outputs are small: absolute tolerances alone could accept
    # an incorrectly zero-filled result, so also bound the relative L2 error.
    out_np = np.array(out.astype(mx.float32))
    ref_np = np.array(ref.astype(mx.float32))
    relative_l2 = np.linalg.norm(out_np - ref_np) / max(np.linalg.norm(ref_np), 1e-10)
    assert (
        relative_l2
        < {
            mx.bfloat16: 2e-2,
            mx.float16: 6e-3,
            mx.float32: 1e-3,
        }[dtype]
    )


def _require_grid(kv_len: int, kv_heads: int) -> None:
    """Positive route tests need the measured grid guard on this GPU."""
    cores = get_ops().detected_gpu_core_count()
    if cores <= 0:
        pytest.skip("GPU core count unavailable: GQA conservatively disabled")
    if ((kv_len + 511) // 512) * kv_heads < 3 * cores:
        pytest.skip("This boundary is below the GQA grid guard on this GPU")


def _eligible_context(kv_heads: int = NUM_KV_HEADS, minimum: int = 32768) -> int:
    cores = get_ops().detected_gpu_core_count()
    if cores <= 0:
        pytest.skip("GPU core count unavailable: GQA conservatively disabled")
    # Round to whole partitions; this helper only sizes positive-test inputs.
    # Expected shape/length decisions are explicit in the boundary tests.
    partitions = (3 * cores + kv_heads - 1) // kv_heads
    n = max(minimum, partitions * 512)
    if n > 131072:
        pytest.skip("No context inside the scoped range meets this GPU's grid guard")
    return n


def _dispatch_family() -> str:
    return get_ops().last_paged_dispatch()


def _assert_fallback() -> None:
    # Existing occupancy routing chooses ps0 or ps512 depending on the GPU.
    assert _dispatch_family() in {"per_token_ps0", "per_token_ps512"}


def _grouped_paged_reference(
    *,
    query: mx.array,
    key_cache: mx.array,
    value_cache: mx.array,
    query_lens: list[int],
    kv_lens: list[int],
    block_tables: np.ndarray,
    scale: float,
    sliding_window: int | None = None,
    soft_cap: float = 0.0,
) -> mx.array:
    """FP32 attention without repeating K/V for each grouped query head.

    Fold the query-token and GQA-group axes into a matrix row axis. Each KV
    head then owns one ordinary QK/PV matrix product, so the 128K reference
    retains only the unique gathered K/V instead of allocating a copy for
    every query head. This is a high-level full-softmax oracle, independent
    of the paged kernel's partitioning and online softmax implementation.
    """
    # GPU matrix-matrix and matrix-vector paths can use different effective
    # precision even for float32 arrays. Keep the oracle's arithmetic on CPU
    # so grouping changes neither its precision nor its error allowance.
    with mx.stream(mx.cpu):
        _, block_size, kv_heads, head_size = key_cache.shape
        query_heads = query.shape[1]
        group = query_heads // kv_heads
        outputs = []
        offset = 0
        for index, (query_len, kv_len) in enumerate(
            zip(query_lens, kv_lens, strict=True)
        ):
            pages = mx.array(
                block_tables[index, : (kv_len + block_size - 1) // block_size]
            )
            keys = (
                key_cache[pages]
                .reshape(-1, kv_heads, head_size)[:kv_len]
                .astype(mx.float32)
                .transpose(1, 0, 2)
            )
            values = (
                value_cache[pages]
                .reshape(-1, kv_heads, head_size)[:kv_len]
                .astype(mx.float32)
                .transpose(1, 0, 2)
            )
            queries = (
                query[offset : offset + query_len]
                .astype(mx.float32)
                .reshape(query_len, kv_heads, group, head_size)
                .transpose(1, 0, 2, 3)
                .reshape(kv_heads, query_len * group, head_size)
            )
            scores = mx.einsum("krd,knd->krn", queries * scale, keys)
            scores = scores.reshape(kv_heads, query_len, group, kv_len)
            if soft_cap > 0:
                scores = soft_cap * mx.tanh(scores / soft_cap)
            query_positions = (kv_len - query_len + mx.arange(query_len))[:, None]
            key_positions = mx.arange(kv_len)[None, :]
            allowed = key_positions <= query_positions
            if sliding_window is not None:
                allowed = allowed & (key_positions > query_positions - sliding_window)
            scores = mx.where(allowed[None, :, None, :], scores, float("-inf"))
            probabilities = mx.softmax(scores, axis=-1).reshape(
                kv_heads, query_len * group, kv_len
            )
            result = mx.einsum("krn,knd->krd", probabilities, values)
            outputs.append(
                result.reshape(kv_heads, query_len, group, head_size)
                .transpose(1, 0, 2, 3)
                .reshape(query_len, query_heads, head_size)
            )
            offset += query_len
        return mx.concatenate(outputs, axis=0)


@pytest.mark.parametrize("query_heads", [3, 12])
@pytest.mark.parametrize(
    "sliding_window,soft_cap", [(None, 0.0), (7, 0.0), (None, 2.0), (7, 2.0)]
)
def test_grouped_reference_matches_expanded_reference(
    query_heads, sliding_window, soft_cap
):
    """Cross-check grouping, causal rows, page gathering and feature masks."""
    mx.random.seed(39)
    query = mx.random.normal((4, query_heads, 16))
    keys = mx.random.normal((12, 8, 3, 16))
    values = mx.random.normal((12, 8, 3, 16))
    kwargs = {
        "query": query,
        "key_cache": keys,
        "value_cache": values,
        "query_lens": [1, 3],
        "kv_lens": [19, 23],
        "block_tables": np.array([[7, 2, 9], [1, 8, 4]], dtype=np.int32),
        "scale": 16**-0.5,
        "sliding_window": sliding_window,
        "soft_cap": soft_cap,
    }
    grouped = _grouped_paged_reference(**kwargs)
    with mx.stream(mx.cpu):
        expanded = ref_paged_attn(**kwargs)
    mx.eval(grouped, expanded)
    np.testing.assert_allclose(
        np.array(grouped), np.array(expanded), atol=2e-6, rtol=2e-5
    )


def _run_primitive(
    kv_lens: list[int],
    dtype: mx.Dtype,
    *,
    interleaved: bool,
    seed: int,
    window_seqlen_q: int = 1,
    query_lens: list[int] | None = None,
    num_decode_requests: int = -1,
    gqa_disabled: bool = False,
    num_query_heads: int = NUM_QUERY_HEADS,
    num_kv_heads: int = NUM_KV_HEADS,
    head_size: int = HEAD_SIZE,
    block_size: int = BLOCK_SIZE,
    softcap: float = 0.0,
    sliding_window: int = -1,
    sink_value: float | None = None,
    turboquant: bool = False,
) -> tuple[mx.array, mx.array]:
    mx.random.seed(seed)
    num_seqs = len(kv_lens)
    max_kv_len = max(kv_lens)
    scale = head_size**-0.5
    if query_lens is None:
        query_lens = [1] * num_seqs
    n_blocks_needed = (max_kv_len + block_size - 1) // block_size
    tables = []
    max_blk = 0
    for s in range(num_seqs):
        if interleaved:
            table = [
                b + s * (n_blocks_needed * 3 + 4)
                for b in _interleaved_table(n_blocks_needed)
            ]
        else:
            table = list(range(s * n_blocks_needed, (s + 1) * n_blocks_needed))
        tables.append(table)
        max_blk = max(max_blk, max(table))
    cache_shape = (max_blk + 4, block_size, num_kv_heads, head_size)
    key_cache = mx.random.normal(cache_shape).astype(dtype)
    value_cache = mx.random.normal(cache_shape).astype(dtype)
    query = mx.random.normal((sum(query_lens), num_query_heads, head_size)).astype(
        dtype
    )
    block_tables = mx.array(tables, dtype=mx.int32)
    kv_lens_arr = mx.array(kv_lens, dtype=mx.int32)
    cu = [0]
    for qlen in query_lens:
        cu.append(cu[-1] + qlen)
    cu_seqlens_q = mx.array(cu, dtype=mx.int32)
    sinks = (
        None
        if sink_value is None
        else mx.full((num_query_heads,), sink_value, dtype=mx.float32)
    )
    mx.eval(key_cache, value_cache, query, block_tables, kv_lens_arr, cu_seqlens_q)

    key_ref, value_ref = key_cache, value_cache
    quant_kwargs = {}
    if turboquant:
        from vllm_metal.attention.caches.turboquant import (
            get_v_centroids,
            turbo_quant_decode,
            turbo_quant_encode,
        )

        (key_cache, k_scale, k_zero), (value_cache, v_scale) = turbo_quant_encode(
            key_cache, value_cache, "q8_0"
        )
        key_ref, value_ref = turbo_quant_decode(
            (key_cache, k_scale, k_zero),
            (value_cache, v_scale),
            output_dtype=dtype,
            key_quant_type="q8_0",
        )
        quant_kwargs = {
            "key_scale_cache": k_scale,
            "value_scale_cache": v_scale,
            "key_zero_cache": k_zero,
            "v_centroids": get_v_centroids(3),
            "use_turboquant": True,
            "quant_type": "q8_0",
            "v_bits": 3,
        }
        mx.eval(key_cache, value_cache, k_scale, k_zero, v_scale, key_ref, value_ref)

    out = mx.array(0)
    get_ops().paged_attention_primitive(
        query,
        key_cache,
        value_cache,
        num_kv_heads,
        scale,
        softcap,
        block_tables,
        kv_lens_arr,
        cu_seqlens_q,
        block_size,
        max_kv_len,
        sliding_window,
        out,
        window_seqlen_q=window_seqlen_q,
        num_decode_requests=num_decode_requests,
        gqa_disabled=gqa_disabled,
        sinks=sinks,
        **quant_kwargs,
    )
    mx.eval(out)
    if sinks is None:
        ref = _grouped_paged_reference(
            query=query,
            key_cache=key_ref,
            value_cache=value_ref,
            query_lens=query_lens,
            kv_lens=kv_lens,
            block_tables=np.array(block_tables),
            scale=scale,
            sliding_window=None if sliding_window < 0 else sliding_window,
            soft_cap=softcap,
        )
    else:
        assert query_lens == [1] and softcap == 0 and sliding_window < 0
        flat_k = key_cache[block_tables[0]].reshape(-1, num_kv_heads, head_size)
        flat_v = value_cache[block_tables[0]].reshape(-1, num_kv_heads, head_size)
        ref = mx.fast.scaled_dot_product_attention(
            query.transpose(1, 0, 2)[None],
            flat_k[:max_kv_len].transpose(1, 0, 2)[None],
            flat_v[:max_kv_len].transpose(1, 0, 2)[None],
            scale=scale,
            sinks=sinks.astype(dtype),
        ).reshape(out.shape)
    mx.eval(ref)
    return out, ref


def test_gqa_decode_kernel_is_in_default_library() -> None:
    ops = get_ops()
    assert ops.has_gqa_decode_kernel(), (
        "default shader library is missing paged_attention_gqa_decode; "
        "rebuild with `python -m vllm_metal.metal.build`"
    )
    assert ops.GQA_DECODE_MIN_SEQ_LEN == 32768
    assert ops.GQA_DECODE_MAX_SEQ_LEN == 131072


def test_unknown_core_count_keeps_measured_shape_on_baseline() -> None:
    if get_ops().detected_gpu_core_count() > 0:
        pytest.skip("This platform exposes GPU core count")
    out, ref = _run_primitive(
        [32768],
        mx.bfloat16,
        interleaved=False,
        seed=37,
        num_decode_requests=1,
    )
    _assert_fallback()
    _assert_close(out, ref, mx.bfloat16)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("offset,interleaved", [(0, False), (17, True)])
def test_gqa_decode_matches_reference(dtype, offset, interleaved) -> None:
    n = _eligible_context() + offset
    if n > 131072:
        pytest.skip("Partial-tail case exceeds this device's eligible range")
    out, ref = _run_primitive([n], dtype, interleaved=interleaved, seed=0)
    assert _dispatch_family() == "gqa_decode"
    _assert_close(out, ref, dtype)


@pytest.mark.parametrize("num_decode_requests", [-1, 1])
def test_one_decode_request_takes_gqa(num_decode_requests) -> None:
    out, ref = _run_primitive(
        [_eligible_context()],
        mx.bfloat16,
        interleaved=True,
        seed=4,
        num_decode_requests=num_decode_requests,
    )
    assert _dispatch_family() == "gqa_decode"
    _assert_close(out, ref, mx.bfloat16)


@pytest.mark.parametrize(
    "q,kv,head,minimum", [(32, 8, 128, 32768), (24, 4, 256, 32768), (16, 2, 128, 65536)]
)
@pytest.mark.parametrize("offset", [-1, 0, 1])
def test_each_geometry_lower_boundary_dispatch(q, kv, head, minimum, offset):
    n = minimum + offset
    if offset >= 0:
        _require_grid(n, kv)
    out, ref = _run_primitive(
        [n],
        mx.bfloat16,
        interleaved=True,
        seed=715,
        num_query_heads=q,
        num_kv_heads=kv,
        head_size=head,
    )
    if offset >= 0:
        assert _dispatch_family() == "gqa_decode"
    else:
        _assert_fallback()
    _assert_close(out, ref, mx.bfloat16)


@pytest.mark.parametrize("q,kv,head", [(32, 8, 128), (24, 4, 256), (16, 2, 128)])
@pytest.mark.parametrize("n", [131072, 131073])
def test_each_geometry_upper_boundary_dispatch(q, kv, head, n):
    if n == 131072:
        _require_grid(n, kv)
    out, ref = _run_primitive(
        [n],
        mx.float16,
        interleaved=False,
        seed=716,
        num_query_heads=q,
        num_kv_heads=kv,
        head_size=head,
    )
    if n == 131072:
        assert _dispatch_family() == "gqa_decode"
    else:
        _assert_fallback()
    _assert_close(out, ref, mx.float16)


@pytest.mark.parametrize(
    "q,kv,head,n",
    [
        (16, 2, 128, 16384),  # Reproduced MiniCPM5 regression.
        (16, 2, 128, 32768),  # Small measured gain deliberately omitted.
        (32, 8, 128, 16384),
        (24, 4, 256, 16384),
        (32, 4, 64, 32768),  # Both narrow-head regressions stay excluded.
        (64, 8, 64, 32768),
        (32, 4, 64, 65536),  # Same old byte proxy/grid as the preceding row.
        (16, 2, 96, 65536),
        (16, 4, 256, 65536),  # Supported shader, unmeasured default geometry.
        (16, 8, 128, 65536),
        (16, 16, 128, 65536),  # MHA.
    ],
)
def test_unselected_shapes_really_use_fallback(q, kv, head, n):
    out, ref = _run_primitive(
        [n],
        mx.bfloat16,
        interleaved=True,
        seed=717,
        num_query_heads=q,
        num_kv_heads=kv,
        head_size=head,
    )
    _assert_fallback()
    _assert_close(out, ref, mx.bfloat16)


@pytest.mark.parametrize("block_size", [8, 32])
def test_unmeasured_kernel_page_sizes_use_fallback(block_size):
    out, ref = _run_primitive(
        [_eligible_context()],
        mx.float16,
        interleaved=True,
        seed=718,
        block_size=block_size,
    )
    _assert_fallback()
    _assert_close(out, ref, mx.float16)


@pytest.mark.parametrize("num_decode_requests", [-1, 2])
def test_two_individually_eligible_requests_stay_off_gqa(num_decode_requests):
    n = _eligible_context()
    out, ref = _run_primitive(
        [n, n + 1],
        mx.float16,
        interleaved=True,
        seed=2,
        num_decode_requests=num_decode_requests,
    )
    _assert_fallback()
    _assert_close(out, ref, mx.float16)


@pytest.mark.parametrize("num_decode_requests", [-2, 0, 2])
def test_scheduler_must_confirm_one_decode_or_omit_count(num_decode_requests):
    out, ref = _run_primitive(
        [_eligible_context()],
        mx.float16,
        interleaved=True,
        seed=3,
        num_decode_requests=num_decode_requests,
    )
    _assert_fallback()
    _assert_close(out, ref, mx.float16)


@pytest.mark.parametrize("query_lens", [[3], [1, 2]])
def test_prefill_and_mixed_batches_stay_off_gqa(query_lens):
    n = _eligible_context()
    out, ref = _run_primitive(
        [n] * len(query_lens),
        mx.float16,
        interleaved=True,
        seed=8,
        query_lens=query_lens,
        num_decode_requests=1 if len(query_lens) == 2 else 0,
    )
    assert _dispatch_family() in {"nax_prefill", "tiled_prefill"}
    _assert_close(out, ref, mx.float16)


def test_spec_window_does_not_switch_kernel_family() -> None:
    out, ref = _run_primitive(
        [_eligible_context() + 4],
        mx.float16,
        interleaved=True,
        seed=9,
        window_seqlen_q=4,
        query_lens=[4],
    )
    assert _dispatch_family().startswith("window_")
    _assert_close(out, ref, mx.float16)


@pytest.mark.parametrize(
    "feature", [{"softcap": 2.0}, {"sliding_window": 4096}, {"sink_value": 12.0}]
)
def test_special_attention_features_use_fallback(feature):
    out, ref = _run_primitive(
        [_eligible_context()],
        mx.float16,
        interleaved=True,
        seed=10,
        **feature,
    )
    _assert_fallback()
    _assert_close(out, ref, mx.float16)


def test_turboquant_uses_fallback_with_dequantized_reference() -> None:
    out, ref = _run_primitive(
        [_eligible_context(kv_heads=2, minimum=65536)],
        mx.float16,
        interleaved=False,
        seed=12,
        num_query_heads=16,
        num_kv_heads=2,
        turboquant=True,
    )
    _assert_fallback()
    _assert_close(out, ref, mx.float16)


def test_matching_float32_uses_fallback() -> None:
    out, ref = _run_primitive(
        [_eligible_context()], mx.float32, interleaved=True, seed=11
    )
    _assert_fallback()
    _assert_close(out, ref, mx.float32)


def test_gqa_disable_flag_forces_established_kernels(monkeypatch) -> None:
    from vllm_metal import envs

    monkeypatch.delenv("VLLM_METAL_DISABLE_GQA_DECODE", raising=False)
    assert envs.VLLM_METAL_DISABLE_GQA_DECODE is False
    n = _eligible_context()
    out_on, ref = _run_primitive([n], mx.bfloat16, interleaved=True, seed=7)
    assert _dispatch_family() == "gqa_decode"
    out_off, _ = _run_primitive(
        [n], mx.bfloat16, interleaved=True, seed=7, gqa_disabled=True
    )
    _assert_fallback()
    _assert_close(out_on, ref, mx.bfloat16)
    _assert_close(out_off, ref, mx.bfloat16)

    monkeypatch.setenv("VLLM_METAL_DISABLE_GQA_DECODE", "1")
    assert envs.VLLM_METAL_DISABLE_GQA_DECODE is True
    out_env, _ = _run_primitive(
        [n],
        mx.bfloat16,
        interleaved=True,
        seed=7,
        gqa_disabled=envs.VLLM_METAL_DISABLE_GQA_DECODE,
    )
    _assert_fallback()
    _assert_close(out_env, ref, mx.bfloat16)


@pytest.mark.parametrize(
    "q,kv,head,n,cores,expected",
    [
        (32, 8, 128, 32767, 40, False),
        (32, 8, 128, 32768, 40, True),
        (24, 4, 256, 32767, 40, False),
        (24, 4, 256, 32768, 40, True),
        (16, 2, 128, 16384, 20, False),
        (16, 2, 128, 32768, 40, False),
        (16, 2, 128, 65535, 40, False),
        (16, 2, 128, 65536, 40, True),
        (16, 2, 128, 65536, 80, True),
        (16, 2, 128, 65536, 86, False),  # Grid256 < 258.
        (16, 2, 128, 65537, 86, True),  # Partial partition raises grid to258.
        (24, 4, 256, 32768, 86, False),
        (24, 4, 256, 32769, 86, True),
        (32, 8, 128, 131072, 40, True),
        (24, 4, 256, 131072, 40, True),
        (16, 2, 128, 131072, 40, True),
        (32, 8, 128, 131073, 40, False),
        (24, 4, 256, 131073, 40, False),
        (16, 2, 128, 131073, 40, False),
        (32, 8, 128, 65536, 0, False),
        (32, 8, 128, 65536, -1, False),
        (32, 4, 64, 65536, 40, False),
        (64, 8, 64, 32768, 40, False),
        (16, 2, 96, 65536, 40, False),
        (16, 4, 256, 65536, 40, False),
        (16, 8, 128, 65536, 40, False),
        (16, 16, 128, 65536, 40, False),
        (8, 1, 128, 65536, 40, False),
        (18, 4, 128, 65536, 40, False),
        (16, 0, 128, 65536, 40, False),
    ],
)
def test_shape_and_device_performance_gate(q, kv, head, n, cores, expected):
    assert get_ops().gqa_decode_shape_eligible(q, kv, head, n, cores) == expected
