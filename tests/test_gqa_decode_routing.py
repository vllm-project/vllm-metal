# SPDX-License-Identifier: Apache-2.0
"""P256/P512 production selection and fallback, verified after native evaluation.

Kernel correctness is covered independently in test_gqa_paged_decode.py.
Tests inject core counts only when hardware detection is unavailable.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from tests.test_gqa_paged_decode import (
    BLOCK_SIZE,
    GQA_GEOMETRIES,
    GQA_PARTITIONS,
    GQA_SIMD_GROUPS_PER_CORE,
    HEAD_SIZE,
    NUM_KV_HEADS,
    NUM_QUERY_HEADS,
    _assert_close,
    _assert_fallback,
    _dispatch_family,
    _grouped_paged_reference,
    _interleaved_table,
)
from vllm_metal.metal import get_ops


def test_native_reports_public_routing_capabilities():
    assert get_ops().paged_attention_capabilities() == {
        "gqa_decode": True,
        "gqa_disable": True,
        "decode_routing_metadata": True,
        "gqa_batch_context_lens": True,
    }


def _restore_test_gpu_cores() -> None:
    ops = get_ops()
    ops._override_detected_gpu_core_count_for_test(-1)
    if ops.detected_gpu_core_count() <= 0:
        ops._override_detected_gpu_core_count_for_test(10)


@pytest.fixture(scope="module", autouse=True)
def _enable_dispatch_diagnostics():
    ops = get_ops()
    previous = ops._set_paged_dispatch_diagnostics(True)
    try:
        yield
    finally:
        mx.synchronize()
        ops._set_paged_dispatch_diagnostics(previous)


@pytest.fixture(scope="module", autouse=True)
def _inject_test_gpu_cores():
    """CI runners often omit IORegistry gpu-core-count; keep routing tests on."""
    _restore_test_gpu_cores()
    yield
    get_ops()._override_detected_gpu_core_count_for_test(-1)


def _require_grid(kv_len: int, query_heads: int) -> None:
    """Positive route tests need the measured grid guard on this GPU."""
    cores = get_ops().detected_gpu_core_count()
    if cores <= 0:
        pytest.skip("GPU core count unavailable: GQA conservatively disabled")
    if (kv_len // min(GQA_PARTITIONS)) * query_heads < GQA_SIMD_GROUPS_PER_CORE * cores:
        pytest.skip("This boundary is below the GQA grid guard on this GPU")


def _eligible_context(query_heads: int = NUM_QUERY_HEADS, minimum: int = 32768) -> int:
    cores = get_ops().detected_gpu_core_count()
    if cores <= 0:
        pytest.skip("GPU core count unavailable: GQA conservatively disabled")
    # Round to whole partitions; this helper only sizes positive-test inputs.
    # Expected shape/length decisions are explicit in the boundary tests.
    partitions = (GQA_SIMD_GROUPS_PER_CORE * cores + query_heads - 1) // query_heads
    n = max(minimum, partitions * min(GQA_PARTITIONS))
    return n


def _run_primitive(
    kv_lens: list[int],
    dtype: mx.Dtype,
    *,
    interleaved: bool,
    seed: int,
    window_seqlen_q: int = 1,
    query_lens: list[int] | None = None,
    num_decode_requests: int = -1,
    num_decode_tokens: int = 0,
    max_decode_context_len: int = 0,
    gqa_disabled: bool = False,
    num_query_heads: int = NUM_QUERY_HEADS,
    num_kv_heads: int = NUM_KV_HEADS,
    head_size: int = HEAD_SIZE,
    block_size: int = BLOCK_SIZE,
    softcap: float = 0.0,
    sliding_window: int = -1,
    sink_value: float | None = None,
    turboquant: bool = False,
    native_reference: bool = False,
    test_partition: int | None = None,
    gqa_context_lens: list[int] | None = None,
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
                b + s * n_blocks_needed for b in _interleaved_table(n_blocks_needed)
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
    if gqa_context_lens is not None:
        quant_kwargs["gqa_context_lens"] = gqa_context_lens
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
        quant_kwargs.update(
            {
                "key_scale_cache": k_scale,
                "value_scale_cache": v_scale,
                "key_zero_cache": k_zero,
                "v_centroids": get_v_centroids(3),
                "use_turboquant": True,
                "quant_type": "q8_0",
                "v_bits": 3,
            }
        )
        mx.eval(key_cache, value_cache, k_scale, k_zero, v_scale, key_ref, value_ref)

    out = mx.array(0)
    if test_partition is not None:
        get_ops()._gqa_paged_attention_for_test(
            query,
            key_cache,
            value_cache,
            scale,
            block_tables,
            kv_lens_arr,
            block_size,
            max_kv_len,
            test_partition,
            out,
        )
    else:
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
            num_decode_tokens=num_decode_tokens,
            max_decode_context_len=max_decode_context_len,
            gqa_disabled=gqa_disabled,
            sinks=sinks,
            **quant_kwargs,
        )
    mx.eval(out)
    if sinks is None and not native_reference:
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
            sinks=None if sinks is None else sinks.astype(dtype),
        ).reshape(out.shape)
    mx.eval(ref)
    return out, ref


def test_core_count_override_restores_hardware_detection() -> None:
    ops = get_ops()
    original = ops.detected_gpu_core_count()
    ops._override_detected_gpu_core_count_for_test(7)
    try:
        assert ops.detected_gpu_core_count() == 7
    finally:
        _restore_test_gpu_cores()
    assert ops.detected_gpu_core_count() == original


def test_core_count_override_changes_default_routing() -> None:
    """Production dispatch reads the override through detected_gpu_core_count()."""
    ops = get_ops()
    # Q=32, KV=8000: 31 full P256 partitions. Eligible iff 31*32 >= 33*cores;
    # forty cores require 10752 KV, while ten cores can select GQA.
    ops._override_detected_gpu_core_count_for_test(40)
    try:
        assert ops.detected_gpu_core_count() == 40
        out, ref = _run_primitive(
            [8000],
            mx.bfloat16,
            interleaved=False,
            seed=41,
            num_decode_requests=1,
        )
        _assert_fallback()
        _assert_close(out, ref, mx.bfloat16)
        ops._override_detected_gpu_core_count_for_test(10)
        out, ref = _run_primitive(
            [8000],
            mx.bfloat16,
            interleaved=False,
            seed=41,
            num_decode_requests=1,
        )
        assert _dispatch_family() == "gqa_decode"
        _assert_close(out, ref, mx.bfloat16)
    finally:
        _restore_test_gpu_cores()


def test_unknown_core_count_keeps_measured_shape_on_baseline() -> None:
    ops = get_ops()
    ops._override_detected_gpu_core_count_for_test(0)
    try:
        assert ops.detected_gpu_core_count() == 0
        out, ref = _run_primitive(
            [32768],
            mx.bfloat16,
            interleaved=False,
            seed=37,
            num_decode_requests=1,
        )
        _assert_fallback()
        _assert_close(out, ref, mx.bfloat16)
    finally:
        _restore_test_gpu_cores()


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("offset,interleaved", [(0, False), (17, True)])
def test_gqa_decode_matches_reference(dtype, offset, interleaved) -> None:
    n = _eligible_context() + offset
    out, ref = _run_primitive(
        [n],
        dtype,
        interleaved=interleaved,
        seed=0,
        native_reference=not interleaved,
    )
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


@pytest.mark.parametrize("q,kv,head,block_size", GQA_GEOMETRIES)
@pytest.mark.parametrize("offset", [-1, 0, 1])
def test_each_geometry_lower_boundary_dispatch(q, kv, head, block_size, offset):
    cores = get_ops().detected_gpu_core_count()
    if cores <= 0:
        pytest.skip("GPU core count unavailable")
    minimum = min(GQA_PARTITIONS) * ((GQA_SIMD_GROUPS_PER_CORE * cores + q - 1) // q)
    n = minimum + offset
    if offset >= 0:
        _require_grid(n, q)
    out, ref = _run_primitive(
        [n],
        mx.bfloat16,
        interleaved=True,
        seed=715,
        num_query_heads=q,
        num_kv_heads=kv,
        head_size=head,
        block_size=block_size,
    )
    if offset >= 0:
        assert _dispatch_family() == "gqa_decode"
    else:
        _assert_fallback()
    _assert_close(out, ref, mx.bfloat16)


@pytest.mark.parametrize("q,kv,head,block_size", GQA_GEOMETRIES)
@pytest.mark.parametrize("n", [131071, 131072, 131073, 196608, 262144, 262145])
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.slow
def test_each_geometry_stays_on_gqa_at_long_context(q, kv, head, block_size, n, dtype):
    _require_grid(n, q)
    out, ref = _run_primitive(
        [n],
        dtype,
        interleaved=True,
        seed=716,
        num_query_heads=q,
        num_kv_heads=kv,
        head_size=head,
        block_size=block_size,
    )
    assert _dispatch_family() == "gqa_decode"
    _assert_close(out, ref, dtype)


@pytest.mark.parametrize(
    "q,kv,head,n",
    [
        (16, 4, 256, 16384),
        (16, 4, 256, 32768),  # 16/4/256 is outside default routing.
        (16, 4, 256, 65536),
        (16, 4, 256, 131073),
        (32, 4, 64, 32768),  # Head dim 64 is not a shipped GQA specialization.
        (64, 8, 64, 32768),
        (32, 4, 64, 65536),
        (16, 2, 96, 65536),
        (32, 4, 256, 65536),  # Unmeasured 32/4/256 stays on the established path.
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


@pytest.mark.parametrize(
    "q,kv,head,block_size",
    [
        (32, 8, 128, 8),
        (32, 8, 128, 32),
        (16, 2, 256, 8),
        (16, 4, 256, 32),
        (24, 4, 256, 32),
    ],
)
def test_unmeasured_kernel_page_sizes_use_fallback(q, kv, head, block_size):
    out, ref = _run_primitive(
        [32768],
        mx.float16,
        interleaved=True,
        seed=718,
        block_size=block_size,
        num_query_heads=q,
        num_kv_heads=kv,
        head_size=head,
    )
    _assert_fallback()
    _assert_close(out, ref, mx.float16)


@pytest.mark.parametrize("num_decode_requests", [-1, 2])
def test_multi_request_decode_stays_off_gqa(num_decode_requests):
    n = 32768
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
        [32768],
        mx.float16,
        interleaved=True,
        seed=3,
        num_decode_requests=num_decode_requests,
    )
    _assert_fallback()
    _assert_close(out, ref, mx.float16)


@pytest.mark.parametrize("query_lens", [[3], [1, 2]])
def test_prefill_and_mixed_batches_stay_off_gqa(query_lens):
    n = 32768
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


@pytest.mark.parametrize("disabled", [False, True])
def test_split_mixed_decode_stays_on_upstream_path(disabled, force_tiled_prefill):
    """Splitting one decode row out of a mixed batch must not enable GQA."""
    out, ref = _run_primitive(
        [8192, 128],
        mx.float16,
        interleaved=True,
        seed=720,
        query_lens=[1, 32],
        num_decode_requests=1,
        num_decode_tokens=1,
        max_decode_context_len=8192,
        gqa_disabled=disabled,
    )
    assert _dispatch_family() == "mixed_prefill_decode"
    assert get_ops().last_gqa_partition_size() == 0
    _assert_close(out, ref, mx.float16)


def test_spec_window_does_not_switch_kernel_family() -> None:
    out, ref = _run_primitive(
        [32768 + 4],
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
        [32768],
        mx.float16,
        interleaved=True,
        seed=10,
        **feature,
    )
    _assert_fallback()
    _assert_close(out, ref, mx.float16)


def test_turboquant_uses_fallback_with_dequantized_reference() -> None:
    out, ref = _run_primitive(
        [65536],
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
    out, ref = _run_primitive([32768], mx.float32, interleaved=True, seed=11)
    _assert_fallback()
    _assert_close(out, ref, mx.float32)


@pytest.mark.parametrize("minimum", [32768, 196609])
@pytest.mark.parametrize("q,kv,head,block_size", GQA_GEOMETRIES)
def test_gqa_disable_flag_forces_established_kernels(
    monkeypatch, minimum, q, kv, head, block_size
) -> None:
    from vllm_metal import envs

    monkeypatch.delenv("VLLM_METAL_DISABLE_GQA_DECODE", raising=False)
    assert envs.VLLM_METAL_DISABLE_GQA_DECODE is False
    n = _eligible_context(q, minimum=minimum)
    shape = {
        "num_query_heads": q,
        "num_kv_heads": kv,
        "head_size": head,
        "block_size": block_size,
    }
    out_on, ref = _run_primitive([n], mx.bfloat16, interleaved=True, seed=7, **shape)
    assert _dispatch_family() == "gqa_decode"
    out_off, _ = _run_primitive(
        [n], mx.bfloat16, interleaved=True, seed=7, gqa_disabled=True, **shape
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
        **shape,
    )
    _assert_fallback()
    _assert_close(out_env, ref, mx.bfloat16)


# Golden boundaries remain independent of the exported budget. Sharing the
# supported geometries must not turn these into an implementation self-check.
_ADMISSION_AT_CORE_COUNT = {
    32: {20: 5376, 40: 10752},
    24: {40: 14080, 86: 30464},
    16: {40: 21248, 80: 42240, 86: 45568},
}


@pytest.mark.parametrize(
    "q,kv,head,block,cores,minimum",
    [
        (*geometry, cores, minimum)
        for geometry in GQA_GEOMETRIES
        for cores, minimum in _ADMISSION_AT_CORE_COUNT[geometry[0]].items()
    ],
)
def test_shape_and_device_performance_gate(q, kv, head, block, cores, minimum):
    ops = get_ops()
    assert not ops.gqa_decode_shape_eligible(q, kv, head, minimum - 1, cores, block)
    assert ops.gqa_decode_shape_eligible(q, kv, head, minimum, cores, block)
    assert ops.gqa_decode_shape_eligible(q, kv, head, minimum + 1, cores, block)


@pytest.mark.parametrize("q,kv,head,block", GQA_GEOMETRIES)
@pytest.mark.parametrize("n,cores", [(0, 40), (-1, 40), (262145, 0), (65536, -1)])
def test_unknown_core_count_or_empty_length_is_not_eligible(
    q, kv, head, block, n, cores
):
    assert not get_ops().gqa_decode_shape_eligible(q, kv, head, n, cores, block)


@pytest.mark.parametrize("q,kv,head,block", GQA_GEOMETRIES)
@pytest.mark.parametrize("n", [131072, 131073, 262145, 524289, 1048576])
def test_policy_has_no_fixed_long_context_ceiling(q, kv, head, block, n):
    assert get_ops().gqa_decode_partition_size(q, kv, head, n, 40, block) == 512


@pytest.mark.parametrize(
    "q,kv,head,block",
    [
        (16, 4, 256, 16),
        (32, 4, 64, 16),
        (64, 8, 64, 16),
        (16, 2, 96, 16),
        (32, 4, 256, 16),
        (16, 8, 128, 16),
        (16, 16, 128, 16),
        (8, 1, 128, 16),
        (18, 4, 128, 16),
        (16, 0, 128, 16),
        (32, 8, 128, 32),
        (24, 4, 256, 32),
    ],
)
def test_policy_rejects_unmeasured_geometries(q, kv, head, block):
    assert not get_ops().gqa_decode_shape_eligible(q, kv, head, 65536, 40, block)
    assert get_ops().gqa_decode_partition_size(q, kv, head, 65536, 40, block) == 0


@pytest.mark.parametrize("q,kv,head,block", GQA_GEOMETRIES)
def test_default_partition_thresholds(q, kv, head, block):
    thresholds = {32: [10752, 21504], 24: [14080, 28160], 16: [21248, 42496]}[q]
    ops = get_ops()
    for previous, part, threshold in zip([0, 256], [256, 512], thresholds, strict=True):
        assert (
            ops.gqa_decode_partition_size(q, kv, head, threshold - 1, 40, block)
            == previous
        )
        assert ops.gqa_decode_partition_size(q, kv, head, threshold, 40, block) == part
        assert (
            ops.gqa_decode_partition_size(q, kv, head, threshold + 1, 40, block) == part
        )


@pytest.mark.parametrize("q,kv,head,block", GQA_GEOMETRIES)
@pytest.mark.parametrize("part", GQA_PARTITIONS)
@pytest.mark.parametrize("offset", [-1, 0, 17])
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_default_partition_executes_and_matches_oracle(
    q, kv, head, block, part, offset, dtype
):
    ops = get_ops()
    cores = ops.detected_gpu_core_count()
    if cores <= 0:
        pytest.skip("GPU core count unavailable")
    threshold = part * ((GQA_SIMD_GROUPS_PER_CORE * cores + q - 1) // q)
    out, ref = _run_primitive(
        [threshold + offset],
        dtype,
        interleaved=True,
        seed=719,
        num_query_heads=q,
        num_kv_heads=kv,
        head_size=head,
        block_size=block,
        num_decode_requests=1,
    )
    expected = part if offset >= 0 else (0 if part == 256 else part // 2)
    if expected:
        assert _dispatch_family() == "gqa_decode"
    else:
        _assert_fallback()
    assert ops.last_gqa_partition_size() == expected
    _assert_close(out, ref, dtype)
