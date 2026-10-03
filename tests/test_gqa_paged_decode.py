# SPDX-License-Identifier: Apache-2.0
"""Numerical and cache-layout coverage for the private P256/P512 GQA kernels.

Tests force each specialization without GPU core detection. Production
selection and fallback are covered separately in test_gqa_decode_routing.py.
"""

from __future__ import annotations

import subprocess
import sys

import mlx.core as mx
import numpy as np
import pytest

from tools.attention_bench_utils import attention_tolerances, ref_paged_attn
from vllm_metal.metal import get_ops

# The native dispatcher owns the supported domain. Parametrize the positive
# matrices from that table, while routing tests keep independent boundary and
# rejection expectations. A missing table must fail rather than skip coverage.
GQA_CONFIG = get_ops()._gqa_decode_config_for_test()
GQA_GEOMETRIES = tuple(tuple(row) for row in GQA_CONFIG["geometries"])
GQA_PARTITIONS = tuple(sorted(GQA_CONFIG["partitions"]))
GQA_SIMD_GROUPS_PER_CORE = GQA_CONFIG["simd_groups_per_core"]
assert GQA_GEOMETRIES and GQA_PARTITIONS
assert len(GQA_GEOMETRIES) == len(set(GQA_GEOMETRIES))

NUM_QUERY_HEADS = 32
NUM_KV_HEADS = 8
HEAD_SIZE = 128
BLOCK_SIZE = 16


def _interleaved_table(n_blocks: int) -> list[int]:
    """Non-contiguous logical pages within a compact physical allocation."""
    return np.random.default_rng(715).permutation(n_blocks).tolist()


def _assert_close(out: mx.array, ref: mx.array, dtype: mx.Dtype) -> None:
    atol, rtol = attention_tolerances(dtype, float32_tolerance=2e-4)
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


@pytest.fixture(scope="module", autouse=True)
def _enable_dispatch_diagnostics():
    ops = get_ops()
    previous = ops._set_paged_dispatch_diagnostics(True)
    try:
        yield
    finally:
        mx.synchronize()
        ops._set_paged_dispatch_diagnostics(previous)


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
    head then owns one ordinary QK/PV matrix product, so the long-context reference
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
    # Cache axes are (pages, block_size, KV heads, head dimension).
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
    num_query_heads: int = NUM_QUERY_HEADS,
    num_kv_heads: int = NUM_KV_HEADS,
    head_size: int = HEAD_SIZE,
    block_size: int = BLOCK_SIZE,
    test_partition: int | None = None,
) -> tuple[mx.array, mx.array]:
    assert len(kv_lens) == 1
    mx.random.seed(seed)
    n = kv_lens[0]
    n_blocks = (n + block_size - 1) // block_size
    pages = _interleaved_table(n_blocks) if interleaved else list(range(n_blocks))
    shape = (n_blocks + 4, block_size, num_kv_heads, head_size)
    key = mx.random.normal(shape).astype(dtype)
    value = mx.random.normal(shape).astype(dtype)
    query = mx.random.normal((1, num_query_heads, head_size)).astype(dtype)
    tables = mx.array([pages], dtype=mx.int32)
    lengths = mx.array(kv_lens, dtype=mx.int32)
    mx.eval(query, key, value, tables, lengths)
    out = mx.array(0)
    if test_partition is not None:
        get_ops()._gqa_paged_attention_for_test(
            query,
            key,
            value,
            head_size**-0.5,
            tables,
            lengths,
            block_size,
            n,
            test_partition,
            out,
        )
    else:
        get_ops().paged_attention_primitive(
            query,
            key,
            value,
            num_kv_heads,
            head_size**-0.5,
            0.0,
            tables,
            lengths,
            mx.array([0, 1], dtype=mx.int32),
            block_size,
            n,
            -1,
            out,
        )
    mx.eval(out)
    ref = _grouped_paged_reference(
        query=query,
        key_cache=key,
        value_cache=value,
        query_lens=[1],
        kv_lens=kv_lens,
        block_tables=np.array(tables),
        scale=head_size**-0.5,
    )
    mx.eval(ref)
    return out, ref


def test_dispatch_diagnostics_are_disabled_in_a_fresh_process() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from vllm_metal.metal import get_ops; "
            "ops = get_ops(); "
            "assert ops._set_paged_dispatch_diagnostics(False) is False; "
            "assert ops.last_paged_dispatch() == ''; "
            "assert ops.last_gqa_partition_size() == 0",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("test_partition", [None, *GQA_PARTITIONS])
def test_dispatch_diagnostics_require_opt_in(test_partition) -> None:
    ops = get_ops()
    mx.synchronize()
    previous = ops._set_paged_dispatch_diagnostics(False)
    kwargs = {
        "kv_lens": [513],
        "dtype": mx.bfloat16,
        "interleaved": True,
        "seed": 715,
        "test_partition": test_partition,
    }
    try:
        unrecorded, reference = _run_primitive(**kwargs)
        assert ops.last_paged_dispatch() == ""
        assert ops.last_gqa_partition_size() == 0
        assert ops._set_paged_dispatch_diagnostics(True) is False
        recorded, _ = _run_primitive(**kwargs)
        if test_partition is None:
            _assert_fallback()
            assert ops.last_gqa_partition_size() == 0
        else:
            assert ops.last_paged_dispatch() == "gqa_decode"
            assert ops.last_gqa_partition_size() == test_partition
        _assert_close(unrecorded, reference, mx.bfloat16)
        np.testing.assert_array_equal(
            np.array(unrecorded.astype(mx.float32)),
            np.array(recorded.astype(mx.float32)),
        )
        assert ops._set_paged_dispatch_diagnostics(False) is True
        assert ops.last_paged_dispatch() == ""
        assert ops.last_gqa_partition_size() == 0
        _run_primitive(**kwargs)
        assert ops.last_paged_dispatch() == ""
        assert ops.last_gqa_partition_size() == 0
    finally:
        mx.synchronize()
        ops._set_paged_dispatch_diagnostics(previous)


def test_gqa_decode_kernel_is_in_default_library() -> None:
    ops = get_ops()
    assert ops._has_gqa_decode_kernel(), (
        "default shader library is missing paged_attention_gqa_decode; "
        "rebuild with `python -m vllm_metal.metal.build`"
    )


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("part", GQA_PARTITIONS)
@pytest.mark.parametrize("tail", [False, True])
@pytest.mark.parametrize("q,kv,head,block", GQA_GEOMETRIES)
def test_gqa_kernel_numerics_without_core_detection(
    dtype, part, tail, q, kv, head, block
):
    """CI executes every specialization, including single and partial partitions."""
    out, ref = _run_primitive(
        [2 * part + 17 if tail else part],
        dtype,
        interleaved=True,
        seed=42,
        num_query_heads=q,
        num_kv_heads=kv,
        head_size=head,
        block_size=block,
        test_partition=part,
    )
    assert _dispatch_family() == "gqa_decode"
    assert get_ops().last_gqa_partition_size() == part
    _assert_close(out, ref, dtype)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("part", GQA_PARTITIONS)
@pytest.mark.parametrize("n_tok", [1, 5, 6, 7, 13, 15])
@pytest.mark.parametrize("q,kv,head,block", GQA_GEOMETRIES)
def test_gqa_partial_page_tails(dtype, part, n_tok, q, kv, head, block):
    """Partition remainder pages, including n_tok % 4 != 0."""
    # One full partition plus a last page of n_tok tokens; the tail falls
    # through the 4-way groups into the per-token loop.
    extra = block if block > 16 else 0
    out, ref = _run_primitive(
        [part + extra + n_tok],
        dtype,
        interleaved=True,
        seed=64 + n_tok,
        num_query_heads=q,
        num_kv_heads=kv,
        head_size=head,
        block_size=block,
        test_partition=part,
    )
    assert _dispatch_family() == "gqa_decode"
    assert get_ops().last_gqa_partition_size() == part
    _assert_close(out, ref, dtype)


@pytest.mark.parametrize("part", [0, 32, 64, 128, 1024])
def test_private_gqa_entry_rejects_unshipped_partitions(part):
    with pytest.raises(ValueError, match="GQA test partition"):
        _run_primitive([65], mx.float16, interleaved=False, seed=0, test_partition=part)


@pytest.mark.parametrize("cache", ["key", "value"])
@pytest.mark.parametrize(
    "query_dtype,cache_dtype",
    [(mx.float16, mx.bfloat16), (mx.bfloat16, mx.float16), (mx.float16, mx.float32)],
)
def test_private_gqa_entry_rejects_mismatched_cache_dtype(
    cache, query_dtype, cache_dtype
):
    query = mx.zeros((1, 32, 128), dtype=query_dtype)
    key = mx.zeros((2, 16, 8, 128), dtype=query_dtype)
    value = mx.zeros(key.shape, dtype=query_dtype)
    if cache == "key":
        key = key.astype(cache_dtype)
    else:
        value = value.astype(cache_dtype)
    # Rejection must happen before a lazy kernel can reinterpret either cache.
    with pytest.raises(ValueError, match="requires one supported decode row"):
        get_ops()._gqa_paged_attention_for_test(
            query,
            key,
            value,
            128**-0.5,
            mx.array([[0, 1]], dtype=mx.int32),
            mx.array([17], dtype=mx.int32),
            16,
            17,
            256,
            mx.array(0),
        )


def test_private_gqa_partition_is_local_to_lazy_primitive():
    """Building another node must not change a pending node or production routing."""
    ops = get_ops()
    q = mx.ones((1, 32, 128), dtype=mx.float16)
    k = mx.ones((40, 16, 8, 128), dtype=mx.float16)
    v = mx.full(k.shape, 2, dtype=mx.float16)
    tables = mx.arange(40, dtype=mx.int32)[None]
    lengths = mx.array([513], dtype=mx.int32)
    outputs = [mx.array(0) for _ in range(3)]
    for part, out in zip([256, 512], outputs[:2], strict=True):
        ops._gqa_paged_attention_for_test(
            q, k, v, 128**-0.5, tables, lengths, 16, 513, part, out
        )
    ops.paged_attention_primitive(
        q,
        k,
        v,
        8,
        128**-0.5,
        0.0,
        tables,
        lengths,
        mx.array([0, 1], dtype=mx.int32),
        16,
        513,
        -1,
        outputs[2],
    )
    for part, out in zip([256, 512, 0], outputs, strict=True):
        mx.eval(out)
        assert ops.last_gqa_partition_size() == part
        assert np.allclose(np.array(out), 2, atol=2e-3)
    _assert_fallback()


@pytest.mark.parametrize("partition", [256])
def test_forced_partition_rejects_oversized_reducer(partition):
    """A small cache can represent a long logical history without a huge allocation."""
    ops = get_ops()
    n = 1048577
    query = mx.ones((1, 16, 128), mx.float16)
    key = mx.ones((1, 16, 2, 128), mx.float16)
    value = mx.full(key.shape, 2, mx.float16)
    table = mx.zeros((1, (n + 15) // 16), mx.int32)
    lengths = mx.array([n], mx.int32)
    out = mx.array(0)
    with pytest.raises(ValueError, match="threadgroup memory limit"):
        ops._gqa_paged_attention_for_test(
            query, key, value, 128**-0.5, table, lengths, 16, n, partition, out
        )
    # A larger partition fits the same history and still computes every token.
    ops._gqa_paged_attention_for_test(
        query, key, value, 128**-0.5, table, lengths, 16, n, 512, out
    )
    mx.eval(out)
    assert ops.last_gqa_partition_size() == 512
    np.testing.assert_allclose(np.array(out), 2, atol=2e-3)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("test_partition", GQA_PARTITIONS)
@pytest.mark.parametrize(
    "q_heads,kv_heads,head,block_size",
    GQA_GEOMETRIES
    + tuple(
        (q, kv, head, {16: 784, 32: 1056}[block])
        for q, kv, head, block in GQA_GEOMETRIES
    ),
)
def test_gqa_reads_upstream_views_after_writes_and_block_copy(
    dtype, test_partition, q_heads, kv_heads, head, block_size
):
    """Exercise every shipped GQA specialization on shared K/V storage.

    A non-contiguous page table survives copy-on-write and a new decode write.
    No eval separates cache updates from attention: their graph dependencies
    must make both writes visible through the strided K/V aliases.
    """
    import torch
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
        KVCacheTensor,
    )

    from vllm_metal.attention.block_tables import build_block_tables
    from vllm_metal.attention.caches.kv_cache import MetalPagedKVCache
    from vllm_metal.attention.caches.storage import KVCacheStorage

    n = 2 * max(block_size, test_partition) + 1
    pages = _interleaved_table((n + 1 + block_size - 1) // block_size)
    num_blocks = max(pages) + 2
    spec = FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=kv_heads,
        head_size=head,
        dtype=torch.float16 if dtype == mx.float16 else torch.bfloat16,
    )
    page_bytes = spec.page_size_bytes
    storage = KVCacheStorage(
        KVCacheConfig(
            num_blocks=num_blocks,
            kv_cache_groups=[
                KVCacheGroupSpec(layer_names=["attn"], kv_cache_spec=spec)
            ],
            kv_cache_tensors=[
                KVCacheTensor(
                    size=num_blocks * page_bytes,
                    layers=["attn"],
                    layer_stride=num_blocks * page_bytes,
                    block_stride=page_bytes,
                )
            ],
            kv_cache_layout="LBNHC",
        )
    )
    cache = MetalPagedKVCache.from_upstream(storage, ["attn"])
    # K/V occupy different offsets of the same physical region, not dense arrays.
    key_desc, value_desc = (
        cache.key_caches.descriptors[0],
        cache.value_caches.descriptors[0],
    )
    assert key_desc[0] == value_desc[0]
    assert key_desc[2] == value_desc[2]
    assert key_desc[2][-2:] == (2 * head, 1)
    assert value_desc[3] - key_desc[3] == head

    mx.random.seed(719)
    keys = mx.random.normal((n + 1, kv_heads, head)).astype(dtype)
    values = mx.random.normal(keys.shape).astype(dtype)
    # These rows dominate softmax, so losing either the copied page or the
    # appended write cannot hide inside long-context numerical tolerances.
    keys[0], keys[n] = 4, 4
    values[0], values[n] = 1, 3
    slots = [pages[i // block_size] * block_size + i % block_size for i in range(n)]
    ops = get_ops()
    written = ops.reshape_and_cache(
        keys[:n],
        values[:n],
        cache.key_caches[0],
        cache.value_caches[0],
        mx.array(slots, dtype=mx.int64),
    )
    cache.replace_layer_cache(0, *written)
    for length in (n, n + 1):
        if length == n + 1:
            # Move a cached prefix page, then append into the partial final page.
            storage.copy_blocks([(pages[0], num_blocks - 1)])
            storage.zero_blocks([pages[0]])
            pages[0] = num_blocks - 1
            slot = pages[n // block_size] * block_size + n % block_size
            written = ops.reshape_and_cache(
                keys[n:],
                values[n:],
                cache.key_caches[0],
                cache.value_caches[0],
                mx.array([slot], dtype=mx.int64),
            )
            cache.replace_layer_cache(0, *written)
        query = mx.ones((1, q_heads, head), dtype=dtype)
        kernel_tables, kernel_block_size = build_block_tables([pages], block_size)
        if block_size == 1056:
            assert kernel_block_size == 32
        # Production keeps dense K/V in scheduler-page views. Translated
        # block IDs must use the kernel-token stride without reshaping them.
        kernel_keys = cache.key_caches[0]
        kernel_values = cache.value_caches[0]
        out = mx.array(0)
        ops._gqa_paged_attention_for_test(
            query,
            kernel_keys,
            kernel_values,
            head**-0.5,
            kernel_tables,
            mx.array([length], dtype=mx.int32),
            kernel_block_size,
            length,
            test_partition,
            out,
        )
        mx.eval(out)
        assert _dispatch_family() == "gqa_decode"
        assert ops.last_gqa_partition_size() == test_partition
        # Independent logical history; never gather the cache under test.
        ref = _grouped_paged_reference(
            query=query,
            key_cache=keys[None],
            value_cache=values[None],
            query_lens=[1],
            kv_lens=[length],
            block_tables=np.array([[0]]),
            scale=head**-0.5,
        )
        _assert_close(out, ref, dtype)


@pytest.mark.parametrize("q,kv,head,block_size", GQA_GEOMETRIES)
@pytest.mark.parametrize("n", [131071, 131072, 131073, 196608, 262144, 262145])
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("part", GQA_PARTITIONS)
@pytest.mark.slow
def test_each_geometry_at_long_context(q, kv, head, block_size, n, dtype, part):
    """Opt-in long-context matrix; regular CI covers every specialization above."""
    out, ref = _run_primitive(
        [n],
        dtype,
        interleaved=True,
        seed=716,
        num_query_heads=q,
        num_kv_heads=kv,
        head_size=head,
        block_size=block_size,
        test_partition=part,
    )
    assert _dispatch_family() == "gqa_decode"
    assert get_ops().last_gqa_partition_size() == part
    _assert_close(out, ref, dtype)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("n", [1024, 32769])
def test_private_kernels_do_not_change_production_routing(dtype, n):
    ops = get_ops()
    out, ref = _run_primitive([n], dtype, interleaved=True, seed=715)
    expected = (ops.last_paged_dispatch(), ops.last_gqa_partition_size())
    _run_primitive([n], dtype, interleaved=True, seed=715, test_partition=256)
    repeated, _ = _run_primitive([n], dtype, interleaved=True, seed=715)
    assert (ops.last_paged_dispatch(), ops.last_gqa_partition_size()) == expected
    _assert_close(out, ref, dtype)
    np.testing.assert_array_equal(
        np.array(out.astype(mx.float32)), np.array(repeated.astype(mx.float32))
    )
