# SPDX-License-Identifier: Apache-2.0
"""Tests for the paged attention primitive and scatter-based cache donation.

Covers two of Eric's review concerns on PR #225:
  1. No test coverage for the primitive path (paged_attention_primitive).
  2. Buffer donation may silently fail, making scatter-based cache writes
     O(entire_cache) instead of O(new_tokens).

Run with:
    python -m pytest tests/test_primitive_and_donation.py -v -s
"""

from __future__ import annotations

import time

import mlx.core as mx
import numpy as np
import pytest

from tools.attention_bench_utils import ref_paged_attn
from vllm_metal.attention.caches.kv_cache import MetalPagedKVCache
from vllm_metal.metal import get_ops

# ── Shared fixtures ──────────────────────────────────────────────────────────

NUM_KV_HEADS_CASES = [2, 4]
NUM_QUERY_HEADS_CASES = [4, 8]  # must be divisible by corresponding kv heads
HEAD_SIZE = 128
BLOCK_SIZE = 16
DTYPE = mx.float16


def _make_cache_and_inputs(
    num_blocks: int,
    num_kv_heads: int,
    num_query_heads: int,
    seq_lens: list[tuple[int, int]],
    *,
    head_size: int = HEAD_SIZE,
    dtype: mx.Dtype = DTYPE,
):
    """Build a populated cache and matching query/metadata tensors."""
    block_size = BLOCK_SIZE
    num_seqs = len(seq_lens)
    query_lens = [s[0] for s in seq_lens]
    kv_lens = [s[1] for s in seq_lens]
    total_q = sum(query_lens)
    max_kv_len = max(kv_lens)
    scale = head_size**-0.5

    key_cache = mx.random.normal(
        shape=(num_blocks, block_size, num_kv_heads, head_size)
    ).astype(dtype)
    value_cache = mx.random.normal(
        shape=(num_blocks, block_size, num_kv_heads, head_size)
    ).astype(dtype)
    query = mx.random.normal(shape=(total_q, num_query_heads, head_size)).astype(dtype)

    max_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = mx.random.randint(
        0, num_blocks, shape=(num_seqs, max_blocks_per_seq)
    ).astype(mx.int32)

    kv_lens_arr = mx.array(kv_lens, dtype=mx.int32)
    cu_seqlens_q = mx.cumsum(mx.array([0] + query_lens, dtype=mx.int32))

    mx.eval(key_cache, value_cache, query, block_tables, kv_lens_arr, cu_seqlens_q)

    return {
        "query": query,
        "key_cache": key_cache,
        "value_cache": value_cache,
        "num_kv_heads": num_kv_heads,
        "scale": scale,
        "block_tables": block_tables,
        "kv_lens_arr": kv_lens_arr,
        "cu_seqlens_q": cu_seqlens_q,
        "query_lens": query_lens,
        "kv_lens": kv_lens,
        "max_kv_len": max_kv_len,
    }


# ── 1. Primitive correctness tests ──────────────────────────────────────────


@pytest.mark.parametrize(
    "seq_lens",
    [
        [(1, 523), (1, 37), (1, 2011)],
        [(1, 1), (1, 128), (1, 2048)],
    ],
)
@pytest.mark.parametrize(
    "num_heads",
    [(4, 4), (8, 2)],
)
@pytest.mark.parametrize("sliding_window", [-1, 128])
@pytest.mark.parametrize("num_blocks", [256])
def test_primitive_vs_reference_decode(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    num_blocks: int,
    sliding_window: int,
) -> None:
    """paged_attention_primitive matches the pure-MLX reference (decode)."""
    mx.random.seed(0)
    num_query_heads, num_kv_heads = num_heads
    d = _make_cache_and_inputs(num_blocks, num_kv_heads, num_query_heads, seq_lens)
    num_decode_requests = len(seq_lens)

    ops = get_ops()
    out = mx.array(0)
    ops.paged_attention_primitive(
        d["query"],
        d["key_cache"],
        d["value_cache"],
        d["num_kv_heads"],
        d["scale"],
        0.0,  # softcap
        d["block_tables"],
        d["kv_lens_arr"],
        d["cu_seqlens_q"],
        BLOCK_SIZE,
        d["max_kv_len"],
        sliding_window,
        out,
        num_decode_requests=num_decode_requests,
        num_decode_tokens=num_decode_requests,
        max_decode_context_len=max(kv_len for _, kv_len in seq_lens),
    )
    mx.eval(out)

    ref = ref_paged_attn(
        query=d["query"],
        key_cache=d["key_cache"],
        value_cache=d["value_cache"],
        query_lens=d["query_lens"],
        kv_lens=d["kv_lens"],
        block_tables=np.array(d["block_tables"]),
        scale=d["scale"],
        sliding_window=sliding_window if sliding_window >= 0 else None,
    )
    mx.eval(ref)

    np.testing.assert_allclose(
        np.array(out),
        np.array(ref),
        atol=1.5e-2,
        rtol=1e-2,
    )


@pytest.mark.parametrize(
    "seq_lens",
    [
        [(1, 1328), (5, 18), (129, 463)],
        [(1, 523), (1, 37), (1, 2011)],
        # Decode prefix crosses the tiled kernel's 32-row block boundary.
        [(1, 4096)] + [(1, 64)] * 32 + [(32, 64)],
        # Prefill-only: all q_len > 1, guarantees tiled kernel dispatch
        # (total_q_tokens > num_seqs).
        [(8, 128), (16, 256)],
        [(32, 512), (64, 1024)],
    ],
)
@pytest.mark.parametrize(
    "num_heads",
    [(4, 4), (8, 2)],
)
@pytest.mark.parametrize("sliding_window", [-1, 100])
@pytest.mark.parametrize("num_blocks", [256])
def test_primitive_vs_reference_varlen(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    num_blocks: int,
    sliding_window: int,
    force_tiled_prefill,
) -> None:
    """paged_attention_primitive matches reference for mixed prefill+decode."""
    mx.random.seed(0)
    num_query_heads, num_kv_heads = num_heads
    d = _make_cache_and_inputs(num_blocks, num_kv_heads, num_query_heads, seq_lens)
    num_decode_requests = 0
    for query_len, _ in seq_lens:
        if query_len != 1:
            break
        num_decode_requests += 1
    max_decode_context_len = max(
        (seq_lens[i][1] for i in range(num_decode_requests)), default=0
    )

    ops = get_ops()
    out = mx.array(0)
    ops.paged_attention_primitive(
        d["query"],
        d["key_cache"],
        d["value_cache"],
        d["num_kv_heads"],
        d["scale"],
        0.0,  # softcap
        d["block_tables"],
        d["kv_lens_arr"],
        d["cu_seqlens_q"],
        BLOCK_SIZE,
        d["max_kv_len"],
        sliding_window,
        out,
        num_decode_requests=num_decode_requests,
        num_decode_tokens=num_decode_requests,
        max_decode_context_len=max_decode_context_len,
    )
    mx.eval(out)

    ref = ref_paged_attn(
        query=d["query"],
        key_cache=d["key_cache"],
        value_cache=d["value_cache"],
        query_lens=d["query_lens"],
        kv_lens=d["kv_lens"],
        block_tables=np.array(d["block_tables"]),
        scale=d["scale"],
        sliding_window=sliding_window if sliding_window >= 0 else None,
    )
    mx.eval(ref)

    np.testing.assert_allclose(
        np.array(out),
        np.array(ref),
        atol=1.5e-2,
        rtol=1e-2,
    )


@pytest.mark.parametrize(
    "decode_context,head_size,split_decode",
    [
        (512, 128, False),
        (4095, 128, False),
        (4096, 128, True),
        (4096, 256, True),
        (4096, 512, True),
    ],
)
def test_mixed_batch_dispatches_decode_by_context(
    decode_context: int,
    head_size: int,
    split_decode: bool,
    force_tiled_prefill,
) -> None:
    """Only long decode contexts pay for a separate kernel dispatch."""
    decode_rows = 33
    seq_lens = [(1, decode_context)] + [(1, 64)] * (decode_rows - 1) + [(32, 64)]
    mx.random.seed(0)
    d = _make_cache_and_inputs(512, 2, 8, seq_lens, head_size=head_size)
    ops = get_ops()

    def run_attention(
        query: mx.array,
        block_tables: mx.array,
        kv_lens: mx.array,
        cu_seqlens: mx.array,
        decode_count: int = 0,
        max_context: int = 0,
    ) -> np.ndarray:
        out = mx.array(0)
        ops.paged_attention_primitive(
            query,
            d["key_cache"],
            d["value_cache"],
            d["num_kv_heads"],
            d["scale"],
            0.0,
            block_tables,
            kv_lens,
            cu_seqlens,
            BLOCK_SIZE,
            d["max_kv_len"],
            -1,
            out,
            num_decode_requests=decode_count,
            num_decode_tokens=decode_count,
            max_decode_context_len=max_context,
        )
        mx.eval(out)
        return np.array(out)

    mixed = run_attention(
        d["query"],
        d["block_tables"],
        d["kv_lens_arr"],
        d["cu_seqlens_q"],
        decode_rows,
        decode_context,
    )
    whole_tiled = run_attention(
        d["query"], d["block_tables"], d["kv_lens_arr"], d["cu_seqlens_q"]
    )
    pure_decode = run_attention(
        d["query"][:decode_rows],
        d["block_tables"][:decode_rows],
        d["kv_lens_arr"][:decode_rows],
        mx.arange(decode_rows + 1, dtype=mx.int32),
        decode_rows,
        decode_context,
    )

    expected_decode = pure_decode if split_decode else whole_tiled[:decode_rows]
    np.testing.assert_array_equal(mixed[:decode_rows], expected_decode)
    np.testing.assert_array_equal(mixed[decode_rows:], whole_tiled[decode_rows:])


@pytest.fixture
def nax_prefill():
    """Route eligible prefill to NAX and record the dispatch family.

    Skips off M5: the NAX kernel cannot run there, and the tiled twin of this
    path is covered by test_mixed_batch_dispatches_decode_by_context.
    """
    ops = get_ops()
    if not (ops.nax_supported() and ops.nax_ready()):
        pytest.skip("NAX prefill needs an M5 GPU and the NAX metallib")
    ops.set_nax_enabled(True)
    previous = ops._set_paged_dispatch_diagnostics(True)
    try:
        yield ops
    finally:
        mx.synchronize()
        ops._set_paged_dispatch_diagnostics(previous)


@pytest.mark.parametrize(
    "decode_rows,decode_context,head_size,dtype,sliding_window,use_sinks",
    [
        (1, 4096, 128, mx.float16, -1, False),
        (9, 16384, 128, mx.bfloat16, -1, False),
        # Decode prefix crosses NAX's 64-row q-block boundary.
        (65, 4096, 128, mx.float16, -1, False),
        (65, 4096, 256, mx.bfloat16, 100, True),
        (3, 4096, 64, mx.float16, 100, True),
        (2, 4096, 512, mx.bfloat16, -1, False),
        # Below the split threshold the whole batch stays on NAX.
        (9, 4095, 128, mx.float16, -1, False),
    ],
)
def test_nax_mixed_batch_splits_long_decode_prefix(
    decode_rows: int,
    decode_context: int,
    head_size: int,
    dtype: mx.Dtype,
    sliding_window: int,
    use_sinks: bool,
    nax_prefill,
) -> None:
    """Long decode rows leave the NAX tile; prefill rows keep NAX bit-for-bit."""
    ops = nax_prefill
    split_decode = decode_context >= 4096
    seq_lens = (
        [(1, decode_context)] + [(1, 64)] * (decode_rows - 1) + [(129, 259), (64, 64)]
    )
    mx.random.seed(0)
    num_kv_heads, num_query_heads = 2, 8
    d = _make_cache_and_inputs(
        512, num_kv_heads, num_query_heads, seq_lens, head_size=head_size, dtype=dtype
    )
    sinks = (
        (mx.random.normal((num_query_heads,)) * 0.5).astype(mx.float32)
        if use_sinks
        else None
    )
    if sinks is not None:
        mx.eval(sinks)

    def run_attention(
        query: mx.array,
        block_tables: mx.array,
        kv_lens: mx.array,
        cu_seqlens: mx.array,
        decode_count: int = 0,
        max_context: int = 0,
    ) -> tuple[np.ndarray, str]:
        out = mx.array(0)
        ops.paged_attention_primitive(
            query,
            d["key_cache"],
            d["value_cache"],
            d["num_kv_heads"],
            d["scale"],
            0.0,
            block_tables,
            kv_lens,
            cu_seqlens,
            BLOCK_SIZE,
            d["max_kv_len"],
            sliding_window,
            out,
            sinks=sinks,
            num_decode_requests=decode_count,
            num_decode_tokens=decode_count,
            max_decode_context_len=max_context,
        )
        mx.eval(out)
        return np.array(out.astype(mx.float32)), ops.last_paged_dispatch()

    mixed, mixed_family = run_attention(
        d["query"],
        d["block_tables"],
        d["kv_lens_arr"],
        d["cu_seqlens_q"],
        decode_rows,
        decode_context,
    )
    whole_nax, whole_family = run_attention(
        d["query"], d["block_tables"], d["kv_lens_arr"], d["cu_seqlens_q"]
    )
    pure_decode, _ = run_attention(
        d["query"][:decode_rows],
        d["block_tables"][:decode_rows],
        d["kv_lens_arr"][:decode_rows],
        mx.arange(decode_rows + 1, dtype=mx.int32),
        decode_rows,
        decode_context,
    )

    assert whole_family == "nax_prefill"
    assert mixed_family == (
        "mixed_nax_prefill_decode" if split_decode else "nax_prefill"
    )
    expected_decode = pure_decode if split_decode else whole_nax[:decode_rows]
    np.testing.assert_array_equal(mixed[:decode_rows], expected_decode)
    np.testing.assert_array_equal(mixed[decode_rows:], whole_nax[decode_rows:])

    if sinks is None:
        # Independent fp32 reference for every row of the mixed batch.
        ref = ref_paged_attn(
            query=d["query"].astype(mx.float32),
            key_cache=d["key_cache"].astype(mx.float32),
            value_cache=d["value_cache"].astype(mx.float32),
            query_lens=d["query_lens"],
            kv_lens=d["kv_lens"],
            block_tables=np.array(d["block_tables"]),
            scale=d["scale"],
            sliding_window=sliding_window if sliding_window >= 0 else None,
        )
        np.testing.assert_allclose(
            mixed, np.array(ref.astype(mx.float32)), atol=2e-2, rtol=1e-2
        )


@pytest.mark.parametrize("gqa_disabled", [False, True])
@pytest.mark.parametrize("prefill_kernel", ["tiled", "nax"])
def test_split_decode_prefix_is_bounded_by_decode_context(
    prefill_kernel: str, gqa_disabled: bool, request
) -> None:
    """The split decode prefix sizes its split-KV work by the decode rows.

    max_seq_len only has to bound every row, so a caller may pass a legal
    allocation bound. At 4M tokens per-token split-KV would plan 8192 P512
    partitions, whose reducer statistics exceed threadgroup memory, while the
    4096-token decode row needs 8. The whole batch on the prefill kernel
    accepts that bound, so the split batch must too, with GQA on or off.
    """
    ops = get_ops()
    if prefill_kernel == "tiled":
        request.getfixturevalue("force_tiled_prefill")
        families = ("mixed_prefill_decode", "tiled_prefill")
    else:
        request.getfixturevalue("nax_prefill")
        families = ("mixed_nax_prefill_decode", "nax_prefill")
    allocation_bound = 4 * 1024 * 1024
    decode_context = 4096
    num_kv_heads, num_query_heads = 8, 32
    mx.random.seed(1061)
    d = _make_cache_and_inputs(
        300,
        num_kv_heads,
        num_query_heads,
        [(1, decode_context), (2, 2)],
        dtype=mx.bfloat16,
    )
    # Rows may address the whole allocation: pad the table to the bound.
    used = d["block_tables"].shape[1]
    tables = mx.concatenate(
        [
            d["block_tables"],
            mx.zeros((2, allocation_bound // BLOCK_SIZE - used), mx.int32),
        ],
        axis=1,
    )
    mx.eval(tables)

    def run_attention(
        query: mx.array,
        block_tables: mx.array,
        kv_lens: mx.array,
        cu_seqlens: mx.array,
        max_seq_len: int,
        decode_count: int = 0,
        disabled: bool = gqa_disabled,
    ) -> tuple[np.ndarray, str, int]:
        out = mx.array(0)
        ops.paged_attention_primitive(
            query,
            d["key_cache"],
            d["value_cache"],
            num_kv_heads,
            d["scale"],
            0.0,
            block_tables,
            kv_lens,
            cu_seqlens,
            BLOCK_SIZE,
            max_seq_len,
            -1,
            out,
            num_decode_requests=decode_count,
            num_decode_tokens=decode_count,
            max_decode_context_len=decode_context if decode_count else 0,
            gqa_disabled=disabled,
        )
        mx.eval(out)
        return (
            np.array(out.astype(mx.float32)),
            ops.last_paged_dispatch(),
            ops.last_gqa_partition_size(),
        )

    previous = ops._set_paged_dispatch_diagnostics(True)
    try:
        mixed, mixed_family, mixed_partition = run_attention(
            d["query"],
            tables,
            d["kv_lens_arr"],
            d["cu_seqlens_q"],
            allocation_bound,
            1,
        )
        whole, whole_family, _ = run_attention(
            d["query"],
            tables,
            d["kv_lens_arr"],
            d["cu_seqlens_q"],
            allocation_bound,
        )
        # The prefix stays on the per-token kernel on every core count.
        pure_decode, _, _ = run_attention(
            d["query"][:1],
            tables[:1],
            d["kv_lens_arr"][:1],
            mx.arange(2, dtype=mx.int32),
            decode_context,
            1,
            True,
        )
    finally:
        mx.synchronize()
        ops._set_paged_dispatch_diagnostics(previous)

    assert (mixed_family, whole_family) == families
    assert mixed_partition == 0
    np.testing.assert_array_equal(mixed[:1], pure_decode)
    np.testing.assert_array_equal(mixed[1:], whole[1:])
    ref = ref_paged_attn(
        query=d["query"].astype(mx.float32),
        key_cache=d["key_cache"].astype(mx.float32),
        value_cache=d["value_cache"].astype(mx.float32),
        query_lens=d["query_lens"],
        kv_lens=d["kv_lens"],
        block_tables=np.array(d["block_tables"]),
        scale=d["scale"],
    )
    np.testing.assert_allclose(
        mixed, np.array(ref.astype(mx.float32)), atol=2e-2, rtol=1e-2
    )


@pytest.mark.slow
def test_sliding_window_bounds_tiled_prefill_work(force_tiled_prefill) -> None:
    """A 1024-token window avoids most KV tiles of an 8K prefill."""
    seq_len = 8192
    window = 1024
    repeats = 5
    minimum_speedup = 2.0
    num_kv_heads = 2
    num_query_heads = 4
    d = _make_cache_and_inputs(
        seq_len // BLOCK_SIZE,
        num_kv_heads,
        num_query_heads,
        [(seq_len, seq_len)],
        head_size=64,
    )
    ops = get_ops()

    def run(sliding_window: int) -> None:
        out = mx.array(0)
        ops.paged_attention_primitive(
            d["query"],
            d["key_cache"],
            d["value_cache"],
            d["num_kv_heads"],
            d["scale"],
            0.0,
            d["block_tables"],
            d["kv_lens_arr"],
            d["cu_seqlens_q"],
            BLOCK_SIZE,
            d["max_kv_len"],
            sliding_window,
            out,
        )
        mx.eval(out)

    def median_runtime(sliding_window: int) -> float:
        run(sliding_window)
        samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            run(sliding_window)
            samples.append(time.perf_counter() - start)
        return sorted(samples)[len(samples) // 2]

    full = median_runtime(-1)
    windowed = median_runtime(window)
    speedup = full / windowed
    assert speedup >= minimum_speedup, f"sliding-window speedup was only {speedup:.2f}x"


# ── 1b. Tiled-kernel head-size coverage ─────────────────────────────────────


@pytest.mark.parametrize("head_size", [64, 96, 128])
def test_tiled_kernel_head_size_parity(head_size: int, force_tiled_prefill) -> None:
    """Tiled prefill kernel matches reference for every routed head size.

    can_use_tiled_kernel() routes head sizes {64, 96, 128} to the BQ=32
    flash kernel; each is a separately compiled template specialization
    (TD = HEAD_SIZE / 8 differs).  The varlen test only exercises 128, so
    this adds one prefill shape per head size to keep CI parity 1:1 with
    can_use_tiled_kernel().
    """
    mx.random.seed(0)
    # Prefill-only (q_len > 1 for every seq) guarantees tiled dispatch.
    seq_lens = [(32, 512), (64, 1024)]
    d = _make_cache_and_inputs(256, 2, 8, seq_lens, head_size=head_size)

    ops = get_ops()
    out = mx.array(0)
    ops.paged_attention_primitive(
        d["query"],
        d["key_cache"],
        d["value_cache"],
        d["num_kv_heads"],
        d["scale"],
        0.0,  # softcap
        d["block_tables"],
        d["kv_lens_arr"],
        d["cu_seqlens_q"],
        BLOCK_SIZE,
        d["max_kv_len"],
        -1,  # sliding_window
        out,
    )
    mx.eval(out)

    ref = ref_paged_attn(
        query=d["query"],
        key_cache=d["key_cache"],
        value_cache=d["value_cache"],
        query_lens=d["query_lens"],
        kv_lens=d["kv_lens"],
        block_tables=np.array(d["block_tables"]),
        scale=d["scale"],
        sliding_window=None,
    )
    mx.eval(ref)

    np.testing.assert_allclose(
        np.array(out),
        np.array(ref),
        atol=1.5e-2,
        rtol=1e-2,
    )


# ── 2. Buffer donation test ─────────────────────────────────────────────────


@pytest.mark.parametrize("num_blocks", [128, 256])
def test_scatter_cache_donation(num_blocks: int) -> None:
    """Verify that scatter-based cache write reuses the buffer (donation).

    MLX's buffer donation is an optimisation, not a contract.  This test
    acts as an early-warning: if donation stops happening (due to MLX
    changes or unexpected reference leaks), the memory delta will spike
    and the assertion will fail.
    """
    num_kv_heads = 4
    head_dim = HEAD_SIZE
    num_layers = 1
    block_size = BLOCK_SIZE
    dtype = DTYPE

    cache = MetalPagedKVCache(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        num_blocks=num_blocks,
        block_size=block_size,
        dtype=dtype,
    )
    # Already eval'd by MetalPagedKVCache.__init__

    cache_nbytes = cache.key_caches[0].nbytes  # size of one K or V cache

    # Simulate a decode step: scatter 3 tokens into random slots
    num_tokens = 3
    slot_indices = mx.array(
        np.random.choice(num_blocks * block_size, size=num_tokens, replace=False),
        dtype=mx.int64,
    )
    new_k = mx.random.normal(shape=(num_tokens, num_kv_heads, head_dim)).astype(dtype)
    new_v = mx.random.normal(shape=(num_tokens, num_kv_heads, head_dim)).astype(dtype)
    mx.eval(slot_indices, new_k, new_v)

    # Warm up: run multiple rounds so the MLX memory pool stabilises.
    # IMPORTANT: delete all locals afterwards — any stray reference to the
    # old cache array bumps use_count and defeats buffer donation.
    for _ in range(5):
        _fk = cache.key_caches[0].reshape(-1, num_kv_heads, head_dim)
        _fk[slot_indices] = new_k
        _wk = _fk.reshape(cache.key_caches[0].shape)
        cache.key_caches[0] = _wk
        _fv = cache.value_caches[0].reshape(-1, num_kv_heads, head_dim)
        _fv[slot_indices] = new_v
        _wv = _fv.reshape(cache.value_caches[0].shape)
        cache.value_caches[0] = _wv
        mx.eval(cache.key_caches[0], cache.value_caches[0])
        del _fk, _wk, _fv, _wv

    # ── Measure at steady state (average of several rounds) ──
    total_delta = 0
    num_rounds = 5
    for _ in range(num_rounds):
        mem_before = mx.get_active_memory()

        # K cache scatter + rebind
        flat_k = cache.key_caches[0].reshape(-1, num_kv_heads, head_dim)
        flat_k[slot_indices] = new_k
        new_k_cache = flat_k.reshape(cache.key_caches[0].shape)
        cache.key_caches[0] = new_k_cache

        # V cache scatter + rebind
        flat_v = cache.value_caches[0].reshape(-1, num_kv_heads, head_dim)
        flat_v[slot_indices] = new_v
        new_v_cache = flat_v.reshape(cache.value_caches[0].shape)
        cache.value_caches[0] = new_v_cache

        mx.eval(cache.key_caches[0], cache.value_caches[0])

        mem_after = mx.get_active_memory()
        total_delta += mem_after - mem_before
        del flat_k, new_k_cache, flat_v, new_v_cache

    avg_delta = total_delta / num_rounds

    # If donation works: avg_delta ≈ 0 (buffers reused in-place).
    # If donation fails: avg_delta ≈ 2 * cache_nbytes (full copy for K + V).
    # Allow generous headroom (one full cache) for pool fluctuations.
    threshold = cache_nbytes
    assert avg_delta < threshold, (
        f"Buffer donation likely failed: avg memory growth {avg_delta:,.0f} "
        f"bytes/round over {num_rounds} rounds, "
        f"but each cache is only {cache_nbytes:,} bytes. "
        f"Expected near-zero growth with donation."
    )
