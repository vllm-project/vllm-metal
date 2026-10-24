# SPDX-License-Identifier: Apache-2.0
"""Tests for the paged attention primitive.

Run with:
    python -m pytest tests/test_primitive_and_donation.py -v -s
"""

from __future__ import annotations

import time

import mlx.core as mx
import numpy as np
import pytest

from tools.attention_bench_utils import ref_paged_attn
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


def _run_attention(
    d,
    *,
    decode_rows: int = 0,
    decode_context: int = 0,
    max_seq_len: int | None = None,
    sliding_window: int = -1,
    sinks: mx.array | None = None,
    decode_lengths: list[int] | None = None,
) -> tuple[np.ndarray, str]:
    ops = get_ops()
    # sdpa passes a length plan describing the decode rows of a mixed batch.
    plan = (
        {}
        if decode_lengths is None
        else {"gqa_length_plan": ops.gqa_decode_length_plan(decode_lengths)}
    )
    out = ops.paged_attention_primitive(
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
        d["max_kv_len"] if max_seq_len is None else max_seq_len,
        sliding_window,
        sinks=sinks,
        num_decode_requests=decode_rows,
        num_decode_tokens=decode_rows,
        max_decode_context_len=decode_context,
        **plan,
    )
    mx.eval(out)
    return np.array(out.astype(mx.float32)), ops.last_paged_dispatch()


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
    out = ops.paged_attention_primitive(
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
    out = ops.paged_attention_primitive(
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
    "decode_lengths,head_size,split_decode",
    [
        # Split once the decode rows' context lengths sum to 512 tokens.
        ([256], 128, False),
        ([511], 128, False),
        ([512], 128, True),
        # Uneven rows count their exact sum, not rows x longest context.
        ([256, 64], 128, False),
        ([447, 64], 128, False),
        ([448, 64], 128, True),
        ([256, 256], 128, True),
        # Many short rows split: they gain the most.
        ([64] * 33, 128, True),
        ([4096] + [64] * 32, 128, True),
        ([4096] + [64] * 32, 256, True),
        ([4096] + [64] * 32, 512, True),
    ],
)
def test_mixed_batch_splits_decode_by_work(
    decode_lengths: list[int],
    head_size: int,
    split_decode: bool,
    force_tiled_prefill,
) -> None:
    """Enough decode work takes the decode kernel; prefill rows keep tiled bit-for-bit."""
    decode_rows = len(decode_lengths)
    decode_context = max(decode_lengths)
    seq_lens = [(1, length) for length in decode_lengths] + [(32, 64)]
    mx.random.seed(0)
    d = _make_cache_and_inputs(512, 2, 8, seq_lens, head_size=head_size)
    ops = get_ops()
    previous = ops._set_paged_dispatch_diagnostics(True)
    try:
        mixed, mixed_family = _run_attention(
            d,
            decode_rows=decode_rows,
            decode_context=decode_context,
            decode_lengths=decode_lengths,
        )
        # Without decode metadata the batch stays on one tiled dispatch.
        whole_tiled, whole_family = _run_attention(d)
    finally:
        mx.synchronize()
        ops._set_paged_dispatch_diagnostics(previous)
    expected_decode = whole_tiled[:decode_rows]
    if split_decode:
        decode = {
            **d,
            "query": d["query"][:decode_rows],
            "block_tables": d["block_tables"][:decode_rows],
            "kv_lens_arr": d["kv_lens_arr"][:decode_rows],
            "cu_seqlens_q": mx.arange(decode_rows + 1, dtype=mx.int32),
        }
        expected_decode, _ = _run_attention(
            decode,
            decode_rows=decode_rows,
            decode_context=decode_context,
            decode_lengths=decode_lengths,
        )
    assert whole_family == "tiled_prefill"
    assert mixed_family == ("mixed_prefill_decode" if split_decode else "tiled_prefill")
    np.testing.assert_array_equal(mixed[:decode_rows], expected_decode)
    np.testing.assert_array_equal(mixed[decode_rows:], whole_tiled[decode_rows:])


@pytest.mark.parametrize(
    "decode_lengths,split_decode",
    [([256, 256], False), ([512, 64], True)],
)
def test_mixed_batch_without_length_plan_bounds_work_by_longest_row(
    decode_lengths: list[int], split_decode: bool, force_tiled_prefill
) -> None:
    """Without a length plan the longest row is a lower bound on the decode work."""
    seq_lens = [(1, length) for length in decode_lengths] + [(32, 64)]
    mx.random.seed(0)
    d = _make_cache_and_inputs(512, 2, 8, seq_lens)
    ops = get_ops()
    previous = ops._set_paged_dispatch_diagnostics(True)
    try:
        _, family = _run_attention(
            d, decode_rows=len(decode_lengths), decode_context=max(decode_lengths)
        )
    finally:
        mx.synchronize()
        ops._set_paged_dispatch_diagnostics(previous)
    assert family == ("mixed_prefill_decode" if split_decode else "tiled_prefill")


@pytest.mark.parametrize(
    "decode_rows,decode_context,head_size,dtype,sliding_window,use_sinks",
    [
        (1, 4096, 128, mx.float16, -1, False),
        (9, 16384, 128, mx.bfloat16, -1, False),
        # Decode prefix lands on and crosses NAX's 64-row q-block boundary.
        (64, 4096, 128, mx.float16, -1, False),
        (65, 4096, 256, mx.bfloat16, 100, True),
        (3, 4096, 64, mx.float16, 100, True),
        (2, 4096, 512, mx.bfloat16, -1, False),
        # Many short decode rows leave the NAX tile too.
        (9, 512, 128, mx.float16, -1, False),
        # Below the decode-work floor the whole batch stays on NAX.
        (1, 256, 128, mx.float16, -1, False),
        # Uneven rows: 256 + 64 stays on NAX; 448 + 64 reaches the floor.
        (2, 256, 128, mx.float16, -1, False),
        (2, 448, 128, mx.float16, -1, False),
    ],
)
def test_nax_mixed_batch_splits_decode_prefix(
    decode_rows: int,
    decode_context: int,
    head_size: int,
    dtype: mx.Dtype,
    sliding_window: int,
    use_sinks: bool,
    nax_prefill,
) -> None:
    """Enough decode work leaves the NAX tile; prefill rows keep NAX bit-for-bit."""
    decode_lengths = [decode_context] + [64] * (decode_rows - 1)
    split_decode = sum(decode_lengths) >= 512
    seq_lens = [(1, length) for length in decode_lengths] + [(129, 259), (64, 64)]
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

    mixed, mixed_family = _run_attention(
        d,
        decode_rows=decode_rows,
        decode_context=decode_context,
        sliding_window=sliding_window,
        sinks=sinks,
        decode_lengths=decode_lengths,
    )
    whole_nax, whole_family = _run_attention(
        d, sliding_window=sliding_window, sinks=sinks
    )
    expected_decode = whole_nax[:decode_rows]
    if split_decode:
        decode = {
            **d,
            "query": d["query"][:decode_rows],
            "block_tables": d["block_tables"][:decode_rows],
            "kv_lens_arr": d["kv_lens_arr"][:decode_rows],
            "cu_seqlens_q": mx.arange(decode_rows + 1, dtype=mx.int32),
        }
        expected_decode, _ = _run_attention(
            decode,
            decode_rows=decode_rows,
            decode_context=decode_context,
            sliding_window=sliding_window,
            sinks=sinks,
            decode_lengths=decode_lengths,
        )

    assert whole_family == "nax_prefill"
    assert mixed_family == (
        "mixed_nax_prefill_decode" if split_decode else "nax_prefill"
    )
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


@pytest.mark.parametrize("prefill_kernel", ["tiled", "nax"])
def test_split_decode_prefix_is_bounded_by_decode_context(
    prefill_kernel: str, request
) -> None:
    """An oversized allocation bound must not enlarge the decode split.

    4M tokens would need 8192 P512 partitions and overflow reducer shared
    memory; the actual 4096-token decode context needs only eight.
    """
    if prefill_kernel == "tiled":
        request.getfixturevalue("force_tiled_prefill")
        expected_family = "mixed_prefill_decode"
    else:
        request.getfixturevalue("nax_prefill")
        expected_family = "mixed_nax_prefill_decode"
    allocation_bound = 4 * 1024 * 1024
    decode_context = 4096
    mx.random.seed(1061)
    d = _make_cache_and_inputs(
        300, 8, 32, [(1, decode_context), (2, 2)], dtype=mx.bfloat16
    )
    # Rows may address the whole allocation: pad the table to the bound.
    used = d["block_tables"].shape[1]
    d["block_tables"] = mx.concatenate(
        [
            d["block_tables"],
            mx.zeros((2, allocation_bound // BLOCK_SIZE - used), mx.int32),
        ],
        axis=1,
    )
    mx.eval(d["block_tables"])

    ops = get_ops()
    previous = ops._set_paged_dispatch_diagnostics(True)
    try:
        tight, tight_family = _run_attention(
            d, decode_rows=1, decode_context=decode_context
        )
        oversized, oversized_family = _run_attention(
            d,
            decode_rows=1,
            decode_context=decode_context,
            max_seq_len=allocation_bound,
        )
    finally:
        mx.synchronize()
        ops._set_paged_dispatch_diagnostics(previous)

    assert tight_family == oversized_family == expected_family
    np.testing.assert_array_equal(oversized, tight)


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
        out = ops.paged_attention_primitive(
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
    out = ops.paged_attention_primitive(
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
