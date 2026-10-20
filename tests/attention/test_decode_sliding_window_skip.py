# SPDX-License-Identifier: Apache-2.0
"""Decode kernel and the sliding window: skip KV blocks left of the window.

The per-token (decode) kernel partitioned the context into PARTITION_SIZE
slices and masked keys outside the sliding window per element, so a decode
step on a Gemma 4 sliding layer (window 1024) read the whole context: at 32K
that is 32x the KV traffic the layer needs, on 25 of 30 layers.  These tests
pin the contract: results match the fp32 reference for the partitioned path
(few query tokens), the non-partitioned path (a large decode batch), a
window start inside a partition and inside a block, and the spec-decode
window-mode rows; and a zero window, whose block range can be empty, writes
a zero output on both paths.

The per-token reference cases run the kernel on caches whose blocks wholly
left of the window hold NaN.  The per-token body adds every token it reads
into the output with its softmax weight, 0 for a masked token, so a kernel
that read those blocks, even only to mask them, would carry 0 * NaN into its
output: a match with the reference proves the block range is bounded by the
window.  Window mode skips V for rows whose weight is 0, so NaN would not
reveal its reads; it takes its block range from the same first-block
computation.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from vllm_metal.metal import get_ops
from vllm_metal.metal.constants import PA_WINDOW_ROWS

BLOCK = 16
DTYPE = mx.float16
ATOL, RTOL = 1.5e-2, 1e-2


def _cache(seed: int, *, seq_lens: list[int], kv_heads: int, hd: int):
    """One cache holding every sequence's blocks; returns caches and tables."""
    mx.random.seed(seed)
    per_seq = [(n + BLOCK - 1) // BLOCK for n in seq_lens]
    num_blocks = sum(per_seq) + 1
    key_cache = mx.random.normal((num_blocks, BLOCK, kv_heads, hd)).astype(DTYPE)
    value_cache = mx.random.normal((num_blocks, BLOCK, kv_heads, hd)).astype(DTYPE)
    rows, nxt = [], 1
    for nb in per_seq:
        rows.append(list(range(nxt, nxt + nb)))
        nxt += nb
    width = max(per_seq)
    table = mx.array([r + [0] * (width - len(r)) for r in rows], dtype=mx.int32)
    mx.eval(key_cache, value_cache, table)
    return key_cache, value_cache, table, rows


def _kernel(
    query,
    key_cache,
    value_cache,
    table,
    *,
    kv_heads,
    kv_lens,
    cu_seqlens_q,
    window,
    **kwargs,
):
    hd = int(query.shape[-1])
    out = get_ops().paged_attention_primitive(
        query,
        key_cache,
        value_cache,
        kv_heads,
        hd**-0.5,
        0.0,
        table,
        mx.array(kv_lens, dtype=mx.int32),
        mx.array(cu_seqlens_q, dtype=mx.int32),
        BLOCK,
        max(kv_lens),
        window if window is not None else -1,
        **kwargs,
    )
    mx.eval(out)
    return out


def _reference(query, key_cache, value_cache, table_row, *, q_lo, seq_len, window):
    q = np.array(query.astype(mx.float32))
    kc = np.array(key_cache.astype(mx.float32))
    vc = np.array(value_cache.astype(mx.float32))
    k = np.stack([kc[table_row[p // BLOCK], p % BLOCK] for p in range(seq_len)])
    v = np.stack([vc[table_row[p // BLOCK], p % BLOCK] for p in range(seq_len)])
    n_rep = q.shape[1] // k.shape[1]
    k = np.repeat(k, n_rep, axis=1)
    v = np.repeat(v, n_rep, axis=1)
    n = q.shape[0]
    hd = q.shape[-1]
    qi = np.arange(q_lo, q_lo + n)[:, None]
    ki = np.arange(seq_len)[None, :]
    allowed = ki <= qi
    if window is not None:
        allowed &= (qi - ki) < window
    scores = np.einsum("qhd,khd->hqk", q, k) * hd**-0.5
    scores = np.where(allowed[None], scores, -1e30)
    m = scores.max(axis=-1, keepdims=True)
    probs = np.exp(scores - m)
    probs /= probs.sum(axis=-1, keepdims=True)
    return np.einsum("hqk,khd->qhd", probs, v)


def _nan_left_of_window(key_cache, value_cache, rows, q_positions, window):
    """Copies of the caches with NaN in each sequence's blocks that lie wholly
    left of its window; q_positions holds each sequence's query position."""
    k, v = np.array(key_cache), np.array(value_cache)
    for row, pos in zip(rows, q_positions, strict=True):
        skipped = row[: max(0, pos - window + 1) // BLOCK]
        k[skipped] = np.nan
        v[skipped] = np.nan
    return mx.array(k), mx.array(v)


@pytest.mark.parametrize("window", [96, 1000, 1024])
@pytest.mark.parametrize("seq_len", [1500, 4096, 4103])
def test_partitioned_decode_with_window_matches_reference(window, seq_len) -> None:
    """One token, few heads: the split-KV (partitioned) kernel.  Windows and
    lengths chosen so the window start lands mid-partition and mid-block."""
    heads, kv_heads, hd = 4, 2, 64
    # The split-KV gate is invisible from Python: assert the inputs that make
    # it take the partitioned kernel, so a gate change fails loudly here.
    ops = get_ops()
    assert heads < ops.min_decode_grid()
    assert seq_len > ops.PARTITION_SIZE
    key_cache, value_cache, table, rows = _cache(
        1, seq_lens=[seq_len], kv_heads=kv_heads, hd=hd
    )
    mx.random.seed(2)
    query = mx.random.normal((1, heads, hd)).astype(DTYPE)
    mx.eval(query)
    nan_keys, nan_values = _nan_left_of_window(
        key_cache, value_cache, rows, [seq_len - 1], window
    )
    got = _kernel(
        query,
        nan_keys,
        nan_values,
        table,
        kv_heads=kv_heads,
        kv_lens=[seq_len],
        cu_seqlens_q=[0, 1],
        window=window,
    )
    ref = _reference(
        query,
        key_cache,
        value_cache,
        rows[0],
        q_lo=seq_len - 1,
        seq_len=seq_len,
        window=window,
    )
    np.testing.assert_allclose(np.array(got), ref, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("magnitude", [1.5, 2.0, 4.0])
def test_partitioned_window_with_strongly_negative_scores(magnitude) -> None:
    """Partitions skipped by the window must be neutral in the split-KV reduce.

    A skipped partition that reports a max logit of 0 pins the reducer's
    global max, so a head whose in-window scores all sit far below 0 comes
    out attenuated or zero while the reference softmax is a near-uniform
    average of V (#837).  q = a*ones and k = -a*ones + noise put every
    in-window scaled score near -8*a*a.
    """
    heads, kv_heads, hd = 4, 2, 64
    seq_len, window = 1500, 96  # three partitions; the window is in the last
    ops = get_ops()
    assert heads < ops.min_decode_grid()
    assert seq_len > 2 * ops.PARTITION_SIZE
    key_cache, value_cache, table, rows = _cache(
        1, seq_lens=[seq_len], kv_heads=kv_heads, hd=hd
    )
    mx.random.seed(3)
    query = (mx.ones((1, heads, hd)) * magnitude).astype(DTYPE)
    key_cache = (
        -magnitude * mx.ones(key_cache.shape) + 0.05 * mx.random.normal(key_cache.shape)
    ).astype(DTYPE)
    mx.eval(query, key_cache)
    got = _kernel(
        query,
        key_cache,
        value_cache,
        table,
        kv_heads=kv_heads,
        kv_lens=[seq_len],
        cu_seqlens_q=[0, 1],
        window=window,
    )
    ref = _reference(
        query,
        key_cache,
        value_cache,
        rows[0],
        q_lo=seq_len - 1,
        seq_len=seq_len,
        window=window,
    )
    np.testing.assert_allclose(np.array(got), ref, atol=ATOL, rtol=RTOL)


def test_large_decode_batch_with_window_matches_reference() -> None:
    """Many single-token sequences: the grid exceeds the split-KV gate and the
    non-partitioned kernel runs; each sequence has its own context and window
    start."""
    heads, kv_heads, hd, window = 4, 2, 64, 200
    # The gate is 8 threadgroups per GPU core: 128 sequences of 4 heads still
    # take the split on a 65+ core GPU, so size the batch from the gate.
    ops = get_ops()
    num_seqs = max(128, -(-ops.min_decode_grid() // heads))  # ceil division
    assert heads * num_seqs >= ops.min_decode_grid()
    seq_lens = [300 + 37 * i for i in range(num_seqs)]
    key_cache, value_cache, table, rows = _cache(
        3, seq_lens=seq_lens, kv_heads=kv_heads, hd=hd
    )
    mx.random.seed(4)
    query = mx.random.normal((len(seq_lens), heads, hd)).astype(DTYPE)
    mx.eval(query)
    nan_keys, nan_values = _nan_left_of_window(
        key_cache, value_cache, rows, [n - 1 for n in seq_lens], window
    )
    got = np.array(
        _kernel(
            query,
            nan_keys,
            nan_values,
            table,
            kv_heads=kv_heads,
            kv_lens=seq_lens,
            cu_seqlens_q=list(range(len(seq_lens) + 1)),
            window=window,
        )
    )
    for i, n in enumerate(seq_lens):
        ref = _reference(
            query[i : i + 1],
            key_cache,
            value_cache,
            rows[i],
            q_lo=n - 1,
            seq_len=n,
            window=window,
        )
        np.testing.assert_allclose(got[i : i + 1], ref, atol=ATOL, rtol=RTOL)


def test_window_mode_rows_with_sliding_window_match_reference() -> None:
    """Spec-decode verify: several query rows per sequence in window mode; the
    rows' windows start at different keys and row 0's is the earliest."""
    heads, kv_heads, hd, window = 4, 2, 64, 96
    seq_len, q_len = 2048, 4
    # Partitioned window mode: the skipped partitions write one partial per row.
    ops = get_ops()
    assert heads * q_len < ops.min_decode_grid()
    assert seq_len > ops.PARTITION_SIZE
    key_cache, value_cache, table, rows = _cache(
        5, seq_lens=[seq_len], kv_heads=kv_heads, hd=hd
    )
    mx.random.seed(6)
    query = mx.random.normal((q_len, heads, hd)).astype(DTYPE)
    mx.eval(query)
    got = _kernel(
        query,
        key_cache,
        value_cache,
        table,
        kv_heads=kv_heads,
        kv_lens=[seq_len],
        cu_seqlens_q=[0, q_len],
        window=window,
        window_seqlen_q=q_len,
    )
    ref = _reference(
        query,
        key_cache,
        value_cache,
        rows[0],
        q_lo=seq_len - q_len,
        seq_len=seq_len,
        window=window,
    )
    np.testing.assert_allclose(np.array(got), ref, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("magnitude", [1.5, 2.0, 4.0])
def test_window_mode_row_with_fully_masked_partition_stays_neutral(
    magnitude,
) -> None:
    """Window mode reads blocks from the window start of its threadgroup's
    first row, so a partition can hold keys that only earlier rows attend to.
    The last row's window here starts exactly at a partition boundary: the
    partition before it is read for the row before it but fully masked for
    the last row, whose partial must not pin its reducer max at 0 (#837).
    Scores as in test_partitioned_window_with_strongly_negative_scores."""
    heads, kv_heads, hd = 4, 2, 64
    window, q_len = 96, 2 * PA_WINDOW_ROWS
    ops = get_ops()
    seq_len = 2 * ops.PARTITION_SIZE + window  # last row's window starts at 2P
    # The last row shares its threadgroup with the row before it, and the
    # batch takes the partitioned kernel.
    assert PA_WINDOW_ROWS >= 2
    assert heads * q_len < ops.min_decode_grid()
    key_cache, value_cache, table, rows = _cache(
        1, seq_lens=[seq_len], kv_heads=kv_heads, hd=hd
    )
    mx.random.seed(3)
    query = (mx.ones((q_len, heads, hd)) * magnitude).astype(DTYPE)
    key_cache = (
        -magnitude * mx.ones(key_cache.shape) + 0.05 * mx.random.normal(key_cache.shape)
    ).astype(DTYPE)
    mx.eval(query, key_cache)
    got = _kernel(
        query,
        key_cache,
        value_cache,
        table,
        kv_heads=kv_heads,
        kv_lens=[seq_len],
        cu_seqlens_q=[0, q_len],
        window=window,
        window_seqlen_q=q_len,
    )
    ref = _reference(
        query,
        key_cache,
        value_cache,
        rows[0],
        q_lo=seq_len - q_len,
        seq_len=seq_len,
        window=window,
    )
    np.testing.assert_allclose(np.array(got), ref, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("magnitude", [1.5, 2.0, 4.0])
def test_window_mode_row_with_fully_masked_block_stays_neutral(
    magnitude,
) -> None:
    """One level below the fully masked partition: window mode reads blocks
    from the window start of the threadgroup's first row, so a block can be
    scanned for an earlier row while lying wholly left of a later row's
    window, inside a partition that still holds real keys for it.  A fully
    masked block must not affect the row's max state or its V output:
    pinning the running max to 0 weights every later in-window key by
    exp2(score) instead of exp2(score - max) (#875), and a masked slot that
    still contributes V turns inf cache elements into NaN.

    seq_len 1506 puts row 0's window start on the last token of a block
    (window start mod BLOCK == BLOCK - 1), so the rest of that block is
    read for row 0 and fully masked for row 1.  Scores as in
    test_partitioned_window_with_strongly_negative_scores."""
    heads, kv_heads, hd = 4, 2, 64
    window, q_len = 96, 2 * PA_WINDOW_ROWS
    seq_len = 1506
    ops = get_ops()
    # The kernel masks token_idx < row_context_len - window with
    # row_context_len = (seq_len - q_len + 1) + r, so row 0's window starts
    # at 1503 - 96 = 1407, the last token of a block: that block is
    # scanned for row 0 and fully masked for row 1.
    row0_win_start = (seq_len - q_len + 1) - window
    assert row0_win_start % BLOCK == BLOCK - 1
    assert PA_WINDOW_ROWS >= 2
    assert heads * q_len < ops.min_decode_grid()
    assert seq_len > ops.PARTITION_SIZE
    key_cache, value_cache, table, rows = _cache(
        1, seq_lens=[seq_len], kv_heads=kv_heads, hd=hd
    )
    mx.random.seed(3)
    query = (mx.ones((q_len, heads, hd)) * magnitude).astype(DTYPE)
    key_cache = (
        -magnitude * mx.ones(key_cache.shape) + 0.05 * mx.random.normal(key_cache.shape)
    ).astype(DTYPE)
    # The masked block's V slots that no row may read get +/-inf; slot
    # BLOCK - 1 stays finite — it is row 0's window start, a legitimate read.
    vc = np.array(value_cache)
    masked_block = rows[0][row0_win_start // BLOCK]
    signs = np.where(np.arange(BLOCK - 1) % 2 == 0, np.inf, -np.inf)
    vc[masked_block, : BLOCK - 1] = signs[:, None, None]
    value_cache = mx.array(vc)
    mx.eval(query, key_cache, value_cache)
    got = _kernel(
        query,
        key_cache,
        value_cache,
        table,
        kv_heads=kv_heads,
        kv_lens=[seq_len],
        cu_seqlens_q=[0, q_len],
        window=window,
        window_seqlen_q=q_len,
    )
    # Masked slots never contribute to the true result, so the NumPy
    # reference runs over a finite copy with them zeroed.
    vc_ref = vc.copy()
    vc_ref[masked_block, : BLOCK - 1] = 0.0
    ref = _reference(
        query,
        key_cache,
        mx.array(vc_ref),
        rows[0],
        q_lo=seq_len - q_len,
        seq_len=seq_len,
        window=window,
    )
    out = np.array(got)
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out, ref, atol=ATOL, rtol=RTOL)


def _zero_window_output(query, key_cache, value_cache, table, **common):
    """Decode with sliding_window == 0, which masks every key: the neutral
    result is a zero output.  A full-attention call of the same shape runs
    first and is dropped, so MLX's buffer cache hands its non-zero buffers to
    the windowed call and anything that call leaves unwritten reads as stale
    data rather than zeros."""
    stale = np.array(
        _kernel(query, key_cache, value_cache, table, window=None, **common)
    )
    assert np.abs(stale).max() > 0
    return np.array(_kernel(query, key_cache, value_cache, table, window=0, **common))


@pytest.mark.parametrize("seq_len", [2048, 2050])
def test_zero_window_partitioned_decode_writes_zeros(seq_len) -> None:
    """On a block-aligned context every partition's block range is empty and
    each must still write its neutral partial; the unaligned length runs one
    fully masked block instead."""
    heads, kv_heads, hd = 4, 2, 64
    ops = get_ops()
    assert heads < ops.min_decode_grid()
    assert seq_len > ops.PARTITION_SIZE
    key_cache, value_cache, table, _ = _cache(
        9, seq_lens=[seq_len], kv_heads=kv_heads, hd=hd
    )
    mx.random.seed(10)
    query = mx.random.normal((1, heads, hd)).astype(DTYPE)
    mx.eval(query)
    got = _zero_window_output(
        query,
        key_cache,
        value_cache,
        table,
        kv_heads=kv_heads,
        kv_lens=[seq_len],
        cu_seqlens_q=[0, 1],
    )
    np.testing.assert_array_equal(got, 0)


def test_zero_window_single_pass_decode_writes_zeros() -> None:
    """The same boundary on the non-partitioned kernel: block-aligned
    sequences have an empty block range, unaligned ones one masked block."""
    heads, kv_heads, hd = 4, 2, 64
    ops = get_ops()
    num_seqs = max(128, -(-ops.min_decode_grid() // heads))  # ceil division
    assert heads * num_seqs >= ops.min_decode_grid()
    seq_lens = [BLOCK * (8 + i) + i % 2 for i in range(num_seqs)]
    key_cache, value_cache, table, _ = _cache(
        11, seq_lens=seq_lens, kv_heads=kv_heads, hd=hd
    )
    mx.random.seed(12)
    query = mx.random.normal((num_seqs, heads, hd)).astype(DTYPE)
    mx.eval(query)
    got = _zero_window_output(
        query,
        key_cache,
        value_cache,
        table,
        kv_heads=kv_heads,
        kv_lens=seq_lens,
        cu_seqlens_q=list(range(num_seqs + 1)),
    )
    np.testing.assert_array_equal(got, 0)
