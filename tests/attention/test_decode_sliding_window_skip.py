# SPDX-License-Identifier: Apache-2.0
"""Decode kernel and the sliding window: skip KV blocks left of the window.

The per-token (decode) kernel partitioned the context into PARTITION_SIZE
slices and masked keys outside the sliding window per element, so a decode
step on a Gemma 4 sliding layer (window 1024) read the whole context: at 32K
that is 32x the KV traffic the layer needs, on 25 of 30 layers.  These tests
pin the contract: results match the fp32 reference for the partitioned path
(few query tokens), the non-partitioned path (a large decode batch), a
window start inside a partition and inside a block, and the spec-decode
window-mode rows; and a windowed decode step over a long context must be
several times cheaper than full attention.
"""

from __future__ import annotations

import time

import mlx.core as mx
import numpy as np
import pytest

from vllm_metal.metal import get_ops

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


def _kernel(query, key_cache, value_cache, table, *, kv_heads, kv_lens, cu_seqlens_q, window, **kwargs):
    hd = int(query.shape[-1])
    out = mx.array(0)
    get_ops().paged_attention_primitive(
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
        out,
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


@pytest.mark.parametrize("window", [96, 1000, 1024])
@pytest.mark.parametrize("seq_len", [1500, 4096, 4103])
def test_partitioned_decode_with_window_matches_reference(window, seq_len) -> None:
    """One token, few heads: the split-KV (partitioned) kernel.  Windows and
    lengths chosen so the window start lands mid-partition and mid-block."""
    heads, kv_heads, hd = 4, 2, 64
    key_cache, value_cache, table, rows = _cache(1, seq_lens=[seq_len], kv_heads=kv_heads, hd=hd)
    mx.random.seed(2)
    query = mx.random.normal((1, heads, hd)).astype(DTYPE)
    mx.eval(query)
    got = _kernel(query, key_cache, value_cache, table, kv_heads=kv_heads, kv_lens=[seq_len], cu_seqlens_q=[0, 1], window=window)
    ref = _reference(query, key_cache, value_cache, rows[0], q_lo=seq_len - 1, seq_len=seq_len, window=window)
    np.testing.assert_allclose(np.array(got), ref, atol=ATOL, rtol=RTOL)


def test_large_decode_batch_with_window_matches_reference() -> None:
    """Many single-token sequences: the grid exceeds the split-KV gate and the
    non-partitioned kernel runs; each sequence has its own context and window
    start."""
    heads, kv_heads, hd, window = 4, 2, 64, 200
    seq_lens = [300 + 37 * i for i in range(128)]
    key_cache, value_cache, table, rows = _cache(3, seq_lens=seq_lens, kv_heads=kv_heads, hd=hd)
    mx.random.seed(4)
    query = mx.random.normal((len(seq_lens), heads, hd)).astype(DTYPE)
    mx.eval(query)
    got = np.array(
        _kernel(query, key_cache, value_cache, table, kv_heads=kv_heads, kv_lens=seq_lens, cu_seqlens_q=list(range(len(seq_lens) + 1)), window=window)
    )
    for i, n in enumerate(seq_lens):
        ref = _reference(query[i : i + 1], key_cache, value_cache, rows[i], q_lo=n - 1, seq_len=n, window=window)
        np.testing.assert_allclose(got[i : i + 1], ref, atol=ATOL, rtol=RTOL)


def test_window_mode_rows_with_sliding_window_match_reference() -> None:
    """Spec-decode verify: several query rows per sequence in window mode; the
    rows' windows start at different keys and row 0's is the earliest."""
    heads, kv_heads, hd, window = 4, 2, 64, 96
    seq_len, q_len = 2048, 4
    key_cache, value_cache, table, rows = _cache(5, seq_lens=[seq_len], kv_heads=kv_heads, hd=hd)
    mx.random.seed(6)
    query = mx.random.normal((q_len, heads, hd)).astype(DTYPE)
    mx.eval(query)
    try:
        got = _kernel(query, key_cache, value_cache, table, kv_heads=kv_heads, kv_lens=[seq_len], cu_seqlens_q=[0, q_len], window=window, window_seqlen_q=q_len)
    except TypeError as exc:  # keyword not exposed by this build
        pytest.skip(f"window mode not exposed: {exc}")
    ref = _reference(query, key_cache, value_cache, rows[0], q_lo=seq_len - q_len, seq_len=seq_len, window=window)
    np.testing.assert_allclose(np.array(got), ref, atol=ATOL, rtol=RTOL)


def _median_seconds(fn, repeats: int = 5) -> float:
    fn()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return sorted(samples)[len(samples) // 2]


@pytest.mark.slow
def test_windowed_decode_reads_a_fraction_of_the_context() -> None:
    """A decode step on a 32K context with window 1024 needs 1/32 of the KV
    a full-attention step reads.  The mask-only kernel ran the whole context
    and landed near 1.0x; require at least 4x."""
    heads, kv_heads, hd, seq_len = 16, 8, 256, 32768
    key_cache, value_cache, table, _ = _cache(7, seq_lens=[seq_len], kv_heads=kv_heads, hd=hd)
    mx.random.seed(8)
    query = mx.random.normal((1, heads, hd)).astype(DTYPE)
    mx.eval(query)
    common = {"kv_heads": kv_heads, "kv_lens": [seq_len], "cu_seqlens_q": [0, 1]}
    full = _median_seconds(lambda: _kernel(query, key_cache, value_cache, table, window=None, **common))
    windowed = _median_seconds(lambda: _kernel(query, key_cache, value_cache, table, window=1024, **common))
    assert full / windowed >= 4.0, (
        f"windowed decode {windowed * 1e3:.2f} ms vs full {full * 1e3:.2f} ms: "
        "the block range is not bounded by the window"
    )
