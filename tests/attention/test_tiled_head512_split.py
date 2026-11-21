# SPDX-License-Identifier: Apache-2.0
"""Tiled prefill kernel at HEAD_SIZE=512: the D-split configuration.

The 32 KB threadgroup budget gave HEAD_SIZE=512 a tile config of 8 query rows,
8 keys per tile and a single simdgroup, with 64 Q and 64 O fragments per
thread: one 512-wide full-attention layer of Gemma 4 cost 2.7 s for a 16K
prompt on an M3 Ultra, 7.7x the 256-wide instantiation for twice the work,
and the five such layers were 63-83% of prefill.  ``D_SPLIT`` lets four
simdgroups share the same 8 rows and split the head dimension between them:
each computes partial scores over its quarter of Q/K and owns a quarter of
O, with Q read from threadgroup memory per tile instead of held in
registers.  These tests pin the 512 path against the fp32 reference for every
feature the row mapping touches (chunking, sliding window, varlen batches,
sinks, mm_prefix image blocks) and require the 512 kernel to cost at most
three times the 256 one.
"""

from __future__ import annotations

import functools
import time

import mlx.core as mx
import numpy as np
import pytest

from vllm_metal.metal import get_ops

BLOCK = 16
HEADS, KV_HEADS = 4, 2
DTYPE = mx.float16
ATOL, RTOL = 1.5e-2, 1e-2


def _setup(seed: int, *, n: int, seq_len: int, hd: int):
    mx.random.seed(seed)
    num_blocks = (seq_len + BLOCK - 1) // BLOCK + 2
    key_cache = mx.random.normal((num_blocks, BLOCK, KV_HEADS, hd)).astype(DTYPE)
    value_cache = mx.random.normal((num_blocks, BLOCK, KV_HEADS, hd)).astype(DTYPE)
    query = mx.random.normal((n, HEADS, hd)).astype(DTYPE)
    nblocks = (seq_len + BLOCK - 1) // BLOCK
    table = mx.array([list(range(1, nblocks + 1))], dtype=mx.int32)
    mx.eval(key_cache, value_cache, query, table)
    return key_cache, value_cache, query, table


def _kernel(
    query, key_cache, value_cache, table, *, kv_lens, cu_seqlens_q, window, **kwargs
):
    hd = int(query.shape[-1])
    out = get_ops().paged_attention_primitive(
        query,
        key_cache,
        value_cache,
        KV_HEADS,
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


def _rows(cache, table_row, lo, hi):
    arr = np.array(cache.astype(mx.float32))
    return np.stack([arr[table_row[p // BLOCK], p % BLOCK] for p in range(lo, hi)])


def _mask(q_lo, n, seq_len, blocks, window):
    q = np.arange(q_lo, q_lo + n)[:, None]
    k = np.arange(seq_len)[None, :]
    allowed = k <= q
    for b0, b1 in blocks:
        allowed |= ((q >= b0) & (q < b1)) & ((k >= b0) & (k < b1))
    if window is not None:
        allowed &= (q - k) < window
    return allowed


def _reference(
    query,
    key_cache,
    value_cache,
    table_row,
    *,
    q_lo,
    seq_len,
    window,
    blocks=(),
    sinks=None,
):
    q = np.array(query.astype(mx.float32))
    k = _rows(key_cache, table_row, 0, seq_len)
    v = _rows(value_cache, table_row, 0, seq_len)
    hd = q.shape[-1]
    n_rep = q.shape[1] // k.shape[1]
    k = np.repeat(k, n_rep, axis=1)
    v = np.repeat(v, n_rep, axis=1)
    scores = np.einsum("qhd,khd->hqk", q, k) * hd**-0.5
    scores = np.where(
        _mask(q_lo, q.shape[0], seq_len, blocks, window)[None], scores, -1e30
    )
    if sinks is None:
        m = scores.max(axis=-1, keepdims=True)
        probs = np.exp(scores - m)
        probs /= probs.sum(axis=-1, keepdims=True)
    else:
        sink = np.asarray(sinks, dtype=scores.dtype).reshape(-1, 1, 1)
        m = np.maximum(scores.max(axis=-1, keepdims=True), sink)
        probs = np.exp(scores - m)
        probs /= probs.sum(axis=-1, keepdims=True) + np.exp(sink - m)
    return np.einsum("hqk,khd->qhd", probs, v)


def _range_rows(cu_seqlens, context_lens, blocks_per_segment) -> mx.array:
    rows = np.full((cu_seqlens[-1], 2), -1, dtype=np.int32)
    for i, blocks in enumerate(blocks_per_segment):
        n = cu_seqlens[i + 1] - cu_seqlens[i]
        q_lo = context_lens[i] - n
        for b0, b1 in blocks or ():
            a, b = max(b0, q_lo), min(b1, q_lo + n)
            if b > a:
                rows[cu_seqlens[i] + a - q_lo : cu_seqlens[i] + b - q_lo] = (b0, b1 - 1)
    return mx.array(rows)


@pytest.mark.parametrize("window", [None, 200])
@pytest.mark.parametrize(
    ("seq_len", "chunks"),
    [(300, [300]), (300, [150, 150]), (420, [200, 100, 120])],
)
def test_head512_matches_reference_for_every_chunking(
    window, seq_len, chunks, force_tiled_prefill
) -> None:
    key_cache, value_cache, query, table = _setup(1, n=seq_len, seq_len=seq_len, hd=512)
    table_row = table[0].tolist()
    done = 0
    for n in chunks:
        q_lo, kv_len = done, done + n
        got = _kernel(
            query[q_lo:kv_len],
            key_cache,
            value_cache,
            table,
            kv_lens=[kv_len],
            cu_seqlens_q=[0, n],
            window=window,
        )
        ref = _reference(
            query[q_lo:kv_len],
            key_cache,
            value_cache,
            table_row,
            q_lo=q_lo,
            seq_len=kv_len,
            window=window,
        )
        np.testing.assert_allclose(np.array(got), ref, atol=ATOL, rtol=RTOL)
        done = kv_len


def test_head512_varlen_batch_matches_reference(force_tiled_prefill) -> None:
    lens = [(180, 100), (330, 90)]  # (kv_len, query_len)
    parts = [_setup(5 + i, n=q, seq_len=kv, hd=512) for i, (kv, q) in enumerate(lens)]
    key_cache = mx.concatenate([parts[0][0], parts[1][0]], axis=0)
    value_cache = mx.concatenate([parts[0][1], parts[1][1]], axis=0)
    offset = int(parts[0][0].shape[0])
    max_blocks = max((kv + BLOCK - 1) // BLOCK for kv, _ in lens)
    rows = []
    for i, (kv, _) in enumerate(lens):
        nb = (kv + BLOCK - 1) // BLOCK
        row = [(offset if i else 0) + b for b in range(1, nb + 1)]
        rows.append(row + [0] * (max_blocks - nb))
    table = mx.array(rows, dtype=mx.int32)
    query = mx.concatenate([parts[0][2], parts[1][2]], axis=0)
    mx.eval(key_cache, value_cache, table, query)
    got = np.array(
        _kernel(
            query,
            key_cache,
            value_cache,
            table,
            kv_lens=[kv for kv, _ in lens],
            cu_seqlens_q=[0, lens[0][1], lens[0][1] + lens[1][1]],
            window=64,
        )
    )
    start = 0
    for i, (kv, q) in enumerate(lens):
        ref = _reference(
            query[start : start + q],
            key_cache,
            value_cache,
            rows[i],
            q_lo=kv - q,
            seq_len=kv,
            window=64,
        )
        np.testing.assert_allclose(got[start : start + q], ref, atol=ATOL, rtol=RTOL)
        start += q


def test_head512_sinks_match_reference(force_tiled_prefill) -> None:
    n = 200
    key_cache, value_cache, query, table = _setup(7, n=n, seq_len=n, hd=512)
    sinks = mx.array([0.5, -1.0, 2.0, 0.0], dtype=mx.float32)
    got = _kernel(
        query,
        key_cache,
        value_cache,
        table,
        kv_lens=[n],
        cu_seqlens_q=[0, n],
        window=None,
        sinks=sinks,
    )
    ref = _reference(
        query,
        key_cache,
        value_cache,
        table[0].tolist(),
        q_lo=0,
        seq_len=n,
        window=None,
        sinks=np.array(sinks),
    )
    np.testing.assert_allclose(np.array(got), ref, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("window", [None, 128])
def test_head512_mm_prefix_rows_match_reference(window, force_tiled_prefill) -> None:
    """Gemma 4 vision: a bidirectional image block on the 512-wide layers."""
    n, seq_len = 256, 400
    block = (300, 400)
    key_cache, value_cache, query, table = _setup(9, n=n, seq_len=seq_len, hd=512)
    ranges = _range_rows([0, n], [seq_len], [[block]])
    got = _kernel(
        query,
        key_cache,
        value_cache,
        table,
        kv_lens=[seq_len],
        cu_seqlens_q=[0, n],
        window=window,
        mm_prefix_ranges=ranges,
    )
    ref = _reference(
        query,
        key_cache,
        value_cache,
        table[0].tolist(),
        q_lo=seq_len - n,
        seq_len=seq_len,
        window=window,
        blocks=[block],
    )
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
def test_head512_costs_at_most_three_times_head256(force_tiled_prefill) -> None:
    """Twice the head dim is twice the MMA work; the old 512 config paid 7.7x
    because one simdgroup and 128 register-resident fragments left the GPU
    idle.  Allow 3x for tile overheads."""
    seq_len = 4096
    times = {}
    for hd in (256, 512):
        key_cache, value_cache, query, table = _setup(
            11, n=seq_len, seq_len=seq_len, hd=hd
        )
        times[hd] = _median_seconds(
            functools.partial(
                _kernel,
                query,
                key_cache,
                value_cache,
                table,
                kv_lens=[seq_len],
                cu_seqlens_q=[0, seq_len],
                window=None,
            )
        )
    ratio = times[512] / times[256]
    assert ratio <= 3.0, (
        f"512: {times[512] * 1e3:.0f} ms vs 256: {times[256] * 1e3:.0f} ms, ratio {ratio:.1f}"
    )
