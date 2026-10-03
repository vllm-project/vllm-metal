# SPDX-License-Identifier: Apache-2.0
"""Bidirectional image-block rows against the Metal kernel and an fp32 reference."""

from __future__ import annotations

import contextlib
import logging

import mlx.core as mx
import numpy as np
import pytest

from vllm_metal.attention.block_tables import build_block_tables
from vllm_metal.attention.context import PagedAttentionContext
from vllm_metal.attention.impls.bidi_prefill import (
    apply_bidirectional_segments,
    build_bidi_mask,
    gather_kv,
    slot_indices,
)
from vllm_metal.metal import get_ops

BLOCK = 16
HEADS, KV_HEADS, HD = 4, 2, 64
DTYPE = mx.float16


def _setup(seed: int, *, n: int, seq_len: int, num_blocks: int = 128):
    mx.random.seed(seed)
    key_cache = mx.random.normal((num_blocks, BLOCK, KV_HEADS, HD)).astype(DTYPE)
    value_cache = mx.random.normal((num_blocks, BLOCK, KV_HEADS, HD)).astype(DTYPE)
    query = mx.random.normal((n, HEADS, HD)).astype(DTYPE)
    nblocks = (seq_len + BLOCK - 1) // BLOCK
    table = mx.array([list(range(1, nblocks + 1))], dtype=mx.int32)
    mx.eval(key_cache, value_cache, query, table)
    return key_cache, value_cache, query, table


def _kernel(query, key_cache, value_cache, table, *, n, seq_len, window):
    out = mx.array(0)
    get_ops().paged_attention_primitive(
        query,
        key_cache,
        value_cache,
        KV_HEADS,
        HD**-0.5,
        0.0,
        table,
        mx.array([seq_len], dtype=mx.int32),
        mx.array([0, n], dtype=mx.int32),
        BLOCK,
        seq_len,
        window if window is not None else -1,
        out,
    )
    mx.eval(out)
    return out


@contextlib.contextmanager
def _captured_bidi_logs():
    """Records emitted by the splice module (its logger may not propagate)."""
    records: list[logging.LogRecord] = []

    class _Sink(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    logger = logging.getLogger("vllm_metal.attention.impls.bidi_prefill")
    sink = _Sink(level=logging.INFO)
    logger.addHandler(sink)
    previous = logger.level
    logger.setLevel(logging.INFO)
    try:
        yield records
    finally:
        logger.setLevel(previous)
        logger.removeHandler(sink)


def _kernel_multi(
    query, key_cache, value_cache, table, *, kv_lens, cu_seqlens_q, window
):
    """``_kernel`` for a whole batch: per-segment kv lens, query offsets, tables."""
    out = mx.array(0)
    get_ops().paged_attention_primitive(
        query,
        key_cache,
        value_cache,
        KV_HEADS,
        HD**-0.5,
        0.0,
        table,
        mx.array(kv_lens, dtype=mx.int32),
        mx.array(cu_seqlens_q, dtype=mx.int32),
        BLOCK,
        max(kv_lens),
        window if window is not None else -1,
        out,
    )
    mx.eval(out)
    return out


def _rows(cache: mx.array, table_row: list[int], lo: int, hi: int) -> np.ndarray:
    arr = np.array(cache.astype(mx.float32))
    return np.stack([arr[table_row[p // BLOCK], p % BLOCK] for p in range(lo, hi)])


def _ref_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray, mask: np.ndarray):
    n_rep = q.shape[1] // k.shape[1]
    k = np.repeat(k, n_rep, axis=1)
    v = np.repeat(v, n_rep, axis=1)
    scores = np.einsum("qhd,khd->hqk", q, k) * HD**-0.5
    scores = np.where(mask[None], scores, -1e30)
    scores -= scores.max(axis=-1, keepdims=True)
    probs = np.exp(scores)
    probs /= probs.sum(axis=-1, keepdims=True)
    return np.einsum("hqk,khd->qhd", probs, v)


def _ctx(n: int, seq_len: int, ranges) -> PagedAttentionContext:
    return PagedAttentionContext(
        slot_mapping=[],
        cu_seqlens=[0, n],
        context_lens=[seq_len],
        segment_bidi_ranges=[ranges],
        bidi_layer_kinds=frozenset({"sliding", "full"}),
    )


@pytest.mark.parametrize("window", [None, 128])
def test_assembled_causal_path_matches_kernel(window) -> None:
    """slot_indices + gather_kv + build_bidi_mask(no block) + SDPA == kernel rows."""
    n, seq_len = 96, 600
    key_cache, value_cache, query, table = _setup(0, n=n, seq_len=seq_len)
    out = _kernel(
        query, key_cache, value_cache, table, n=n, seq_len=seq_len, window=window
    )
    q_lo = seq_len - n
    k_lo = max(0, q_lo - window + 1) if window is not None else 0
    slots = slot_indices(table[0], BLOCK, k_lo, seq_len)
    keys = gather_kv(key_cache, slots, HD)
    values = gather_kv(value_cache, slots, HD)
    mask = mx.array(build_bidi_mask(q_lo, n, k_lo, seq_len - k_lo, (0, 0), window))
    got = mx.fast.scaled_dot_product_attention(
        query.transpose(1, 0, 2)[None],
        keys.transpose(1, 0, 2)[None],
        values.transpose(1, 0, 2)[None],
        scale=HD**-0.5,
        mask=mask[None, None],
    )[0].transpose(1, 0, 2)
    mx.eval(got)
    np.testing.assert_allclose(np.array(got), np.array(out), atol=1.5e-2, rtol=1e-2)


@pytest.mark.parametrize("window", [128, None])
def test_block_rows_are_recomputed_with_the_bidirectional_mask(window) -> None:
    """A 200-row block inside a 256-row chunk, longer than the 128 window."""
    n, seq_len = 256, 600
    b0, b1 = 400, 600
    key_cache, value_cache, query, table = _setup(1, n=n, seq_len=seq_len)
    out = _kernel(
        query, key_cache, value_cache, table, n=n, seq_len=seq_len, window=window
    )
    ctx = _ctx(n, seq_len, [(b0, b1)])
    got = apply_bidirectional_segments(
        out,
        query,
        key_cache,
        value_cache,
        block_tables=table,
        block_size=BLOCK,
        cu_seqlens=[0, n],
        context_lens=[seq_len],
        ctx=ctx,
        window=window,
        scale=HD**-0.5,
        head_dim=HD,
        softcap=0.0,
        sinks=None,
        turboquant=False,
    )
    mx.eval(got)
    q_lo = seq_len - n
    k_lo = max(0, b0 - window + 1) if window is not None else 0
    row_table = table[0].tolist()
    ref = _ref_attention(
        np.array(query.astype(mx.float32))[b0 - q_lo :],
        _rows(key_cache, row_table, k_lo, b1),
        _rows(value_cache, row_table, k_lo, b1),
        build_bidi_mask(b0, b1 - b0, k_lo, b1 - k_lo, (b0, b1), window),
    )
    np.testing.assert_allclose(np.array(got)[b0 - q_lo :], ref, atol=1.5e-2, rtol=1e-2)
    # Rows before the block keep the kernel result bit-for-bit.
    np.testing.assert_array_equal(
        np.array(got)[: b0 - q_lo], np.array(out)[: b0 - q_lo]
    )
    assert ctx.bidi_logged is True


def test_block_head_in_context_is_read_from_the_cache() -> None:
    """Prefix-hit shape: the block starts below q_lo, only its tail is recomputed."""
    n, seq_len, window = 64, 300, 128
    b0, b1 = 200, 300  # q_lo = 236: 36 block rows are already in the cache
    key_cache, value_cache, query, table = _setup(2, n=n, seq_len=seq_len)
    out = _kernel(
        query, key_cache, value_cache, table, n=n, seq_len=seq_len, window=window
    )
    got = apply_bidirectional_segments(
        out,
        query,
        key_cache,
        value_cache,
        block_tables=table,
        block_size=BLOCK,
        cu_seqlens=[0, n],
        context_lens=[seq_len],
        ctx=_ctx(n, seq_len, [(b0, b1)]),
        window=window,
        scale=HD**-0.5,
        head_dim=HD,
        softcap=0.0,
        sinks=None,
        turboquant=False,
    )
    mx.eval(got)
    q_lo = seq_len - n
    k_lo = max(0, q_lo - window + 1)
    row_table = table[0].tolist()
    ref = _ref_attention(
        np.array(query.astype(mx.float32)),
        _rows(key_cache, row_table, k_lo, b1),
        _rows(value_cache, row_table, k_lo, b1),
        build_bidi_mask(q_lo, n, k_lo, b1 - k_lo, (b0, b1), window),
    )
    np.testing.assert_allclose(np.array(got), ref, atol=1.5e-2, rtol=1e-2)


def test_ranges_outside_the_queries_leave_the_output_untouched() -> None:
    n, seq_len = 32, 200
    key_cache, value_cache, query, table = _setup(3, n=n, seq_len=seq_len)
    out = _kernel(
        query, key_cache, value_cache, table, n=n, seq_len=seq_len, window=128
    )
    ctx = _ctx(n, seq_len, [(10, 50)])
    got = apply_bidirectional_segments(
        out,
        query,
        key_cache,
        value_cache,
        block_tables=table,
        block_size=BLOCK,
        cu_seqlens=[0, n],
        context_lens=[seq_len],
        ctx=ctx,
        window=128,
        scale=HD**-0.5,
        head_dim=HD,
        softcap=0.0,
        sinks=None,
        turboquant=False,
    )
    assert got is out
    assert ctx.bidi_logged is False


def test_gather_follows_hybrid_block_size_translation() -> None:
    """vLLM block 64 → kernel block 32: slots must index the reshaped view."""
    cache64 = mx.random.normal((8, 64, KV_HEADS, HD)).astype(DTYPE)
    tables, kernel_bs = build_block_tables([[3, 5]], 64)
    view = cache64.reshape(-1, kernel_bs, KV_HEADS, HD)
    got = gather_kv(view, slot_indices(tables[0], kernel_bs, 10, 100), HD)
    expected = mx.stack([cache64[[3, 5][p // 64], p % 64] for p in range(10, 100)])
    mx.eval(got, expected)
    assert mx.array_equal(got, expected)


def test_softcap_and_turboquant_are_refused() -> None:
    n, seq_len = 8, 8
    key_cache, value_cache, query, table = _setup(4, n=n, seq_len=seq_len)
    out = mx.zeros((n, HEADS, HD), dtype=DTYPE)
    common = {
        "block_tables": table,
        "block_size": BLOCK,
        "cu_seqlens": [0, n],
        "context_lens": [seq_len],
        "ctx": _ctx(n, seq_len, [(0, 8)]),
        "window": None,
        "scale": 1.0,
        "head_dim": HD,
        "sinks": None,
    }
    with pytest.raises(NotImplementedError, match="softcap"):
        apply_bidirectional_segments(
            out, query, key_cache, value_cache, softcap=30.0, turboquant=False, **common
        )
    with pytest.raises(RuntimeError, match="TurboQuant"):
        apply_bidirectional_segments(
            out, query, key_cache, value_cache, softcap=0.0, turboquant=True, **common
        )


def test_two_blocks_in_one_segment_are_spliced_in_order() -> None:
    """Two disjoint image blocks in one segment are each recomputed in place."""
    n, seq_len, window = 256, 600, 128
    ranges = [(360, 420), (500, 560)]
    key_cache, value_cache, query, table = _setup(5, n=n, seq_len=seq_len)
    out = _kernel(
        query, key_cache, value_cache, table, n=n, seq_len=seq_len, window=window
    )
    ctx = _ctx(n, seq_len, ranges)
    got = apply_bidirectional_segments(
        out,
        query,
        key_cache,
        value_cache,
        block_tables=table,
        block_size=BLOCK,
        cu_seqlens=[0, n],
        context_lens=[seq_len],
        ctx=ctx,
        window=window,
        scale=HD**-0.5,
        head_dim=HD,
        softcap=0.0,
        sinks=None,
        turboquant=False,
    )
    mx.eval(got)
    got_np = np.array(got)
    out_np = np.array(out)
    q_lo = seq_len - n
    row_table = table[0].tolist()
    query_np = np.array(query.astype(mx.float32))

    for b0, b1 in ranges:
        row_start, row_end = b0 - q_lo, b1 - q_lo
        k_lo = max(0, b0 - window + 1)
        ref = _ref_attention(
            query_np[row_start:row_end],
            _rows(key_cache, row_table, k_lo, b1),
            _rows(value_cache, row_table, k_lo, b1),
            build_bidi_mask(b0, b1 - b0, k_lo, b1 - k_lo, (b0, b1), window),
        )
        np.testing.assert_allclose(
            got_np[row_start:row_end], ref, atol=1.5e-2, rtol=1e-2
        )

    # Rows outside both blocks keep the kernel result bit-for-bit.
    outside = [(0, 16), (76, 156), (216, n)]
    for lo, hi in outside:
        np.testing.assert_array_equal(got_np[lo:hi], out_np[lo:hi])

    assert ctx.bidi_logged is True


def test_narrow_head_dim_rows_are_repadded() -> None:
    """A caller-supplied ``head_dim`` narrower than the cache pads the tail."""
    n, seq_len, window = 256, 600, 128
    b0, b1 = 400, 600
    narrow = 32
    key_cache, value_cache, query, table = _setup(6, n=n, seq_len=seq_len)
    out = _kernel(
        query, key_cache, value_cache, table, n=n, seq_len=seq_len, window=window
    )
    got = apply_bidirectional_segments(
        out,
        query,
        key_cache,
        value_cache,
        block_tables=table,
        block_size=BLOCK,
        cu_seqlens=[0, n],
        context_lens=[seq_len],
        ctx=_ctx(n, seq_len, [(b0, b1)]),
        window=window,
        scale=HD**-0.5,
        head_dim=narrow,
        softcap=0.0,
        sinks=None,
        turboquant=False,
    )
    mx.eval(got)
    assert got.shape == out.shape
    assert got.dtype == out.dtype

    got_np = np.array(got)
    q_lo = seq_len - n
    row_start, row_end = b0 - q_lo, b1 - q_lo
    k_lo = max(0, b0 - window + 1)
    row_table = table[0].tolist()
    ref = _ref_attention(
        np.array(query.astype(mx.float32))[row_start:row_end, :, :narrow],
        _rows(key_cache, row_table, k_lo, b1)[..., :narrow],
        _rows(value_cache, row_table, k_lo, b1)[..., :narrow],
        build_bidi_mask(b0, b1 - b0, k_lo, b1 - k_lo, (b0, b1), window),
    )
    np.testing.assert_array_equal(got_np[row_start:row_end, :, narrow:], 0)
    np.testing.assert_allclose(
        got_np[row_start:row_end, :, :narrow], ref, atol=1.5e-2, rtol=1e-2
    )


def test_two_segments_with_blocks_are_spliced_independently() -> None:
    """A decode row plus two prefill segments, each with its own block-table row.

    Every segment reads its own KV pages, so a splice must use segment ``i``'s
    block table and its own ``q_lo`` — mixing them up would silently attend to
    another request's cache.
    """
    window = 128
    # (query rows, context len, image block) per segment; the decode row first.
    segments = [(1, 50, None), (128, 300, (220, 280)), (96, 200, (150, 200))]
    cu_seqlens = [0, 1, 129, 225]
    context_lens = [50, 300, 200]
    assert cu_seqlens == [0, *np.cumsum([n for n, _, _ in segments]).tolist()]
    assert context_lens == [seq_len for _, seq_len, _ in segments]

    # One cache, disjoint page ranges per row: block 0 stays unused so a padded
    # table entry cannot alias a real page.
    mx.random.seed(7)
    row_tables: list[list[int]] = []
    next_block = 1
    for _, seq_len, _ in segments:
        count = (seq_len + BLOCK - 1) // BLOCK
        row_tables.append(list(range(next_block, next_block + count)))
        next_block += count
    width = max(len(row) for row in row_tables)
    table = mx.array(
        [row + [0] * (width - len(row)) for row in row_tables], dtype=mx.int32
    )
    key_cache = mx.random.normal((next_block, BLOCK, KV_HEADS, HD)).astype(DTYPE)
    value_cache = mx.random.normal((next_block, BLOCK, KV_HEADS, HD)).astype(DTYPE)
    query = mx.random.normal((cu_seqlens[-1], HEADS, HD)).astype(DTYPE)
    mx.eval(key_cache, value_cache, query, table)

    out = _kernel_multi(
        query,
        key_cache,
        value_cache,
        table,
        kv_lens=context_lens,
        cu_seqlens_q=cu_seqlens,
        window=window,
    )
    ctx = PagedAttentionContext(
        slot_mapping=[],
        cu_seqlens=cu_seqlens,
        context_lens=context_lens,
        segment_bidi_ranges=[
            None if block is None else [block] for _, _, block in segments
        ],
        bidi_layer_kinds=frozenset({"sliding", "full"}),
    )
    with _captured_bidi_logs() as records:
        got = apply_bidirectional_segments(
            out,
            query,
            key_cache,
            value_cache,
            block_tables=table,
            block_size=BLOCK,
            cu_seqlens=cu_seqlens,
            context_lens=context_lens,
            ctx=ctx,
            window=window,
            scale=HD**-0.5,
            head_dim=HD,
            softcap=0.0,
            sinks=None,
            turboquant=False,
        )
    mx.eval(got)
    got_np = np.array(got)
    out_np = np.array(out)
    query_np = np.array(query.astype(mx.float32))

    spliced: list[tuple[int, int]] = []
    for i, (n, seq_len, block) in enumerate(segments):
        if block is None:
            continue
        b0, b1 = block
        q_lo = seq_len - n
        a, b = max(q_lo, b0), min(seq_len, b1)
        row_start = cu_seqlens[i] + (a - q_lo)
        row_end = cu_seqlens[i] + (b - q_lo)
        k_lo = max(0, a - window + 1)
        ref = _ref_attention(
            query_np[row_start:row_end],
            _rows(key_cache, row_tables[i], k_lo, b),
            _rows(value_cache, row_tables[i], k_lo, b),
            build_bidi_mask(a, b - a, k_lo, b - k_lo, (b0, b1), window),
        )
        np.testing.assert_allclose(
            got_np[row_start:row_end], ref, atol=1.5e-2, rtol=1e-2
        )
        spliced.append((row_start, row_end))

    assert spliced == [(49, 109), (175, 225)]
    # Every row outside a block — the decode row included — is untouched.
    untouched = [(0, 49), (109, 175)]
    for lo, hi in untouched:
        np.testing.assert_array_equal(got_np[lo:hi], out_np[lo:hi])

    assert ctx.bidi_logged is True
    assert len(records) == 1
    assert "2 segment(s), 2 block(s), 110 row(s)" in records[0].getMessage()


@pytest.mark.parametrize("magnitude", [1.5, 2.0, 4.0])
@pytest.mark.parametrize("poisoned_group", [0, 1])
def test_tiled_prefill_rows_behind_masked_first_tile_stay_neutral(
    magnitude, poisoned_group
) -> None:
    """A KV tile fully masked for a row must not affect that row's max
    state or its V output (#876).  The masked V slots carry +/-fp16-max:
    a leaked weight swings the output far past the reference, while the
    correct kernel contributes exactly 0.  (inf would poison the
    *correct* kernel too — the PV MMA computes 0 * inf = NaN for
    P == 0 elements.)

    One threadgroup's first tile is poisoned per case.  Rows of an earlier
    threadgroup read those keys inside their window, so their reference is
    dominated by the poison and a relative tolerance says nothing about
    them; the tight comparison covers the poisoned group and every group
    after it.
    """
    n, seq_len, window = 64, 216, 96
    cfg = get_ops().tile_config(HD)
    assert cfg is not None
    bq, tile_kv = cfg
    assert n == 2 * bq
    row0_win_start = (seq_len - n) + 1 - window
    assert row0_win_start % tile_kv == 25
    key_cache, value_cache, _, table = _setup(0, n=n, seq_len=seq_len)
    mx.random.seed(3)
    query = (mx.ones((n, HEADS, HD)) * magnitude).astype(DTYPE)
    key_cache = (
        -magnitude * mx.ones(key_cache.shape) + 0.05 * mx.random.normal(key_cache.shape)
    ).astype(DTYPE)
    # Keys left of a threadgroup's first row's window start are masked for
    # every row in the group; poison their V slots.
    table_row = table[0].tolist()
    huge = np.finfo(np.float16).max
    vc = np.array(value_cache)
    first_row = poisoned_group * bq
    w0 = row0_win_start + first_row
    for p in range(w0 - w0 % tile_kv, w0):
        vc[table_row[p // BLOCK], p % BLOCK] = huge if p % 2 == 0 else -huge
    value_cache = mx.array(vc)
    mx.eval(query, key_cache, value_cache)
    got = _kernel(
        query, key_cache, value_cache, table, n=n, seq_len=seq_len, window=window
    )
    qi = np.arange(seq_len - n, seq_len)[:, None]
    ki = np.arange(seq_len)[None, :]
    mask = (ki <= qi) & (qi - ki < window)
    ref = _ref_attention(
        np.array(query.astype(mx.float32)),
        _rows(key_cache, table_row, 0, seq_len),
        _rows(value_cache, table_row, 0, seq_len),
        mask,
    )
    out = np.array(got)
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out[first_row:], ref[first_row:], atol=1.5e-2, rtol=1e-2)
