# SPDX-License-Identifier: Apache-2.0
"""Tiled prefill kernel with per-row image-block ranges (mm_prefix) on Metal."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from vllm_metal.attention.block_tables import build_block_tables
from vllm_metal.attention.context import PagedAttentionContext
from vllm_metal.attention.impls.bidi_prefill import apply_bidirectional_segments
from vllm_metal.attention.impls.mm_prefix import mm_prefix_path
from vllm_metal.metal import get_ops

BLOCK = 16
HEADS, KV_HEADS = 4, 2
DTYPE = mx.float16
ATOL, RTOL = 1.5e-2, 1e-2


def _setup(seed: int, *, n: int, seq_len: int, hd: int = 64, num_blocks: int = 128):
    mx.random.seed(seed)
    key_cache = mx.random.normal((num_blocks, BLOCK, KV_HEADS, hd)).astype(DTYPE)
    value_cache = mx.random.normal((num_blocks, BLOCK, KV_HEADS, hd)).astype(DTYPE)
    query = mx.random.normal((n, HEADS, hd)).astype(DTYPE)
    nblocks = (seq_len + BLOCK - 1) // BLOCK
    table = mx.array([list(range(1, nblocks + 1))], dtype=mx.int32)
    mx.eval(key_cache, value_cache, query, table)
    return key_cache, value_cache, query, table


def _kernel(
    query,
    key_cache,
    value_cache,
    table,
    *,
    kv_lens: list[int],
    cu_seqlens_q: list[int],
    window: int | None,
    ranges: mx.array | None = None,
    block_size: int = BLOCK,
    **kwargs,
):
    hd = int(query.shape[-1])
    out = mx.array(0)
    get_ops().paged_attention_primitive(
        query,
        key_cache,
        value_cache,
        KV_HEADS,
        hd**-0.5,
        0.0,
        table,
        mx.array(kv_lens, dtype=mx.int32),
        mx.array(cu_seqlens_q, dtype=mx.int32),
        block_size,
        max(kv_lens),
        window if window is not None else -1,
        out,
        mm_prefix_ranges=ranges,
        **kwargs,
    )
    mx.eval(out)
    return out


def _range_rows(cu_seqlens, context_lens, blocks_per_segment) -> mx.array:
    """Inclusive ``(L, 2)`` int32 rows from half-open blocks, ``(-1, -1)`` elsewhere.

    Built inline so the kernel tests do not depend on the host module.
    """
    rows = np.full((cu_seqlens[-1], 2), -1, dtype=np.int32)
    for i, blocks in enumerate(blocks_per_segment):
        n = cu_seqlens[i + 1] - cu_seqlens[i]
        q_lo = context_lens[i] - n
        for b0, b1 in blocks or ():
            a, b = max(b0, q_lo), min(b1, q_lo + n)
            if b > a:
                rows[cu_seqlens[i] + a - q_lo : cu_seqlens[i] + b - q_lo] = (
                    b0,
                    b1 - 1,
                )
    return mx.array(rows)


def _hf_mask(q_lo: int, n: int, seq_len: int, blocks, window: int | None):
    """``(n, seq_len)`` bool: ``(causal OR same block) AND window`` over all keys."""
    q = np.arange(q_lo, q_lo + n)[:, None]
    k = np.arange(seq_len)[None, :]
    allowed = k <= q
    for b0, b1 in blocks:
        allowed |= ((q >= b0) & (q < b1)) & ((k >= b0) & (k < b1))
    if window is not None:
        allowed &= (q - k) < window
    return allowed


def _rows(cache: mx.array, table_row: list[int], lo: int, hi: int) -> np.ndarray:
    arr = np.array(cache.astype(mx.float32))
    return np.stack([arr[table_row[p // BLOCK], p % BLOCK] for p in range(lo, hi)])


def _ref_attention(q, k, v, mask, sinks=None):
    hd = q.shape[-1]
    n_rep = q.shape[1] // k.shape[1]
    k = np.repeat(k, n_rep, axis=1)
    v = np.repeat(v, n_rep, axis=1)
    scores = np.einsum("qhd,khd->hqk", q, k) * hd**-0.5
    scores = np.where(mask[None], scores, -1e30)
    if sinks is None:
        m = scores.max(axis=-1, keepdims=True)
        probs = np.exp(scores - m)
        probs /= probs.sum(axis=-1, keepdims=True)
    else:
        # The sink logit joins the running max and the denominator but
        # contributes no value row -- the softmax update the kernel runs
        # (see tests/test_paged_attention_sinks.py).
        sink = np.asarray(sinks, dtype=scores.dtype).reshape(-1, 1, 1)
        m = np.maximum(scores.max(axis=-1, keepdims=True), sink)
        probs = np.exp(scores - m)
        probs /= probs.sum(axis=-1, keepdims=True) + np.exp(sink - m)
    return np.einsum("hqk,khd->qhd", probs, v)


def _reference(
    query,
    key_cache,
    value_cache,
    table_row,
    *,
    q_lo,
    seq_len,
    blocks,
    window,
    sinks=None,
):
    """fp32 attention of rows ``[q_lo, q_lo + n)`` over keys ``[0, seq_len)``."""
    n = int(query.shape[0])
    return _ref_attention(
        np.array(query.astype(mx.float32)),
        _rows(key_cache, table_row, 0, seq_len),
        _rows(value_cache, table_row, 0, seq_len),
        _hf_mask(q_lo, n, seq_len, blocks, window),
        sinks=None if sinks is None else np.array(sinks),
    )


def test_supports_mm_prefix(monkeypatch) -> None:
    """The compiled ops advertise mm_prefix, and the dispatch probe sees it.

    If either side drifted, every image block would take the recompute with
    only the one-time warning to show for it.
    """
    monkeypatch.delenv("VLLM_METAL_MM_PREFIX_PATH", raising=False)
    ops = get_ops()
    assert ops.supports_mm_prefix() is True
    assert mm_prefix_path(ops) == "kernel"


@pytest.mark.parametrize("window", [128, None])
def test_block_longer_than_window_matches_reference_and_recompute(
    window, force_tiled_prefill
) -> None:
    """A 200-row block in a 256-row chunk; its tail lies right of the causal frontier.

    Without the extended tile-loop exit the kernel never visits the block's
    last tiles (error 0.05–0.29 against atol 0.015).  ``force_tiled_prefill``
    keeps the ranges-free baseline on the tiled kernel too: on an M5 it would
    otherwise run on NAX, and the bit-for-bit comparison below would compare
    two different kernels.
    """
    n, seq_len = 256, 600
    block = (400, 600)
    key_cache, value_cache, query, table = _setup(1, n=n, seq_len=seq_len)
    common = {"kv_lens": [seq_len], "cu_seqlens_q": [0, n], "window": window}
    plain = _kernel(query, key_cache, value_cache, table, **common)
    ranges = _range_rows([0, n], [seq_len], [[block]])
    got = _kernel(query, key_cache, value_cache, table, ranges=ranges, **common)
    q_lo = seq_len - n
    ref = _reference(
        query,
        key_cache,
        value_cache,
        table[0].tolist(),
        q_lo=q_lo,
        seq_len=seq_len,
        blocks=[block],
        window=window,
    )
    got_np, plain_np = np.array(got), np.array(plain)
    # fp32 HF reference.
    np.testing.assert_allclose(got_np, ref, atol=ATOL, rtol=RTOL)
    # the recompute path over the same cache agrees.
    ctx = PagedAttentionContext(
        slot_mapping=[],
        cu_seqlens=[0, n],
        context_lens=[seq_len],
        segment_bidi_ranges=[[block]],
        bidi_layer_kinds=frozenset({"sliding", "full"}),
    )
    recomputed = apply_bidirectional_segments(
        plain,
        query,
        key_cache,
        value_cache,
        block_tables=table,
        block_size=BLOCK,
        cu_seqlens=[0, n],
        context_lens=[seq_len],
        ctx=ctx,
        window=window,
        scale=64**-0.5,
        head_dim=64,
        softcap=0.0,
        sinks=None,
        turboquant=False,
    )
    mx.eval(recomputed)
    np.testing.assert_allclose(got_np, np.array(recomputed), atol=ATOL, rtol=RTOL)
    # rows outside the block are the plain kernel's, bit for bit.
    split = block[0] - q_lo
    np.testing.assert_array_equal(got_np[:split], plain_np[:split])
    # block rows really changed: a silently ignored buffer passes the check above.
    assert np.abs(got_np[split:] - plain_np[split:]).max() > 0.1


def test_sinks_with_ranges_use_the_sinks_input_slot(force_tiled_prefill) -> None:
    """Sinks and ranges together drive the ``_sk1_mp1`` pipeline.

    The ranges buffer is read from ``inputs[use_sinks_ ? 7 : 6]``: with the
    wrong index the kernel would take the sinks array for ranges and never
    say so.  Both terms are pinned against one fp32 reference that folds the
    sink into the softmax denominator.
    """
    n, seq_len, window = 256, 600, 128
    block = (400, 600)
    key_cache, value_cache, query, table = _setup(1, n=n, seq_len=seq_len)
    # One non-zero logit per query head, large enough that a dropped or
    # misread sink cannot hide inside ATOL/RTOL (it moves ~8% of the
    # elements past the tolerance band).
    sinks = mx.array([2.5, -1.0, 4.0, 1.5], dtype=mx.float32)
    assert sinks.size == HEADS
    common = {
        "kv_lens": [seq_len],
        "cu_seqlens_q": [0, n],
        "window": window,
        "sinks": sinks,
    }
    sinks_only = _kernel(query, key_cache, value_cache, table, **common)
    got = _kernel(
        query,
        key_cache,
        value_cache,
        table,
        ranges=_range_rows([0, n], [seq_len], [[block]]),
        **common,
    )
    q_lo = seq_len - n
    ref = _reference(
        query,
        key_cache,
        value_cache,
        table[0].tolist(),
        q_lo=q_lo,
        seq_len=seq_len,
        blocks=[block],
        window=window,
        sinks=sinks,
    )
    got_np, sinks_np = np.array(got), np.array(sinks_only)
    # fp32 HF reference, sink folded into the denominator.
    np.testing.assert_allclose(got_np, ref, atol=ATOL, rtol=RTOL)
    split = block[0] - q_lo
    # rows outside the block are the sinks-only kernel's, bit for bit.
    np.testing.assert_array_equal(got_np[:split], sinks_np[:split])
    # block rows really changed: a silently ignored buffer passes the check above.
    assert np.abs(got_np[split:] - sinks_np[split:]).max() > 0.1


def test_ranges_are_validated_before_dispatch() -> None:
    n, seq_len = 32, 64
    key_cache, value_cache, query, table = _setup(2, n=n, seq_len=seq_len)
    common = {"kv_lens": [seq_len], "cu_seqlens_q": [0, n], "window": None}
    ranges = _range_rows([0, n], [seq_len], [[(40, 64)]])
    with pytest.raises(ValueError, match="mm_prefix_ranges must be int32"):
        _kernel(
            query,
            key_cache,
            value_cache,
            table,
            ranges=ranges.astype(mx.int64),
            **common,
        )
    with pytest.raises(ValueError, match=r"shape \(query rows, 2\)"):
        _kernel(
            query,
            key_cache,
            value_cache,
            table,
            ranges=mx.concatenate([ranges, mx.zeros((1, 2), dtype=mx.int32)]),
            **common,
        )
    with pytest.raises(ValueError, match="tiled prefill kernel"):
        _kernel(
            query.astype(mx.float32),
            key_cache,
            value_cache,
            table,
            ranges=ranges,
            **common,
        )
    with pytest.raises(ValueError, match="tiled prefill kernel"):
        _kernel(
            query,
            key_cache.astype(mx.bfloat16),
            value_cache.astype(mx.bfloat16),
            table,
            ranges=ranges,
            **common,
        )
    with pytest.raises(ValueError, match="tiled prefill kernel"):
        _kernel(
            query,
            key_cache,
            value_cache,
            table,
            ranges=ranges,
            window_seqlen_q=n,
            **common,
        )
    # A pure-decode batch (one query row per segment) never reaches the tiled kernel.
    decode_ranges = _range_rows([0, 1], [seq_len], [[(seq_len - 1, seq_len)]])
    with pytest.raises(ValueError, match="tiled prefill kernel"):
        _kernel(
            query[:1],
            key_cache,
            value_cache,
            table,
            ranges=decode_ranges,
            kv_lens=[seq_len],
            cu_seqlens_q=[0, 1],
            window=None,
        )


def test_two_blocks_in_one_segment(force_tiled_prefill) -> None:
    n, seq_len, window = 256, 600, 128
    blocks = [(360, 420), (500, 560)]
    key_cache, value_cache, query, table = _setup(5, n=n, seq_len=seq_len)
    common = {"kv_lens": [seq_len], "cu_seqlens_q": [0, n], "window": window}
    plain = _kernel(query, key_cache, value_cache, table, **common)
    got = _kernel(
        query,
        key_cache,
        value_cache,
        table,
        ranges=_range_rows([0, n], [seq_len], [blocks]),
        **common,
    )
    q_lo = seq_len - n
    ref = _reference(
        query,
        key_cache,
        value_cache,
        table[0].tolist(),
        q_lo=q_lo,
        seq_len=seq_len,
        blocks=blocks,
        window=window,
    )
    got_np, plain_np = np.array(got), np.array(plain)
    np.testing.assert_allclose(got_np, ref, atol=ATOL, rtol=RTOL)
    for lo, hi in [(0, 16), (76, 156), (216, n)]:  # text rows between the blocks
        np.testing.assert_array_equal(got_np[lo:hi], plain_np[lo:hi])
    for b0, b1 in blocks:
        assert (
            np.abs(
                got_np[b0 - q_lo : b1 - q_lo] - plain_np[b0 - q_lo : b1 - q_lo]
            ).max()
            > 0.1
        )


def test_batch_with_a_decode_row_and_two_prefill_segments(force_tiled_prefill) -> None:
    """Segments with blocks at different positions and their own block-table rows.

    A segment-local row index instead of the global one would read another
    segment's ranges; a wrong table row would attend to another request's cache.
    """
    window = 128
    segments = [(1, 4096, None), (128, 300, (220, 280)), (96, 200, (150, 200))]
    cu_seqlens = [0, 1, 129, 225]
    context_lens = [4096, 300, 200]
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
    key_cache = mx.random.normal((next_block, BLOCK, KV_HEADS, 64)).astype(DTYPE)
    value_cache = mx.random.normal((next_block, BLOCK, KV_HEADS, 64)).astype(DTYPE)
    query = mx.random.normal((cu_seqlens[-1], HEADS, 64)).astype(DTYPE)
    mx.eval(key_cache, value_cache, query, table)
    ranges = _range_rows(
        cu_seqlens, context_lens, [None if b is None else [b] for _, _, b in segments]
    )
    common = {
        "kv_lens": context_lens,
        "cu_seqlens_q": cu_seqlens,
        "window": window,
        "num_decode_requests": 1,
        "num_decode_tokens": 1,
        "max_decode_context_len": 4096,
    }
    plain = _kernel(query, key_cache, value_cache, table, **common)
    got = _kernel(query, key_cache, value_cache, table, ranges=ranges, **common)
    got_np, plain_np = np.array(got), np.array(plain)
    for i, (n, seq_len, block) in enumerate(segments):
        lo, hi = cu_seqlens[i], cu_seqlens[i + 1]
        ref = _reference(
            query[lo:hi],
            key_cache,
            value_cache,
            row_tables[i],
            q_lo=seq_len - n,
            seq_len=seq_len,
            blocks=[] if block is None else [block],
            window=window,
        )
        np.testing.assert_allclose(got_np[lo:hi], ref, atol=ATOL, rtol=RTOL)
    for lo, hi in [(0, 1), (1, 49), (109, 129), (129, 175)]:  # decode + text rows
        np.testing.assert_array_equal(got_np[lo:hi], plain_np[lo:hi])
    for lo, hi in [(49, 109), (175, 225)]:  # block rows
        assert np.abs(got_np[lo:hi] - plain_np[lo:hi]).max() > 0.1


def test_block_head_in_context_is_read_from_the_cache(force_tiled_prefill) -> None:
    """Prefix-hit shape: the block starts below q_lo, only its tail is in the chunk."""
    n, seq_len, window = 64, 300, 128
    block = (200, 300)
    key_cache, value_cache, query, table = _setup(2, n=n, seq_len=seq_len)
    common = {"kv_lens": [seq_len], "cu_seqlens_q": [0, n], "window": window}
    plain = _kernel(query, key_cache, value_cache, table, **common)
    got = _kernel(
        query,
        key_cache,
        value_cache,
        table,
        ranges=_range_rows([0, n], [seq_len], [[block]]),
        **common,
    )
    ref = _reference(
        query,
        key_cache,
        value_cache,
        table[0].tolist(),
        q_lo=seq_len - n,
        seq_len=seq_len,
        blocks=[block],
        window=window,
    )
    np.testing.assert_allclose(np.array(got), ref, atol=ATOL, rtol=RTOL)
    # Every chunk row is inside the block; all but the last one gain keys.
    assert np.abs(np.array(got)[:-1] - np.array(plain)[:-1]).max() > 0.1


@pytest.mark.parametrize("hd", [256, 512])
def test_wide_heads_use_their_own_tile_config(hd, force_tiled_prefill) -> None:
    """HEAD_SIZE 256 (BQ 16) and 512 (BQ 8) instantiations carry the buffer too."""
    n, seq_len, window = 128, 400, 128
    block = (300, 400)
    key_cache, value_cache, query, table = _setup(
        9, n=n, seq_len=seq_len, hd=hd, num_blocks=32
    )
    common = {"kv_lens": [seq_len], "cu_seqlens_q": [0, n], "window": window}
    plain = _kernel(query, key_cache, value_cache, table, **common)
    got = _kernel(
        query,
        key_cache,
        value_cache,
        table,
        ranges=_range_rows([0, n], [seq_len], [[block]]),
        **common,
    )
    q_lo = seq_len - n
    ref = _reference(
        query,
        key_cache,
        value_cache,
        table[0].tolist(),
        q_lo=q_lo,
        seq_len=seq_len,
        blocks=[block],
        window=window,
    )
    got_np, plain_np = np.array(got), np.array(plain)
    np.testing.assert_allclose(got_np, ref, atol=ATOL, rtol=RTOL)
    split = block[0] - q_lo
    np.testing.assert_array_equal(got_np[:split], plain_np[:split])
    assert np.abs(got_np[split:] - plain_np[split:]).max() > 0.1


def test_hybrid_block_size_translation() -> None:
    """vLLM block 64 → kernel block 32: ranges are positions, tables are translated."""
    n, seq_len, window = 128, 256, 64
    block = (160, 256)
    vllm_table = [3, 5, 1, 6]  # four 64-token pages = 256 positions
    mx.random.seed(8)
    cache64_k = mx.random.normal((8, 64, KV_HEADS, 64)).astype(DTYPE)
    cache64_v = mx.random.normal((8, 64, KV_HEADS, 64)).astype(DTYPE)
    query = mx.random.normal((n, HEADS, 64)).astype(DTYPE)
    tables, kernel_bs = build_block_tables([vllm_table], 64)
    k_view = cache64_k.reshape(-1, kernel_bs, KV_HEADS, 64)
    v_view = cache64_v.reshape(-1, kernel_bs, KV_HEADS, 64)
    mx.eval(cache64_k, cache64_v, query, tables, k_view, v_view)
    got = _kernel(
        query,
        k_view,
        v_view,
        tables,
        ranges=_range_rows([0, n], [seq_len], [[block]]),
        kv_lens=[seq_len],
        cu_seqlens_q=[0, n],
        window=window,
        block_size=kernel_bs,
    )

    def rows64(cache: mx.array) -> np.ndarray:
        arr = np.array(cache.astype(mx.float32))
        return np.stack([arr[vllm_table[p // 64], p % 64] for p in range(seq_len)])

    ref = _ref_attention(
        np.array(query.astype(mx.float32)),
        rows64(cache64_k),
        rows64(cache64_v),
        _hf_mask(seq_len - n, n, seq_len, [block], window),
    )
    np.testing.assert_allclose(np.array(got), ref, atol=ATOL, rtol=RTOL)


def test_all_minus_one_rows_match_the_plain_kernel_bitwise(force_tiled_prefill) -> None:
    """A buffer with no block is inert: identical tiles, identical math."""
    n, seq_len = 96, 300
    key_cache, value_cache, query, table = _setup(3, n=n, seq_len=seq_len)
    common = {"kv_lens": [seq_len], "cu_seqlens_q": [0, n], "window": 128}
    plain = _kernel(query, key_cache, value_cache, table, **common)
    got = _kernel(
        query,
        key_cache,
        value_cache,
        table,
        ranges=mx.full((n, 2), -1, dtype=mx.int32),
        **common,
    )
    np.testing.assert_array_equal(np.array(got), np.array(plain))
