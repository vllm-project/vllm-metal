# SPDX-License-Identifier: Apache-2.0
"""Bidirectional attention inside image blocks for prefill segments.

Gemma 4 (HF ``create_masks_for_vision_model``) lets the soft tokens of one
image attend to each other on sliding-window layers: a query row inside an
image block ``[b0, b1)`` may see key ``k`` iff
``(k <= q or b0 <= k < b1) and (q - k < window)``.  The Metal paged kernel
only knows the causal + window mask, so the rows of image blocks are
recomputed here with ``mx.fast.scaled_dot_product_attention`` over K/V
gathered from the paged cache, and spliced into the kernel output.  Text
rows, decode rows and full-attention layers never enter this module.

DiffusionGemma's decoder canvas is a block too, but its sliding window is
anchored at the block start (``ctx.bidi_window_at_block_start``): every canvas
row sees the last ``window - 1`` keys before the canvas plus the whole canvas,
i.e. ``(b0 - k < window)`` in place of ``(q - k < window)``.  The kernel has no
such rule, so those rows always come through here on sliding layers.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import numpy as np
from vllm.logger import init_logger

from vllm_metal.attention.caches.storage import gather_cache_rows

logger = init_logger(__name__)


def intersecting_ranges(
    q_lo: int, q_hi: int, ranges: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Ranges (half-open, absolute) containing at least one query in [q_lo, q_hi)."""
    return [(start, end) for start, end in ranges if start < q_hi and end > q_lo]


def build_bidi_mask(
    q_lo: int,
    n: int,
    k_lo: int,
    num_keys: int,
    block: tuple[int, int],
    window: int | None,
    *,
    window_at_block_start: bool = False,
) -> np.ndarray:
    """``(n, num_keys)`` bool mask for query rows ``[q_lo, q_lo + n)`` of one block.

    Keys are absolute positions ``[k_lo, k_lo + num_keys)``.  ``(k <= q or
    b0 <= k < b1) and (q - k < window)`` — HF's ``(causal OR blockwise) AND
    sliding_window`` for rows that lie inside the block.  With
    ``window_at_block_start`` the window term is ``(b0 - k < window)``: every
    row keeps the same ``window - 1`` keys before the block and the whole block
    (mlx_vlm DiffusionGemma ``_make_decoder_masks``).
    """
    q = np.arange(q_lo, q_lo + n, dtype=np.int64)[:, None]
    k = np.arange(k_lo, k_lo + num_keys, dtype=np.int64)[None, :]
    b0, b1 = block
    allowed = (k <= q) | ((k >= b0) & (k < b1))
    if window is not None:
        allowed &= ((b0 if window_at_block_start else q) - k) < window
    return allowed


def slot_indices(
    block_table_row: mx.array, block_size: int, k_lo: int, k_hi: int
) -> mx.array:
    """Flat cache rows for absolute positions ``[k_lo, k_hi)``.

    ``bt[p // bs] * bs + p % bs`` with MLX ops on the device-side block table
    row (kernel format, kernel block size) — no host sync.
    """
    positions = mx.arange(k_lo, k_hi, dtype=mx.int32)
    blocks = block_table_row[positions // block_size]
    return blocks * block_size + positions % block_size


def gather_kv(
    cache: mx.array, slots: mx.array, head_dim: int, block_stride: int
) -> mx.array:
    """Rows ``slots`` of a ``(num_blocks, block_size, kv_heads, cache_hd)`` cache.

    ``block_stride`` is the cache's element stride between blocks.  Returns
    ``(len(slots), kv_heads, head_dim)``: the cache's zero-padded tail beyond
    the layer's real ``head_dim`` is sliced off.
    """
    return gather_cache_rows(cache, slots, block_stride)[:, :, :head_dim]


def apply_bidirectional_segments(
    out: mx.array,
    q_3d: mx.array,
    k_cache: mx.array,
    v_cache: mx.array,
    *,
    block_tables: mx.array,
    block_size: int,
    block_stride: int,
    cu_seqlens: list[int],
    context_lens: list[int],
    ctx: Any,
    window: int | None,
    scale: float,
    head_dim: int,
    softcap: float,
    sinks: mx.array | None,
    turboquant: bool,
) -> mx.array:
    """Recompute the image-block rows of prefill segments and splice them in.

    ``out`` and ``q_3d`` are ``(L, heads, cache_head_dim)``; the caches are the
    kernel-format arrays the kernel just read (``block_size`` is the kernel
    block size, ``block_stride`` their element stride between blocks and
    ``block_tables`` their tables).  Only rows inside an active
    block are recomputed; every other row keeps the kernel result.  The window
    runs from each row, or from the block start under
    ``ctx.bidi_window_at_block_start``.
    """
    if turboquant:
        raise RuntimeError(
            "bidirectional image attention cannot read a TurboQuant KV cache"
        )
    if softcap > 0 or sinks is not None:
        raise NotImplementedError(
            "bidirectional image attention does not support logit softcap or "
            "attention sinks"
        )
    per_segment = ctx.segment_bidi_ranges
    if per_segment is None:
        return out
    anchored = ctx.bidi_window_at_block_start

    width = int(out.shape[-1])
    pieces: list[mx.array] = []
    cursor = 0
    n_segments = n_blocks = n_rows = 0
    for i, ranges in enumerate(per_segment):
        if not ranges:
            continue
        q_start, q_end = cu_seqlens[i], cu_seqlens[i + 1]
        n = q_end - q_start
        seq_len = int(context_lens[i])
        q_lo, q_hi = seq_len - n, seq_len
        active = intersecting_ranges(q_lo, q_hi, ranges)
        if not active:
            continue
        n_segments += 1
        for b0, b1 in active:
            a, b = max(q_lo, b0), min(q_hi, b1)
            if b <= a:
                continue
            origin = b0 if anchored else a
            k_lo = max(0, origin - window + 1) if window is not None else 0
            slots = slot_indices(block_tables[i], block_size, k_lo, b)
            keys = gather_kv(k_cache, slots, head_dim, block_stride)
            values = gather_kv(v_cache, slots, head_dim, block_stride)
            mask = mx.array(
                build_bidi_mask(
                    a,
                    b - a,
                    k_lo,
                    b - k_lo,
                    (b0, b1),
                    window,
                    window_at_block_start=anchored,
                )
            )
            row_start = q_start + (a - q_lo)
            row_end = q_start + (b - q_lo)
            if row_start < cursor:
                raise RuntimeError("image ranges must be sorted and disjoint")
            q = q_3d[row_start:row_end, :, :head_dim].transpose(1, 0, 2)[None]
            seg = (
                mx.fast.scaled_dot_product_attention(
                    q,
                    keys.transpose(1, 0, 2)[None],
                    values.transpose(1, 0, 2)[None],
                    scale=scale,
                    mask=mask[None, None],
                )[0]
                .transpose(1, 0, 2)
                .astype(out.dtype)
            )
            if head_dim < width:
                seg = mx.pad(seg, [(0, 0), (0, 0), (0, width - head_dim)])
            pieces.append(out[cursor:row_start])
            pieces.append(seg)
            cursor = row_end
            n_blocks += 1
            n_rows += b - a
    if not pieces:
        return out
    pieces.append(out[cursor:])
    out = mx.concatenate(pieces, axis=0)
    if not ctx.bidi_logged:
        logger.info(
            "Metal: bidirectional image attention: %d segment(s), %d block(s), "
            "%d row(s)",
            n_segments,
            n_blocks,
            n_rows,
        )
        ctx.bidi_logged = True
    return out
