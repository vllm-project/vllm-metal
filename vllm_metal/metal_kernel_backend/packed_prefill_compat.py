# SPDX-License-Identifier: Apache-2.0
# Per-request RoPE helper for packed / unified forward passes.

from __future__ import annotations

from collections.abc import Callable

import mlx.core as mx


def _apply_mrope_segment(
    rotary_emb: Callable[..., tuple[mx.array, mx.array]],
    q_seg: mx.array,
    k_seg: mx.array,
    offset: int,
) -> tuple[mx.array, mx.array]:
    """Apply M-RoPE (multimodal rotary embedding) to one packed segment.

    mlx_vlm's Qwen3_5RotaryEmbedding expects ``(x, position_ids)`` and returns
    ``(cos, sin)``; actual rotation is via ``apply_multimodal_rotary_pos_emb``.
    For text-only requests, position_ids are simple sequential integers tiled
    across the 3 M-RoPE sections.
    """
    # Model-specific import: only Qwen3.5 uses M-RoPE; must stay lazy so
    # non-Qwen3.5 models do not require mlx_vlm.models.qwen3_5 at import time.
    from mlx_vlm.models.qwen3_5.language import apply_multimodal_rotary_pos_emb

    seg_len = q_seg.shape[2]
    pos = mx.arange(offset, offset + seg_len)
    # M-RoPE: (3, 1, seg_len) — 3 sections, batch=1
    position_ids = mx.broadcast_to(pos[None, None, :], (3, 1, seg_len))
    cos, sin = rotary_emb(q_seg, position_ids)  # type: ignore[operator]
    return apply_multimodal_rotary_pos_emb(q_seg, k_seg, cos, sin)


def apply_packed_rope(
    attn_module: object,
    queries: mx.array,
    keys: mx.array,
    cu_seqlens: list[int],
    offsets: list[int] | None = None,
) -> tuple[mx.array, mx.array]:
    """Apply per-request RoPE for packed sequences.

    Each segment delimited by ``cu_seqlens`` gets its own RoPE application
    starting at the corresponding offset.  When *offsets* is ``None`` every
    segment starts at position 0 (pure prefill).  For unified prefill+decode
    batches, decode segments carry ``offset=seq_len`` while prefill segments
    keep ``offset=0``.

    Supports both mlx_lm's ``rope(x, offset=)`` API and mlx_vlm's
    ``rotary_emb(x, position_ids)`` M-RoPE API (Qwen3.5).
    """
    rope_fn = getattr(attn_module, "rope", None)
    rotary_emb = getattr(attn_module, "rotary_emb", None) if rope_fn is None else None

    q_parts = []
    k_parts = []
    for i in range(len(cu_seqlens) - 1):
        start = cu_seqlens[i]
        end = cu_seqlens[i + 1]
        off = offsets[i] if offsets is not None else 0
        q_seg = queries[:, :, start:end, :]
        k_seg = keys[:, :, start:end, :]

        if rope_fn is not None:
            # mlx_lm API: rope(x, offset=off) → rotated_x
            q_parts.append(rope_fn(q_seg, offset=off))
            k_parts.append(rope_fn(k_seg, offset=off))
        else:
            # mlx_vlm M-RoPE API: rotary_emb(x, position_ids) → (cos, sin)
            q_rot, k_rot = _apply_mrope_segment(rotary_emb, q_seg, k_seg, off)
            q_parts.append(q_rot)
            k_parts.append(k_rot)

    return mx.concatenate(q_parts, axis=2), mx.concatenate(k_parts, axis=2)
