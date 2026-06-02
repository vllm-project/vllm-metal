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


def _apply_mrope_segment_with_positions(
    rotary_emb: Callable[..., tuple[mx.array, mx.array]],
    q_seg: mx.array,
    k_seg: mx.array,
    position_ids: mx.array,
) -> tuple[mx.array, mx.array]:
    """Apply M-RoPE with caller-supplied ``(3, 1, seg_len)`` positions.

    Used for multimodal segments where the position policy is non-trivial
    (image M-RoPE alignment shifts every section independently); the
    caller computes positions through ``adapter.get_mrope_input_positions``
    and slices the relevant chunk into this segment.
    """
    from mlx_vlm.models.qwen3_5.language import apply_multimodal_rotary_pos_emb

    cos, sin = rotary_emb(q_seg, position_ids)  # type: ignore[operator]
    return apply_multimodal_rotary_pos_emb(q_seg, k_seg, cos, sin)


def apply_packed_rope(
    attn_module: object,
    queries: mx.array,
    keys: mx.array,
    cu_seqlens: list[int],
    offsets: list[int] | None = None,
    apply_keys: bool = True,
    *,
    positions: list[mx.array | None] | None = None,
) -> tuple[mx.array, mx.array]:
    """Apply per-request RoPE for packed sequences.

    Each segment delimited by ``cu_seqlens`` gets its own RoPE application.
    When *offsets* is ``None`` every segment starts at position 0 (pure
    prefill).  For unified prefill+decode batches, decode segments carry
    ``offset=seq_len`` while prefill segments keep ``offset=0``.

    When *positions* is supplied, an entry of ``positions[i]`` that is not
    ``None`` takes precedence over ``offsets[i]`` and is fed to the
    M-RoPE rotary embedding directly as the ``(3, 1, seg_len)`` position
    array.  This is the path multimodal prefill takes: the caller computes
    Qwen3-VL M-RoPE positions on the full prompt then slices the relevant
    chunk for this segment.  Caller-supplied positions are only honored on
    the M-RoPE path (``attn_module.rotary_emb``); the mlx_lm ``rope(x,
    offset=)`` API has no position-array slot and raises if combined.

    When ``apply_keys`` is False, keys are returned unchanged (used by
    YOCO KV sharing where keys arrive already RoPE'd from a prior layer).

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
        seg_pos = positions[i] if positions is not None else None
        q_seg = queries[:, :, start:end, :]

        if rope_fn is not None:
            # mlx_lm API: rope(x, offset=off) → rotated_x
            if seg_pos is not None:
                raise NotImplementedError(
                    "apply_packed_rope: caller-provided positions are only "
                    "supported for M-RoPE (rotary_emb); the mlx_lm rope(x, "
                    "offset=) API has no position-array slot."
                )
            q_parts.append(rope_fn(q_seg, offset=off))
            if apply_keys:
                k_seg = keys[:, :, start:end, :]
                k_parts.append(rope_fn(k_seg, offset=off))
        else:
            # mlx_vlm M-RoPE API: rotary_emb(x, position_ids) → (cos, sin)
            k_seg = keys[:, :, start:end, :]
            if seg_pos is not None:
                q_rot, k_rot = _apply_mrope_segment_with_positions(
                    rotary_emb, q_seg, k_seg, seg_pos
                )
            else:
                q_rot, k_rot = _apply_mrope_segment(rotary_emb, q_seg, k_seg, off)
            q_parts.append(q_rot)
            if apply_keys:
                k_parts.append(k_rot)

    rotated_q = mx.concatenate(q_parts, axis=2) if q_parts else queries
    rotated_k = mx.concatenate(k_parts, axis=2) if apply_keys and k_parts else keys
    return rotated_q, rotated_k
