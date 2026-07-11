# SPDX-License-Identifier: Apache-2.0
# Per-request RoPE helper for packed / unified forward passes.

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import mlx.core as mx
import mlx.nn as nn


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


def _supports_batched_offsets(rope_fn: object) -> bool:
    """Return whether a RoPE callable supports vectorized MLX offsets."""
    return isinstance(rope_fn, nn.RoPE) or (
        isinstance(rope_fn, partial) and rope_fn.func is mx.fast.rope
    )


def _apply_equal_length_batched_rope(
    rope_fn: Callable[..., mx.array],
    x: mx.array,
    segment_count: int,
    segment_len: int,
    batch_offsets: int | mx.array,
) -> mx.array:
    """Apply native RoPE to equal-length packed segments as one batch."""
    batch, heads, _, head_dim = x.shape

    if segment_len == 1:
        batched = mx.reshape(
            mx.transpose(x, (0, 2, 1, 3)),
            (batch * segment_count, heads, 1, head_dim),
        )
    else:
        batched = mx.reshape(
            mx.transpose(
                mx.reshape(
                    x,
                    (batch, heads, segment_count, segment_len, head_dim),
                ),
                (0, 2, 1, 3, 4),
            ),
            (batch * segment_count, heads, segment_len, head_dim),
        )

    if batch == 1 or isinstance(batch_offsets, int):
        effective_offsets = batch_offsets
    else:
        effective_offsets = mx.reshape(
            mx.broadcast_to(
                batch_offsets[None, :],
                (batch, segment_count),
            ),
            (-1,),
        )

    rotated = rope_fn(batched, offset=effective_offsets)

    if segment_len == 1:
        return mx.transpose(
            mx.reshape(
                rotated,
                (batch, segment_count, heads, head_dim),
            ),
            (0, 2, 1, 3),
        )

    return mx.reshape(
        mx.transpose(
            mx.reshape(
                rotated,
                (batch, segment_count, heads, segment_len, head_dim),
            ),
            (0, 2, 1, 3, 4),
        ),
        x.shape,
    )


def apply_precomputed_mrope(
    attn_module: object,
    queries: mx.array,
    keys: mx.array,
    position_embeddings: tuple[mx.array, mx.array],
) -> tuple[mx.array, mx.array]:
    """Apply caller-precomputed M-RoPE ``(cos, sin)`` embeddings to Q/K.

    Some VLM attention modules receive the rotary embeddings from their parent
    language model instead of exposing a ``rope`` or ``rotary_emb`` callable on
    the attention layer itself.  Keep that model-specific apply policy in this
    compat module so SDPA projection/cache code can stay model-agnostic.
    """
    cos, sin = position_embeddings

    rotary_emb = getattr(attn_module, "rotary_emb", None)
    if rotary_emb is not None and getattr(rotary_emb, "style", None) == "interleaved":
        from mlx_vlm.models.qwen3_5.language import apply_multimodal_rotary_pos_emb

        return apply_multimodal_rotary_pos_emb(queries, keys, cos, sin)

    rope_parameters = getattr(attn_module, "rope_parameters", None)
    if rope_parameters is not None and "mrope_section" in rope_parameters:
        from mlx_vlm.models.paddleocr_vl.language import (
            apply_multimodal_rotary_pos_emb,
        )

        return apply_multimodal_rotary_pos_emb(
            queries,
            keys,
            cos,
            sin,
            rope_parameters["mrope_section"],
        )

    if rotary_emb is not None:
        raise NotImplementedError(
            f"Attention module {type(attn_module).__name__} received "
            "precomputed M-RoPE embeddings but its rotary_emb style "
            f"{getattr(rotary_emb, 'style', None)!r} is not supported."
        )
    raise NotImplementedError(
        f"Attention module {type(attn_module).__name__} received precomputed "
        "M-RoPE embeddings but does not expose a supported `rotary_emb` or "
        "rope_parameters['mrope_section']."
    )


def apply_attention_rope(
    attn_module: object,
    queries: mx.array,
    keys: mx.array,
    cu_seqlens: list[int],
    offsets: list[int] | None = None,
    apply_keys: bool = True,
    *,
    positions: list[mx.array | None] | None = None,
    position_embeddings: tuple[mx.array, mx.array] | None = None,
) -> tuple[mx.array, mx.array]:
    """Apply the attention module's RoPE contract to packed Q/K tensors."""
    if position_embeddings is not None:
        rotated_q, rotated_k = apply_precomputed_mrope(
            attn_module, queries, keys, position_embeddings
        )
        return rotated_q, (rotated_k if apply_keys else keys)

    if not hasattr(attn_module, "rope") and not hasattr(attn_module, "rotary_emb"):
        raise NotImplementedError(
            f"Attention module {type(attn_module).__name__} does not have a "
            "'rope' or 'rotary_emb' attribute."
        )

    return apply_packed_rope(
        attn_module,
        queries,
        keys,
        cu_seqlens,
        offsets=offsets,
        apply_keys=apply_keys,
        positions=positions,
    )


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

    segment_count = len(cu_seqlens) - 1
    if (
        rope_fn is not None
        and positions is None
        and segment_count > 1
        and cu_seqlens[0] == 0
        and cu_seqlens[-1] == queries.shape[2]
        and (not apply_keys or cu_seqlens[-1] == keys.shape[2])
        and _supports_batched_offsets(rope_fn)
    ):
        segment_len = cu_seqlens[1]
        if segment_len > 0 and all(
            cu_seqlens[i + 1] == (i + 1) * segment_len for i in range(1, segment_count)
        ):
            batch_offsets: int | mx.array
            # MLX 0.31.2 corrupts rows after the first for [B, H, 1, D]
            # RoPE with a scalar offset (MLX #3494). Vector offsets select the
            # correct kernel, including when every request has the same offset.
            if segment_len == 1:
                batch_offsets = (
                    mx.zeros((segment_count,), dtype=mx.int32)
                    if offsets is None
                    else mx.array(offsets[:segment_count])
                )
            elif offsets is None:
                batch_offsets = 0
            elif all(offsets[i] == offsets[0] for i in range(1, segment_count)):
                batch_offsets = offsets[0]
            else:
                batch_offsets = mx.array(offsets[:segment_count])

            rotated_q = _apply_equal_length_batched_rope(
                rope_fn, queries, segment_count, segment_len, batch_offsets
            )
            if not apply_keys:
                return rotated_q, keys
            rotated_k = _apply_equal_length_batched_rope(
                rope_fn, keys, segment_count, segment_len, batch_offsets
            )
            return rotated_q, rotated_k

    q_parts = []
    k_parts = []
    for i in range(segment_count):
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
