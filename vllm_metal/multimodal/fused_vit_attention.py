# SPDX-License-Identifier: Apache-2.0
"""Qwen3-VL vision-tower attention wiring for encoder varlen attention."""

from __future__ import annotations

import logging

import mlx.core as mx

logger = logging.getLogger(__name__)

_ENCODER_VARLEN_HEAD_DIMS = frozenset({64, 80, 96, 128})
_FUSED_SDPA_PAD_TARGETS = (64, 80, 128)
_PATCHED = False


def _baseline_segment_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    cu_seqlens: mx.array,
    scale: float,
) -> mx.array:
    """Per-segment ``ensure_fused_sdpa`` loop (mlx_vlm baseline)."""
    from mlx_vlm.models.base import ensure_fused_sdpa

    splits = [
        mx.split(tensor, cu_seqlens[1:-1].tolist(), axis=2) for tensor in (q, k, v)
    ]
    attn_outputs = [
        ensure_fused_sdpa(q_seg, k_seg, v_seg, scale)
        for q_seg, k_seg, v_seg in zip(*splits, strict=True)
    ]
    return mx.concatenate(attn_outputs, axis=2)


def _pad_head_dim_if_needed(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    head_dim: int,
) -> tuple[mx.array, mx.array, mx.array, int, int | None]:
    """Pad Q/K/V head dim to a supported encoder-varlen specialization."""
    if head_dim in _ENCODER_VARLEN_HEAD_DIMS:
        return q, k, v, head_dim, None

    target = next((t for t in _FUSED_SDPA_PAD_TARGETS if head_dim <= t), None)
    if target is None:
        raise ValueError(
            "fused ViT attention: unsupported head_dim "
            f"{head_dim} (supported: {sorted(_ENCODER_VARLEN_HEAD_DIMS)})."
        )

    pad = [(0, 0)] * (q.ndim - 1) + [(0, target - head_dim)]
    return mx.pad(q, pad), mx.pad(k, pad), mx.pad(v, pad), target, head_dim


def _fused_varlen_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    cu_seqlens: mx.array,
    scale: float,
) -> mx.array:
    """Single-call dense varlen encoder attention (Metal primitive)."""
    import numpy as np

    from vllm_metal.metal import encoder_varlen_attention

    head_dim = int(q.shape[-1])
    q, k, v, head_dim, orig_head_dim = _pad_head_dim_if_needed(q, k, v, head_dim)

    q_pack = q[0].transpose(1, 0, 2)
    k_pack = k[0].transpose(1, 0, 2)
    v_pack = v[0].transpose(1, 0, 2)

    orig_dtype = q.dtype
    kernel_dtype = (
        orig_dtype if orig_dtype in (mx.float16, mx.bfloat16) else mx.bfloat16
    )
    if q_pack.dtype != kernel_dtype:
        q_pack = q_pack.astype(kernel_dtype)
        k_pack = k_pack.astype(kernel_dtype)
        v_pack = v_pack.astype(kernel_dtype)

    bounds = [int(x) for x in np.asarray(cu_seqlens)]
    max_seqlen = max(bounds[i + 1] - bounds[i] for i in range(len(bounds) - 1))

    out = encoder_varlen_attention(
        q_pack,
        k_pack,
        v_pack,
        cu_seqlens,
        max_seqlen=max_seqlen,
        softmax_scale=scale,
    )
    if orig_head_dim is not None:
        out = out[..., :orig_head_dim]
    if out.dtype != orig_dtype:
        out = out.astype(orig_dtype)

    return out.transpose(1, 0, 2)[None]


def qwen3_vl_vision_attention_forward(
    attn,
    x: mx.array,
    cu_seqlens: mx.array,
    rotary_pos_emb: mx.array | None = None,
    *,
    use_fused_varlen: bool = False,
) -> mx.array:
    """Qwen3-VL vision ``Attention`` forward with selectable attention core."""
    from mlx_vlm.models.qwen3_vl.vision import apply_rotary_pos_emb_vision

    seq_length = x.shape[0]
    qkv = attn.qkv(x).reshape(seq_length, 3, attn.num_heads, -1).transpose(1, 0, 2, 3)
    q, k, v = mx.split(qkv, 3)

    q = apply_rotary_pos_emb_vision(mx.expand_dims(q, 0), rotary_pos_emb)[0]
    k = apply_rotary_pos_emb_vision(mx.expand_dims(k, 0), rotary_pos_emb)[0]

    q = q.transpose(0, 2, 1, 3)
    k = k.transpose(0, 2, 1, 3)
    v = v.transpose(0, 2, 1, 3)

    if use_fused_varlen:
        core = _fused_varlen_attention(q, k, v, cu_seqlens, attn.scale)
    else:
        core = _baseline_segment_attention(q, k, v, cu_seqlens, attn.scale)

    output = core.transpose(0, 2, 1, 3).reshape(seq_length, -1)
    return attn.proj(output)


def patch_qwen3_vl_vision_attention() -> None:
    """Replace mlx_vlm Qwen3-VL vision ``Attention.__call__`` with varlen core."""
    global _PATCHED
    if _PATCHED:
        return

    from mlx_vlm.models.qwen3_vl import vision

    def _patched_call(
        self,
        x: mx.array,
        cu_seqlens: mx.array,
        rotary_pos_emb: mx.array | None = None,
    ) -> mx.array:
        return qwen3_vl_vision_attention_forward(
            self,
            x,
            cu_seqlens,
            rotary_pos_emb,
            use_fused_varlen=True,
        )

    vision.Attention.__call__ = _patched_call
    _PATCHED = True
    logger.info(
        "Patched mlx_vlm Qwen3-VL vision Attention to use encoder_varlen_attention"
    )
