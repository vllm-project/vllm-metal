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
    """Single-call dense varlen encoder attention (Metal primitive).

    Expects Q/K/V in kernel-native ``[total_tokens, num_heads, head_dim]`` layout.
    """
    import numpy as np

    from vllm_metal.metal import encoder_varlen_attention

    head_dim = int(q.shape[-1])
    q, k, v, head_dim, orig_head_dim = _pad_head_dim_if_needed(q, k, v, head_dim)

    orig_dtype = q.dtype
    kernel_dtype = (
        orig_dtype if orig_dtype in (mx.float16, mx.bfloat16) else mx.bfloat16
    )
    if q.dtype != kernel_dtype:
        q = q.astype(kernel_dtype)
        k = k.astype(kernel_dtype)
        v = v.astype(kernel_dtype)

    bounds = [int(x) for x in np.asarray(cu_seqlens)]
    max_seqlen = max(bounds[i + 1] - bounds[i] for i in range(len(bounds) - 1))

    out = encoder_varlen_attention(
        q,
        k,
        v,
        cu_seqlens,
        max_seqlen=max_seqlen,
        softmax_scale=scale,
    )
    if orig_head_dim is not None:
        out = out[..., :orig_head_dim]
    if out.dtype != orig_dtype:
        out = out.astype(orig_dtype)

    return out


def _apply_rotary_pos_emb_vision_shd(
    tensor: mx.array,
    freqs: mx.array,
) -> mx.array:
    """RoPE for ``[total_tokens, num_heads, head_dim]`` without batch dim churn."""
    from mlx_vlm.models.qwen3_vl.vision import rotate_half

    orig_dtype = tensor.dtype
    cos = mx.tile(mx.expand_dims(mx.cos(freqs), axis=1), (1, 1, 2))
    sin = mx.tile(mx.expand_dims(mx.sin(freqs), axis=1), (1, 1, 2))
    output = (tensor * cos) + (rotate_half(tensor) * sin)
    return output.astype(orig_dtype)


def _qkv_to_shd(qkv: mx.array) -> tuple[mx.array, mx.array, mx.array]:
    """``[S, 3, H, D]`` linear output -> Q/K/V in ``[S, H, D]``."""
    return qkv[:, 0], qkv[:, 1], qkv[:, 2]


def _shd_to_bhsd(
    q: mx.array,
    k: mx.array,
    v: mx.array,
) -> tuple[mx.array, mx.array, mx.array]:
    """``[S, H, D]`` -> ``[1, H, S, D]`` for baseline SDPA."""
    return (
        q.transpose(1, 0, 2)[None],
        k.transpose(1, 0, 2)[None],
        v.transpose(1, 0, 2)[None],
    )


def _fused_rope_varlen_core(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    freqs: mx.array,
    cu_seqlens: mx.array,
    scale: float,
) -> mx.array:
    """RoPE + varlen attention on ``[S, H, D]`` tensors."""
    q = _apply_rotary_pos_emb_vision_shd(q, freqs)
    k = _apply_rotary_pos_emb_vision_shd(k, freqs)
    return _fused_varlen_attention(q, k, v, cu_seqlens, scale)


def qwen3_vl_vision_attention_forward(
    attn,
    x: mx.array,
    cu_seqlens: mx.array,
    rotary_pos_emb: mx.array | None = None,
    *,
    use_fused_varlen: bool = False,
    use_fused_rope_varlen: bool = False,
) -> mx.array:
    """Qwen3-VL vision ``Attention`` forward with selectable attention core."""
    from mlx_vlm.models.qwen3_vl.vision import apply_rotary_pos_emb_vision

    seq_length = x.shape[0]
    qkv = attn.qkv(x).reshape(seq_length, 3, attn.num_heads, -1)
    q, k, v = _qkv_to_shd(qkv)

    if use_fused_varlen and use_fused_rope_varlen:
        core = _fused_rope_varlen_core(q, k, v, rotary_pos_emb, cu_seqlens, attn.scale)
        output = core.reshape(seq_length, -1)
    elif use_fused_varlen:
        q = apply_rotary_pos_emb_vision(mx.expand_dims(q, 0), rotary_pos_emb)[0]
        k = apply_rotary_pos_emb_vision(mx.expand_dims(k, 0), rotary_pos_emb)[0]
        core = _fused_varlen_attention(q, k, v, cu_seqlens, attn.scale)
        output = core.reshape(seq_length, -1)
    else:
        q = apply_rotary_pos_emb_vision(mx.expand_dims(q, 0), rotary_pos_emb)[0]
        k = apply_rotary_pos_emb_vision(mx.expand_dims(k, 0), rotary_pos_emb)[0]
        q, k, v = _shd_to_bhsd(q, k, v)
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
            use_fused_rope_varlen=True,
        )

    vision.Attention.__call__ = _patched_call
    _PATCHED = True
    logger.info(
        "Patched mlx_vlm Qwen3-VL vision Attention to use encoder_varlen_attention"
    )
