# SPDX-License-Identifier: Apache-2.0
# SCAFFOLDING: remove when varlen kernel is ready.
#
# Dense causal mask and per-request RoPE helpers for packed prefill.
# These are temporary — the varlen kernel will handle masking and
# position encoding natively, making this module unnecessary.

from __future__ import annotations

import mlx.core as mx


def build_packed_causal_mask(
    cu_seqlens: list[int],
    total_len: int,
    dtype: mx.Dtype = mx.float32,
) -> mx.array:
    """Build a block-diagonal causal mask for packed prefill.

    Each request only attends to its own tokens (causally).  Returns an
    additive mask of shape ``(1, 1, total_len, total_len)`` with 0 for
    allowed positions and ``-inf`` for blocked positions, suitable for
    ``mx.fast.scaled_dot_product_attention``.

    Args:
        dtype: Construct the mask directly in this dtype to avoid a
            transient float32 allocation followed by a cast.

    SCAFFOLDING: remove when varlen kernel is ready.
    """
    neg_inf = mx.array(-mx.inf, dtype=dtype)
    # Start with all-blocked, then open causal windows per request
    mask = mx.full((total_len, total_len), neg_inf)
    for i in range(len(cu_seqlens) - 1):
        start = cu_seqlens[i]
        end = cu_seqlens[i + 1]
        seq_len = end - start
        # Causal mask for this request's tokens
        causal = mx.triu(mx.full((seq_len, seq_len), neg_inf), k=1)
        mask[start:end, start:end] = causal
    return mask.reshape(1, 1, total_len, total_len)


def apply_packed_rope(
    attn_module: object,
    queries: mx.array,
    keys: mx.array,
    cu_seqlens: list[int],
) -> tuple[mx.array, mx.array]:
    """Apply per-request RoPE with position reset for packed prefill.

    SCAFFOLDING: remove when varlen kernel is ready.
    """
    q_parts = []
    k_parts = []
    for i in range(len(cu_seqlens) - 1):
        start = cu_seqlens[i]
        end = cu_seqlens[i + 1]
        q_parts.append(attn_module.rope(queries[:, :, start:end, :], offset=0))
        k_parts.append(attn_module.rope(keys[:, :, start:end, :], offset=0))
    return mx.concatenate(q_parts, axis=2), mx.concatenate(k_parts, axis=2)
