# SPDX-License-Identifier: Apache-2.0
# SCAFFOLDING: remove when varlen kernel handles position encoding natively.
#
# Per-request RoPE helper for packed prefill.

from __future__ import annotations

import mlx.core as mx


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
