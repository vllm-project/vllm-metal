# SPDX-License-Identifier: Apache-2.0
"""Standard multi-head attention (Qwen3, Llama, Mistral, etc.) on Metal.

Handles models whose attention module exposes:
- ``q_proj``, ``k_proj``, ``v_proj``, ``o_proj`` linear projections
- ``rope`` for rotary position embeddings
- ``n_heads``, ``n_kv_heads`` head counts
- Optionally ``q_norm``, ``k_norm`` (Qwen3 per-head RMSNorm before RoPE)

All operations use MLX arrays end-to-end — no PyTorch MPS bridge.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn

from vllm_metal.metal import get_ops
from vllm_metal.metal_kernel_backend.cache import MetalPagedKVCache
from vllm_metal.metal_kernel_backend.packed_prefill_compat import (
    apply_packed_rope,
)
from vllm_metal.paged_attention_common import PagedAttentionContext


def is_standard_mha(module: nn.Module) -> bool:
    """Return True if *module* looks like a standard MHA attention layer."""
    return (
        hasattr(module, "q_proj")
        and hasattr(module, "k_proj")
        and hasattr(module, "v_proj")
        and hasattr(module, "o_proj")
    )


def standard_mha_forward(
    inner: nn.Module,
    x: mx.array,
    ctx: PagedAttentionContext,
    kv_cache: MetalPagedKVCache,
    layer_idx: int,
) -> mx.array:
    """Full forward pass for standard MHA: project → norm → RoPE → Metal kernel.

    This combines the projection/reshape step (previously in the wrapper's
    ``__call__``) with the Metal kernel dispatch (previously
    ``_metal_kernel_prefill_attention``), so the entire attention-type-specific
    logic lives in one place.
    """
    B, L, D = x.shape  # noqa: N806

    # --- Projections + reshape ---
    queries = inner.q_proj(x).reshape(B, L, inner.n_heads, -1)
    keys = inner.k_proj(x).reshape(B, L, inner.n_kv_heads, -1)
    values = inner.v_proj(x).reshape(B, L, inner.n_kv_heads, -1)

    # Qwen3 per-head RMSNorm before RoPE
    if hasattr(inner, "q_norm"):
        queries = inner.q_norm(queries)
    if hasattr(inner, "k_norm"):
        keys = inner.k_norm(keys)

    # transpose → (B, heads, L, head_dim)
    queries = queries.transpose(0, 2, 1, 3)
    keys = keys.transpose(0, 2, 1, 3)
    values = values.transpose(0, 2, 1, 3)

    # --- RoPE (per-request position reset) ---
    if not hasattr(inner, "rope"):
        raise NotImplementedError(
            f"Attention module {type(inner).__name__} does not have a 'rope' "
            "attribute. Only RoPE-based models are supported by paged attention."
        )

    queries, keys = apply_packed_rope(
        inner,
        queries,
        keys,
        ctx.cu_seqlens,
        offsets=ctx.offsets if ctx.offsets else None,
    )

    # --- Metal kernel dispatch ---
    n_heads = queries.shape[1]
    head_dim = queries.shape[3]

    # Reshape to 3D: (1, heads, L, hd) → (L, heads, hd)
    q_3d = mx.contiguous(queries[0].transpose(1, 0, 2).astype(kv_cache.dtype))
    k_3d = mx.contiguous(keys[0].transpose(1, 0, 2).astype(kv_cache.dtype))
    v_3d = mx.contiguous(values[0].transpose(1, 0, 2).astype(kv_cache.dtype))

    slot_mapping = mx.array(ctx.slot_mapping, dtype=mx.int64)

    # Build block_tables and seq_lens from context
    max_blocks_per_seq = max(len(bt) for bt in ctx.block_tables)
    block_tables_list = [
        bt + [0] * (max_blocks_per_seq - len(bt)) for bt in ctx.block_tables
    ]
    block_tables = mx.array(block_tables_list, dtype=mx.int32)
    seq_lens = mx.array(ctx.context_lens, dtype=mx.int32)
    cu_seqlens_q = mx.array(ctx.cu_seqlens, dtype=mx.int32)

    # Allocate output buffer before eval so we can materialize everything in one call
    out = mx.zeros((L, n_heads, head_dim), dtype=kv_cache.dtype)
    mx.eval(q_3d, k_3d, v_3d, slot_mapping, block_tables, seq_lens, cu_seqlens_q, out)

    ops = get_ops()

    # Write K/V into paged cache BEFORE attention — the kernel reads from
    # the paged cache via block_table, not from raw tensors.
    ops.reshape_and_cache(
        k_3d,
        v_3d,
        kv_cache.key_caches[layer_idx],
        kv_cache.value_caches[layer_idx],
        slot_mapping,
    )

    max_seq_len = max(ctx.context_lens)

    ops.paged_attention_v2_online(
        out,
        q_3d,
        kv_cache.key_caches[layer_idx],
        kv_cache.value_caches[layer_idx],
        kv_cache.num_kv_heads,
        inner.scale,
        0.0,  # softcap (0 = disabled)
        block_tables,
        seq_lens,
        cu_seqlens_q,
        kv_cache.block_size,
        max_seq_len,
        -1,  # sliding_window (-1 = disabled)
    )

    mx.synchronize()

    # output: (L, n_heads, head_dim) → (B, L, n_heads * head_dim)
    out = out.reshape(B, L, n_heads * head_dim)
    return inner.o_proj(out)
