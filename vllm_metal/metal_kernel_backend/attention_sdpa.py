# SPDX-License-Identifier: Apache-2.0
"""Scaled dot-product attention (SDPA) on Metal.

Supports MHA, GQA, and MQA as variants of the same kernel — the head ratio
between ``n_heads`` (queries) and ``n_kv_heads`` (keys/values) is handled
transparently by the Metal paged attention kernel.

Handles models whose attention module exposes:
- ``q_proj``, ``k_proj``, ``v_proj``, ``o_proj`` linear projections
- ``rope`` for rotary position embeddings
- ``n_heads``, ``n_kv_heads`` head counts
- Optionally ``q_norm``, ``k_norm`` (Qwen3 per-head RMSNorm before RoPE)

Covers: Qwen3, Llama, Mistral, and other standard transformer architectures.

All operations use MLX arrays end-to-end — no PyTorch MPS bridge.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from vllm_metal.metal import get_ops
from vllm_metal.metal_kernel_backend.cache import MetalPagedKVCache
from vllm_metal.metal_kernel_backend.packed_prefill_compat import (
    apply_packed_rope,
)
from vllm_metal.paged_attention_common import PagedAttentionContext


def is_sdpa(module: nn.Module) -> bool:
    """Return True if *module* is an SDPA attention layer (MHA, GQA, or MQA)."""
    return (
        hasattr(module, "q_proj")
        and hasattr(module, "k_proj")
        and hasattr(module, "v_proj")
        and hasattr(module, "o_proj")
    )


def sdpa_forward(
    inner: nn.Module,
    x: mx.array,
    ctx: PagedAttentionContext,
    kv_cache: MetalPagedKVCache,
    layer_idx: int,
) -> mx.array:
    """Full SDPA forward pass: project → norm → RoPE → Metal kernel.

    Handles MHA, GQA, and MQA uniformly — the head ratio between
    ``inner.n_heads`` and ``inner.n_kv_heads`` is passed to the Metal
    kernel which handles the broadcast internally.
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

    ops = get_ops()
    max_seq_len = max(ctx.context_lens)

    # Paged SDPA primitive: reshape_and_cache + paged_attention in one
    # eval_gpu.  Both Metal kernel dispatches go to the same command
    # encoder, so Metal guarantees the cache write completes before the
    # attention read.
    updated_k_cache = mx.array(0)
    updated_v_cache = mx.array(0)
    out = mx.array(0)
    ops.paged_sdpa_primitive(
        q_3d,
        k_3d,
        v_3d,
        kv_cache.key_caches[layer_idx],
        kv_cache.value_caches[layer_idx],
        slot_mapping,
        kv_cache.num_kv_heads,
        inner.scale,
        0.0,  # softcap (0 = disabled)
        block_tables,
        seq_lens,
        cu_seqlens_q,
        kv_cache.block_size,
        max_seq_len,
        -1,  # sliding_window (-1 = disabled)
        updated_k_cache,
        updated_v_cache,
        out,
    )

    # Evaluate the paged SDPA primitive for this layer.  Required because
    # copy_shared_buffer cache aliasing is not safe across a fully-lazy
    # 28-layer graph — MLX's buffer management may reorder or reuse
    # aliased buffers.  Removing this eval is tracked as future work.
    mx.eval(updated_k_cache, updated_v_cache, out)

    # Rebind cache references for next layer / decode step
    kv_cache.key_caches[layer_idx] = updated_k_cache
    kv_cache.value_caches[layer_idx] = updated_v_cache

    # output: (L, n_heads, head_dim) → (B, L, n_heads * head_dim)
    out = out.reshape(B, L, n_heads * head_dim)
    return inner.o_proj(out)
