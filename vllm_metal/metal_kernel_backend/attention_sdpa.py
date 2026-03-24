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

import os
import time

import mlx.core as mx
import mlx.nn as nn

from vllm_metal.metal import get_ops, sync_profile
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


def _phase_name(ctx: PagedAttentionContext) -> str:
    if ctx.cu_seqlens and ctx.offsets:
        q_lens = [
            ctx.cu_seqlens[i + 1] - ctx.cu_seqlens[i]
            for i in range(len(ctx.cu_seqlens) - 1)
        ]
        if (
            q_lens
            and all(q_len == 1 for q_len in q_lens)
            and all(offset > 0 for offset in ctx.offsets)
        ):
            return "decode"
    return "prefill"


def _use_attention_primitive() -> bool:
    return os.getenv("VLLM_METAL_USE_ATTENTION_PRIMITIVE") == "1"


def _use_reshape_cache_primitive() -> bool:
    return os.getenv("VLLM_METAL_USE_RESHAPE_CACHE_PRIMITIVE") == "1"


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
    phase = _phase_name(ctx)
    sync_profile.ensure_registered()
    max_seq_len = max(ctx.context_lens)

    if _use_attention_primitive() and _use_reshape_cache_primitive():
        t0 = time.perf_counter()
        cache_write_token = ops.reshape_and_cache_primitive(
            k_3d,
            v_3d,
            kv_cache.key_caches[layer_idx],
            kv_cache.value_caches[layer_idx],
            slot_mapping,
        )
        sync_profile.record(
            f"{phase}.ops.reshape_and_cache_primitive",
            time.perf_counter() - t0,
        )

        t0 = time.perf_counter()
        out = ops.paged_attention_v2_online_primitive_with_dep(
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
            cache_write_token,
        )
        sync_profile.record(
            f"{phase}.ops.paged_attention_v2_online_primitive",
            time.perf_counter() - t0,
        )
    elif _use_attention_primitive():
        t0 = time.perf_counter()
        mx.eval(
            q_3d,
            k_3d,
            v_3d,
            slot_mapping,
            block_tables,
            seq_lens,
            cu_seqlens_q,
        )
        sync_profile.record(f"{phase}.eval", time.perf_counter() - t0)

        # Write K/V into paged cache BEFORE attention — the kernel reads from
        # the paged cache via block_table, not from raw tensors.
        t0 = time.perf_counter()
        ops.reshape_and_cache(
            k_3d,
            v_3d,
            kv_cache.key_caches[layer_idx],
            kv_cache.value_caches[layer_idx],
            slot_mapping,
        )
        sync_profile.record(f"{phase}.ops.reshape_and_cache", time.perf_counter() - t0)

        t0 = time.perf_counter()
        out = ops.paged_attention_v2_online_primitive(
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
        sync_profile.record(
            f"{phase}.ops.paged_attention_v2_online_primitive",
            time.perf_counter() - t0,
        )
    else:
        # Allocate output buffer before eval so we can materialize everything
        # in one call on the raw path.
        out = mx.zeros((L, n_heads, head_dim), dtype=kv_cache.dtype)
        t0 = time.perf_counter()
        mx.eval(
            q_3d,
            k_3d,
            v_3d,
            slot_mapping,
            block_tables,
            seq_lens,
            cu_seqlens_q,
            out,
        )
        sync_profile.record(f"{phase}.eval", time.perf_counter() - t0)

        # Write K/V into paged cache BEFORE attention — the kernel reads from
        # the paged cache via block_table, not from raw tensors.
        t0 = time.perf_counter()
        ops.reshape_and_cache(
            k_3d,
            v_3d,
            kv_cache.key_caches[layer_idx],
            kv_cache.value_caches[layer_idx],
            slot_mapping,
        )
        sync_profile.record(f"{phase}.ops.reshape_and_cache", time.perf_counter() - t0)

        t0 = time.perf_counter()
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
        sync_profile.record(
            f"{phase}.ops.paged_attention_v2_online", time.perf_counter() - t0
        )

        t0 = time.perf_counter()
        mx.synchronize()
        sync_profile.record(f"{phase}.synchronize", time.perf_counter() - t0)

    # output: (L, n_heads, head_dim) → (B, L, n_heads * head_dim)
    out = out.reshape(B, L, n_heads * head_dim)
    return inner.o_proj(out)
