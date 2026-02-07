# SPDX-License-Identifier: Apache-2.0
"""Paged attention using the HF ``kernels-community/paged-attention`` Metal
kernel for zero-copy decode.

Prefill: MLX inline SDPA (causal), then bridge K/V to MPS and call
``reshape_and_cache`` to write into the paged cache.

Decode: MLX projections + per-request RoPE, bridge Q/K/V to MPS, call
``reshape_and_cache`` then ``paged_attention_v1`` (zero-copy read from
block tables), bridge output back to MLX.

Reuses ``PagedAttentionContext``, ``OffsetCache``, ``prepare_prefill``,
``prepare_decode``, ``clear_context`` from ``mlx_backend.paged_attention``.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn
import torch

from vllm_metal.metal_kernel_backend.cache import MPSPagedKVCache
from vllm_metal.metal_kernel_backend.kernel_loader import get_paged_attention_ops
from vllm_metal.mlx_backend.paged_attention import (
    PagedAttentionContext,
    _find_layers_and_attr,
    get_context,
)
from vllm_metal.pytorch_backend.tensor_bridge import mlx_to_torch, torch_to_mlx


# ---------------------------------------------------------------------------
# Bridge helpers
# ---------------------------------------------------------------------------


def _mlx_to_mps(arr: mx.array, dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """Convert an MLX array to an MPS PyTorch tensor."""
    mx.eval(arr)
    t = mlx_to_torch(arr, device="cpu")
    return t.to(dtype=dtype, device="mps")


def _mps_to_mlx(t: torch.Tensor) -> mx.array:
    """Convert an MPS PyTorch tensor back to an MLX array."""
    return torch_to_mlx(t.cpu())


# ---------------------------------------------------------------------------
# Prefill attention (MLX SDPA + reshape_and_cache write)
# ---------------------------------------------------------------------------


def _metal_kernel_prefill_attention(
    attn_module: Any,
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    cache: MPSPagedKVCache,
    layer_idx: int,
    ctx: PagedAttentionContext,
    offset_cache: Any,
) -> mx.array:
    """Prefill: B=1, L=prompt_len.

    Inline causal SDPA in MLX, then write K/V to MPS paged cache via
    ``reshape_and_cache``.
    """
    B, _, L, _ = queries.shape

    # RoPE
    offset = offset_cache.offset if offset_cache is not None else 0
    queries = attn_module.rope(queries, offset=offset)
    keys = attn_module.rope(keys, offset=offset)

    # Causal SDPA (inline — K/V already in hand)
    attn_mask = "causal" if L > 1 else None
    output = mx.fast.scaled_dot_product_attention(
        queries, keys, values, scale=attn_module.scale, mask=attn_mask
    )

    # Write K/V into paged MPS cache via reshape_and_cache
    # keys/values: (1, kv_heads, L, head_dim) → (L, kv_heads, head_dim)
    k_flat = keys[0].transpose(1, 0, 2)  # (L, kv_heads, head_dim)
    v_flat = values[0].transpose(1, 0, 2)

    k_mps = _mlx_to_mps(k_flat, dtype=cache.dtype)
    v_mps = _mlx_to_mps(v_flat, dtype=cache.dtype)

    slot_mapping_mps = torch.tensor(ctx.slot_mapping, dtype=torch.long, device="mps")

    ops = get_paged_attention_ops()
    ops.reshape_and_cache(
        k_mps,
        v_mps,
        cache.key_caches[layer_idx],
        cache.value_caches[layer_idx],
        slot_mapping_mps,
        "auto",
        cache.k_scale_tensor,
        cache.v_scale_tensor,
    )

    # output: (B, heads, L, head_dim) → (B, L, heads, head_dim) → (B, L, D)
    output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
    return attn_module.o_proj(output)


# ---------------------------------------------------------------------------
# Decode attention (reshape_and_cache + paged_attention_v1)
# ---------------------------------------------------------------------------


def _metal_kernel_decode_attention(
    attn_module: Any,
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    cache: MPSPagedKVCache,
    layer_idx: int,
    ctx: PagedAttentionContext,
) -> mx.array:
    """Batched decode: B=batch_size, L=1.

    Per-request RoPE, write new token via ``reshape_and_cache``,
    then zero-copy attention via ``paged_attention_v1``.
    """
    B = queries.shape[0]
    n_heads = queries.shape[1]
    head_dim = queries.shape[3]

    # Per-request RoPE
    q_parts = []
    k_parts = []
    for i in range(B):
        q_parts.append(attn_module.rope(queries[i : i + 1], offset=ctx.offsets[i]))
        k_parts.append(attn_module.rope(keys[i : i + 1], offset=ctx.offsets[i]))
    queries = mx.concatenate(q_parts, axis=0)  # (B, heads, 1, head_dim)
    keys_new = mx.concatenate(k_parts, axis=0)  # (B, kv_heads, 1, head_dim)

    # Bridge Q, new K/V to MPS
    # (B, heads, 1, hd) → squeeze seq dim → (B, heads, hd)
    q_mps = _mlx_to_mps(queries[:, :, 0, :], dtype=cache.dtype)
    k_mps = _mlx_to_mps(keys_new[:, :, 0, :], dtype=cache.dtype)
    v_mps = _mlx_to_mps(values[:, :, 0, :], dtype=cache.dtype)

    slot_mapping_mps = torch.tensor(ctx.slot_mapping, dtype=torch.long, device="mps")

    ops = get_paged_attention_ops()

    # Write new K/V tokens into paged cache
    ops.reshape_and_cache(
        k_mps,
        v_mps,
        cache.key_caches[layer_idx],
        cache.value_caches[layer_idx],
        slot_mapping_mps,
        "auto",
        cache.k_scale_tensor,
        cache.v_scale_tensor,
    )

    # Build block_tables and seq_lens tensors
    max_blocks_per_seq = max(len(bt) for bt in ctx.block_tables)
    block_tables_list = [
        bt + [0] * (max_blocks_per_seq - len(bt)) for bt in ctx.block_tables
    ]
    block_tables_mps = torch.tensor(block_tables_list, dtype=torch.int32, device="mps")
    seq_lens_mps = torch.tensor(ctx.context_lens, dtype=torch.int32, device="mps")

    # Allocate output tensor
    out = torch.zeros(B, n_heads, head_dim, dtype=cache.dtype, device="mps")

    max_seq_len = max(ctx.context_lens)
    scale = attn_module.scale

    # Zero-copy paged attention
    ops.paged_attention_v1(
        out,
        q_mps,
        cache.key_caches[layer_idx],
        cache.value_caches[layer_idx],
        cache.num_kv_heads,
        scale,
        block_tables_mps,
        seq_lens_mps,
        cache.block_size,
        max_seq_len,
        None,  # alibi_slopes
        "auto",  # kv_cache_dtype
        cache.k_scale_tensor,
        cache.v_scale_tensor,
        0,  # tp_rank
        0,  # blocksparse_local_blocks
        0,  # blocksparse_vert_stride
        64,  # blocksparse_block_size
        0,  # blocksparse_head_sliding_step
    )

    # Bridge output back to MLX: (B, heads, hd) → (B, 1, heads*hd)
    out_mlx = _mps_to_mlx(out)  # (B, heads, head_dim)
    out_mlx = out_mlx.reshape(B, 1, n_heads * head_dim)
    return attn_module.o_proj(out_mlx)


# ---------------------------------------------------------------------------
# Wrapper nn.Module
# ---------------------------------------------------------------------------


class MetalKernelPagedAttentionWrapper(nn.Module):
    """Wraps an mlx_lm Attention module to use the HF Metal kernel for paged KV.

    Same pattern as ``PagedAttentionWrapper`` in ``mlx_backend/paged_attention.py``:
    uses ``object.__setattr__`` to bypass MLX nn.Module's ``__setattr__``.

    When no ``PagedAttentionContext`` is set, falls back to original attention.
    """

    def __init__(
        self,
        inner: nn.Module,
        layer_idx: int,
        kv_cache: MPSPagedKVCache,
        block_size: int,
    ) -> None:
        super().__init__()
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_mk_layer_idx", layer_idx)
        object.__setattr__(self, "_mk_kv_cache", kv_cache)
        object.__setattr__(self, "_mk_block_size", block_size)

    def __call__(self, x: mx.array, mask: Any = None, cache: Any = None) -> mx.array:
        ctx = get_context()
        if ctx is None:
            # No paged context → delegate to original attention
            return self._inner(x, mask=mask, cache=cache)

        inner = self._inner
        kv_cache = self._mk_kv_cache
        layer_idx = self._mk_layer_idx

        B, L, D = x.shape

        # Projections + reshape
        queries = inner.q_proj(x)
        keys = inner.k_proj(x)
        values = inner.v_proj(x)

        queries = queries.reshape(B, L, inner.n_heads, -1)
        keys = keys.reshape(B, L, inner.n_kv_heads, -1)
        values = values.reshape(B, L, inner.n_kv_heads, -1)

        # Qwen3 per-head RMSNorm before RoPE
        if hasattr(inner, "q_norm"):
            queries = inner.q_norm(queries)
        if hasattr(inner, "k_norm"):
            keys = inner.k_norm(keys)

        # transpose → (B, heads, L, head_dim)
        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        if ctx.is_prefill:
            return _metal_kernel_prefill_attention(
                inner, queries, keys, values, kv_cache, layer_idx, ctx, cache
            )
        else:
            return _metal_kernel_decode_attention(
                inner, queries, keys, values, kv_cache, layer_idx, ctx
            )


# ---------------------------------------------------------------------------
# Model patching
# ---------------------------------------------------------------------------


def patch_model_attention_metal_kernel(
    model: Any,
    kv_cache: MPSPagedKVCache,
    block_size: int,
) -> int:
    """Walk model layers and replace each attention module with a
    ``MetalKernelPagedAttentionWrapper``.

    Returns the number of patched layers.
    """
    layer_list, attn_attr = _find_layers_and_attr(model)
    patched = 0

    for layer_idx, layer in enumerate(layer_list):
        attn = getattr(layer, attn_attr)
        if isinstance(attn, MetalKernelPagedAttentionWrapper):
            # Already patched — update cache reference
            object.__setattr__(attn, "_mk_kv_cache", kv_cache)
            object.__setattr__(attn, "_mk_block_size", block_size)
            patched += 1
            continue

        wrapper = MetalKernelPagedAttentionWrapper(
            attn, layer_idx, kv_cache, block_size
        )
        setattr(layer, attn_attr, wrapper)
        patched += 1

    return patched
