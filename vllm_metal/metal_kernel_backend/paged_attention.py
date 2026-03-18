# SPDX-License-Identifier: Apache-2.0
"""Paged attention using vendored Metal kernels dispatched through MLX.

Prefill: ``reshape_and_cache`` to write projected K/V into the paged cache,
then varlen Metal kernel (``paged_attention_v2_online``) for attention.

Decode: MLX projections + per-request RoPE, ``reshape_and_cache`` to write
the new token, then ``paged_attention_v1`` for zero-copy attention over
all cached K/V blocks.

All operations use MLX arrays end-to-end — no PyTorch MPS bridge.

Reuses ``PagedAttentionContext``, ``OffsetCache``, ``prepare_prefill_packed``,
``prepare_decode``, ``clear_context`` from ``paged_attention_common``.

Backend replacement guide
-------------------------
This module exists because there is no flash attention library for Apple
Silicon.  To swap in a new attention backend:

1. **Cache**: Create a new cache class that allocates per-layer KV storage
   addressable by block index.  Block allocation is managed externally
   by the scheduler's KV cache manager; the cache only stores tensors.

2. **Prefill**: Receives ``(queries, keys, values)`` after projection and
   RoPE, all as MLX arrays shaped ``(1, heads, seq_len, head_dim)``.
   Must compute attention output AND write K/V into the paged cache at
   positions given by ``ctx.slot_mapping``.

3. **Decode**: Receives ``(queries, keys, values)`` for the new token
   only, shaped ``(B, heads, 1, head_dim)``.  Must write the new K/V
   into the cache at ``ctx.slot_mapping``, then compute attention against
   ALL previously cached K/V using ``ctx.block_tables`` (list of block
   ids per request) and ``ctx.context_lens`` (total length including the
   new token).

4. **RoPE**: Currently applied inside this wrapper with per-request
   offsets (``ctx.offsets``).  If your kernel expects pre-RoPE'd inputs,
   keep this logic.  If it handles RoPE internally, remove it.

5. **OffsetCache**: The model runner passes ``OffsetCache`` objects as
   ``cache=`` to the model forward call.  These are NOT real KV caches —
   they are shims that satisfy mlx_lm's ``create_attention_mask`` /
   RoPE offset protocol.  The wrapper reads the real paged cache from
   its own ``_mk_kv_cache`` attribute, not from this argument.

6. **Patch function**: ``patch_model_attention_*(model, cache,
   block_size)`` walks transformer layers and replaces each attention
   module with a wrapper.  Keep this pattern — the worker calls it once
   at startup (``worker._setup_paged_attention``).

Files that do NOT need changes when replacing the backend:
- ``paged_attention_common.py`` (shared context / prepare functions)
- ``model_runner.py`` (only uses prepare/clear API)

Files that DO need changes:
- This module (attention wrapper + patch function)
- ``cache.py`` (cache class + layout)
- ``worker.py`` (``_setup_paged_attention`` — one function)
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
from vllm_metal.paged_attention_common import (
    PagedAttentionContext,
    find_layers_and_attr,
    get_context,
)

# ---------------------------------------------------------------------------
# Prefill attention (reshape_and_cache write + varlen Metal kernel)
# ---------------------------------------------------------------------------


def _metal_kernel_prefill_attention(
    attn_module: Any,
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    cache: MetalPagedKVCache,
    layer_idx: int,
    ctx: PagedAttentionContext,
) -> mx.array:
    """Prefill: B=1, L=prompt_len (single) or L=total_tokens (packed).

    Write K/V to paged cache via ``reshape_and_cache``, then compute
    attention using the varlen Metal kernel (``paged_attention_v2_online``).
    The kernel uses ``cu_seqlens_q`` to locate each sequence's query tokens
    and enforces causal masking internally — no dense mask needed.
    """
    B, n_heads, L, head_dim = queries.shape  # noqa: N806

    # RoPE — per-request position reset
    if not hasattr(attn_module, "rope"):
        raise NotImplementedError(
            f"Attention module {type(attn_module).__name__} does not have a 'rope' "
            "attribute. Only RoPE-based models are supported by paged attention."
        )

    # NOTE: apply_packed_rope always uses offset=0 per request. Chunked
    # prefill will need per-request offsets (like decode) for continuation chunks.
    queries, keys = apply_packed_rope(attn_module, queries, keys, ctx.cu_seqlens)

    # Reshape to 3D: (1, heads, L, hd) → (L, heads, hd)
    q_3d = mx.contiguous(queries[0].transpose(1, 0, 2).astype(cache.dtype))
    k_3d = mx.contiguous(keys[0].transpose(1, 0, 2).astype(cache.dtype))
    v_3d = mx.contiguous(values[0].transpose(1, 0, 2).astype(cache.dtype))

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
    out = mx.zeros((L, n_heads, head_dim), dtype=cache.dtype)
    mx.eval(q_3d, k_3d, v_3d, slot_mapping, block_tables, seq_lens, cu_seqlens_q, out)

    ops = get_ops()

    # Write K/V into paged cache BEFORE attention — the kernel reads from
    # the paged cache via block_table, not from raw tensors.
    ops.reshape_and_cache(
        k_3d,
        v_3d,
        cache.key_caches[layer_idx],
        cache.value_caches[layer_idx],
        slot_mapping,
    )

    max_seq_len = max(ctx.context_lens)

    ops.paged_attention_v2_online(
        out,
        q_3d,
        cache.key_caches[layer_idx],
        cache.value_caches[layer_idx],
        cache.num_kv_heads,
        attn_module.scale,
        0.0,  # softcap (0 = disabled)
        block_tables,
        seq_lens,
        cu_seqlens_q,
        cache.block_size,
        max_seq_len,
        -1,  # sliding_window (-1 = disabled)
    )

    mx.synchronize()

    # output: (L, n_heads, head_dim) → (B, L, n_heads * head_dim)
    out = out.reshape(B, L, n_heads * head_dim)
    return attn_module.o_proj(out)


# ---------------------------------------------------------------------------
# Decode attention (reshape_and_cache + paged_attention_v1)
# ---------------------------------------------------------------------------


def _metal_kernel_decode_attention(
    attn_module: Any,
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    cache: MetalPagedKVCache,
    layer_idx: int,
    ctx: PagedAttentionContext,
) -> mx.array:
    """Batched decode: B=batch_size, L=1.

    Per-request RoPE, write new token via ``reshape_and_cache``,
    then zero-copy attention via ``paged_attention_v1``.
    """
    B = queries.shape[0]  # noqa: N806
    n_heads = queries.shape[1]
    head_dim = queries.shape[3]

    # Per-request RoPE
    if not hasattr(attn_module, "rope"):
        raise NotImplementedError(
            f"Attention module {type(attn_module).__name__} does not have a 'rope' "
            "attribute. Only RoPE-based models are supported by paged attention."
        )
    q_parts = []
    k_parts = []
    for i in range(B):
        q_parts.append(attn_module.rope(queries[i : i + 1], offset=ctx.offsets[i]))
        k_parts.append(attn_module.rope(keys[i : i + 1], offset=ctx.offsets[i]))
    queries = mx.concatenate(q_parts, axis=0)  # (B, heads, 1, head_dim)
    keys_new = mx.concatenate(k_parts, axis=0)  # (B, kv_heads, 1, head_dim)

    # Squeeze seq dim: (B, heads, 1, hd) → (B, heads, hd)
    q_3d = mx.contiguous(queries[:, :, 0, :].astype(cache.dtype))
    k_3d = mx.contiguous(keys_new[:, :, 0, :].astype(cache.dtype))
    v_3d = mx.contiguous(values[:, :, 0, :].astype(cache.dtype))

    slot_mapping = mx.array(ctx.slot_mapping, dtype=mx.int64)

    # Build block_tables and seq_lens
    max_blocks_per_seq = max(len(bt) for bt in ctx.block_tables)
    block_tables_list = [
        bt + [0] * (max_blocks_per_seq - len(bt)) for bt in ctx.block_tables
    ]
    block_tables = mx.array(block_tables_list, dtype=mx.int32)
    seq_lens = mx.array(ctx.context_lens, dtype=mx.int32)

    # Eval all inputs before kernel dispatch
    mx.eval(q_3d, k_3d, v_3d, slot_mapping, block_tables, seq_lens)

    ops = get_ops()

    # Write new K/V tokens into paged cache
    ops.reshape_and_cache(
        k_3d,
        v_3d,
        cache.key_caches[layer_idx],
        cache.value_caches[layer_idx],
        slot_mapping,
    )

    # Allocate output
    out = mx.zeros((B, n_heads, head_dim), dtype=cache.dtype)
    mx.eval(out)

    max_seq_len = max(ctx.context_lens)
    scale = attn_module.scale

    # Build cu_seqlens_q for varlen dispatch: decode has q_len=1 per sequence.
    cu_seqlens_q = mx.arange(B + 1, dtype=mx.int32)
    mx.eval(cu_seqlens_q)

    # Zero-copy paged attention (v2, online softmax, varlen-capable)
    ops.paged_attention_v2_online(
        out,
        q_3d,
        cache.key_caches[layer_idx],
        cache.value_caches[layer_idx],
        cache.num_kv_heads,
        scale,
        0.0,  # softcap (0 = disabled)
        block_tables,
        seq_lens,
        cu_seqlens_q,
        cache.block_size,
        max_seq_len,
        -1,  # sliding_window (-1 = disabled)
    )

    # Synchronize GPU: paged_attention_v2_online wrote to out's buffer via a raw
    # Metal dispatch that MLX's lazy graph doesn't track.  mx.eval(out) would
    # be a no-op here (out was already evaluated as zeros), so we must use
    # mx.synchronize() to flush the command encoder and wait for the kernel.
    mx.synchronize()
    out = out.reshape(B, 1, n_heads * head_dim)
    return attn_module.o_proj(out)


# ---------------------------------------------------------------------------
# Wrapper nn.Module
# ---------------------------------------------------------------------------


class MetalKernelPagedAttentionWrapper(nn.Module):
    """Wraps an mlx_lm Attention module to use native Metal paged KV.

    Uses ``object.__setattr__`` to bypass MLX nn.Module's ``__setattr__``.

    When no ``PagedAttentionContext`` is set, falls back to original attention.
    """

    def __init__(
        self,
        inner: nn.Module,
        layer_idx: int,
        kv_cache: MetalPagedKVCache,
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

        B, L, D = x.shape  # noqa: N806

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
                inner, queries, keys, values, kv_cache, layer_idx, ctx
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
    kv_cache: MetalPagedKVCache,
    block_size: int,
) -> int:
    """Walk model layers and replace each attention module with a
    ``MetalKernelPagedAttentionWrapper``.

    Returns the number of patched layers.
    """
    layer_list, attn_attr = find_layers_and_attr(model)
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
