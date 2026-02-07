# SPDX-License-Identifier: Apache-2.0
"""Paged Attention v3 — patched MLX attention with block-paged KV cache.

Binds a PagedKVCache block pool to each attention layer and intercepts
the standard mlx_lm Attention.__call__ to scatter/gather KV through
the paged pool instead of using mlx_lm's built-in KVCache.

Usage:
    1. patch_model_attention(model, block_pool, block_size)
    2. Before each forward pass call prepare_prefill() or prepare_decode()
    3. Run model(input_ids, cache=offset_caches) as normal
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_causal_mask

from vllm_metal.mlx_backend.cache import PagedKVCache

# ---------------------------------------------------------------------------
# Global context (thread-local)
# ---------------------------------------------------------------------------

_thread_local = threading.local()


@dataclass
class PagedAttentionContext:
    """Context set before each forward pass, read by patched attention."""

    is_prefill: bool
    slot_mapping: list[int]
    # decode-only fields
    block_tables: list[list[int]] = field(default_factory=list)
    context_lens: list[int] = field(default_factory=list)
    offsets: list[int] = field(default_factory=list)


def set_context(ctx: PagedAttentionContext) -> None:
    _thread_local.paged_ctx = ctx


def get_context() -> PagedAttentionContext | None:
    return getattr(_thread_local, "paged_ctx", None)


def clear_context() -> None:
    _thread_local.paged_ctx = None


# ---------------------------------------------------------------------------
# OffsetCache — thin shim so the model's create_attention_mask / RoPE work
# ---------------------------------------------------------------------------


class OffsetCache:
    """Minimal cache-like object that provides ``offset`` and ``make_mask``.

    The mlx_lm model reads ``cache.offset`` for RoPE and calls
    ``cache.make_mask(N)`` (delegated from ``create_attention_mask``).
    We satisfy both without storing any KV data.
    """

    def __init__(self, offset: int) -> None:
        self.offset = offset

    # --- satisfy KVCache protocol expected by create_attention_mask ---------

    def make_mask(
        self,
        N: int,
        return_array: bool = False,
        window_size: int | None = None,
    ) -> Any:
        if N == 1:
            return None
        if return_array:
            return create_causal_mask(N, self.offset, window_size=window_size)
        return "causal"


# ---------------------------------------------------------------------------
# KV write / gather helpers
# ---------------------------------------------------------------------------


def write_kv_to_pool(
    block_pool: mx.array,
    layer_idx: int,
    keys: mx.array,
    values: mx.array,
    slot_mapping: list[int],
    block_size: int,
) -> None:
    """Scatter new K/V tokens into the block pool via *slot_mapping*.

    Args:
        block_pool: shape (num_blocks, num_layers, 2, block_size, kv_heads, head_dim)
        layer_idx: transformer layer index
        keys:   shape (num_tokens, kv_heads, head_dim)
        values: shape (num_tokens, kv_heads, head_dim)
        slot_mapping: flat slot indices, one per token
        block_size: tokens per block
    """
    for tok_idx, slot in enumerate(slot_mapping):
        block_idx = slot // block_size
        slot_offset = slot % block_size
        block_pool[block_idx, layer_idx, 0, slot_offset] = keys[tok_idx]
        block_pool[block_idx, layer_idx, 1, slot_offset] = values[tok_idx]


def gather_kv_batched(
    block_pool: mx.array,
    layer_idx: int,
    block_tables: list[list[int]],
    context_lens: list[int],
    block_size: int,
) -> tuple[mx.array, mx.array]:
    """Gather K/V for a batch of requests and left-pad to max length.

    Returns:
        keys:   (B, kv_heads, max_len, head_dim)
        values: (B, kv_heads, max_len, head_dim)
    """
    max_len = max(context_lens)
    all_keys: list[mx.array] = []
    all_values: list[mx.array] = []

    for req_idx, (blocks, ctx_len) in enumerate(zip(block_tables, context_lens)):
        # Gather this request's KV from its blocks
        k_parts: list[mx.array] = []
        v_parts: list[mx.array] = []
        remaining = ctx_len
        for blk in blocks:
            n = min(remaining, block_size)
            # block_pool[blk, layer_idx, 0/1, :n] → (n, kv_heads, head_dim)
            k_parts.append(block_pool[blk, layer_idx, 0, :n])
            v_parts.append(block_pool[blk, layer_idx, 1, :n])
            remaining -= n
            if remaining <= 0:
                break

        # (ctx_len, kv_heads, head_dim)
        k_seq = mx.concatenate(k_parts, axis=0)
        v_seq = mx.concatenate(v_parts, axis=0)

        # Left-pad to max_len
        pad_len = max_len - ctx_len
        if pad_len > 0:
            kv_heads = k_seq.shape[1]
            head_dim = k_seq.shape[2]
            pad = mx.zeros((pad_len, kv_heads, head_dim), dtype=k_seq.dtype)
            k_seq = mx.concatenate([pad, k_seq], axis=0)
            v_seq = mx.concatenate([pad, v_seq], axis=0)

        # (kv_heads, max_len, head_dim)
        all_keys.append(k_seq.transpose(1, 0, 2))
        all_values.append(v_seq.transpose(1, 0, 2))

    # Stack → (B, kv_heads, max_len, head_dim)
    return mx.stack(all_keys, axis=0), mx.stack(all_values, axis=0)


def _build_left_pad_mask(
    context_lens: list[int],
    max_len: int,
    dtype: mx.Dtype = mx.float32,
) -> mx.array:
    """Build a broadcastable mask for left-padded KV sequences.

    Returns shape (B, 1, 1, max_len) — True where KV is valid.
    """
    # For each request, positions [max_len - ctx_len, max_len) are valid
    pos = mx.arange(max_len)[None, :]  # (1, max_len)
    lens = mx.array(context_lens)[:, None]  # (B, 1)
    # valid where pos >= (max_len - ctx_len)
    mask = pos >= (max_len - lens)  # (B, max_len)
    return mask[:, None, None, :]  # (B, 1, 1, max_len)


# ---------------------------------------------------------------------------
# Patched Attention.__call__
# ---------------------------------------------------------------------------


def _paged_prefill_attention(
    attn_module,
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    pool: PagedKVCache,
    layer_idx: int,
    block_size: int,
    ctx: PagedAttentionContext,
    cache,
) -> mx.array:
    """Prefill: B=1, L=prompt_len. Inline SDPA, then write KV to blocks."""
    B, _, L, _ = queries.shape

    # RoPE (offset=0 for fresh prefill, or cache.offset if resuming)
    offset = cache.offset if cache is not None else 0
    queries = attn_module.rope(queries, offset=offset)
    keys = attn_module.rope(keys, offset=offset)

    # Standard causal SDPA with inline K/V (no gathering from pool needed)
    if L > 1:
        attn_mask = "causal"
    else:
        attn_mask = None

    output = mx.fast.scaled_dot_product_attention(
        queries, keys, values, scale=attn_module.scale, mask=attn_mask
    )

    # Write K/V to block pool
    # keys/values: (1, kv_heads, L, head_dim) → squeeze batch → (L, kv_heads, head_dim)
    k_flat = keys[0].transpose(1, 0, 2)  # (L, kv_heads, head_dim)
    v_flat = values[0].transpose(1, 0, 2)
    write_kv_to_pool(
        pool.block_pool, layer_idx, k_flat, v_flat, ctx.slot_mapping, block_size
    )

    # output: (B, heads, L, head_dim) → (B, L, heads, head_dim) → (B, L, D)
    output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
    return attn_module.o_proj(output)


def _paged_decode_attention(
    attn_module,
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    pool: PagedKVCache,
    layer_idx: int,
    block_size: int,
    ctx: PagedAttentionContext,
) -> mx.array:
    """Batched decode: B=batch_size, L=1. Per-request RoPE, scatter, gather."""
    B = queries.shape[0]

    # Per-request RoPE (each request has a different offset)
    q_parts = []
    k_parts = []
    for i in range(B):
        q_parts.append(attn_module.rope(queries[i : i + 1], offset=ctx.offsets[i]))
        k_parts.append(attn_module.rope(keys[i : i + 1], offset=ctx.offsets[i]))
    queries = mx.concatenate(q_parts, axis=0)  # (B, heads, 1, head_dim)
    keys_new = mx.concatenate(k_parts, axis=0)  # (B, kv_heads, 1, head_dim)

    # Write each request's new K/V token to block pool
    for i in range(B):
        k_tok = keys_new[i, :, 0, :]  # (kv_heads, head_dim)
        v_tok = values[i, :, 0, :]
        slot = ctx.slot_mapping[i]
        blk_idx = slot // block_size
        slot_off = slot % block_size
        pool.block_pool[blk_idx, layer_idx, 0, slot_off] = k_tok
        pool.block_pool[blk_idx, layer_idx, 1, slot_off] = v_tok

    # Gather K/V from block pool (all historical + new token)
    gathered_k, gathered_v = gather_kv_batched(
        pool.block_pool, layer_idx, ctx.block_tables, ctx.context_lens, block_size
    )
    # gathered_k/v: (B, kv_heads, max_len, head_dim)

    # Build left-padding mask
    max_len = max(ctx.context_lens)
    pad_mask = _build_left_pad_mask(ctx.context_lens, max_len)

    # SDPA: queries (B, heads, 1, hd) × gathered (B, kv_heads, max_len, hd)
    output = mx.fast.scaled_dot_product_attention(
        queries, gathered_k, gathered_v, scale=attn_module.scale, mask=pad_mask
    )

    # (B, heads, 1, head_dim) → (B, 1, D)
    output = output.transpose(0, 2, 1, 3).reshape(B, 1, -1)
    return attn_module.o_proj(output)


# ---------------------------------------------------------------------------
# Wrapper module for paged attention
# ---------------------------------------------------------------------------


class PagedAttentionWrapper(nn.Module):
    """Wraps an mlx_lm Attention module to intercept __call__ for paged KV.

    nn.Module dispatches __call__ via the class, so instance-level overrides
    don't work. This wrapper replaces self_attn on each TransformerBlock.
    """

    def __init__(
        self,
        inner: nn.Module,
        layer_idx: int,
        kv_pool: PagedKVCache,
        block_size: int,
    ) -> None:
        super().__init__()
        # Use object.__setattr__ to bypass MLX nn.Module's __setattr__
        # which calls hasattr → __getattr__ → _inner before it's set.
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_paged_layer_idx", layer_idx)
        object.__setattr__(self, "_paged_kv_pool", kv_pool)
        object.__setattr__(self, "_paged_block_size", block_size)
        object.__setattr__(self, "_paged_call_count", 0)
        object.__setattr__(self, "_fallback_call_count", 0)

    def __call__(self, x: mx.array, mask=None, cache=None) -> mx.array:
        ctx = get_context()
        if ctx is None:
            # No paged context → delegate to original
            object.__setattr__(
                self, "_fallback_call_count", self._fallback_call_count + 1
            )
            return self._inner(x, mask=mask, cache=cache)

        object.__setattr__(self, "_paged_call_count", self._paged_call_count + 1)
        layer_idx = self._paged_layer_idx
        pool = self._paged_kv_pool
        blk_size = self._paged_block_size
        inner = self._inner

        B, L, D = x.shape

        # --- projections + reshape (same as original) ---
        queries = inner.q_proj(x)
        keys = inner.k_proj(x)
        values = inner.v_proj(x)

        queries = queries.reshape(B, L, inner.n_heads, -1)
        keys = keys.reshape(B, L, inner.n_kv_heads, -1)
        values = values.reshape(B, L, inner.n_kv_heads, -1)

        # Qwen3 applies per-head RMSNorm before RoPE
        if hasattr(inner, "q_norm"):
            queries = inner.q_norm(queries)
        if hasattr(inner, "k_norm"):
            keys = inner.k_norm(keys)

        # transpose → (B, heads, L, head_dim)
        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        if ctx.is_prefill:
            return _paged_prefill_attention(
                inner, queries, keys, values, pool, layer_idx, blk_size, ctx, cache
            )
        else:
            return _paged_decode_attention(
                inner, queries, keys, values, pool, layer_idx, blk_size, ctx
            )


# ---------------------------------------------------------------------------
# Model patching
# ---------------------------------------------------------------------------


def patch_model_attention(
    model: Any,
    block_pool: PagedKVCache,
    block_size: int,
) -> int:
    """Walk model layers and replace each attention module with a PagedAttentionWrapper.

    Returns the number of patched layers.
    """
    layer_list, attn_attr = _find_layers_and_attr(model)
    patched = 0

    for layer_idx, layer in enumerate(layer_list):
        attn = getattr(layer, attn_attr)
        if isinstance(attn, PagedAttentionWrapper):
            # Already patched (e.g. second call) — update pool reference
            object.__setattr__(attn, "_paged_kv_pool", block_pool)
            object.__setattr__(attn, "_paged_block_size", block_size)
            patched += 1
            continue

        wrapper = PagedAttentionWrapper(attn, layer_idx, block_pool, block_size)
        setattr(layer, attn_attr, wrapper)
        patched += 1

    return patched


def get_paged_call_counts(
    model: Any,
) -> list[tuple[int, int]]:
    """Return per-layer (paged_calls, fallback_calls) for diagnostics.

    Only works after ``patch_model_attention`` has been called.
    """
    layer_list, attn_attr = _find_layers_and_attr(model)
    counts = []
    for layer in layer_list:
        attn = getattr(layer, attn_attr)
        if isinstance(attn, PagedAttentionWrapper):
            counts.append((attn._paged_call_count, attn._fallback_call_count))
        else:
            counts.append((0, 0))
    return counts


def reset_paged_call_counts(model: Any) -> None:
    """Reset all per-layer call counters to zero."""
    layer_list, attn_attr = _find_layers_and_attr(model)
    for layer in layer_list:
        attn = getattr(layer, attn_attr)
        if isinstance(attn, PagedAttentionWrapper):
            object.__setattr__(attn, "_paged_call_count", 0)
            object.__setattr__(attn, "_fallback_call_count", 0)


def _find_layers_and_attr(model: Any) -> tuple[list[Any], str]:
    """Find transformer layers and the attention attribute name.

    Returns (layer_list, attn_attr_name) where each layer has
    getattr(layer, attn_attr_name) pointing to the attention module.

    Supports mlx_lm model structures like:
        model.model.layers[i].self_attn
        model.layers[i].self_attn
    """
    # Try model.model.layers (Qwen3 Model wrapper)
    layers_container = getattr(model, "model", model)
    if hasattr(layers_container, "layers"):
        layer_list = layers_container.layers
    elif hasattr(model, "layers"):
        layer_list = model.layers
    else:
        raise ValueError(
            f"Cannot find transformer layers in model of type {type(model)}"
        )

    # Determine attribute name
    if layer_list:
        sample = layer_list[0]
        if hasattr(sample, "self_attn"):
            return layer_list, "self_attn"
        elif hasattr(sample, "attention"):
            return layer_list, "attention"
        else:
            raise ValueError(f"Cannot find attention module in layer {type(sample)}")
    return layer_list, "self_attn"


# ---------------------------------------------------------------------------
# Prepare functions — called before each forward pass
# ---------------------------------------------------------------------------


def prepare_prefill(
    block_ids: list[int],
    num_tokens: int,
    block_size: int,
) -> None:
    """Compute slot_mapping for prefill and set global context."""
    slot_mapping = []
    for pos in range(num_tokens):
        block_idx = block_ids[pos // block_size]
        slot = block_idx * block_size + (pos % block_size)
        slot_mapping.append(slot)

    set_context(
        PagedAttentionContext(
            is_prefill=True,
            slot_mapping=slot_mapping,
        )
    )


def prepare_decode(
    requests: list[tuple[list[int], int]],
    block_size: int,
) -> None:
    """Compute slot_mapping, block_tables, context_lens, offsets for decode.

    Args:
        requests: list of (block_ids, seq_len) per request.
                  seq_len = number of tokens already stored (before this step).
        block_size: tokens per block
    """
    slot_mapping: list[int] = []
    block_tables: list[list[int]] = []
    context_lens: list[int] = []
    offsets: list[int] = []

    for block_ids, seq_len in requests:
        # Slot for the new token at position seq_len
        new_pos = seq_len
        block_idx = block_ids[new_pos // block_size]
        slot = block_idx * block_size + (new_pos % block_size)
        slot_mapping.append(slot)
        block_tables.append(block_ids)
        context_lens.append(seq_len + 1)  # including new token
        offsets.append(seq_len)  # RoPE position = seq_len

    set_context(
        PagedAttentionContext(
            is_prefill=False,
            slot_mapping=slot_mapping,
            block_tables=block_tables,
            context_lens=context_lens,
            offsets=offsets,
        )
    )
