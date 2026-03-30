# SPDX-License-Identifier: Apache-2.0
"""Paged attention wrapper and dispatch for native Metal kernels.

The wrapper intercepts mlx_lm attention modules and dispatches to the
appropriate Metal attention backend based on the module's structure:

- SDPA (Qwen3, Llama, Mistral, …) → ``attention_sdpa.py``
- Linear attention (Qwen3.5 GatedDeltaNet, …) → ``attention_linear.py`` (stub)
- Future attention types (MLA, …) → add detection + forward function

All operations use MLX arrays end-to-end — no PyTorch MPS bridge.

Reuses ``PagedAttentionContext``, ``OffsetCache``, ``prepare_unified``,
``clear_context`` from ``paged_attention_common``.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn

from vllm_metal.metal_kernel_backend.attention_linear import (
    is_linear_attention,
    linear_attention_forward,
)
from vllm_metal.metal_kernel_backend.attention_sdpa import (
    is_sdpa,
    sdpa_forward,
)
from vllm_metal.metal_kernel_backend.cache import MetalPagedKVCache
from vllm_metal.paged_attention_common import (
    find_attn_attr,
    find_layers,
    get_context,
)

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
        *,
        cache_idx: int | None = None,
    ) -> None:
        super().__init__()
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_mk_layer_idx", layer_idx)
        object.__setattr__(self, "_mk_kv_cache", kv_cache)
        object.__setattr__(self, "_mk_block_size", block_size)
        # For compact caches (hybrid models), cache_idx maps to the
        # per-type cache array.  Defaults to layer_idx for non-hybrid.
        object.__setattr__(
            self, "_mk_cache_idx", cache_idx if cache_idx is not None else layer_idx
        )

    def __call__(self, x: mx.array, mask: Any = None, cache: Any = None) -> mx.array:
        ctx = get_context()
        if ctx is None:
            # No paged context → delegate to original attention
            return self._inner(x, mask=mask, cache=cache)

        inner = self._inner

        # Dispatch to the right attention backend
        cache_idx = self._mk_cache_idx
        if is_sdpa(inner):
            return sdpa_forward(inner, x, ctx, self._mk_kv_cache, cache_idx)
        elif is_linear_attention(inner):
            return linear_attention_forward(inner, x, ctx, self._mk_kv_cache, cache_idx)
        else:
            raise NotImplementedError(
                f"No Metal attention backend for {type(inner).__name__}. "
                f"Module attributes: {[a for a in dir(inner) if not a.startswith('_')]}. "
                "Supported: SDPA (q_proj + k_proj + v_proj + o_proj)."
            )


# ---------------------------------------------------------------------------
# Model patching
# ---------------------------------------------------------------------------


def patch_model_attention_metal_kernel(
    model: Any,
    kv_cache: MetalPagedKVCache,
    block_size: int,
    *,
    cache_idx_map: dict[int, int] | None = None,
    only_layers: list[int] | None = None,
) -> int:
    """Walk model layers and replace each attention module with a
    ``MetalKernelPagedAttentionWrapper``.

    Supports hybrid models (e.g. Qwen3.5) where different layers use
    different attribute names (``self_attn``, ``linear_attn``, etc.).

    Args:
        cache_idx_map: Optional mapping from model layer_idx to compact
            cache index.  Used for hybrid models so that a compact
            ``MetalPagedKVCache`` (SDPA layers only) is indexed correctly.
            When ``None``, ``layer_idx`` is used directly.
        only_layers: If provided, only patch these layer indices and skip
            the rest.  Used by hybrid backend to avoid wrapping linear
            attention layers that have no kernel implementation yet.

    Returns the number of patched layers.
    """
    layer_list = find_layers(model)
    only_set = set(only_layers) if only_layers is not None else None
    patched = 0

    for layer_idx, layer in enumerate(layer_list):
        if only_set is not None and layer_idx not in only_set:
            continue

        attn_attr = find_attn_attr(layer)
        if attn_attr is None:
            continue

        attn = getattr(layer, attn_attr)
        if isinstance(attn, MetalKernelPagedAttentionWrapper):
            # Already patched — update cache reference
            object.__setattr__(attn, "_mk_kv_cache", kv_cache)
            object.__setattr__(attn, "_mk_block_size", block_size)
            patched += 1
            continue

        cache_idx = (
            cache_idx_map[layer_idx]
            if cache_idx_map is not None and layer_idx in cache_idx_map
            else layer_idx
        )
        wrapper = MetalKernelPagedAttentionWrapper(
            attn, layer_idx, kv_cache, block_size, cache_idx=cache_idx
        )
        setattr(layer, attn_attr, wrapper)
        patched += 1

    return patched
