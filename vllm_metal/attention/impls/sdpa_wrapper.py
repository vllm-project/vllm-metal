# SPDX-License-Identifier: Apache-2.0
"""Paged attention wrapper and dispatch for native Metal kernels.

The wrapper intercepts mlx_lm attention modules and dispatches to the
appropriate Metal attention impl based on the module's structure:

- SDPA (Qwen3, Llama, Mistral, …) → ``sdpa.py``
- Linear attention (Qwen3.5 GatedDeltaNet, …) → ``linear.py`` (stub)
- Future attention types (MLA, …) → add detection + forward function

All operations use MLX arrays end-to-end — no PyTorch MPS bridge.

Reuses ``PagedAttentionContext``, ``OffsetCache``, ``prepare_grouped``,
``prepare_unified``, and ``clear_context`` from ``context``.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn

from vllm_metal.attention.attention_contracts import attention_contract_for
from vllm_metal.attention.caches.kv_cache import MetalPagedKVCache
from vllm_metal.attention.context import OffsetCache, get_context
from vllm_metal.attention.impls.sdpa import (
    _float32_sinks,
    sdpa_forward,
)
from vllm_metal.attention.patching import find_layers, find_num_loops, walk_and_wrap

# ---------------------------------------------------------------------------
# Wrapper nn.Module
# ---------------------------------------------------------------------------


def _is_rope_embedding_pair(value: Any) -> bool:
    """Return True for caller-precomputed RoPE ``(cos, sin)`` embeddings."""
    return (
        isinstance(value, (tuple, list))
        and len(value) == 2
        and all(isinstance(item, mx.array) for item in value)
    )


class SDPAPagedAttentionWrapper(nn.Module):
    """Wraps an mlx_lm Attention module to use native Metal paged KV.

    Uses ``object.__setattr__`` to bypass MLX nn.Module's ``__setattr__``.

    When no ``PagedAttentionContext`` is set, falls back to original attention.

    Return contract:
        - Standard models: returns ``mx.array`` (attention output).
        - YOCO models (e.g. Gemma4): when ``shared_kv`` is in kwargs,
          returns ``(output, kv_pair, offset)`` so the caller can forward
          the K/V pair to the next same-type layer.  The return type is
          selected by kwarg presence rather than a class flag because
          mlx_lm's attention signature is fixed.
    """

    def __init__(
        self,
        inner: nn.Module,
        layer_idx: int,
        kv_cache: MetalPagedKVCache,
        block_size: int,
        *,
        cache_idx: int | None = None,
        looped: bool = False,
        llama4_scaling: tuple[float, int] | None = None,
    ) -> None:
        super().__init__()
        contract = attention_contract_for(inner)
        if contract.llama4_query_scaling and llama4_scaling is None:
            raise ValueError(
                f"{type(inner).__module__}.{type(inner).__name__} scales "
                "queries by position but no llama4_scaling was given"
            )
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_mk_layer_idx", layer_idx)
        object.__setattr__(self, "_mk_kv_cache", kv_cache)
        object.__setattr__(self, "_mk_block_size", block_size)
        object.__setattr__(self, "_mk_attention_contract", contract)
        object.__setattr__(self, "_mk_llama4_scaling", llama4_scaling)
        object.__setattr__(self, "_mk_looped", looped)
        # For compact caches (hybrid models), cache_idx maps to the
        # per-type cache array.  Defaults to layer_idx for non-hybrid.
        object.__setattr__(
            self, "_mk_cache_idx", cache_idx if cache_idx is not None else layer_idx
        )
        # Warm the float32 sink cache at patch time (GPT-OSS stores fp16/bf16
        # sinks) so the first forward does not pay the cast in the hot path.
        # The cast is lazy, so evaluate it here to materialize the array now.
        sinks_f32 = _float32_sinks(inner, contract.sinks_attr)
        if sinks_f32 is not None:
            mx.eval(sinks_f32)

    @property
    def rotary_emb(self) -> Any:
        """Expose the wrapped attention's M-RoPE module."""
        return self._inner.rotary_emb

    @property
    def rope(self) -> Any:
        """Expose the wrapped attention's RoPE module."""
        return self._inner.rope

    @property
    def is_local(self) -> Any:
        """Expose the wrapped attention's local/global attention flag.

        EXAONE 4.0's host model forward reads ``self_attn.is_local`` to select
        the sliding-window vs global mask per layer. Attention modules without
        the attribute (the common case) default to ``False`` — global
        attention — so wrapping them is a no-op for this flag.
        """
        return getattr(self._inner, "is_local", False)

    @property
    def layer_type(self) -> Any:
        """Expose the wrapped attention's layer type.

        Muse Glimmer's host model forward (mlx_lm) reads
        ``self_attn.layer_type`` to select the sliding-window vs full mask
        per layer.
        """
        return self._inner.layer_type

    def rebind_cache(
        self,
        kv_cache: MetalPagedKVCache,
        block_size: int,
        *,
        cache_idx: int | None = None,
        looped: bool = False,
    ) -> None:
        """Rebind wrapper-owned paged-cache state in one place."""
        object.__setattr__(self, "_mk_kv_cache", kv_cache)
        object.__setattr__(self, "_mk_block_size", block_size)
        object.__setattr__(self, "_mk_looped", looped)
        object.__setattr__(
            self,
            "_mk_cache_idx",
            cache_idx if cache_idx is not None else self._mk_layer_idx,
        )

    def _paged_cache_idx(self, cache: Any) -> int:
        """Return the KV cache layer this call reads and writes."""
        if not self._mk_looped:
            return self._mk_cache_idx
        # A looped model calls this module once per loop and passes each call
        # its own cache entry; the entry's index is that call's KV layer.
        if not isinstance(cache, OffsetCache) or cache.index is None:
            raise RuntimeError(
                f"Looped attention layer {self._mk_layer_idx} needs an indexed "
                f"OffsetCache, got {cache!r}."
            )
        return cache.index

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: nn.Module | None = None,
        position_ids: mx.array | None = None,
        **kwargs: Any,
    ) -> Any:
        if self._mk_llama4_scaling is not None:
            return self._call_llama4_scaled(x, mask, cache, position_ids)

        position_embeddings = kwargs.pop("position_embeddings", None)
        if _is_rope_embedding_pair(position_ids):
            # Some attention contracts pass precomputed ``(cos, sin)`` RoPE
            # embeddings positionally as the fourth argument. Preserve that
            # payload instead of treating it as Qwen-style ``position_ids``.
            position_embeddings = position_ids
            position_ids = None

        ctx = get_context()
        if ctx is None:
            # No paged context → delegate to original attention.
            if position_embeddings is not None:
                return self._inner(
                    x,
                    mask=mask,
                    cache=cache,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )
            # Only pass position_ids when provided (mlx_vlm models);
            # mlx_lm models (e.g. Gemma4) use offset via **kwargs instead.
            if position_ids is not None:
                return self._inner(
                    x, mask=mask, cache=cache, position_ids=position_ids, **kwargs
                )
            return self._inner(x, mask=mask, cache=cache, **kwargs)

        inner = self._inner

        # SDPA attention via Metal kernel.
        # Pass shared_kv for YOCO KV sharing (Gemma4: later layers reuse
        # K/V from earlier same-type layers instead of projecting their own).
        has_shared_kv = "shared_kv" in kwargs
        shared_kv = kwargs.get("shared_kv")

        output, kv_pair = sdpa_forward(
            inner,
            x,
            ctx,
            self._mk_kv_cache,
            self._paged_cache_idx(cache),
            shared_kv=shared_kv,
            position_embeddings=position_embeddings,
            attention_contract=self._mk_attention_contract,
        )

        # YOCO models (Gemma4) expect (output, shared_kv, offset) return.
        # Key off shared_kv presence to match mlx_lm's fixed attention
        # signature for Gemma4 YOCO layers.
        if has_shared_kv:
            return (output, kv_pair, kwargs.get("offset", 0))

        return output

    def _call_llama4_scaled(
        self,
        x: mx.array,
        attn_scale: mx.array,
        mask: mx.array | None,
        cache: nn.Module | None,
    ) -> mx.array:
        """Ministral 3: the layer passes ``(x, attn_scale, mask, cache)``.

        The model computes ``attn_scale`` from one offset for the whole
        batch, which is wrong for packed requests, so the paged path derives
        it per row from the context positions instead.
        """
        ctx = get_context()
        if ctx is None:
            return self._inner(x, attn_scale, mask, cache)
        output, _ = sdpa_forward(
            self._inner,
            x,
            ctx,
            self._mk_kv_cache,
            self._paged_cache_idx(cache),
            attention_contract=self._mk_attention_contract,
            llama4_scaling=self._mk_llama4_scaling,
        )
        return output


# ---------------------------------------------------------------------------
# Model patching
# ---------------------------------------------------------------------------


def _llama4_scaling(model: Any) -> tuple[float, int] | None:
    """Return Ministral 3's ``(beta, original_max_position_embeddings)``.

    mlx_lm keeps them only in the text model's ``rope_parameters``.
    """
    root = getattr(model, "language_model", model)
    rope_parameters = getattr(getattr(root, "args", None), "rope_parameters", None)
    if not rope_parameters or "llama_4_scaling_beta" not in rope_parameters:
        return None
    return (
        float(rope_parameters["llama_4_scaling_beta"]),
        int(rope_parameters["original_max_position_embeddings"]),
    )


def patch_sdpa_attention(
    model: Any,
    kv_cache: MetalPagedKVCache,
    block_size: int,
    *,
    cache_idx_map: dict[int, int] | None = None,
    only_layers: list[int] | None = None,
) -> int:
    """Walk model layers and replace each attention module with a
    ``SDPAPagedAttentionWrapper``.

    Supports hybrid models (e.g. Qwen3.5) where different layers use
    different attribute names (``self_attn``, ``linear_attn``, etc.).

    Args:
        cache_idx_map: Optional mapping from model layer_idx to compact
            cache index.  Used for hybrid models so that a compact
            ``MetalPagedKVCache`` (SDPA layers only) is indexed correctly.
            When ``None``, ``layer_idx`` is used directly.
        only_layers: If provided, only patch these layer indices and skip
            the rest.  Used by hybrid runtime to avoid wrapping linear
            attention layers that have no kernel implementation yet.

    Returns the number of patched layers.
    """

    llama4_scaling = _llama4_scaling(model)
    num_loops = find_num_loops(model)
    looped = num_loops > 1
    if looped:
        if cache_idx_map is not None or only_layers is not None:
            raise NotImplementedError(
                "Looped models (num_loops > 1) need one KV cache layer per "
                "loop of every layer; cache_idx_map / only_layers are unsupported."
            )
        num_layers = len(find_layers(model))
        if kv_cache.num_layers != num_loops * num_layers:
            raise ValueError(
                f"Looped model needs {num_loops * num_layers} KV cache layers "
                f"({num_loops} loops x {num_layers} layers), "
                f"got {kv_cache.num_layers}."
            )

    def wrap_layer(layer_idx: int, attn: Any) -> Any:
        cache_idx = (
            cache_idx_map[layer_idx]
            if cache_idx_map is not None and layer_idx in cache_idx_map
            else layer_idx
        )
        if isinstance(attn, SDPAPagedAttentionWrapper):
            # Already patched — refresh cache reference in place.
            attn.rebind_cache(kv_cache, block_size, cache_idx=cache_idx, looped=looped)
            return attn
        return SDPAPagedAttentionWrapper(
            attn,
            layer_idx,
            kv_cache,
            block_size,
            cache_idx=cache_idx,
            looped=looped,
            llama4_scaling=(
                llama4_scaling
                if attention_contract_for(attn).llama4_query_scaling
                else None
            ),
        )

    return walk_and_wrap(model, wrap_layer, only_layers=only_layers)
