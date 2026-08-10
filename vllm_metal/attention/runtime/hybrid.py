# SPDX-License-Identifier: Apache-2.0
"""Paged attention runtime for hybrid models (SDPA + linear attention).

Handles models like Qwen3.5 where some layers use standard dot-product
attention (paged KV cache) and others use GDN linear attention (fixed-size
recurrent state).

SDPA layers use the native Metal SDPA kernel (same as ``MHAPagedAttentionRuntime``).
GDN layers use MLX-native state management via ``GDNPagedAttentionWrapper``.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn
from vllm.logger import init_logger

from vllm_metal.attention.caches.gdn_cache import GDNPagedStateCache
from vllm_metal.attention.caches.kv_cache import MetalPagedKVCache
from vllm_metal.attention.impls.linear import (
    GDNPagedAttentionWrapper,
    is_linear_attention,
)
from vllm_metal.attention.impls.sdpa import is_sdpa
from vllm_metal.attention.impls.sdpa_wrapper import (
    SDPAPagedAttentionWrapper,
)
from vllm_metal.attention.patching import walk_and_wrap
from vllm_metal.attention.runtime.base import StateHybridRuntimeBase
from vllm_metal.attention.state import AlignGDNStateManager, HybridGDNStateManager

logger = init_logger(__name__)


class HybridPagedAttentionRuntime(StateHybridRuntimeBase):
    """Paged attention runtime for hybrid SDPA + linear attention models.

    SDPA layers: paged Metal kernel (via SDPAPagedAttentionWrapper)
    GDN layers: MLX-native state management (via GDNPagedAttentionWrapper)
    """

    SUPPORTED_MAMBA_CACHE_MODES = ("none", "align")

    def __init__(
        self,
        *,
        num_layers: int,
        full_attention_interval: int,
        max_num_seqs: int,
        # SDPA dims
        num_kv_heads: int,
        head_dim: int,
        # GDN dims
        linear_num_v_heads: int,
        linear_key_head_dim: int,
        linear_value_head_dim: int,
        linear_conv_kernel_dim: int,
        linear_conv_dim: int,
        # Common
        block_size: int,
        dtype: mx.Dtype,
        # Scheduler-side mamba caching strategy ("none" or "align").
        mamba_cache_mode: str = "none",
        # TurboQuant (SDPA layers only)
        turboquant: bool = False,
        k_quant: str | None = None,
        v_quant: str | None = None,
    ) -> None:
        self._init_state_hybrid(
            block_size=block_size,
            max_num_seqs=max_num_seqs,
            mamba_cache_mode=mamba_cache_mode,
        )
        self._dtype = dtype

        # SDPA params
        self._num_kv_heads = num_kv_heads
        self._head_dim = head_dim

        # GDN params
        self._linear_num_v_heads = linear_num_v_heads
        self._linear_key_head_dim = linear_key_head_dim
        self._linear_value_head_dim = linear_value_head_dim
        self._linear_conv_kernel_dim = linear_conv_kernel_dim
        self._linear_conv_dim = linear_conv_dim

        # TurboQuant params (only applies to SDPA layers)
        self._turboquant = turboquant
        self._k_quant = k_quant
        self._v_quant = v_quant

        # Classify layers
        self._sdpa_indices: list[int] = []
        self._linear_indices: list[int] = []
        for i in range(num_layers):
            if (i + 1) % full_attention_interval == 0:
                self._sdpa_indices.append(i)
            else:
                self._linear_indices.append(i)

        self._cache = None

    def initialize(self, num_blocks: int) -> None:
        self._init_sdpa_kv_cache(num_blocks)

        self._initialize_state_lifecycle(num_blocks)

        logger.info(
            "Hybrid cache initialized: %d SDPA layers (%d blocks), "
            "%d linear layers (%d/%d GDN slots allocated, mamba_cache_mode=%s)",
            len(self._sdpa_indices),
            num_blocks,
            len(self._linear_indices),
            self._state_cache.allocated_seqs,
            self._state_cache.max_seqs,
            self._mamba_cache_mode,
        )

    def patch_model(self, model: nn.Module) -> int:
        kv_cache = self._require_initialized("patch_model")
        if self._state_cache is None:
            raise RuntimeError("patch_model() called before initialize()")
        state_cache = self._state_cache

        sdpa_cache_map = {
            layer_idx: cache_idx
            for cache_idx, layer_idx in enumerate(self._sdpa_indices)
        }
        linear_cache_map = {
            layer_idx: cache_idx
            for cache_idx, layer_idx in enumerate(self._linear_indices)
        }

        def wrap_layer(layer_idx: int, attn: Any) -> Any:
            if isinstance(attn, SDPAPagedAttentionWrapper):
                # Already patched (cached model reuse) — refresh cache refs.
                cache_idx = sdpa_cache_map.get(layer_idx, layer_idx)
                attn.rebind_cache(kv_cache, self._block_size, cache_idx=cache_idx)
                return attn
            if isinstance(attn, GDNPagedAttentionWrapper):
                # Already patched — refresh state cache ref.
                cache_idx = linear_cache_map.get(layer_idx, layer_idx)
                object.__setattr__(attn, "_gdn_cache_idx", cache_idx)
                object.__setattr__(attn, "_gdn_state_cache", state_cache)
                return attn
            if is_sdpa(attn):
                cache_idx = sdpa_cache_map.get(layer_idx, layer_idx)
                return SDPAPagedAttentionWrapper(
                    attn, layer_idx, kv_cache, self._block_size, cache_idx=cache_idx
                )
            if is_linear_attention(attn):
                cache_idx = linear_cache_map.get(layer_idx, layer_idx)
                return GDNPagedAttentionWrapper(attn, layer_idx, cache_idx, state_cache)
            raise RuntimeError(
                f"Hybrid patch_model: layer {layer_idx} attention "
                f"{type(attn).__name__} is neither SDPA nor linear attention; "
                "refusing to leave it unpatched (it would silently run unpaged)."
            )

        return walk_and_wrap(model, wrap_layer)

    @property
    def kv_cache(self) -> MetalPagedKVCache:
        return self._require_initialized("kv_cache")

    @property
    def gdn_state_manager(self) -> HybridGDNStateManager | AlignGDNStateManager:
        """Alias for :attr:`state_manager` under this runtime's GDN name."""
        return self.state_manager

    def _new_state_cache(
        self, *, max_seqs: int, initial_seqs: int
    ) -> GDNPagedStateCache:
        return GDNPagedStateCache(
            num_layers=len(self._linear_indices),
            max_seqs=max_seqs,
            conv_kernel_dim=self._linear_conv_kernel_dim,
            conv_dim=self._linear_conv_dim,
            num_v_heads=self._linear_num_v_heads,
            value_head_dim=self._linear_value_head_dim,
            key_head_dim=self._linear_key_head_dim,
            initial_seqs=initial_seqs,
            dtype=self._dtype,
        )
