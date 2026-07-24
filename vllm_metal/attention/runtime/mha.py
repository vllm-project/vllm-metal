# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

import mlx.core as mx

from vllm_metal.attention.caches.kv_cache import MetalPagedKVCache
from vllm_metal.attention.caches.mha_layout import MHAKVCacheLayout
from vllm_metal.attention.impls.sdpa_wrapper import (
    patch_sdpa_attention,
)
from vllm_metal.attention.runtime.base import PagedAttentionRuntimeBase


class MHAPagedAttentionRuntime(PagedAttentionRuntimeBase):
    """Paged attention runtime for standard MHA models.

    Orchestrates native Metal SDPA attention: allocates MetalPagedKVCache, patches
    model attention layers with the vendored C++/Metal kernel, and warms up
    the kernel before the first request.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int,
        dtype: mx.Dtype,
        turboquant: bool = False,
        k_quant: str | None = None,
        v_quant: str | None = None,
        cache_idx_map: dict[int, int] | None = None,
        kv_heads_per_layer: list[int] | None = None,
        head_dim_per_layer: list[int] | None = None,
        sliding_window_per_layer: list[int] | None = None,
    ) -> None:
        self._num_layers = num_layers
        self._num_kv_heads = num_kv_heads
        self._head_dim = head_dim
        self._block_size = block_size
        self._dtype = dtype
        self._cache = None
        self._turboquant = turboquant
        self._k_quant = k_quant
        self._v_quant = v_quant
        self._cache_idx_map = cache_idx_map
        self._kv_heads_per_layer = kv_heads_per_layer
        self._head_dim_per_layer = head_dim_per_layer
        self._sliding_window_per_layer = sliding_window_per_layer
        self._layout: MHAKVCacheLayout | None = None

    def initialize(self, num_blocks: int) -> None:
        self._cache = MetalPagedKVCache(
            num_layers=self._num_layers,
            num_kv_heads=self._num_kv_heads,
            head_dim=self._head_dim,
            num_blocks=num_blocks,
            block_size=self._block_size,
            dtype=self._dtype,
            turboquant=self._turboquant,
            k_quant=self._k_quant,
            v_quant=self._v_quant,
            kv_heads_per_layer=self._kv_heads_per_layer,
            head_dim_per_layer=self._head_dim_per_layer,
            sliding_window_per_layer=self._sliding_window_per_layer,
        )

    def adopt_layout(self, layout: MHAKVCacheLayout) -> None:
        """Replace the legacy dense cache with vLLM's grouped MHA layout."""
        self._require_initialized("adopt_layout")
        if self._turboquant:
            raise NotImplementedError(
                "layout-backed MHA runtime does not support TurboQuant"
            )

        self._layout = layout
        self._block_size = layout.group_block_sizes[0]
        self._cache = MetalPagedKVCache.from_layout(layout, self._dtype)

    def kv_scheduler_group_indices(self) -> tuple[int, ...]:
        """Return scheduler KV groups consumed by this runtime."""
        if self._layout is None:
            return super().kv_scheduler_group_indices()
        self._require_initialized("kv_scheduler_group_indices")
        return tuple(range(len(self._layout.group_block_sizes)))

    def kv_group_block_sizes(self) -> tuple[int, ...]:
        """Return layout-owned group sizes or the legacy scalar page size."""
        if self._layout is None:
            return super().kv_group_block_sizes()
        self._require_initialized("kv_group_block_sizes")
        return self._layout.group_block_sizes

    def patch_model(self, model: Any) -> int:
        cache = self._require_initialized("patch_model")

        return patch_sdpa_attention(
            model, cache, self._block_size, cache_idx_map=self._cache_idx_map
        )

    @property
    def kv_cache(self) -> MetalPagedKVCache:
        return self._require_initialized("kv_cache")
