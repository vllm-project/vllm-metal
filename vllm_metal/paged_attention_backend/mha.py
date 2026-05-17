# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import mlx.core as mx

from vllm_metal.metal import warm_up_kernels

if TYPE_CHECKING:
    from vllm_metal.metal_kernel_backend.cache import MetalPagedKVCache


class MHAPagedAttentionBackend:
    """Paged attention backend for standard MHA models.

    Orchestrates metal_kernel_backend: allocates MetalPagedKVCache, patches
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
        self._cache: MetalPagedKVCache | None = None
        self._turboquant = turboquant
        self._k_quant = k_quant
        self._v_quant = v_quant
        self._cache_idx_map = cache_idx_map
        self._kv_heads_per_layer = kv_heads_per_layer
        self._head_dim_per_layer = head_dim_per_layer
        self._sliding_window_per_layer = sliding_window_per_layer

    def _require_initialized(self, caller: str) -> MetalPagedKVCache:
        if self._cache is None:
            raise RuntimeError(f"{caller}() called before initialize()")
        return self._cache

    def initialize(self, num_blocks: int) -> None:
        from vllm_metal.metal_kernel_backend.cache import MetalPagedKVCache

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

    def patch_model(self, model: Any) -> int:
        cache = self._require_initialized("patch_model")

        from vllm_metal.metal_kernel_backend.paged_attention import (
            patch_model_attention_metal_kernel,
        )

        return patch_model_attention_metal_kernel(
            model, cache, self._block_size, cache_idx_map=self._cache_idx_map
        )

    def warm_up(self) -> None:
        self._require_initialized("warm_up")
        warm_up_kernels()

    def num_blocks(self) -> int:
        return self._require_initialized("num_blocks").num_blocks
