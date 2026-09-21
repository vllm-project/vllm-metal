# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

import mlx.core as mx
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import KVCacheConfig

from vllm_metal.attention.caches.kv_cache import MetalPagedKVCache
from vllm_metal.attention.caches.storage import KVCacheStorage
from vllm_metal.attention.impls.sdpa_wrapper import (
    patch_sdpa_attention,
)
from vllm_metal.attention.runtime.base import PagedAttentionRuntimeBase

logger = init_logger(__name__)


class SDPAPagedAttentionRuntime(PagedAttentionRuntimeBase):
    """Paged attention runtime for SDPA attention models (MHA, GQA, MQA).

    Bind upstream cache storage and patch model layers for native Metal kernels.
    Model-based speculative decoding still uses count-initialized storage.
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
        self._storage: KVCacheStorage | None = None
        self._scheduler_group_indices = (0,)
        self._group_block_sizes = (block_size,)

    def initialize(self, num_blocks: int) -> None:
        # TODO: Move model-based speculation's target and committed draft KV to
        # initialize_from_config, handling lookahead scratch separately. Then
        # remove this initializer and MetalPagedKVCache's count-based allocation.
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

    def initialize_from_config(
        self, config: KVCacheConfig, layer_names: tuple[str, ...]
    ) -> None:
        """Use the physical allocation and views planned by vLLM."""
        self._storage = KVCacheStorage(config)
        self._cache = MetalPagedKVCache.from_upstream(
            self._storage, layer_names, dtype=self._dtype
        )
        # Allocation specs may be merged into full attention when grouping is
        # disabled; the model still determines each layer's attention window.
        if self._sliding_window_per_layer is not None:
            self._cache.sliding_window_per_layer = self._sliding_window_per_layer
        self.adopt_scheduler_groups(config, layer_names)
        logger.info(
            "Shared attention cache: %d blocks, %.2f GiB across %d Metal regions",
            config.num_blocks,
            self._storage.nbytes / (1 << 30),
            len(self._storage.buffers),
        )

    def adopt_scheduler_groups(
        self, config: KVCacheConfig, layer_names: tuple[str, ...]
    ) -> None:
        groups = [
            (i, group)
            for i, group in enumerate(config.kv_cache_groups)
            if any(name in layer_names for name in group.layer_names)
        ]
        self._scheduler_group_indices = tuple(i for i, _ in groups)
        self._group_block_sizes = tuple(
            group.kv_cache_spec.block_size for _, group in groups
        )
        self._block_size = self._group_block_sizes[0]

    def kv_scheduler_group_indices(self) -> tuple[int, ...]:
        """Return scheduler KV groups consumed by this runtime."""
        self._require_initialized("kv_scheduler_group_indices")
        return self._scheduler_group_indices

    def kv_group_block_sizes(self) -> tuple[int, ...]:
        """Return the upstream scheduler page sizes."""
        self._require_initialized("kv_group_block_sizes")
        return self._group_block_sizes

    def extend_forward_eval_outputs(self, outputs: list[mx.array]) -> None:
        if self._storage is not None:
            self._storage.depend(outputs)
            outputs.extend(self._storage.buffers)

    def patch_model(self, model: Any) -> int:
        cache = self._require_initialized("patch_model")

        return patch_sdpa_attention(
            model, cache, self._block_size, cache_idx_map=self._cache_idx_map
        )

    @property
    def kv_cache(self) -> MetalPagedKVCache:
        return self._require_initialized("kv_cache")
