# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

import mlx.core as mx

from vllm_metal.attention.caches.mla_cache import MLAPagedLatentCache
from vllm_metal.attention.impls.mla import MLAPagedAttentionWrapper
from vllm_metal.attention.patching import walk_and_wrap
from vllm_metal.attention.runtime.base import PagedAttentionRuntimeBase


class MLAPagedAttentionRuntime(PagedAttentionRuntimeBase):
    """Paged attention runtime for MLA models.

    Implements the PagedAttentionRuntime protocol. Uses MLX-native
    scatter/gather (cache I/O only; attention is MLX SDPA by default
    or an opt-in single-pass Metal kernel, RFC #360) because MLA
    latents do not fit the standard (num_heads, head_dim) kernel
    layout.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        latent_dim: int,
        block_size: int,
        dtype: mx.Dtype,
    ) -> None:
        self._num_layers = num_layers
        self._latent_dim = latent_dim
        self._block_size = block_size
        self._dtype = dtype
        self._cache = None

    def initialize(self, num_blocks: int) -> None:
        self._cache = MLAPagedLatentCache(
            num_layers=self._num_layers,
            latent_dim=self._latent_dim,
            num_blocks=num_blocks,
            block_size=self._block_size,
            dtype=self._dtype,
        )

    def patch_model(self, model: Any) -> int:
        cache = self._require_initialized("patch_model")
        return self._patch_model(model, cache)

    def _patch_model(self, model: Any, latent_cache: MLAPagedLatentCache) -> int:
        def wrap_layer(layer_idx: int, attn: Any) -> Any:
            if isinstance(attn, MLAPagedAttentionWrapper):
                # Already patched — refresh cache reference in place.
                object.__setattr__(attn, "_mla_latent_cache", latent_cache)
                return attn
            return MLAPagedAttentionWrapper(attn, layer_idx, latent_cache)

        return walk_and_wrap(model, wrap_layer)
