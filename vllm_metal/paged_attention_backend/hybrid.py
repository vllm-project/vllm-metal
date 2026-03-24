# SPDX-License-Identifier: Apache-2.0
"""Paged attention backend for hybrid models (SDPA + linear attention).

Handles models like Qwen3.5 where some layers use standard dot-product
attention (paged KV cache) and others use GDN linear attention (fixed-size
recurrent state).

Allocates ``MetalPagedKVCache`` for SDPA layers and
``LinearAttentionCache`` for GDN layers, both indexed by scheduler-managed
block IDs.
"""

from __future__ import annotations

import platform
from typing import Any

import mlx.core as mx
from vllm.logger import init_logger

logger = init_logger(__name__)

_METAL_LANGUAGE_VERSION_ERROR = "language version"


class HybridPagedAttentionBackend:
    """Paged attention backend for hybrid SDPA + linear attention models."""

    def __init__(
        self,
        *,
        num_layers: int,
        full_attention_interval: int,
        # SDPA dims
        num_kv_heads: int,
        head_dim: int,
        # GDN dims
        linear_num_k_heads: int,
        linear_num_v_heads: int,
        linear_key_head_dim: int,
        linear_value_head_dim: int,
        linear_conv_kernel_dim: int,
        # Common
        block_size: int,
        dtype: mx.Dtype,
    ) -> None:
        self._num_layers = num_layers
        self._full_attention_interval = full_attention_interval
        self._block_size = block_size
        self._dtype = dtype

        # SDPA params
        self._num_kv_heads = num_kv_heads
        self._head_dim = head_dim

        # GDN params
        self._linear_num_k_heads = linear_num_k_heads
        self._linear_num_v_heads = linear_num_v_heads
        self._linear_key_head_dim = linear_key_head_dim
        self._linear_value_head_dim = linear_value_head_dim
        self._linear_conv_kernel_dim = linear_conv_kernel_dim

        # Classify layers
        self._sdpa_indices: list[int] = []
        self._linear_indices: list[int] = []
        for i in range(num_layers):
            if (i + 1) % full_attention_interval == 0:
                self._sdpa_indices.append(i)
            else:
                self._linear_indices.append(i)

        self._kv_cache: Any = None  # MetalPagedKVCache
        self._linear_cache: Any = None  # LinearAttentionCache
        self._num_blocks: int = 0

    def initialize(self, num_blocks: int) -> None:
        from vllm_metal.metal_kernel_backend.cache import MetalPagedKVCache
        from vllm_metal.metal_kernel_backend.cache_linear import (
            LinearAttentionCache,
        )

        self._num_blocks = num_blocks

        # SDPA cache: paged KV for attention layers
        self._kv_cache = MetalPagedKVCache(
            num_layers=len(self._sdpa_indices),
            num_kv_heads=self._num_kv_heads,
            head_dim=self._head_dim,
            num_blocks=num_blocks,
            block_size=self._block_size,
            dtype=self._dtype,
        )

        # Linear cache: recurrent state for GDN layers, same num_blocks
        # so the scheduler's block management works uniformly.
        conv_dim = (
            self._linear_num_k_heads * self._linear_key_head_dim * 2
            + self._linear_num_v_heads * self._linear_value_head_dim
        )
        self._linear_cache = LinearAttentionCache(
            num_layers=len(self._linear_indices),
            num_blocks=num_blocks,
            conv_kernel_dim=self._linear_conv_kernel_dim,
            conv_dim=conv_dim,
            num_v_heads=self._linear_num_v_heads,
            value_head_dim=self._linear_value_head_dim,
            key_head_dim=self._linear_key_head_dim,
            dtype=self._dtype,
        )

        logger.info(
            "Hybrid cache initialized: %d SDPA layers + %d linear layers, "
            "%d blocks allocated",
            len(self._sdpa_indices),
            len(self._linear_indices),
            num_blocks,
        )

    def patch_model(self, model: Any) -> int:
        if self._kv_cache is None:
            raise RuntimeError("patch_model() called before initialize()")

        from vllm_metal.metal_kernel_backend.paged_attention import (
            patch_model_attention_metal_kernel,
        )

        # Map SDPA layer indices to compact cache indices so the wrapper
        # indexes into the compact MetalPagedKVCache correctly.
        sdpa_cache_idx = {
            layer_idx: cache_idx
            for cache_idx, layer_idx in enumerate(self._sdpa_indices)
        }
        return patch_model_attention_metal_kernel(
            model, self._kv_cache, self._block_size, cache_idx_map=sdpa_cache_idx
        )

    def warm_up(self) -> None:
        if self._kv_cache is None:
            raise RuntimeError("warm_up() called before initialize()")

        from vllm_metal.metal import get_ops

        macos_version = platform.mac_ver()[0]
        logger.info("Warming up hybrid paged attention Metal kernel...")

        try:
            ops = get_ops()
        except Exception as e:
            raise RuntimeError(
                f"Failed to load Metal kernel: {e}. macOS {macos_version}"
            ) from e

        try:
            cache = self._kv_cache
            dummy_k = mx.zeros(
                (1, cache.num_kv_heads, cache.head_dim), dtype=cache.dtype
            )
            dummy_v = mx.zeros(
                (1, cache.num_kv_heads, cache.head_dim), dtype=cache.dtype
            )
            dummy_slot = mx.zeros((1,), dtype=mx.int64)
            mx.eval(dummy_k, dummy_v, dummy_slot)
            ops.reshape_and_cache(
                dummy_k,
                dummy_v,
                cache.key_caches[0],
                cache.value_caches[0],
                dummy_slot,
            )
            mx.eval(cache.key_caches[0])
            logger.info("Hybrid paged attention Metal kernel warm-up complete")
        except RuntimeError as e:
            if _METAL_LANGUAGE_VERSION_ERROR in str(e):
                raise RuntimeError(
                    f"Metal kernel incompatible with macOS {macos_version}: {e}"
                ) from e
            raise

    def num_blocks(self) -> int:
        return self._num_blocks

    @property
    def kv_cache(self) -> Any:
        return self._kv_cache

    @property
    def linear_cache(self) -> Any:
        return self._linear_cache
