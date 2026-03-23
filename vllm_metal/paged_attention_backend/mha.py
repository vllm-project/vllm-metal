# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import platform
from typing import Any

import mlx.core as mx
from vllm.logger import init_logger

logger = init_logger(__name__)


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
    ) -> None:
        self._num_layers = num_layers
        self._num_kv_heads = num_kv_heads
        self._head_dim = head_dim
        self._block_size = block_size
        self._dtype = dtype
        self._cache: Any = None

    def initialize(self, num_blocks: int) -> None:
        from vllm_metal.metal_kernel_backend.cache import MetalPagedKVCache

        self._cache = MetalPagedKVCache(
            num_layers=self._num_layers,
            num_kv_heads=self._num_kv_heads,
            head_dim=self._head_dim,
            num_blocks=num_blocks,
            block_size=self._block_size,
            dtype=self._dtype,
        )

    def patch_model(self, model: Any) -> int:
        from vllm_metal.metal_kernel_backend.paged_attention import (
            patch_model_attention_metal_kernel,
        )

        return patch_model_attention_metal_kernel(model, self._cache, self._block_size)

    def warm_up(self) -> None:
        from vllm_metal.metal import get_ops

        cache = self._cache
        logger.info("Warming up paged attention Metal kernel...")

        try:
            ops = get_ops()
        except Exception as e:
            raise RuntimeError(
                f"Failed to load Metal kernel: {e}. macOS {platform.mac_ver()[0]}"
            ) from e

        try:
            dummy_k = mx.zeros((1, cache.num_kv_heads, cache.head_dim), dtype=cache.dtype)
            dummy_v = mx.zeros((1, cache.num_kv_heads, cache.head_dim), dtype=cache.dtype)
            dummy_slot = mx.zeros((1,), dtype=mx.int64)
            mx.eval(dummy_k, dummy_v, dummy_slot)
            ops.reshape_and_cache(
                dummy_k, dummy_v, cache.key_caches[0], cache.value_caches[0], dummy_slot
            )
            mx.eval(cache.key_caches[0])
            logger.info("Paged attention Metal kernel warm-up complete")
        except RuntimeError as e:
            if "language version" in str(e):
                raise RuntimeError(
                    f"Metal kernel incompatible with macOS {platform.mac_ver()[0]}: {e}"
                ) from e
            raise

    def num_blocks(self) -> int:
        return self._cache.num_blocks
