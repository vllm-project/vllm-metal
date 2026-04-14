# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import platform
from typing import TYPE_CHECKING, Any

import mlx.core as mx
from vllm.logger import init_logger

from vllm_metal.metal import get_ops

if TYPE_CHECKING:
    from vllm_metal.metal_kernel_backend.cache import MetalPagedKVCache

logger = init_logger(__name__)

# Substring present in Metal shader compilation errors when the OS's Metal
# language version is too old for the kernel. Matched against str(e) because
# the C++/nanobind layer does not raise a typed exception for this case.
_METAL_LANGUAGE_VERSION_ERROR = "language version"


def warm_up_paged_cache(cache: MetalPagedKVCache) -> None:
    """Trigger Metal shader compilation with a dummy reshape_and_cache call.

    Shared by MHA and Hybrid backends to avoid duplicating warm-up logic.
    """
    macos_version = platform.mac_ver()[0]
    logger.info("Warming up paged attention Metal kernel...")

    try:
        ops = get_ops()
    except Exception as e:
        raise RuntimeError(
            f"Failed to load Metal kernel: {e}. macOS {macos_version}"
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
        if _METAL_LANGUAGE_VERSION_ERROR in str(e):
            raise RuntimeError(
                f"Metal kernel incompatible with macOS {macos_version}: {e}"
            ) from e
        raise


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
        cache_idx_map: dict[int, int] | None = None,
    ) -> None:
        self._num_layers = num_layers
        self._num_kv_heads = num_kv_heads
        self._head_dim = head_dim
        self._block_size = block_size
        self._dtype = dtype
        self._cache: MetalPagedKVCache | None = None
        self._cache_idx_map = cache_idx_map

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
        warm_up_paged_cache(self._require_initialized("warm_up"))

    def num_blocks(self) -> int:
        return self._require_initialized("num_blocks").num_blocks
