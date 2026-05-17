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


def warm_up_paged_cache(cache: MetalPagedKVCache) -> None:
    """Front-load v2 paged-attention kernel compilation at startup.

    ``get_ops()`` JIT-builds the C++ ``_paged_ops`` extension and eagerly
    compiles the v2 / GDN / MLA Metal libraries: MLX's
    ``Device::get_library`` compiles the source synchronously inside each
    ``init_*_library`` call. Calling it here moves that cost off the first
    request and fails fast at startup if the kernels cannot compile on this
    macOS (e.g. an unsupported Metal language version).

    The legacy v1 ``reshape_and_cache`` probe is gone: KV writes are now
    MLX-native scatter, and the probe was only needed in v1 because the
    dispatch itself was the compile trigger. Kept as the stable warm-up
    entry point for the MHA and Hybrid backends (``hybrid.py`` /
    ``MHAPagedAttentionBackend``); ``cache`` is intentionally unused.
    """
    del cache
    macos_version = platform.mac_ver()[0]
    logger.info("Warming up v2 paged-attention Metal kernels...")
    try:
        get_ops()
    except Exception as e:
        raise RuntimeError(
            f"Failed to compile paged-attention Metal kernels on "
            f"macOS {macos_version}: {e}"
        ) from e
    logger.info("Paged-attention Metal kernel warm-up complete")


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
        warm_up_paged_cache(self._require_initialized("warm_up"))

    def num_blocks(self) -> int:
        return self._require_initialized("num_blocks").num_blocks
