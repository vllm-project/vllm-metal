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

from typing import Any

import mlx.core as mx
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import MambaSpec

from vllm_metal.metal_kernel_backend.cache import MetalPagedKVCache
from vllm_metal.metal_kernel_backend.cache_linear import LinearAttentionCache
from vllm_metal.metal_kernel_backend.paged_attention import (
    patch_model_attention_metal_kernel,
)
from vllm_metal.paged_attention_backend.mha import warm_up_paged_cache

logger = init_logger(__name__)


def build_linear_layer_spec(runner: Any, torch_dtype: Any) -> Any:
    """Build a MambaSpec for one GDN linear attention layer.

    Keeps model-family-specific shape construction out of ModelRunner.
    """
    return MambaSpec(
        shapes=(
            (runner.linear_conv_kernel_dim - 1, runner.linear_conv_dim),
            (
                runner.linear_num_v_heads,
                runner.linear_value_head_dim,
                runner.linear_key_head_dim,
            ),
        ),
        dtypes=(torch_dtype, torch_dtype),
        block_size=1,
    )


class HybridPagedAttentionBackend:
    """Paged attention backend for hybrid SDPA + linear attention models."""

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
        linear_num_k_heads: int,
        linear_num_v_heads: int,
        linear_key_head_dim: int,
        linear_value_head_dim: int,
        linear_conv_kernel_dim: int,
        linear_conv_dim: int,
        # Common
        block_size: int,
        dtype: mx.Dtype,
    ) -> None:
        self._max_num_seqs = max_num_seqs
        self._block_size = block_size
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

        # Classify layers
        self._sdpa_indices: list[int] = []
        self._linear_indices: list[int] = []
        for i in range(num_layers):
            if (i + 1) % full_attention_interval == 0:
                self._sdpa_indices.append(i)
            else:
                self._linear_indices.append(i)

        self._kv_cache: MetalPagedKVCache | None = None
        self._linear_cache: LinearAttentionCache | None = None

    def _require_initialized(self, caller: str) -> MetalPagedKVCache:
        if self._kv_cache is None:
            raise RuntimeError(f"{caller}() called before initialize()")
        return self._kv_cache

    def initialize(self, num_blocks: int) -> None:
        # SDPA cache: paged KV for attention layers
        self._kv_cache = MetalPagedKVCache(
            num_layers=len(self._sdpa_indices),
            num_kv_heads=self._num_kv_heads,
            head_dim=self._head_dim,
            num_blocks=num_blocks,
            block_size=self._block_size,
            dtype=self._dtype,
        )

        # Linear cache: fixed-size recurrent state, one slot per concurrent
        # request.  Uses max_num_seqs (not num_blocks) because linear state
        # is O(1) per request, unlike SDPA KV which grows with seq_len.
        self._linear_cache = LinearAttentionCache(
            num_layers=len(self._linear_indices),
            num_blocks=self._max_num_seqs,
            conv_kernel_dim=self._linear_conv_kernel_dim,
            conv_dim=self._linear_conv_dim,
            num_v_heads=self._linear_num_v_heads,
            value_head_dim=self._linear_value_head_dim,
            key_head_dim=self._linear_key_head_dim,
            dtype=self._dtype,
        )

        logger.info(
            "Hybrid cache initialized: %d SDPA layers (%d blocks), "
            "%d linear layers (%d slots, %.1f MB)",
            len(self._sdpa_indices),
            num_blocks,
            len(self._linear_indices),
            self._max_num_seqs,
            self._linear_cache.bytes_per_block() * self._max_num_seqs / 1024 / 1024,
        )

    def patch_model(self, model: Any) -> int:
        cache = self._require_initialized("patch_model")

        # Map SDPA layer indices to compact cache indices so the wrapper
        # indexes into the compact MetalPagedKVCache correctly.
        sdpa_cache_idx = {
            layer_idx: cache_idx
            for cache_idx, layer_idx in enumerate(self._sdpa_indices)
        }
        return patch_model_attention_metal_kernel(
            model, cache, self._block_size, cache_idx_map=sdpa_cache_idx
        )

    def warm_up(self) -> None:
        warm_up_paged_cache(self._require_initialized("warm_up"))

    def num_blocks(self) -> int:
        return self._require_initialized("num_blocks").num_blocks

    @property
    def kv_cache(self) -> MetalPagedKVCache:
        return self._require_initialized("kv_cache")

    @property
    def linear_cache(self) -> LinearAttentionCache:
        if self._linear_cache is None:
            raise RuntimeError("linear_cache accessed before initialize()")
        return self._linear_cache
