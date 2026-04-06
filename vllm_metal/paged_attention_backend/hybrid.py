# SPDX-License-Identifier: Apache-2.0
"""Paged attention backend for hybrid models (SDPA + linear attention).

Handles models like Qwen3.5 where some layers use standard dot-product
attention (paged KV cache) and others use GDN linear attention (fixed-size
recurrent state).

SDPA layers use the Metal kernel backend (same as ``MHAPagedAttentionBackend``).
GDN layers use MLX-native state management via ``GDNPagedAttentionWrapper``.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import torch
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import MambaSpec

from vllm_metal.metal_kernel_backend.attention_linear import (
    GDNPagedAttentionWrapper,
    is_linear_attention,
)
from vllm_metal.metal_kernel_backend.attention_sdpa import is_sdpa
from vllm_metal.metal_kernel_backend.cache import MetalPagedKVCache
from vllm_metal.metal_kernel_backend.paged_attention import (
    MetalKernelPagedAttentionWrapper,
)
from vllm_metal.mlx_backend.gdn_cache import GDNPagedStateCache
from vllm_metal.paged_attention_backend.mha import warm_up_paged_cache
from vllm_metal.paged_attention_common import find_attn_attr, find_layers

logger = init_logger(__name__)


def _build_linear_layer_spec(
    *,
    conv_kernel_dim: int,
    conv_dim: int,
    num_v_heads: int,
    value_head_dim: int,
    key_head_dim: int,
    torch_dtype: torch.dtype,
    page_size_padded: int | None = None,
) -> MambaSpec:
    """Build a MambaSpec for one GDN linear attention layer.

    Args:
        page_size_padded: Optional padded page size from cache_config to
            align Mamba page size with attention page size in hybrid models.
    """
    return MambaSpec(
        shapes=(
            (conv_kernel_dim - 1, conv_dim),
            (num_v_heads, value_head_dim, key_head_dim),
        ),
        dtypes=(torch_dtype, torch_dtype),
        block_size=1,
        page_size_padded=page_size_padded,
    )


class HybridPagedAttentionBackend:
    """Paged attention backend for hybrid SDPA + linear attention models.

    SDPA layers: paged Metal kernel (via MetalKernelPagedAttentionWrapper)
    GDN layers: MLX-native state management (via GDNPagedAttentionWrapper)
    """

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
        self._state_cache: GDNPagedStateCache | None = None

    def _require_initialized(self, caller: str) -> MetalPagedKVCache:
        if self._kv_cache is None:
            raise RuntimeError(f"{caller}() called before initialize()")
        return self._kv_cache

    def initialize(self, num_blocks: int) -> None:
        self._kv_cache = MetalPagedKVCache(
            num_layers=len(self._sdpa_indices),
            num_kv_heads=self._num_kv_heads,
            head_dim=self._head_dim,
            num_blocks=num_blocks,
            block_size=self._block_size,
            dtype=self._dtype,
        )

        self._state_cache = GDNPagedStateCache(
            num_layers=len(self._linear_indices),
            max_seqs=self._max_num_seqs,
            conv_kernel_dim=self._linear_conv_kernel_dim,
            conv_dim=self._linear_conv_dim,
            num_v_heads=self._linear_num_v_heads,
            value_head_dim=self._linear_value_head_dim,
            key_head_dim=self._linear_key_head_dim,
            dtype=self._dtype,
        )

        logger.info(
            "Hybrid cache initialized: %d SDPA layers (%d blocks), "
            "%d linear layers (%d slots)",
            len(self._sdpa_indices),
            num_blocks,
            len(self._linear_indices),
            self._max_num_seqs,
        )

    def patch_model(self, model: nn.Module) -> int:
        kv_cache = self._require_initialized("patch_model")
        if self._state_cache is None:
            raise RuntimeError("patch_model() called before initialize()")

        sdpa_cache_map = {
            layer_idx: cache_idx
            for cache_idx, layer_idx in enumerate(self._sdpa_indices)
        }
        linear_cache_map = {
            layer_idx: cache_idx
            for cache_idx, layer_idx in enumerate(self._linear_indices)
        }

        patched = 0
        for layer_idx, layer in enumerate(find_layers(model)):
            attn_attr = find_attn_attr(layer)
            if attn_attr is None:
                continue

            attn = getattr(layer, attn_attr)

            if isinstance(attn, MetalKernelPagedAttentionWrapper):
                # Already patched (cached model reuse) — refresh cache refs
                object.__setattr__(attn, "_mk_kv_cache", kv_cache)
                object.__setattr__(attn, "_mk_block_size", self._block_size)
                cache_idx = sdpa_cache_map.get(layer_idx, layer_idx)
                object.__setattr__(attn, "_mk_cache_idx", cache_idx)
                patched += 1
            elif isinstance(attn, GDNPagedAttentionWrapper):
                # Already patched — refresh state cache ref
                cache_idx = linear_cache_map.get(layer_idx, layer_idx)
                object.__setattr__(attn, "_gdn_cache_idx", cache_idx)
                object.__setattr__(attn, "_gdn_state_cache", self._state_cache)
                patched += 1
            elif is_sdpa(attn):
                cache_idx = sdpa_cache_map.get(layer_idx, layer_idx)
                wrapper = MetalKernelPagedAttentionWrapper(
                    attn, layer_idx, kv_cache, self._block_size, cache_idx=cache_idx
                )
                setattr(layer, attn_attr, wrapper)
                patched += 1
            elif is_linear_attention(attn):
                cache_idx = linear_cache_map.get(layer_idx, layer_idx)
                wrapper = GDNPagedAttentionWrapper(
                    attn, layer_idx, cache_idx, self._state_cache
                )
                setattr(layer, attn_attr, wrapper)
                patched += 1

        return patched

    def warm_up(self) -> None:
        warm_up_paged_cache(self._require_initialized("warm_up"))

    def num_blocks(self) -> int:
        return self._require_initialized("num_blocks").num_blocks

    @property
    def kv_cache(self) -> MetalPagedKVCache:
        return self._require_initialized("kv_cache")

    @property
    def state_cache(self) -> GDNPagedStateCache:
        if self._state_cache is None:
            raise RuntimeError("state_cache accessed before initialize()")
        return self._state_cache
