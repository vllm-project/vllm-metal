# SPDX-License-Identifier: Apache-2.0
"""Paged attention runtime for conv hybrid models (SDPA + ShortConv).

Handles models like LFM2/LFM2.5 where ``config.layer_types`` mixes
``"full_attention"`` layers (paged KV cache) and ``"conv"`` layers
(fixed-size causal-conv state).

SDPA layers use the native Metal SDPA kernel (same as
``HybridPagedAttentionRuntime``).  Conv layers use MLX-native state
management via ``ShortConvPagedWrapper`` against a ``ShortConvStateCache``,
driven by the same per-request slot lifecycle GDN hybrids use in
``mamba_cache_mode="none"`` (``HybridGDNStateManager``).

Conv hybrids run none mode only; prefix caching for them is rejected at
config time until the align-mode conv state work lands.
"""

from __future__ import annotations

from collections.abc import Sequence

import mlx.core as mx
from vllm.logger import init_logger

from vllm_metal.attention.caches.kv_cache import MetalPagedKVCache
from vllm_metal.attention.caches.shortconv_cache import ShortConvStateCache
from vllm_metal.attention.impls.shortconv import (
    ShortConvPagedWrapper,
    is_shortconv,
)
from vllm_metal.attention.runtime.base import StateHybridRuntimeBase

logger = init_logger(__name__)


class ShortConvHybridPagedAttentionRuntime(StateHybridRuntimeBase):
    """Paged attention runtime for hybrid SDPA + ShortConv models.

    SDPA layers: paged Metal kernel (via SDPAPagedAttentionWrapper)
    Conv layers: MLX-native state management (via ShortConvPagedWrapper)

    Conv state is not block-keyed yet, so only ``none`` mode is supported;
    align-mode conv caching would add the pooling layout and ``copy_blocks``
    the GDN state cache already carries.
    """

    SUPPORTED_MAMBA_CACHE_MODES = ("none",)
    STATE_WRAPPER = ShortConvPagedWrapper
    STATE_LAYER_DETECTOR = staticmethod(is_shortconv)
    STATE_LAYER_LABEL = "ShortConv"

    def __init__(
        self,
        *,
        layer_types: Sequence[str],
        max_num_seqs: int,
        # SDPA dims
        num_kv_heads: int,
        head_dim: int,
        # ShortConv dims
        conv_kernel_dim: int,
        conv_dim: int,
        # Common
        block_size: int,
        dtype: mx.Dtype,
        # Scheduler-side mamba caching strategy; conv hybrids are none-only.
        mamba_cache_mode: str = "none",
        # TurboQuant (SDPA layers only)
        turboquant: bool = False,
        k_quant: str | None = None,
        v_quant: str | None = None,
    ) -> None:
        self._init_state_hybrid(
            block_size=block_size,
            max_num_seqs=max_num_seqs,
            mamba_cache_mode=mamba_cache_mode,
        )
        self._dtype = dtype

        # SDPA params
        self._num_kv_heads = num_kv_heads
        self._head_dim = head_dim

        # ShortConv params
        self._conv_kernel_dim = conv_kernel_dim
        self._conv_dim = conv_dim

        # TurboQuant params (only applies to SDPA layers)
        self._turboquant = turboquant
        self._k_quant = k_quant
        self._v_quant = v_quant

        # Classify layers from the config's layer_types list.
        self._sdpa_indices: list[int] = []
        self._state_indices: list[int] = []
        for i, layer_type in enumerate(layer_types):
            if layer_type == "full_attention":
                self._sdpa_indices.append(i)
            elif layer_type == "conv":
                self._state_indices.append(i)
            else:
                raise NotImplementedError(
                    f"conv hybrid paged attention: unsupported layer_types "
                    f"entry {layer_type!r} at index {i} (expected "
                    "'full_attention' or 'conv')"
                )

        self._cache = None

    def initialize(self, num_blocks: int) -> None:
        self._init_sdpa_kv_cache(num_blocks)

        self._initialize_state_lifecycle(num_blocks)

        logger.info(
            "ShortConv hybrid cache initialized: %d SDPA layers (%d blocks), "
            "%d conv layers (%d/%d state slots allocated, mamba_cache_mode=%s)",
            len(self._sdpa_indices),
            num_blocks,
            len(self._state_indices),
            self._state_cache.allocated_seqs,
            self._state_cache.max_seqs,
            self._mamba_cache_mode,
        )

    @property
    def kv_cache(self) -> MetalPagedKVCache:
        return self._require_initialized("kv_cache")

    def _new_state_cache(
        self, *, max_seqs: int, initial_seqs: int
    ) -> ShortConvStateCache:
        return ShortConvStateCache(
            num_layers=len(self._state_indices),
            max_seqs=max_seqs,
            conv_kernel_dim=self._conv_kernel_dim,
            conv_dim=self._conv_dim,
            initial_seqs=initial_seqs,
            dtype=self._dtype,
        )
