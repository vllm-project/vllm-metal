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
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import torch
from vllm.logger import init_logger
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import MambaSpec

from vllm_metal.attention.caches.kv_cache import MetalPagedKVCache
from vllm_metal.attention.caches.shortconv_cache import ShortConvStateCache
from vllm_metal.attention.context import PagedAttentionContext
from vllm_metal.attention.impls.sdpa import is_sdpa
from vllm_metal.attention.impls.sdpa_wrapper import SDPAPagedAttentionWrapper
from vllm_metal.attention.impls.shortconv import (
    ShortConvPagedWrapper,
    is_shortconv,
)
from vllm_metal.attention.patching import walk_and_wrap
from vllm_metal.attention.runtime.base import PagedAttentionRuntimeBase
from vllm_metal.attention.state import HybridGDNStateManager

logger = init_logger(__name__)


def _build_shortconv_layer_spec(
    *,
    conv_kernel_dim: int,
    conv_dim: int,
    torch_dtype: torch.dtype,
    page_size_padded: int | None = None,
    mamba_block_size: int,
    mamba_cache_mode: str = "none",
) -> MambaSpec:
    """Build the scheduler-visible ShortConv state spec.

    Mirrors upstream's ``ShortConv.get_kv_cache_spec``: one state tensor of
    ``(conv_kernel - 1, conv_dim)`` at the model dtype.  See
    ``_build_linear_layer_spec`` for the ``mamba_block_size`` semantics.
    """
    return MambaSpec(
        shapes=((conv_kernel_dim - 1, conv_dim),),
        dtypes=(torch_dtype,),
        block_size=mamba_block_size,
        page_size_padded=page_size_padded,
        mamba_type=MambaAttentionBackendEnum.SHORT_CONV,
        mamba_cache_mode=mamba_cache_mode,
    )


class ShortConvHybridPagedAttentionRuntime(PagedAttentionRuntimeBase):
    """Paged attention runtime for hybrid SDPA + ShortConv models.

    SDPA layers: paged Metal kernel (via SDPAPagedAttentionWrapper)
    Conv layers: MLX-native state management (via ShortConvPagedWrapper)
    """

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
        if mamba_cache_mode != "none":
            raise NotImplementedError(
                "conv hybrid paged attention does not support "
                f"mamba_cache_mode={mamba_cache_mode!r} (only 'none'); "
                "align-mode conv state caching is not implemented yet"
            )
        self._mamba_cache_mode = mamba_cache_mode
        self._max_num_seqs = max_num_seqs
        self._block_size = block_size
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
        self._conv_indices: list[int] = []
        for i, layer_type in enumerate(layer_types):
            if layer_type == "full_attention":
                self._sdpa_indices.append(i)
            elif layer_type == "conv":
                self._conv_indices.append(i)
            else:
                raise NotImplementedError(
                    f"conv hybrid paged attention: unsupported layer_types "
                    f"entry {layer_type!r} at index {i} (expected "
                    "'full_attention' or 'conv')"
                )

        self._cache = None
        self._state_cache: ShortConvStateCache | None = None
        self._state_manager: HybridGDNStateManager | None = None
        self._scheduler_group_indices = (0,)
        self._group_block_sizes = (block_size,)

    def initialize(self, num_blocks: int) -> None:
        self._cache = MetalPagedKVCache(
            num_layers=len(self._sdpa_indices),
            num_kv_heads=self._num_kv_heads,
            head_dim=self._head_dim,
            num_blocks=num_blocks,
            block_size=self._block_size,
            dtype=self._dtype,
            turboquant=self._turboquant,
            k_quant=self._k_quant,
            v_quant=self._v_quant,
        )

        # None mode keeps one slab per resident request and grows on demand.
        self._state_cache = ShortConvStateCache(
            num_layers=len(self._conv_indices),
            max_seqs=self._max_num_seqs,
            conv_kernel_dim=self._conv_kernel_dim,
            conv_dim=self._conv_dim,
            initial_seqs=0,
            dtype=self._dtype,
        )
        self._state_manager = HybridGDNStateManager(self._state_cache)

        logger.info(
            "ShortConv hybrid cache initialized: %d SDPA layers (%d blocks), "
            "%d conv layers (%d/%d state slots allocated, mamba_cache_mode=%s)",
            len(self._sdpa_indices),
            num_blocks,
            len(self._conv_indices),
            self._state_cache.allocated_seqs,
            self._state_cache.max_seqs,
            self._mamba_cache_mode,
        )

    def adopt_scheduler_group(
        self,
        group_index: int,
        block_size: int,
        *,
        state_group_indices: tuple[int, ...] = (),
        layer_group_ordinals: list[int] | None = None,
        layer_pool_ordinals: list[int] | None = None,
    ) -> None:
        """Select the vLLM scheduler group that owns SDPA KV blocks.

        Conv hybrids run none mode, where conv state lives in a private
        per-request slot pool rather than scheduler blocks, so the mamba
        group arguments the GDN runtime consumes must stay empty here.
        """
        self._require_initialized("adopt_scheduler_group")
        if block_size != self._block_size:
            raise NotImplementedError(
                "conv hybrid paged attention requires the SDPA scheduler group "
                f"block size to stay {self._block_size}, got {block_size}"
            )
        if state_group_indices or layer_group_ordinals or layer_pool_ordinals:
            raise NotImplementedError(
                "conv hybrid paged attention does not consume scheduler mamba "
                "cache groups (none mode keeps a private per-request pool)"
            )
        self._scheduler_group_indices = (group_index,)
        self._group_block_sizes = (block_size,)

    def kv_scheduler_group_indices(self) -> tuple[int, ...]:
        """Return scheduler KV groups consumed by SDPA layers."""
        self._require_initialized("kv_scheduler_group_indices")
        return self._scheduler_group_indices

    def kv_group_block_sizes(self) -> tuple[int, ...]:
        """Return SDPA scheduler group page sizes."""
        self._require_initialized("kv_group_block_sizes")
        return self._group_block_sizes

    def patch_model(self, model: nn.Module) -> int:
        kv_cache = self._require_initialized("patch_model")
        if self._state_cache is None:
            raise RuntimeError("patch_model() called before initialize()")
        state_cache = self._state_cache

        sdpa_cache_map = {
            layer_idx: cache_idx
            for cache_idx, layer_idx in enumerate(self._sdpa_indices)
        }
        conv_cache_map = {
            layer_idx: cache_idx
            for cache_idx, layer_idx in enumerate(self._conv_indices)
        }

        def wrap_layer(layer_idx: int, attn: Any) -> Any:
            if isinstance(attn, SDPAPagedAttentionWrapper):
                # Already patched (cached model reuse) — refresh cache refs.
                cache_idx = sdpa_cache_map.get(layer_idx, layer_idx)
                attn.rebind_cache(kv_cache, self._block_size, cache_idx=cache_idx)
                return attn
            if isinstance(attn, ShortConvPagedWrapper):
                # Already patched — refresh state cache ref.
                cache_idx = conv_cache_map.get(layer_idx, layer_idx)
                object.__setattr__(attn, "_sc_cache_idx", cache_idx)
                object.__setattr__(attn, "_sc_state_cache", state_cache)
                return attn
            if is_sdpa(attn):
                cache_idx = sdpa_cache_map.get(layer_idx, layer_idx)
                return SDPAPagedAttentionWrapper(
                    attn, layer_idx, kv_cache, self._block_size, cache_idx=cache_idx
                )
            if is_shortconv(attn):
                cache_idx = conv_cache_map.get(layer_idx, layer_idx)
                return ShortConvPagedWrapper(attn, cache_idx, state_cache)
            raise RuntimeError(
                f"ShortConv hybrid patch_model: layer {layer_idx} attention "
                f"{type(attn).__name__} is neither SDPA nor ShortConv; "
                "refusing to leave it unpatched (it would silently run unpaged)."
            )

        return walk_and_wrap(model, wrap_layer)

    @property
    def kv_cache(self) -> MetalPagedKVCache:
        return self._require_initialized("kv_cache")

    @property
    def state_cache(self) -> ShortConvStateCache:
        if self._state_cache is None:
            raise RuntimeError("state_cache accessed before initialize()")
        return self._state_cache

    @property
    def state_manager(self) -> HybridGDNStateManager:
        if self._state_manager is None:
            raise RuntimeError("state_manager accessed before initialize()")
        return self._state_manager

    def needs_step_context(self) -> bool:
        return True

    def populate_step_context(
        self,
        *,
        req_ids: list[str],
        ctx: PagedAttentionContext,
        state_block_ids: list[list[list[int]]] | None = None,
        step_positions: list[tuple[int, int]] | None = None,
    ) -> None:
        self.state_manager.populate_step_context(
            req_ids=req_ids,
            ctx=ctx,
            state_block_ids=state_block_ids,
            step_positions=step_positions,
        )

    def extend_forward_eval_outputs(self, outputs: list[mx.array]) -> None:
        self.state_manager.extend_forward_eval_outputs(outputs)

    def release_requests(self, req_ids: set[str]) -> None:
        self.state_manager.release_requests(req_ids)

    def materialize_pending_state(self) -> None:
        self.state_manager.materialize_pending_state()
