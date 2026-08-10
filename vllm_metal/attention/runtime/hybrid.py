# SPDX-License-Identifier: Apache-2.0
"""Paged attention runtime for hybrid models (SDPA + linear attention).

Handles models like Qwen3.5 where some layers use standard dot-product
attention (paged KV cache) and others use GDN linear attention (fixed-size
recurrent state).

SDPA layers use the native Metal SDPA kernel (same as ``MHAPagedAttentionRuntime``).
GDN layers use MLX-native state management via ``GDNPagedAttentionWrapper``.
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

from vllm_metal.attention.caches.gdn_cache import GDNPagedStateCache
from vllm_metal.attention.caches.kv_cache import MetalPagedKVCache
from vllm_metal.attention.context import PagedAttentionContext
from vllm_metal.attention.impls.linear import (
    GDNPagedAttentionWrapper,
    is_linear_attention,
)
from vllm_metal.attention.impls.sdpa import is_sdpa
from vllm_metal.attention.impls.sdpa_wrapper import (
    SDPAPagedAttentionWrapper,
)
from vllm_metal.attention.patching import walk_and_wrap
from vllm_metal.attention.runtime.base import PagedAttentionRuntimeBase
from vllm_metal.attention.state import AlignGDNStateManager, HybridGDNStateManager

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
    mamba_block_size: int,
    mamba_cache_mode: str = "none",
) -> MambaSpec:
    """Build the scheduler-visible GDN state spec.

    A max-length Mamba block gives each request one block in ``none`` mode.
    """
    return MambaSpec(
        shapes=(
            (conv_kernel_dim - 1, conv_dim),
            (num_v_heads, value_head_dim, key_head_dim),
        ),
        # Match the fp32 recurrent pool used for kernel accumulation.
        dtypes=(torch_dtype, torch.float32),
        block_size=mamba_block_size,
        page_size_padded=page_size_padded,
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
        mamba_cache_mode=mamba_cache_mode,
    )


class HybridPagedAttentionRuntime(PagedAttentionRuntimeBase):
    """Paged attention runtime for hybrid SDPA + linear attention models.

    SDPA layers: paged Metal kernel (via SDPAPagedAttentionWrapper)
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
        # Scheduler-side mamba caching strategy ("none" or "align").
        mamba_cache_mode: str = "none",
        # TurboQuant (SDPA layers only)
        turboquant: bool = False,
        k_quant: str | None = None,
        v_quant: str | None = None,
    ) -> None:
        self._max_num_seqs = max_num_seqs
        self._block_size = block_size
        self._dtype = dtype
        if mamba_cache_mode not in ("none", "align"):
            raise NotImplementedError(
                f"hybrid paged attention does not support mamba_cache_mode="
                f"{mamba_cache_mode!r} (only 'none' and 'align')"
            )
        self._mamba_cache_mode = mamba_cache_mode

        # SDPA params
        self._num_kv_heads = num_kv_heads
        self._head_dim = head_dim

        # GDN params
        self._linear_num_v_heads = linear_num_v_heads
        self._linear_key_head_dim = linear_key_head_dim
        self._linear_value_head_dim = linear_value_head_dim
        self._linear_conv_kernel_dim = linear_conv_kernel_dim
        self._linear_conv_dim = linear_conv_dim

        # TurboQuant params (only applies to SDPA layers)
        self._turboquant = turboquant
        self._k_quant = k_quant
        self._v_quant = v_quant

        # Classify layers
        self._sdpa_indices: list[int] = []
        self._linear_indices: list[int] = []
        for i in range(num_layers):
            if (i + 1) % full_attention_interval == 0:
                self._sdpa_indices.append(i)
            else:
                self._linear_indices.append(i)

        self._cache = None
        self._state_cache: GDNPagedStateCache | None = None
        self._gdn_state_manager: HybridGDNStateManager | AlignGDNStateManager | None = (
            None
        )
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

        # Align-mode slabs are addressed directly by scheduler block id; any
        # of the pool's blocks can become a mamba state block (the block pool
        # is fungible across cache groups), so the id space is
        # [0, num_blocks).  The paged plan charges every block for its state
        # bytes (admission worst case), but the pool materializes lazily by
        # high-water block id — vLLM's BlockPool hands out low ids first, so
        # resident state memory tracks the live + cached set instead of
        # wiring the whole worst case up front. Start empty so the scheduler's
        # shared tensor layout is adopted before any physical pool exists.
        # None mode keeps one slab per resident request and grows on demand.
        align = self._mamba_cache_mode == "align"
        state_slots = num_blocks if align else self._max_num_seqs
        self._state_cache = GDNPagedStateCache(
            num_layers=len(self._linear_indices),
            max_seqs=state_slots,
            conv_kernel_dim=self._linear_conv_kernel_dim,
            conv_dim=self._linear_conv_dim,
            num_v_heads=self._linear_num_v_heads,
            value_head_dim=self._linear_value_head_dim,
            key_head_dim=self._linear_key_head_dim,
            initial_seqs=0,
            dtype=self._dtype,
        )
        self._gdn_state_manager = (
            AlignGDNStateManager(self._state_cache, self._block_size)
            if align
            else HybridGDNStateManager(self._state_cache)
        )

        logger.info(
            "Hybrid cache initialized: %d SDPA layers (%d blocks), "
            "%d linear layers (%d/%d GDN slots allocated, mamba_cache_mode=%s)",
            len(self._sdpa_indices),
            num_blocks,
            len(self._linear_indices),
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
        """Select the vLLM scheduler groups backing this runtime.

        ``group_index`` is the group owning SDPA KV blocks (kernel block
        tables); ``state_group_indices`` are the mamba cache groups whose
        block ids key the GDN recurrent state slabs;
        ``layer_group_ordinals[cache_idx]`` records which of those groups
        each linear layer belongs to (the engine stripes same-spec layers
        across several groups) and ``layer_pool_ordinals[cache_idx]`` which
        physical state pool it shares (one pool per within-group position,
        following ``kv_cache_tensors.shared_by``).
        """
        self._require_initialized("adopt_scheduler_group")
        if block_size != self._block_size:
            raise NotImplementedError(
                "hybrid paged attention requires the SDPA scheduler group "
                f"block size to stay {self._block_size}, got {block_size}"
            )
        self._scheduler_group_indices = (group_index,)
        self._group_block_sizes = (block_size,)
        self._state_group_indices = tuple(state_group_indices)
        if layer_group_ordinals is not None:
            pool_ordinals = (
                layer_pool_ordinals
                if layer_pool_ordinals is not None
                else list(range(len(layer_group_ordinals)))
            )
            self.state_cache.set_layer_layout(layer_group_ordinals, pool_ordinals)

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
        linear_cache_map = {
            layer_idx: cache_idx
            for cache_idx, layer_idx in enumerate(self._linear_indices)
        }

        def wrap_layer(layer_idx: int, attn: Any) -> Any:
            if isinstance(attn, SDPAPagedAttentionWrapper):
                # Already patched (cached model reuse) — refresh cache refs.
                cache_idx = sdpa_cache_map.get(layer_idx, layer_idx)
                attn.rebind_cache(kv_cache, self._block_size, cache_idx=cache_idx)
                return attn
            if isinstance(attn, GDNPagedAttentionWrapper):
                # Already patched — refresh state cache ref.
                cache_idx = linear_cache_map.get(layer_idx, layer_idx)
                object.__setattr__(attn, "_gdn_cache_idx", cache_idx)
                object.__setattr__(attn, "_gdn_state_cache", state_cache)
                return attn
            if is_sdpa(attn):
                cache_idx = sdpa_cache_map.get(layer_idx, layer_idx)
                return SDPAPagedAttentionWrapper(
                    attn, layer_idx, kv_cache, self._block_size, cache_idx=cache_idx
                )
            if is_linear_attention(attn):
                cache_idx = linear_cache_map.get(layer_idx, layer_idx)
                return GDNPagedAttentionWrapper(attn, layer_idx, cache_idx, state_cache)
            raise RuntimeError(
                f"Hybrid patch_model: layer {layer_idx} attention "
                f"{type(attn).__name__} is neither SDPA nor linear attention; "
                "refusing to leave it unpatched (it would silently run unpaged)."
            )

        return walk_and_wrap(model, wrap_layer)

    @property
    def kv_cache(self) -> MetalPagedKVCache:
        return self._require_initialized("kv_cache")

    @property
    def state_cache(self) -> GDNPagedStateCache:
        if self._state_cache is None:
            raise RuntimeError("state_cache accessed before initialize()")
        return self._state_cache

    @property
    def gdn_state_manager(self) -> HybridGDNStateManager | AlignGDNStateManager:
        if self._gdn_state_manager is None:
            raise RuntimeError("gdn_state_manager accessed before initialize()")
        return self._gdn_state_manager

    def needs_step_context(self) -> bool:
        return True

    def copy_blocks(self, block_copies: Sequence[tuple[int, int]]) -> None:
        """Apply scheduler CoW copies to SDPA KV and align-mode GDN state."""
        self.kv_cache.copy_blocks(block_copies)
        if self._mamba_cache_mode == "align":
            self.state_cache.copy_blocks(block_copies)

    def populate_step_context(
        self,
        *,
        req_ids: list[str],
        ctx: PagedAttentionContext,
        state_block_ids: list[list[list[int]]] | None = None,
        step_positions: list[tuple[int, int]] | None = None,
    ) -> None:
        self.gdn_state_manager.populate_step_context(
            req_ids=req_ids,
            ctx=ctx,
            state_block_ids=state_block_ids,
            step_positions=step_positions,
        )

    def extend_forward_eval_outputs(self, outputs: list[mx.array]) -> None:
        self.gdn_state_manager.extend_forward_eval_outputs(outputs)

    def release_requests(self, req_ids: set[str]) -> None:
        self.gdn_state_manager.release_requests(req_ids)

    def materialize_pending_state(self) -> None:
        self.gdn_state_manager.materialize_pending_state()
