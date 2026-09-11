# SPDX-License-Identifier: Apache-2.0
"""Paged attention runtimes for hybrid full and recurrent attention models.

Qwen hybrids combine SDPA with GDN linear attention. Bailing V3 combines MLA
with KDA. Full-attention layers use paged KV or latent caches, while recurrent
layers use scheduler-managed fixed-size state.

The hybrid plan selects recurrent wrappers; subclasses select full-attention
caches and wrappers.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from vllm.logger import init_logger

from vllm_metal.attention.caches.kv_cache import MetalPagedKVCache
from vllm_metal.attention.caches.mla_cache import MLAPagedLatentCache
from vllm_metal.attention.caches.protocol import PagedStateCache
from vllm_metal.attention.context import PagedAttentionContext
from vllm_metal.attention.impls.mla import MLAPagedAttentionWrapper
from vllm_metal.attention.impls.sdpa import is_sdpa
from vllm_metal.attention.impls.sdpa_wrapper import (
    SDPAPagedAttentionWrapper,
)
from vllm_metal.attention.patching import DEFAULT_ATTN_ATTR_NAMES, walk_and_wrap
from vllm_metal.attention.runtime.base import PagedAttentionRuntimeBase
from vllm_metal.attention.runtime.hybrid_plan import HybridRuntimePlan
from vllm_metal.attention.state import AlignStateManager, RequestStateManager
from vllm_metal.pytorch_backend.tensor_bridge import TORCH_TO_MLX_DTYPE

logger = init_logger(__name__)


class HybridPagedAttentionRuntime(PagedAttentionRuntimeBase):
    """Shared hybrid state management with SDPA as the default attention path.

    State geometry and wrappers come from the family plan. Subclasses override
    attention cache allocation and wrapping without duplicating state handling.
    """

    def __init__(
        self,
        *,
        hybrid_plan: HybridRuntimePlan,
        max_num_seqs: int,
        # Full-attention cache dimensions
        num_kv_heads: int,
        head_dim: int,
        # Common
        block_size: int,
        dtype: mx.Dtype,
        # Scheduler-side mamba caching strategy.
        mamba_cache_mode: str = "none",
        # TurboQuant (SDPA attention only)
        turboquant: bool = False,
        k_quant: str | None = None,
        v_quant: str | None = None,
    ) -> None:
        self._hybrid_plan = hybrid_plan
        self._max_num_seqs = max_num_seqs
        self._block_size = block_size
        self._dtype = dtype
        state_family = self._hybrid_plan.family
        if mamba_cache_mode not in state_family.supported_cache_modes:
            raise NotImplementedError(
                f"hybrid paged attention does not support mamba_cache_mode="
                f"{mamba_cache_mode!r} for the {state_family.label!r} state "
                f"family (supported: {state_family.supported_cache_modes})"
            )
        self._mamba_cache_mode = mamba_cache_mode

        # Full-attention cache parameters
        self._num_kv_heads = num_kv_heads
        self._head_dim = head_dim

        # TurboQuant parameters (SDPA attention only)
        self._turboquant = turboquant
        self._k_quant = k_quant
        self._v_quant = v_quant

        self._cache = None
        self._state_cache: PagedStateCache | None = None
        self._state_manager: RequestStateManager | AlignStateManager | None = None
        self._scheduler_group_indices = (0,)
        self._group_block_sizes = (block_size,)

    def _create_attention_cache(
        self, num_blocks: int
    ) -> MetalPagedKVCache | MLAPagedLatentCache:
        return MetalPagedKVCache(
            num_layers=self._hybrid_plan.layers.num_attention,
            num_kv_heads=self._num_kv_heads,
            head_dim=self._head_dim,
            num_blocks=num_blocks,
            block_size=self._block_size,
            dtype=self._dtype,
            turboquant=self._turboquant,
            k_quant=self._k_quant,
            v_quant=self._v_quant,
        )

    def initialize(self, num_blocks: int) -> None:
        self._cache = self._create_attention_cache(num_blocks)

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
        self._state_cache = self._hybrid_plan.family.create_state_cache(
            geometry=self._hybrid_plan.geometry,
            num_layers=self._hybrid_plan.layers.num_state,
            max_seqs=state_slots,
            initial_seqs=0,
            dtypes=tuple(
                TORCH_TO_MLX_DTYPE[dtype] for dtype in self._hybrid_plan.state_dtypes
            ),
        )
        self._state_manager = (
            AlignStateManager(self._state_cache, self._block_size)
            if align
            else RequestStateManager(self._state_cache)
        )

        logger.info(
            "Hybrid cache initialized: %d attention layers (%d blocks), "
            "%d %s layers (%d/%d state slots allocated, mamba_cache_mode=%s)",
            self._hybrid_plan.layers.num_attention,
            num_blocks,
            self._hybrid_plan.layers.num_state,
            self._hybrid_plan.family.label,
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

        ``group_index`` is the group owning full-attention cache blocks;
        ``state_group_indices`` are the mamba cache groups whose block ids key
        the recurrent state slabs;
        ``layer_group_ordinals[cache_idx]`` records which of those groups
        each state layer belongs to (the engine stripes same-spec layers
        across several groups) and ``layer_pool_ordinals[cache_idx]`` which
        physical state pool it shares (one pool per within-group position,
        following the ``kv_cache_tensors`` byte addresses).
        """
        self._require_initialized("adopt_scheduler_group")
        if block_size != self._block_size:
            raise NotImplementedError(
                "hybrid paged attention requires the full-attention scheduler group "
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
        """Return the scheduler group consumed by full-attention layers."""
        self._require_initialized("kv_scheduler_group_indices")
        return self._scheduler_group_indices

    def kv_group_block_sizes(self) -> tuple[int, ...]:
        """Return full-attention scheduler group page sizes."""
        self._require_initialized("kv_group_block_sizes")
        return self._group_block_sizes

    def patch_model(self, model: nn.Module) -> int:
        self._require_initialized("patch_model")
        state_cache = self.state_cache
        layer_plan = self._hybrid_plan.layers
        state_family = self._hybrid_plan.family

        def wrap_layer(layer_idx: int, attn: Any) -> Any:
            if layer_plan.is_state_layer(layer_idx):
                cache_idx = layer_plan.state_cache_index(layer_idx)
                if isinstance(attn, state_family.wrapper_cls):
                    attn.rebind_state_cache(state_cache, cache_idx=cache_idx)
                    return attn
                if state_family.is_state_module(attn):
                    return state_family.wrapper_cls(
                        attn, layer_idx, cache_idx, state_cache
                    )
                raise RuntimeError(
                    f"Hybrid patch_model: layer {layer_idx} is a state layer in "
                    f"the hybrid plan but {type(attn).__name__} is not a "
                    f"{state_family.label!r} state module."
                )
            cache_idx = layer_plan.attention_cache_index(layer_idx)
            return self._wrap_attention_layer(layer_idx, attn, cache_idx)

        # Stateless layers keep their module; only plan-owned layers are probed.
        return walk_and_wrap(
            model,
            wrap_layer,
            only_layers=[*layer_plan.attention_indices, *layer_plan.state_indices],
            attr_names=(*DEFAULT_ATTN_ATTR_NAMES, self._hybrid_plan.family.layer_name),
        )

    def _wrap_attention_layer(self, layer_idx: int, attn: Any, cache_idx: int) -> Any:
        kv_cache = self.kv_cache
        if isinstance(attn, SDPAPagedAttentionWrapper):
            attn.rebind_cache(kv_cache, self._block_size, cache_idx=cache_idx)
            return attn
        if is_sdpa(attn):
            return SDPAPagedAttentionWrapper(
                attn, layer_idx, kv_cache, self._block_size, cache_idx=cache_idx
            )
        raise RuntimeError(
            f"Hybrid patch_model: layer {layer_idx} is an attention layer in "
            f"the hybrid plan but {type(attn).__name__} is not SDPA."
        )

    @property
    def kv_cache(self) -> MetalPagedKVCache | MLAPagedLatentCache:
        return self._require_initialized("kv_cache")

    @property
    def state_cache(self) -> PagedStateCache:
        if self._state_cache is None:
            raise RuntimeError("state_cache accessed before initialize()")
        return self._state_cache

    @property
    def state_manager(self) -> RequestStateManager | AlignStateManager:
        if self._state_manager is None:
            raise RuntimeError("state_manager accessed before initialize()")
        return self._state_manager

    def needs_step_context(self) -> bool:
        return True

    def copy_blocks(self, block_copies: Sequence[tuple[int, int]]) -> None:
        """Apply scheduler CoW copies to paged and align-mode recurrent state."""
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


class MLAHybridPagedAttentionRuntime(HybridPagedAttentionRuntime):
    """Hybrid state management with an MLA latent attention cache."""

    def _create_attention_cache(self, num_blocks: int) -> MLAPagedLatentCache:
        return MLAPagedLatentCache(
            num_layers=self._hybrid_plan.layers.num_attention,
            latent_dim=self._head_dim,
            num_blocks=num_blocks,
            block_size=self._block_size,
            dtype=self._dtype,
        )

    def _wrap_attention_layer(
        self, layer_idx: int, attn: Any, cache_idx: int
    ) -> MLAPagedAttentionWrapper:
        latent_cache = self.kv_cache
        if isinstance(attn, MLAPagedAttentionWrapper):
            attn.rebind_cache(latent_cache, cache_idx=cache_idx)
            return attn
        return MLAPagedAttentionWrapper(attn, cache_idx, latent_cache)

    @property
    def kv_cache(self) -> MLAPagedLatentCache:
        return self._require_initialized("kv_cache")
