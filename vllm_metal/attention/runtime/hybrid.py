# SPDX-License-Identifier: Apache-2.0
"""Paged SDPA and state-family execution for hybrid models.

The family plan supplies layer roles and wrappers. vLLM supplies the shared
physical allocation and scheduler-owned block IDs for both state and KV.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import KVCacheConfig, MambaSpec

from vllm_metal.attention.caches.kv_cache import MetalPagedKVCache
from vllm_metal.attention.caches.state_cache import PagedStateCache
from vllm_metal.attention.caches.storage import KVCacheStorage
from vllm_metal.attention.context import PagedAttentionContext
from vllm_metal.attention.impls.sdpa import is_sdpa
from vllm_metal.attention.impls.sdpa_wrapper import (
    SDPAPagedAttentionWrapper,
)
from vllm_metal.attention.patching import DEFAULT_ATTN_ATTR_NAMES, walk_and_wrap
from vllm_metal.attention.runtime.base import PagedAttentionRuntimeBase
from vllm_metal.attention.runtime.hybrid_plan import HybridRuntimePlan
from vllm_metal.attention.state import AlignStateManager

logger = init_logger(__name__)


def _declared_state_shapes(cache: Any) -> tuple[tuple[int, ...], ...] | None:
    """Return the loaded model's declared per-request state shapes, or None.

    Multi-array caches (e.g. mlx-lm ``ArraysCache``) expose their components
    through a ``.cache`` sequence whose entries gain a leading batch axis once
    the layer has run; ``None`` entries are slots the layer never filled.
    Caches without that sequence (a plain KV cache) declare no state arrays.
    """
    components = getattr(cache, "cache", None)
    if components is None:
        return None
    return tuple(
        tuple(int(dim) for dim in state.shape[1:])
        for state in components
        if state is not None
    )


class HybridPagedAttentionRuntime(PagedAttentionRuntimeBase):
    """Execute attention and state layers through their registered wrappers."""

    def __init__(
        self,
        *,
        hybrid_plan: HybridRuntimePlan,
        dtype: mx.Dtype,
        # Scheduler-side mamba caching strategy.
        mamba_cache_mode: str = "none",
    ) -> None:
        self._hybrid_plan = hybrid_plan
        self._dtype = dtype
        state_family = self._hybrid_plan.family
        if mamba_cache_mode not in state_family.supported_cache_modes:
            raise NotImplementedError(
                f"hybrid paged attention does not support mamba_cache_mode="
                f"{mamba_cache_mode!r} for the {state_family.label!r} state "
                f"family (supported: {state_family.supported_cache_modes})"
            )
        self._cache = None
        self._state_cache: PagedStateCache | None = None
        self._state_manager: AlignStateManager | None = None
        self._scheduler_group_indices: tuple[int, ...] = ()
        self._group_block_sizes: tuple[int, ...] = ()
        self._geometry_validated = False

    def initialize_from_config(self, config: KVCacheConfig) -> None:
        """Use the upstream allocation for both KV and recurrent state."""
        self.storage = KVCacheStorage(config)
        layer_plan = self._hybrid_plan.layers
        attention_names = [
            f"layers.{i}.self_attn" for i in layer_plan.attention_indices
        ]
        state_names = [
            f"layers.{i}.{self._hybrid_plan.family.layer_name}"
            for i in layer_plan.state_indices
        ]
        self._cache = MetalPagedKVCache.from_upstream(
            self.storage, attention_names, dtype=self._dtype
        )
        self._scheduler_group_indices = tuple(
            i
            for i, group in enumerate(config.kv_cache_groups)
            if any(name in attention_names for name in group.layer_names)
        )
        self._group_block_sizes = tuple(
            config.kv_cache_groups[i].kv_cache_spec.block_size
            for i in self._scheduler_group_indices
        )
        self._state_group_indices = tuple(
            i
            for i, group in enumerate(config.kv_cache_groups)
            if any(name in state_names for name in group.layer_names)
        )
        group_ordinals = [
            next(
                ordinal
                for ordinal, group_id in enumerate(self._state_group_indices)
                if name in config.kv_cache_groups[group_id].layer_names
            )
            for name in state_names
        ]
        self._state_cache = PagedStateCache(
            self.storage.state_views(state_names), group_ordinals
        )
        state_spec = self.storage.specs[state_names[0]]
        assert isinstance(state_spec, MambaSpec)
        self._state_manager = AlignStateManager(
            self._state_cache,
            state_spec.block_size,
            mamba_cache_mode=state_spec.mamba_cache_mode,
        )
        logger.info(
            "Hybrid shared cache: %d blocks, %.2f GiB backing for KV and state",
            config.num_blocks,
            self.storage.nbytes / (1 << 30),
        )

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
        self._validate_declared_state_geometry(model)
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
            if isinstance(attn, SDPAPagedAttentionWrapper):
                attn.rebind_cache(kv_cache, kv_cache.block_size, cache_idx=cache_idx)
                return attn
            if is_sdpa(attn):
                return SDPAPagedAttentionWrapper(
                    attn, layer_idx, kv_cache, kv_cache.block_size, cache_idx=cache_idx
                )
            raise RuntimeError(
                f"Hybrid patch_model: layer {layer_idx} is an attention layer in "
                f"the hybrid plan but {type(attn).__name__} is not SDPA."
            )

        # Stateless layers keep their module; only plan-owned layers are probed.
        return walk_and_wrap(
            model,
            wrap_layer,
            only_layers=[*layer_plan.attention_indices, *layer_plan.state_indices],
            attr_names=(*DEFAULT_ATTN_ATTR_NAMES, self._hybrid_plan.family.layer_name),
        )

    @property
    def kv_cache(self) -> MetalPagedKVCache:
        return self._require_initialized("kv_cache")

    def _validate_declared_state_geometry(self, model: Any) -> None:
        """Cross-check the plan's state geometry against the loaded model.

        The plan derives state shapes and dtypes from model config fields, and
        they now size the engine's shared cache allocation and parameterize the
        strided views into it. The cache the loaded model itself declares (its
        ``make_cache`` shapes after one token) is what generation actually
        feeds, and nothing else consults it — so config-level drift would
        otherwise surface as wrong numerics instead of an error. Skipped for
        models that declare no cache: the plan is the only authority there.
        """
        if self._geometry_validated:
            return
        make_cache = getattr(model, "make_cache", None)
        if make_cache is None:
            logger.debug(
                "%s state geometry check skipped: the loaded model declares "
                "no cache",
                self._hybrid_plan.family.label,
            )
            return
        caches = make_cache()
        plan = self._hybrid_plan
        # mlx-lm models declare one cache per cache-bearing layer in layer
        # order; pure-MLP blocks are skipped (nemotron_h) or don't exist
        # (qwen3_next). The plan's owned layers (state + attention) are
        # exactly the cache-bearing set, so zip them in order.
        owned_indices = sorted(
            [*plan.layers.attention_indices, *plan.layers.state_indices]
        )
        if len(caches) != len(owned_indices):
            raise ValueError(
                f"The loaded model declares {len(caches)} layer caches but "
                f"the {plan.family.label} hybrid plan owns {len(owned_indices)} "
                "cache-bearing layers; the plan and the model disagree on "
                "topology."
            )
        state_caches = {
            layer_idx: caches[owned_indices.index(layer_idx)]
            for layer_idx in plan.layers.state_indices
        }
        try:
            mx.eval(model(mx.array([[0]], dtype=mx.int32), cache=caches))
        except Exception as exc:
            raise RuntimeError(
                "Probing the loaded model's declared cache geometry failed; "
                "the hybrid runtime cross-checks the plan against it at "
                "startup. The same call serves every request, so this is a "
                "model/runtime mismatch, not a probe artifact."
            ) from exc
        geometry = plan.geometry
        for layer_idx in plan.layers.state_indices:
            declared = _declared_state_shapes(state_caches[layer_idx])
            if declared == geometry.state_shapes:
                continue
            declared_text = (
                "no state arrays (a plain KV cache)"
                if declared is None
                else str(declared)
            )
            raise ValueError(
                f"{self._hybrid_plan.family.label} state geometry drift on "
                f"layers.{layer_idx}: the loaded model declares per-request "
                f"state shapes {declared_text}, but the hybrid plan resolved "
                f"{geometry.state_shapes}. The plan sizes the shared cache "
                "allocation and its views, so serving would read and write "
                "wrong offsets. Align the plan and the model config before "
                "starting."
            )
        self._geometry_validated = True

    @property
    def state_cache(self) -> PagedStateCache:
        if self._state_cache is None:
            raise RuntimeError("state_cache accessed before initialize()")
        return self._state_cache

    @property
    def state_manager(self) -> AlignStateManager:
        if self._state_manager is None:
            raise RuntimeError("state_manager accessed before initialize()")
        return self._state_manager

    def needs_step_context(self) -> bool:
        return True

    def zero_blocks(self, block_ids: Sequence[int]) -> None:
        self.state_cache.apply_pending_states(block_ids)
        self.storage.zero_blocks(block_ids)

    def copy_blocks(self, block_copies: Sequence[tuple[int, int]]) -> None:
        """Apply scheduler CoW copies to SDPA KV and align-mode state."""
        self.state_cache.apply_pending_states(
            [slot for pair in block_copies for slot in pair]
        )
        self.storage.copy_blocks(block_copies)

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
        self.storage.depend(outputs)
        outputs.append(self.storage.buffer)
        self.state_manager.extend_forward_eval_outputs(outputs)

    def release_requests(self, req_ids: set[str]) -> None:
        self.state_manager.release_requests(req_ids)

    def materialize_pending_state(self) -> None:
        self.state_manager.materialize_pending_state()
