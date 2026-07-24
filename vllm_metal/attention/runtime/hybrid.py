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
from vllm.v1.kv_cache_interface import KVCacheConfig, MambaSpec

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
from vllm_metal.attention.state import HybridGDNStateManager

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
    block_size: int,
    mamba_cache_mode: str = "none",
) -> MambaSpec:
    """Build a MambaSpec for one GDN linear attention layer.

    Args:
        page_size_padded: Optional padded page size from cache_config to
            align Mamba page size with attention page size in hybrid models.
        block_size: Tokens per block.  Must match the SDPA block_size so
            the scheduler's unified block pool can serve both layer types
            without running out of blocks prematurely.
    """
    return MambaSpec(
        shapes=(
            (conv_kernel_dim - 1, conv_dim),
            (num_v_heads, value_head_dim, key_head_dim),
        ),
        # The Metal GDN runtime accumulates recurrent state in float32 even when
        # convolution state follows the model/KV dtype. Scheduler sizing must
        # describe the arrays that are actually snapshotted.
        dtypes=(torch_dtype, torch.float32),
        block_size=block_size,
        page_size_padded=page_size_padded,
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
        # TurboQuant (SDPA layers only)
        turboquant: bool = False,
        k_quant: str | None = None,
        v_quant: str | None = None,
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
        self._gdn_state_manager: HybridGDNStateManager | None = None

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

        self._state_cache = GDNPagedStateCache(
            num_layers=len(self._linear_indices),
            max_seqs=self._max_num_seqs,
            conv_kernel_dim=self._linear_conv_kernel_dim,
            conv_dim=self._linear_conv_dim,
            num_v_heads=self._linear_num_v_heads,
            value_head_dim=self._linear_value_head_dim,
            key_head_dim=self._linear_key_head_dim,
            initial_seqs=0,
            dtype=self._dtype,
        )
        self._gdn_state_manager = HybridGDNStateManager(
            self._state_cache,
            block_size=self._block_size,
        )

        logger.info(
            "Hybrid cache initialized: %d SDPA layers (%d blocks), "
            "%d linear layers (%d/%d GDN slots allocated)",
            len(self._sdpa_indices),
            num_blocks,
            len(self._linear_indices),
            self._state_cache.allocated_seqs,
            self._max_num_seqs,
        )

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
    def gdn_state_manager(self) -> HybridGDNStateManager:
        if self._gdn_state_manager is None:
            raise RuntimeError("gdn_state_manager accessed before initialize()")
        return self._gdn_state_manager

    def needs_step_context(self) -> bool:
        return True

    def populate_step_context(
        self, *, req_ids: list[str], ctx: PagedAttentionContext
    ) -> None:
        self.gdn_state_manager.populate_step_context(req_ids=req_ids, ctx=ctx)

    def extend_forward_eval_outputs(self, outputs: list[mx.array]) -> None:
        self.gdn_state_manager.extend_forward_eval_outputs(outputs)

    def release_requests(self, req_ids: set[str]) -> None:
        self.gdn_state_manager.release_requests(req_ids)

    def configure_cache_groups(self, kv_cache_config: KVCacheConfig) -> None:
        linear_name_to_cache_idx = {
            f"layers.{model_layer_idx}.linear_attn": cache_idx
            for cache_idx, model_layer_idx in enumerate(self._linear_indices)
        }
        mamba_group_layers: dict[int, tuple[int, ...]] = {}
        for group_idx, group in enumerate(kv_cache_config.kv_cache_groups):
            if not isinstance(group.kv_cache_spec, MambaSpec):
                continue
            if group.kv_cache_spec.mamba_cache_mode != "align":
                continue
            unknown_layers = set(group.layer_names) - set(linear_name_to_cache_idx)
            if unknown_layers:
                raise RuntimeError(
                    "Scheduler Mamba group contains unknown Metal GDN layers: "
                    f"{sorted(unknown_layers)}"
                )
            layer_indices = tuple(
                linear_name_to_cache_idx[layer_name] for layer_name in group.layer_names
            )
            if not layer_indices:
                raise RuntimeError("Scheduler Mamba cache group has no GDN layers")
            mamba_group_layers[group_idx] = layer_indices

        if not mamba_group_layers:
            return
        self.gdn_state_manager.configure_cache_groups(
            num_blocks=kv_cache_config.num_blocks,
            block_size=self._block_size,
            mamba_group_layers=mamba_group_layers,
        )

    def invalidate_blocks(self, block_ids: Sequence[int]) -> None:
        self.gdn_state_manager.invalidate_blocks(block_ids)

    def restore_prefix(
        self,
        req_id: str,
        block_tables: Sequence[Sequence[int]],
        num_computed_tokens: int,
    ) -> bool:
        return self.gdn_state_manager.restore_prefix(
            req_id,
            block_tables,
            num_computed_tokens,
        )

    def checkpoint_blocks(
        self,
        checkpoints: Sequence[tuple[str, Sequence[Sequence[int]], int]],
    ) -> None:
        self.gdn_state_manager.checkpoint_blocks(checkpoints)

    def materialize_pending_state(self) -> None:
        self.gdn_state_manager.materialize_pending_state()
