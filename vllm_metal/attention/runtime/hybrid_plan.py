# SPDX-License-Identifier: Apache-2.0
"""Typed carriers for the hybrid paged runtime.

``HybridLayerPlan`` owns layer topology, ``RecurrentStateGeometry`` owns
state dimensions and ``StateFamilySpec`` owns per-family policy; a family
owner in ``families/`` builds all three into one ``HybridRuntimePlan``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypeAlias

import mlx.nn as nn
import torch
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import MambaSpec

from vllm_metal.attention.caches.gdn_cache import GDNPagedStateCache

LayerRole: TypeAlias = Literal["attention", "state"]

ATTENTION_LAYER: LayerRole = "attention"
STATE_LAYER: LayerRole = "state"


class PagedStateWrapper(Protocol):
    """Construction and rebind contract for a state-family wrapper class."""

    def __init__(
        self,
        inner: nn.Module,
        layer_idx: int,
        cache_idx: int,
        state_cache: GDNPagedStateCache,
    ) -> None: ...

    def rebind_state_cache(
        self, state_cache: GDNPagedStateCache, *, cache_idx: int
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class HybridLayerPlan:
    """Which model layers own paged attention KV pages and which own state.

    ``layer_roles`` is the single source of truth; index tuples and counts
    are derived from it on access.
    """

    layer_roles: tuple[LayerRole, ...]

    def __post_init__(self) -> None:
        if not self.layer_roles:
            raise ValueError("hybrid layer plan requires at least one layer")

    @property
    def attention_indices(self) -> tuple[int, ...]:
        return tuple(
            i for i, role in enumerate(self.layer_roles) if role == ATTENTION_LAYER
        )

    @property
    def state_indices(self) -> tuple[int, ...]:
        return tuple(
            i for i, role in enumerate(self.layer_roles) if role == STATE_LAYER
        )

    @property
    def num_attention(self) -> int:
        return len(self.attention_indices)

    @property
    def num_state(self) -> int:
        return len(self.state_indices)

    def layer_role(self, layer_idx: int) -> LayerRole:
        """Return the role owning ``layer_idx``; consumers dispatch on it."""
        if not 0 <= layer_idx < len(self.layer_roles):
            raise ValueError(
                f"hybrid layer plan covers layers 0..{len(self.layer_roles) - 1}, "
                f"got layer {layer_idx}"
            )
        return self.layer_roles[layer_idx]

    def attention_cache_index(self, layer_idx: int) -> int:
        """Return the compact KV cache ordinal owned by an attention layer."""
        return self._cache_index(self.attention_indices, layer_idx, ATTENTION_LAYER)

    def state_cache_index(self, layer_idx: int) -> int:
        """Return the compact state ordinal owned by a state layer."""
        return self._cache_index(self.state_indices, layer_idx, STATE_LAYER)

    @staticmethod
    def _cache_index(indices: tuple[int, ...], layer_idx: int, role: LayerRole) -> int:
        try:
            return indices.index(layer_idx)
        except ValueError:
            raise ValueError(
                f"hybrid layer plan has no {role} cache index for layer "
                f"{layer_idx}; {role} layers are {indices}"
            ) from None


@dataclass(frozen=True, slots=True)
class RecurrentStateGeometry:
    """Per-layer recurrent state dimensions shared by conv and SSM pools."""

    conv_kernel_dim: int
    conv_dim: int
    num_v_heads: int
    value_head_dim: int
    key_head_dim: int


@dataclass(frozen=True, slots=True)
class StateFamilySpec:
    """Per-family policy: how state layers are detected, wrapped and typed."""

    label: str
    wrapper_cls: type[PagedStateWrapper]
    is_state_module: Callable[[Any], bool]
    mamba_type: MambaAttentionBackendEnum
    recurrent_dtype: torch.dtype
    supported_cache_modes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class HybridRuntimePlan:
    """Everything the hybrid runtime and cache sizing need for one model."""

    layers: HybridLayerPlan
    family: StateFamilySpec
    geometry: RecurrentStateGeometry

    def state_cache_spec(
        self,
        *,
        conv_dtype: torch.dtype,
        mamba_block_size: int,
        page_size_padded: int | None = None,
        mamba_cache_mode: str = "none",
    ) -> MambaSpec:
        """Build the scheduler-visible state spec for one state layer."""
        geometry = self.geometry
        return MambaSpec(
            shapes=(
                (geometry.conv_kernel_dim - 1, geometry.conv_dim),
                (
                    geometry.num_v_heads,
                    geometry.value_head_dim,
                    geometry.key_head_dim,
                ),
            ),
            dtypes=(conv_dtype, self.family.recurrent_dtype),
            block_size=mamba_block_size,
            page_size_padded=page_size_padded,
            mamba_type=self.family.mamba_type,
            mamba_cache_mode=mamba_cache_mode,
        )
