# SPDX-License-Identifier: Apache-2.0
"""Typed carriers for the hybrid paged runtime.

``HybridLayerPlan`` owns layer topology, ``RecurrentStateGeometry`` owns
state dimensions and ``StateFamilySpec`` owns per-family policy; a family
owner in ``families/`` builds all three into one ``HybridRuntimePlan``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import prod
from typing import Any, Literal, Protocol, TypeAlias

import mlx.core as mx
import mlx.nn as nn
import torch
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import MambaSpec

from vllm_metal.attention.caches.protocol import PagedStateCache

LayerRole: TypeAlias = Literal["attention", "state", "stateless"]

ATTENTION_LAYER: LayerRole = "attention"
STATE_LAYER: LayerRole = "state"
STATELESS_LAYER: LayerRole = "stateless"


class PagedStateWrapper(Protocol):
    """Construction and rebind contract for a state-family wrapper class."""

    def __init__(
        self,
        inner: nn.Module,
        layer_idx: int,
        cache_idx: int,
        state_cache: PagedStateCache,
    ) -> None: ...

    def rebind_state_cache(
        self, state_cache: PagedStateCache, *, cache_idx: int
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class HybridLayerPlan:
    """Which model layers own paged KV pages, which own state, and which own neither.

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
        return self.layer_roles[layer_idx]

    def is_state_layer(self, layer_idx: int) -> bool:
        """Return whether ``layer_idx`` is owned by the state runtime."""
        return self.layer_roles[layer_idx] == STATE_LAYER

    def is_attention_layer(self, layer_idx: int) -> bool:
        """Return whether ``layer_idx`` owns paged attention KV pages."""
        return self.layer_roles[layer_idx] == ATTENTION_LAYER

    def attention_cache_index(self, layer_idx: int) -> int:
        """Return the compact KV cache ordinal owned by an attention layer."""
        return self.attention_indices.index(layer_idx)

    def state_cache_index(self, layer_idx: int) -> int:
        """Return the compact state ordinal owned by a state layer."""
        return self.state_indices.index(layer_idx)


@dataclass(frozen=True, slots=True)
class RecurrentStateGeometry:
    """Per-layer recurrent state dimensions shared by conv and SSM pools."""

    conv_kernel_dim: int
    conv_dim: int
    num_v_heads: int
    value_head_dim: int
    key_head_dim: int

    @property
    def state_shapes(self) -> tuple[tuple[int, ...], ...]:
        return (
            (self.conv_kernel_dim - 1, self.conv_dim),
            (self.num_v_heads, self.value_head_dim, self.key_head_dim),
        )


@dataclass(frozen=True, slots=True)
class ConvStateGeometry:
    """Convolution tail dimensions for a family with no recurrent SSM pool."""

    conv_kernel_dim: int
    conv_dim: int

    @property
    def state_shapes(self) -> tuple[tuple[int, ...], ...]:
        return ((self.conv_kernel_dim - 1, self.conv_dim),)


StateGeometry: TypeAlias = RecurrentStateGeometry | ConvStateGeometry


class StateCacheFactory(Protocol):
    """Allocate a family's state pools using its resolved geometry."""

    def __call__(
        self,
        *,
        geometry: StateGeometry,
        num_layers: int,
        max_seqs: int,
        initial_seqs: int,
        dtype: mx.Dtype,
    ) -> PagedStateCache: ...


@dataclass(frozen=True, slots=True)
class StateFamilySpec:
    """Per-family policy: how state layers are detected, wrapped and typed."""

    label: str
    wrapper_cls: type[PagedStateWrapper]
    is_state_module: Callable[[Any], bool]
    mamba_type: MambaAttentionBackendEnum
    # None follows the runtime's convolution dtype; fixed dtypes describe
    # accumulation state such as GDN's fp32 recurrent matrix.
    state_dtypes: tuple[torch.dtype | None, ...]
    supported_cache_modes: tuple[str, ...]
    supports_decode_pipeline: bool
    layer_name: str
    create_state_cache: StateCacheFactory


@dataclass(frozen=True, slots=True)
class HybridRuntimePlan:
    """Everything the hybrid runtime and cache sizing need for one model."""

    layers: HybridLayerPlan
    family: StateFamilySpec
    geometry: StateGeometry

    def state_bytes_per_layer(self, conv_dtype_size: int) -> int:
        """Bytes one request holds in one recurrent state layer."""
        return sum(
            prod(shape) * (conv_dtype_size if dtype is None else dtype.itemsize)
            for shape, dtype in zip(
                self.geometry.state_shapes, self.family.state_dtypes, strict=True
            )
        )

    def state_cache_spec(
        self,
        *,
        conv_dtype: torch.dtype,
        mamba_block_size: int,
        page_size_padded: int | None,
        mamba_cache_mode: str,
    ) -> MambaSpec:
        """Build the scheduler-visible state spec for one state layer."""
        return MambaSpec(
            shapes=self.geometry.state_shapes,
            dtypes=tuple(
                conv_dtype if dtype is None else dtype
                for dtype in self.family.state_dtypes
            ),
            block_size=mamba_block_size,
            page_size_padded=page_size_padded,
            mamba_type=self.family.mamba_type,
            mamba_cache_mode=mamba_cache_mode,
        )
