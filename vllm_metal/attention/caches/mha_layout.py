# SPDX-License-Identifier: Apache-2.0
"""Immutable Metal layout translated from vLLM's standard MHA cache DTOs.

vLLM owns KV-cache grouping and capacity planning. This module only validates
and translates the resulting ``KVCacheConfig`` for the standard mixed
full/sliding-window MHA path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheTensor,
    SlidingWindowSpec,
)

NO_SLIDING_WINDOW = -1
StandardMHASpec: TypeAlias = FullAttentionSpec | SlidingWindowSpec


@dataclass(frozen=True, slots=True)
class MHAGroupLayout:
    """vLLM cache-group specs and layer-to-group mapping."""

    specs: tuple[StandardMHASpec, ...]
    layer_indices: dict[str, int]


@dataclass(frozen=True, slots=True)
class MHATensorLayout:
    """vLLM physical tensor slots and layer-to-slot mapping."""

    layer_indices: dict[str, int]
    slot_layers: tuple[tuple[int, ...], ...]


@dataclass(frozen=True, slots=True)
class MHALayerKVLayout:
    """KV-cache shape and vLLM mapping for one model layer."""

    tensor_index: int
    group_index: int
    block_size: int
    num_kv_heads: int
    head_dim: int
    sliding_window: int

    def cache_shape(self, num_blocks: int) -> tuple[int, int, int, int]:
        """Return the key or value cache shape for ``num_blocks`` pages."""
        return (num_blocks, self.block_size, self.num_kv_heads, self.head_dim)


@dataclass(frozen=True, slots=True)
class MHAKVCacheLayout:
    """Immutable standard-MHA cache layout derived from vLLM's DTOs."""

    num_blocks: int
    tensor_sizes: tuple[int, ...]
    layers: tuple[MHALayerKVLayout, ...]
    group_block_sizes: tuple[int, ...]
    slot_layers: tuple[tuple[int, ...], ...]

    @property
    def total_bytes(self) -> int:
        """Return the aggregate storage vLLM allocated for KV tensors."""
        return sum(self.tensor_sizes)

    @classmethod
    def from_config(
        cls, config: KVCacheConfig, model_layer_names: tuple[str, ...]
    ) -> MHAKVCacheLayout:
        """Translate a standard mixed-MHA ``KVCacheConfig`` without regrouping.

        ``model_layer_names`` is the runner's ordered attention-layer sequence.
        Each layer must occur exactly once in vLLM's group and tensor mappings.
        """
        return MHAKVCacheLayoutTranslator(config, model_layer_names).translate()


@dataclass(frozen=True, slots=True)
class MHAKVCacheLayoutTranslator:
    """Translate vLLM's standard-MHA KV cache config into Metal's layout DTO."""

    config: KVCacheConfig
    model_layer_names: tuple[str, ...]

    def translate(self) -> MHAKVCacheLayout:
        """Translate without changing vLLM's grouping."""
        group_layout = self._group_layout()
        self._require_model_layers(group_layout.layer_indices, "group")

        tensor_layout = self._tensor_layout(group_layout)
        self._require_model_layers(tensor_layout.layer_indices, "tensor")

        return MHAKVCacheLayout(
            num_blocks=self.config.num_blocks,
            tensor_sizes=tuple(tensor.size for tensor in self.config.kv_cache_tensors),
            layers=self._layer_layouts(group_layout, tensor_layout),
            group_block_sizes=tuple(spec.block_size for spec in group_layout.specs),
            slot_layers=tensor_layout.slot_layers,
        )

    @property
    def _model_layer_indices(self) -> dict[str, int]:
        return {name: index for index, name in enumerate(self.model_layer_names)}

    def _group_layout(self) -> MHAGroupLayout:
        specs: list[StandardMHASpec] = []
        layer_indices: dict[str, int] = {}
        for group_index, group in enumerate(self.config.kv_cache_groups):
            spec = group.kv_cache_spec
            if not isinstance(spec, (FullAttentionSpec, SlidingWindowSpec)):
                raise NotImplementedError(
                    "standard MHA layout requires FullAttentionSpec or "
                    "SlidingWindowSpec groups"
                )
            if spec.head_size_v != spec.head_size:
                raise NotImplementedError(
                    "standard MHA layout requires matching key and value head sizes"
                )

            specs.append(spec)
            for layer_name in group.layer_names:
                layer_indices[layer_name] = group_index
        return MHAGroupLayout(specs=tuple(specs), layer_indices=layer_indices)

    def _tensor_layout(self, group_layout: MHAGroupLayout) -> MHATensorLayout:
        layer_indices: dict[str, int] = {}
        slot_layers: list[tuple[int, ...]] = []
        model_layer_indices = self._model_layer_indices

        for tensor_index, tensor in enumerate(self.config.kv_cache_tensors):
            if tensor.offset != 0 or tensor.block_stride != 0:
                raise NotImplementedError(
                    "standard MHA layout does not support KV tensors with "
                    "offset and block_stride"
                )

            tensor_layer_indices: list[int] = []
            for layer_name in tensor.shared_by:
                group_index = group_layout.layer_indices[layer_name]
                group_spec = group_layout.specs[group_index]
                self._require_tensor_size(tensor, tensor_index, group_spec, layer_name)
                layer_indices[layer_name] = tensor_index
                tensor_layer_indices.append(model_layer_indices[layer_name])
            slot_layers.append(tuple(tensor_layer_indices))

        return MHATensorLayout(
            layer_indices=layer_indices,
            slot_layers=tuple(slot_layers),
        )

    def _layer_layouts(
        self,
        group_layout: MHAGroupLayout,
        tensor_layout: MHATensorLayout,
    ) -> tuple[MHALayerKVLayout, ...]:
        return tuple(
            self._layer_layout(layer_name, group_layout, tensor_layout)
            for layer_name in self.model_layer_names
        )

    def _layer_layout(
        self,
        layer_name: str,
        group_layout: MHAGroupLayout,
        tensor_layout: MHATensorLayout,
    ) -> MHALayerKVLayout:
        group_index = group_layout.layer_indices[layer_name]
        spec = group_layout.specs[group_index]
        return MHALayerKVLayout(
            tensor_index=tensor_layout.layer_indices[layer_name],
            group_index=group_index,
            block_size=spec.block_size,
            num_kv_heads=spec.num_kv_heads,
            head_dim=spec.head_size,
            sliding_window=(
                spec.sliding_window
                if isinstance(spec, SlidingWindowSpec)
                else NO_SLIDING_WINDOW
            ),
        )

    def _require_model_layers(self, layer_mapping: dict[str, int], source: str) -> None:
        if set(layer_mapping) != set(self.model_layer_names):
            raise ValueError(
                f"{source} layer mapping must contain the same layers as "
                "model_layer_names"
            )

    def _require_tensor_size(
        self,
        tensor: KVCacheTensor,
        tensor_index: int,
        group_spec: StandardMHASpec,
        layer_name: str,
    ) -> None:
        expected_size = self.config.num_blocks * group_spec.page_size_bytes
        if tensor.size != expected_size:
            raise ValueError(
                f"KV cache tensor {tensor_index} size {tensor.size} does not "
                f"match {self.config.num_blocks} blocks of "
                f"{group_spec.page_size_bytes} bytes for layer {layer_name!r}"
            )
