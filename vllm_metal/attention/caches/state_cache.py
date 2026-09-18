# SPDX-License-Identifier: Apache-2.0
"""Operator-state views and deferred MLX updates over scheduler-owned pages."""

from __future__ import annotations

import functools
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import mlx.core as mx

if TYPE_CHECKING:
    from vllm_metal.attention.context import PagedAttentionContext


@functools.cache
def _native_row_scatter() -> Callable[..., mx.array]:
    """Return the required in-place row-scatter primitive."""
    from vllm_metal.metal import get_ops

    return get_ops().gdn_state_scatter


@dataclass(frozen=True)
class GDNDecodeStateView:
    """State array and slot mappings for one lazy decode kernel launch."""

    state: mx.array
    state_slot_ids: mx.array
    cache_slot_ids: mx.array
    uses_compact_state: bool


class PagedStateCache:
    """Bind state components; allocation and block lifetime belong to vLLM."""

    def __init__(self, states, group_ordinals=None):
        if len(states) not in (1, 2):
            raise ValueError("state cache requires conv and optional recurrent views")
        self.conv_states = states[0]
        self.recurrent_states = states[1] if len(states) > 1 else []
        self.num_layers = len(self.conv_states)
        self.max_seqs = self.conv_states[0].shape[0]
        self.conv_dim = self.conv_states[0].shape[2]
        self.dtype = self.conv_states[0].dtype
        self._layer_group_ordinals = group_ordinals or [0] * self.num_layers
        self.pending_conv_states = [None] * self.num_layers
        self.pending_conv_slot_ids = [None] * self.num_layers
        self.pending_recurrent_states = [None] * self.num_layers
        self.pending_recurrent_slot_ids = [None] * self.num_layers

    def store_conv_state(self, layer_idx: int, array: mx.array) -> None:
        """Store a layer's updated conv pool, keeping pool siblings aliased."""
        self.conv_states[layer_idx] = array

    def store_recurrent_state(self, layer_idx: int, array: mx.array) -> None:
        """Store a layer's updated recurrent pool, keeping siblings aliased."""
        self.recurrent_states[layer_idx] = array

    def require_mixer_dtype(self, mixer_dtype: mx.Dtype, *, layer_idx: int) -> None:
        """Reject a conv pool whose dtype differs from the mixer feeding it."""
        if mixer_dtype != self.dtype:
            raise ValueError(
                f"state pool dtype {self.dtype} does not match the layer "
                f"{layer_idx} mixer dtype {mixer_dtype}; pass --dtype matching "
                "the checkpoint so state and activations keep one dtype."
            )

    def step_slot_ids(
        self, ctx: PagedAttentionContext, cache_idx: int, num_requests: int
    ) -> list[int]:
        """Return this step's state slot per request for ``cache_idx``."""
        if ctx.state_group_slot_mappings is not None:
            slot_ids = ctx.state_group_slot_mappings[
                self.layer_group_ordinal(cache_idx)
            ]
        elif ctx.state_slot_mapping is not None:
            slot_ids = ctx.state_slot_mapping
        else:
            raise RuntimeError("state cache requires state_slot_mapping in context")
        if len(slot_ids) != num_requests:
            raise RuntimeError("state cache requires one slot per request")
        if len(set(slot_ids)) != len(slot_ids):
            raise RuntimeError("state cache requires unique slots per request")
        self.require_allocated_slots(slot_ids)
        return slot_ids

    def require_allocated_slots(self, slot_ids: list[int]) -> None:
        """Validate slots against both the scheduler cap and allocated rows."""
        if any(slot < 0 or slot >= self.max_seqs for slot in slot_ids):
            raise RuntimeError("state cache received out-of-range slot mapping")

    def layer_group_ordinal(self, cache_idx: int) -> int:
        """Return the mamba-cache-group ordinal for one linear layer."""
        return self._layer_group_ordinals[cache_idx]

    def layers_for_group_ordinal(self, ordinal: int) -> list[int]:
        """Return the cache indices of layers in one mamba cache group."""
        return [idx for idx, o in enumerate(self._layer_group_ordinals) if o == ordinal]

    def copy_slots(
        self, src_ids: list[int], dst_ids: list[int], layer_indices: list[int]
    ) -> None:
        """Copy state slabs ``src → dst`` for the given layers (batched, lazy).

        Sources are left untouched — align-mode prefix caching relies on a
        checkpointed block's slab staying immutable once the request advances
        to its next block.
        """
        if not src_ids or not layer_indices:
            return
        self.require_allocated_slots(src_ids)
        self.require_allocated_slots(dst_ids)
        self.apply_pending_states([*src_ids, *dst_ids])
        src = mx.array(src_ids, dtype=mx.int32)
        dst = mx.array(dst_ids, dtype=mx.int32)
        for layer_idx in layer_indices:
            # Gather first (its output is O(rows)), then write the rows back.
            # Reading the sources into their own array keeps the copy atomic
            # when one pair's destination is another pair's source.
            self.write_conv_rows(layer_idx, self.conv_states[layer_idx][src], dst)
            if self.recurrent_states:
                self.write_recurrent_rows(
                    layer_idx, self.recurrent_states[layer_idx][src], dst
                )

    def zero_slots(self, slot_ids: list[int], layer_indices: list[int]) -> None:
        """Zero state slabs for the given layers (batched, lazy).

        Align-mode slabs are addressed by scheduler block id, so a freshly
        allocated block may carry a previous request's bytes; fresh requests
        must start from zero state.
        """
        if not slot_ids or not layer_indices:
            return
        self.require_allocated_slots(slot_ids)
        self.apply_pending_states(slot_ids)
        ids = mx.array(slot_ids, dtype=mx.int32)
        for layer_idx in layer_indices:
            for states in (self.conv_states, self.recurrent_states):
                if states:
                    pool = states[layer_idx]
                    states[layer_idx] = _native_row_scatter()(
                        pool, pool, ids, zero=True
                    )

    def set_pending_conv_state(
        self, layer_idx: int, slot_ids: list[int], state_updates: mx.array
    ) -> None:
        """Store compact conv updates to be consumed by the next decode."""
        self.require_allocated_slots(slot_ids)
        if self.has_pending_conv_state(layer_idx):
            self.apply_pending_conv_state(layer_idx)
        self.pending_conv_states[layer_idx] = state_updates
        self.pending_conv_slot_ids[layer_idx] = list(slot_ids)

    def pending_conv_state(
        self, layer_idx: int, slot_ids: list[int]
    ) -> mx.array | None:
        """Return pending compact conv state when it exactly matches *slot_ids*."""
        pending_slots = self.pending_conv_slot_ids[layer_idx]
        if pending_slots != slot_ids:
            return None
        return self.pending_conv_states[layer_idx]

    def clear_pending_conv_state(self, layer_idx: int) -> None:
        """Drop compact conv updates after they have been consumed."""
        self.pending_conv_states[layer_idx] = None
        self.pending_conv_slot_ids[layer_idx] = None

    def has_pending_conv_state(self, layer_idx: int) -> bool:
        """Return whether a layer has deferred conv updates."""
        return self.pending_conv_states[layer_idx] is not None

    def conv_state_for_decode(
        self, layer_idx: int, slot_ids: list[int]
    ) -> GDNDecodeStateView:
        """Return authoritative conv state and slot ids for a decode kernel."""
        self.require_allocated_slots(slot_ids)
        if not self.has_pending_conv_state(layer_idx):
            return self._decode_state_view(
                self.conv_states[layer_idx], slot_ids, uses_compact_state=False
            )
        pending_state = self.pending_conv_state(layer_idx, slot_ids)
        if pending_state is not None:
            return self._decode_state_view(
                pending_state, slot_ids, uses_compact_state=True
            )
        self.apply_pending_conv_state(layer_idx)
        return self._decode_state_view(
            self.conv_states[layer_idx], slot_ids, uses_compact_state=False
        )

    def set_pending_recurrent_state(
        self, layer_idx: int, slot_ids: list[int], state_updates: mx.array
    ) -> None:
        """Store compact recurrent updates to be consumed by the next decode."""
        self.require_allocated_slots(slot_ids)
        if self.has_pending_recurrent_state(layer_idx):
            self.apply_pending_recurrent_state(layer_idx)
        self.pending_recurrent_states[layer_idx] = state_updates
        self.pending_recurrent_slot_ids[layer_idx] = list(slot_ids)

    def pending_recurrent_state(
        self, layer_idx: int, slot_ids: list[int]
    ) -> mx.array | None:
        """Return pending compact state when it exactly matches *slot_ids*."""
        pending_slots = self.pending_recurrent_slot_ids[layer_idx]
        if pending_slots != slot_ids:
            return None
        return self.pending_recurrent_states[layer_idx]

    def recurrent_state_for_decode(
        self, layer_idx: int, slot_ids: list[int]
    ) -> GDNDecodeStateView:
        """Return authoritative recurrent state and slot ids for a decode kernel."""
        self.require_allocated_slots(slot_ids)
        if not self.has_pending_recurrent_state(layer_idx):
            return self._decode_state_view(
                self.recurrent_states[layer_idx], slot_ids, uses_compact_state=False
            )
        pending_state = self.pending_recurrent_state(layer_idx, slot_ids)
        if pending_state is not None:
            return self._decode_state_view(
                pending_state, slot_ids, uses_compact_state=True
            )
        self.apply_pending_recurrent_state(layer_idx)
        return self._decode_state_view(
            self.recurrent_states[layer_idx], slot_ids, uses_compact_state=False
        )

    def _decode_state_view(
        self,
        state: mx.array,
        slot_ids: list[int],
        *,
        uses_compact_state: bool,
    ) -> GDNDecodeStateView:
        cache_slot_ids = mx.array(slot_ids, dtype=mx.int32)
        compact_order = list(range(len(slot_ids)))
        state_slot_ids = (
            mx.arange(len(slot_ids), dtype=mx.int32)
            if uses_compact_state and slot_ids != compact_order
            else cache_slot_ids
        )
        return GDNDecodeStateView(
            state=state,
            state_slot_ids=state_slot_ids,
            cache_slot_ids=cache_slot_ids,
            uses_compact_state=uses_compact_state,
        )

    def clear_pending_recurrent_state(self, layer_idx: int) -> None:
        """Drop compact recurrent updates after they have been consumed."""
        self.pending_recurrent_states[layer_idx] = None
        self.pending_recurrent_slot_ids[layer_idx] = None

    def has_pending_recurrent_state(self, layer_idx: int) -> bool:
        """Return whether a layer has deferred recurrent updates."""
        return self.pending_recurrent_states[layer_idx] is not None

    def updated_state_arrays(self) -> list[mx.array]:
        """Return state views and deferred updates needed by this forward."""
        arrays = list(self.conv_states)
        arrays.extend(p for p in self.pending_conv_states if p is not None)
        arrays.extend(self.recurrent_states)
        arrays.extend(p for p in self.pending_recurrent_states if p is not None)
        return arrays

    def _scatter_rows(self, pool: mx.array, rows: mx.array, ids: mx.array) -> mx.array:
        """Write distinct rows in place and return the rebound pool handle."""
        # MLX's indexed assignment converts the source implicitly and callers
        # rely on it; the primitive requires an exact match.  ``astype`` is a
        # no-op when the dtypes already agree, and O(rows) otherwise.
        return _native_row_scatter()(pool, rows.astype(pool.dtype), ids)

    def write_conv_rows(self, layer_idx: int, rows: mx.array, ids: mx.array) -> None:
        """Write conv rows and rebind every sibling to the returned handle."""
        self.store_conv_state(
            layer_idx, self._scatter_rows(self.conv_states[layer_idx], rows, ids)
        )

    def write_recurrent_rows(
        self, layer_idx: int, rows: mx.array, ids: mx.array
    ) -> None:
        """Write recurrent rows and rebind siblings to the returned handle."""
        self.store_recurrent_state(
            layer_idx,
            self._scatter_rows(self.recurrent_states[layer_idx], rows, ids),
        )

    def apply_pending_conv_state(self, layer_idx: int) -> None:
        """Scatter deferred conv updates into the stable state pool."""
        pending_state = self.pending_conv_states[layer_idx]
        pending_slots = self.pending_conv_slot_ids[layer_idx]
        if pending_state is None or pending_slots is None:
            return
        self.require_allocated_slots(pending_slots)

        self.write_conv_rows(
            layer_idx, pending_state, mx.array(pending_slots, dtype=mx.int32)
        )
        self.clear_pending_conv_state(layer_idx)

    def apply_pending_conv_states(self) -> None:
        """Scatter all deferred conv updates into stable state pools."""
        for layer_idx in range(self.num_layers):
            self.apply_pending_conv_state(layer_idx)

    def apply_pending_recurrent_state(self, layer_idx: int) -> None:
        """Scatter deferred recurrent updates into the stable state pool."""
        pending_state = self.pending_recurrent_states[layer_idx]
        pending_slots = self.pending_recurrent_slot_ids[layer_idx]
        if pending_state is None or pending_slots is None:
            return
        self.require_allocated_slots(pending_slots)

        self.write_recurrent_rows(
            layer_idx, pending_state, mx.array(pending_slots, dtype=mx.int32)
        )
        self.clear_pending_recurrent_state(layer_idx)

    def apply_pending_recurrent_states(self) -> None:
        """Scatter all deferred recurrent updates into stable state pools."""
        for layer_idx in range(self.num_layers):
            self.apply_pending_recurrent_state(layer_idx)

    def apply_pending_states(self, slot_ids: Sequence[int] | None = None) -> None:
        """Flush pending components touching these slots, or all when omitted."""
        slots = None if slot_ids is None else set(slot_ids)
        for layer_idx in range(self.num_layers):
            if slots is None or slots.intersection(
                self.pending_conv_slot_ids[layer_idx] or ()
            ):
                self.apply_pending_conv_state(layer_idx)
            if slots is None or slots.intersection(
                self.pending_recurrent_slot_ids[layer_idx] or ()
            ):
                self.apply_pending_recurrent_state(layer_idx)
