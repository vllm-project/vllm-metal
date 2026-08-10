# SPDX-License-Identifier: Apache-2.0
"""Per-request recurrent state cache for GDN linear attention layers.

Unlike ``MetalPagedKVCache`` which stores per-token KV that grows with
sequence length, GDN linear attention uses fixed-size recurrent state
per request: a convolution buffer and a hidden state matrix.

Layout per linear layer:
  - conv_state:      [allocated_seqs, conv_kernel - 1, conv_dim]
  - recurrent_state: [allocated_seqs, num_v_heads, value_head_dim, key_head_dim]

Each request occupies one slot (indexed by request position in the batch).
State is managed by the GDN wrapper, not by the scheduler's block system.
``max_seqs`` remains the scheduler-visible hard cap; ``allocated_seqs`` grows
on request admission and never exceeds that cap.

Pending state handoff:
  - At most one compact pending conv or recurrent update may exist per linear
    layer.
  - Lazy decode may consume that compact update directly only when the active
    slot order exactly matches the pending slot order.
  - Slot-order mismatches, fallback execution, new prefill work, or slot release
    must scatter the pending update into the stable state pool first.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import mlx.core as mx


@dataclass(frozen=True)
class GDNDecodeStateView:
    """State array and slot mappings for one lazy decode kernel launch."""

    state: mx.array
    state_slot_ids: mx.array
    cache_slot_ids: mx.array
    uses_compact_state: bool


class GDNPagedStateCache:
    """Per-layer MLX arrays for GDN linear attention recurrent state."""

    def __init__(
        self,
        *,
        num_layers: int,
        max_seqs: int,
        conv_kernel_dim: int,
        conv_dim: int,
        num_v_heads: int,
        value_head_dim: int,
        key_head_dim: int,
        initial_seqs: int | None = None,
        dtype: mx.Dtype = mx.float16,
    ) -> None:
        if dtype not in (mx.float16, mx.bfloat16, mx.float32):
            raise ValueError(f"Unsupported dtype for GDN state cache: {dtype}")
        if max_seqs < 0:
            raise ValueError("max_seqs must be non-negative")
        if initial_seqs is None:
            initial_seqs = max_seqs
        if initial_seqs < 0 or initial_seqs > max_seqs:
            raise ValueError(
                "initial_seqs must be between 0 and max_seqs "
                f"(got {initial_seqs}, max_seqs={max_seqs})"
            )

        self.num_layers = num_layers
        self.max_seqs = max_seqs
        self.allocated_seqs = initial_seqs
        self.conv_kernel_dim = conv_kernel_dim
        self.conv_dim = conv_dim
        self.num_v_heads = num_v_heads
        self.value_head_dim = value_head_dim
        self.key_head_dim = key_head_dim
        self.dtype = dtype

        self.conv_states: list[mx.array] = [
            mx.zeros(self._conv_shape(initial_seqs), dtype=dtype)
            for _ in range(num_layers)
        ]
        # Recurrent state uses float32 to avoid overflow in kernel accumulation.
        self.recurrent_states: list[mx.array] = [
            mx.zeros(self._recurrent_shape(initial_seqs), dtype=mx.float32)
            for _ in range(num_layers)
        ]
        self.pending_conv_states: list[mx.array | None] = [
            None for _ in range(num_layers)
        ]
        self.pending_conv_slot_ids: list[list[int] | None] = [
            None for _ in range(num_layers)
        ]
        self.pending_recurrent_states: list[mx.array | None] = [
            None for _ in range(num_layers)
        ]
        self.pending_recurrent_slot_ids: list[list[int] | None] = [
            None for _ in range(num_layers)
        ]
        # Scheduler layout, adopted via ``set_layer_layout``; until then every
        # layer is its own pool (none mode keeps this identity layout).
        self._layer_group_ordinals: list[int] = [0] * num_layers
        self._pool_siblings: list[list[int]] = [[i] for i in range(num_layers)]
        self._canonical_layers: list[int] = list(range(num_layers))
        self._eval_state_arrays()

    def _conv_shape(self, num_seqs: int) -> tuple[int, int, int]:
        return (num_seqs, self.conv_kernel_dim - 1, self.conv_dim)

    def _recurrent_shape(self, num_seqs: int) -> tuple[int, int, int, int]:
        return (
            num_seqs,
            self.num_v_heads,
            self.value_head_dim,
            self.key_head_dim,
        )

    def _eval_state_arrays(self) -> None:
        arrays = [
            array
            for layer_idx in self._canonical_layers
            for array in (self.conv_states[layer_idx], self.recurrent_states[layer_idx])
        ]
        if arrays:
            mx.eval(*arrays)

    @property
    def num_state_pools(self) -> int:
        """Number of distinct physical state pools under the adopted layout."""
        return len(self._canonical_layers)

    def store_conv_state(self, layer_idx: int, array: mx.array) -> None:
        """Store a layer's updated conv pool, keeping pool siblings aliased."""
        for sibling in self._pool_siblings[layer_idx]:
            self.conv_states[sibling] = array

    def store_recurrent_state(self, layer_idx: int, array: mx.array) -> None:
        """Store a layer's updated recurrent pool, keeping siblings aliased."""
        for sibling in self._pool_siblings[layer_idx]:
            self.recurrent_states[sibling] = array

    def ensure_capacity(self, num_seqs: int) -> None:
        """Grow stable state pools so slots ``[0, num_seqs)`` are valid."""
        if num_seqs < 0:
            raise ValueError("num_seqs must be non-negative")
        if num_seqs > self.max_seqs:
            raise RuntimeError(
                "GDN state cache requested more slots than max_num_seqs "
                f"({num_seqs} > {self.max_seqs})"
            )
        if num_seqs <= self.allocated_seqs:
            return

        self.apply_pending_states()
        old_allocated = self.allocated_seqs

        for layer_idx in self._canonical_layers:
            old_conv = self.conv_states[layer_idx]
            conv = mx.zeros(self._conv_shape(num_seqs), dtype=self.dtype)
            if old_allocated:
                conv[:old_allocated] = old_conv

            old_recurrent = self.recurrent_states[layer_idx]
            recurrent = mx.zeros(self._recurrent_shape(num_seqs), dtype=mx.float32)
            if old_allocated:
                recurrent[:old_allocated] = old_recurrent

            # Materialize now so each old pool is released as it is copied.
            # The memory plan reserves one old physical pool for this overlap.
            mx.eval(conv, recurrent)
            self.store_conv_state(layer_idx, conv)
            self.store_recurrent_state(layer_idx, recurrent)

        self.allocated_seqs = num_seqs

    def require_allocated_slots(self, slot_ids: list[int]) -> None:
        """Validate slots against both the scheduler cap and allocated rows."""
        if any(slot < 0 or slot >= self.max_seqs for slot in slot_ids):
            raise RuntimeError("GDN wrapper received out-of-range slot mapping")
        if any(slot >= self.allocated_seqs for slot in slot_ids):
            raise RuntimeError(
                "GDN wrapper received slot mapping beyond allocated state cache"
            )

    def reset_slot(self, slot: int) -> None:
        """Clear state for one allocated slot before it is reused."""
        self.require_allocated_slots([slot])
        self.apply_pending_states()
        for layer_idx in self._canonical_layers:
            conv = self.conv_states[layer_idx]
            conv[slot] = mx.zeros_like(conv[slot])
            self.store_conv_state(layer_idx, conv)

            recurrent = self.recurrent_states[layer_idx]
            recurrent[slot] = mx.zeros_like(recurrent[slot])
            self.store_recurrent_state(layer_idx, recurrent)

    def set_layer_layout(
        self, group_ordinals: list[int], pool_ordinals: list[int]
    ) -> None:
        """Adopt the scheduler's layer layout: group + physical pool per layer.

        ``group_ordinals[cache_idx]`` selects the block-table row addressing a
        layer's slabs; ``pool_ordinals[cache_idx]`` names its physical pool
        (vLLM's ``kv_cache_tensors.shared_by``: one pool per within-group
        position).  Layers sharing a pool must belong to different groups —
        their groups then own disjoint block ids, so slab rows never collide.
        Must be called before any state is written (pool arrays are rebuilt).
        """
        if not (len(group_ordinals) == len(pool_ordinals) == self.num_layers):
            raise ValueError(
                f"expected one group and one pool ordinal per linear layer "
                f"({self.num_layers}), got {len(group_ordinals)}/"
                f"{len(pool_ordinals)}"
            )
        pool_members: dict[int, list[int]] = {}
        for layer_idx, pool in enumerate(pool_ordinals):
            pool_members.setdefault(pool, []).append(layer_idx)
        for pool, members in pool_members.items():
            groups = [group_ordinals[i] for i in members]
            if len(set(groups)) != len(groups):
                raise ValueError(
                    f"state pool {pool} is shared by two layers of the same "
                    f"mamba cache group (layers {members}); their slab rows "
                    "would collide"
                )
        if any(
            self.pending_conv_states[i] is not None
            or self.pending_recurrent_states[i] is not None
            for i in range(self.num_layers)
        ):
            raise RuntimeError("set_layer_layout() called with pending state")

        self._layer_group_ordinals = list(group_ordinals)
        self._pool_siblings = [pool_members[pool] for pool in pool_ordinals]
        self._canonical_layers = sorted(members[0] for members in pool_members.values())
        # Rebuild storage so pool siblings alias one array.  Pre-adopt state
        # is all zeros, so aliasing to the canonical member's array is exact.
        for members in pool_members.values():
            canonical = members[0]
            for layer_idx in members:
                self.conv_states[layer_idx] = self.conv_states[canonical]
                self.recurrent_states[layer_idx] = self.recurrent_states[canonical]

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
        src = mx.array(src_ids, dtype=mx.int32)
        dst = mx.array(dst_ids, dtype=mx.int32)
        for layer_idx in layer_indices:
            conv = self.conv_states[layer_idx]
            conv[dst] = conv[src]
            self.store_conv_state(layer_idx, conv)

            recurrent = self.recurrent_states[layer_idx]
            recurrent[dst] = recurrent[src]
            self.store_recurrent_state(layer_idx, recurrent)

    def copy_blocks(self, block_copies: Sequence[tuple[int, int]]) -> None:
        """Apply scheduler copy-on-write operations to every physical pool."""
        if not block_copies:
            return

        src_ids, dst_ids = zip(*block_copies, strict=True)
        self.apply_pending_states()
        high_water = max(*src_ids, *dst_ids) + 1
        self.ensure_capacity(high_water)
        self.copy_slots(list(src_ids), list(dst_ids), self._canonical_layers)

    def zero_slots(self, slot_ids: list[int], layer_indices: list[int]) -> None:
        """Zero state slabs for the given layers (batched, lazy).

        Align-mode slabs are addressed by scheduler block id, so a freshly
        allocated block may carry a previous request's bytes; fresh requests
        must start from zero state.
        """
        if not slot_ids or not layer_indices:
            return
        self.require_allocated_slots(slot_ids)
        ids = mx.array(slot_ids, dtype=mx.int32)
        # Every layer's pool has the same per-slab shape, so build one zeros
        # block per pool and reuse it across layers.
        conv_zeros = mx.zeros(
            (len(slot_ids),) + self.conv_states[0].shape[1:],
            dtype=self.conv_states[0].dtype,
        )
        recurrent_zeros = mx.zeros(
            (len(slot_ids),) + self.recurrent_states[0].shape[1:],
            dtype=self.recurrent_states[0].dtype,
        )
        for layer_idx in layer_indices:
            conv = self.conv_states[layer_idx]
            conv[ids] = conv_zeros
            self.store_conv_state(layer_idx, conv)

            recurrent = self.recurrent_states[layer_idx]
            recurrent[ids] = recurrent_zeros
            self.store_recurrent_state(layer_idx, recurrent)

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

    def updated_conv_state_array(self, layer_idx: int) -> mx.array:
        """Return the authoritative conv state array for submission."""
        pending_state = self.pending_conv_states[layer_idx]
        return (
            pending_state if pending_state is not None else self.conv_states[layer_idx]
        )

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
        """Return the GDN state arrays to submit after a forward.

        One entry per physical pool plus any pending compact updates — with
        shared pools a stable array carries several layers' state, so it is
        submitted even when one of its layers also has a pending update.
        """
        arrays = [self.conv_states[i] for i in self._canonical_layers]
        arrays.extend(p for p in self.pending_conv_states if p is not None)
        arrays.extend(self.recurrent_states[i] for i in self._canonical_layers)
        arrays.extend(p for p in self.pending_recurrent_states if p is not None)
        return arrays

    def apply_pending_conv_state(self, layer_idx: int) -> None:
        """Scatter deferred conv updates into the stable state pool."""
        pending_state = self.pending_conv_states[layer_idx]
        pending_slots = self.pending_conv_slot_ids[layer_idx]
        if pending_state is None or pending_slots is None:
            return
        self.require_allocated_slots(pending_slots)

        slot_ids_arr = mx.array(pending_slots, dtype=mx.int32)
        conv_state = self.conv_states[layer_idx]
        conv_state[slot_ids_arr] = pending_state
        self.store_conv_state(layer_idx, conv_state)
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

        slot_ids_arr = mx.array(pending_slots, dtype=mx.int32)
        recurrent_state = self.recurrent_states[layer_idx]
        recurrent_state[slot_ids_arr] = pending_state
        self.store_recurrent_state(layer_idx, recurrent_state)
        self.clear_pending_recurrent_state(layer_idx)

    def apply_pending_recurrent_states(self) -> None:
        """Scatter all deferred recurrent updates into stable state pools."""
        for layer_idx in range(self.num_layers):
            self.apply_pending_recurrent_state(layer_idx)

    def apply_pending_states(self) -> None:
        """Scatter all deferred conv and recurrent updates into stable pools."""
        self.apply_pending_conv_states()
        self.apply_pending_recurrent_states()
