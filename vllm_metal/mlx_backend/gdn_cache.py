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
        arrays = [*self.conv_states, *self.recurrent_states]
        if arrays:
            mx.eval(*arrays)

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

        conv_states: list[mx.array] = []
        recurrent_states: list[mx.array] = []
        for layer_idx in range(self.num_layers):
            old_conv = self.conv_states[layer_idx]
            conv = mx.zeros(self._conv_shape(num_seqs), dtype=self.dtype)
            if old_allocated:
                conv[:old_allocated] = old_conv
            conv_states.append(conv)

            old_recurrent = self.recurrent_states[layer_idx]
            recurrent = mx.zeros(self._recurrent_shape(num_seqs), dtype=mx.float32)
            if old_allocated:
                recurrent[:old_allocated] = old_recurrent
            recurrent_states.append(recurrent)

        self.conv_states = conv_states
        self.recurrent_states = recurrent_states
        self.allocated_seqs = num_seqs
        self._eval_state_arrays()

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
        for layer_idx in range(self.num_layers):
            conv = self.conv_states[layer_idx]
            conv[slot] = mx.zeros_like(conv[slot])
            self.conv_states[layer_idx] = conv

            recurrent = self.recurrent_states[layer_idx]
            recurrent[slot] = mx.zeros_like(recurrent[slot])
            self.recurrent_states[layer_idx] = recurrent

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
        """Return the minimal GDN state arrays to submit after a forward."""
        arrays = [
            self.updated_conv_state_array(layer_idx)
            for layer_idx in range(self.num_layers)
        ]
        for layer_idx, recurrent_state in enumerate(self.recurrent_states):
            pending_state = self.pending_recurrent_states[layer_idx]
            arrays.append(
                pending_state if pending_state is not None else recurrent_state
            )
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
        self.conv_states[layer_idx] = conv_state
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
        self.recurrent_states[layer_idx] = recurrent_state
        self.clear_pending_recurrent_state(layer_idx)

    def apply_pending_recurrent_states(self) -> None:
        """Scatter all deferred recurrent updates into stable state pools."""
        for layer_idx in range(self.num_layers):
            self.apply_pending_recurrent_state(layer_idx)

    def apply_pending_states(self) -> None:
        """Scatter all deferred conv and recurrent updates into stable pools."""
        self.apply_pending_conv_states()
        self.apply_pending_recurrent_states()
