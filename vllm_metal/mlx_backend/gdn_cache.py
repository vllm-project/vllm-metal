# SPDX-License-Identifier: Apache-2.0
"""Per-request recurrent state cache for GDN linear attention layers.

Unlike ``MetalPagedKVCache`` which stores per-token KV that grows with
sequence length, GDN linear attention uses fixed-size recurrent state
per request: a convolution buffer and a hidden state matrix.

Layout per linear layer:
  - conv_state:      [max_seqs, conv_kernel - 1, conv_dim]
  - recurrent_state: [max_seqs, num_v_heads, value_head_dim, key_head_dim]

Each request occupies one slot (indexed by request position in the batch).
State is managed by the GDN wrapper, not by the scheduler's block system.

Pending recurrent state handoff:
  - At most one compact pending recurrent update may exist per linear layer.
  - Lazy decode may consume that compact update directly only when the active
    slot order exactly matches the pending slot order.
  - Slot-order mismatches, fallback execution, new prefill work, or slot release
    must scatter the pending update into the stable recurrent state pool first.
"""

from __future__ import annotations

import mlx.core as mx


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
        dtype: mx.Dtype = mx.float16,
    ) -> None:
        if dtype not in (mx.float16, mx.bfloat16, mx.float32):
            raise ValueError(f"Unsupported dtype for GDN state cache: {dtype}")

        self.num_layers = num_layers
        self.max_seqs = max_seqs
        self.conv_kernel_dim = conv_kernel_dim
        self.conv_dim = conv_dim
        self.num_v_heads = num_v_heads
        self.value_head_dim = value_head_dim
        self.key_head_dim = key_head_dim
        self.dtype = dtype

        conv_shape = (max_seqs, conv_kernel_dim - 1, conv_dim)
        recurrent_shape = (max_seqs, num_v_heads, value_head_dim, key_head_dim)

        self.conv_states: list[mx.array] = [
            mx.zeros(conv_shape, dtype=dtype) for _ in range(num_layers)
        ]
        # Recurrent state uses float32 to avoid overflow in kernel accumulation.
        self.recurrent_states: list[mx.array] = [
            mx.zeros(recurrent_shape, dtype=mx.float32) for _ in range(num_layers)
        ]
        self.pending_recurrent_states: list[mx.array | None] = [
            None for _ in range(num_layers)
        ]
        self.pending_recurrent_slot_ids: list[list[int] | None] = [
            None for _ in range(num_layers)
        ]
        mx.eval(*self.conv_states, *self.recurrent_states)

    def set_pending_recurrent_state(
        self, layer_idx: int, slot_ids: list[int], state_updates: mx.array
    ) -> None:
        """Store compact recurrent updates to be consumed by the next decode."""
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

    def clear_pending_recurrent_state(self, layer_idx: int) -> None:
        """Drop compact recurrent updates after they have been consumed."""
        self.pending_recurrent_states[layer_idx] = None
        self.pending_recurrent_slot_ids[layer_idx] = None

    def has_pending_recurrent_state(self, layer_idx: int) -> bool:
        """Return whether a layer has deferred recurrent updates."""
        return self.pending_recurrent_states[layer_idx] is not None

    def updated_state_arrays(self) -> list[mx.array]:
        """Return the minimal GDN state arrays to submit after a forward."""
        arrays = [*self.conv_states]
        for layer_idx, recurrent_state in enumerate(self.recurrent_states):
            pending_state = self.pending_recurrent_states[layer_idx]
            arrays.append(
                pending_state if pending_state is not None else recurrent_state
            )
        return arrays

    def apply_pending_recurrent_state(self, layer_idx: int) -> None:
        """Scatter deferred recurrent updates into the stable state pool."""
        pending_state = self.pending_recurrent_states[layer_idx]
        pending_slots = self.pending_recurrent_slot_ids[layer_idx]
        if pending_state is None or pending_slots is None:
            return

        slot_ids_arr = mx.array(pending_slots, dtype=mx.int32)
        recurrent_state = self.recurrent_states[layer_idx]
        recurrent_state[slot_ids_arr] = pending_state
        self.recurrent_states[layer_idx] = recurrent_state
        self.clear_pending_recurrent_state(layer_idx)

    def apply_pending_recurrent_states(self) -> None:
        """Scatter all deferred recurrent updates into stable state pools."""
        for layer_idx in range(self.num_layers):
            self.apply_pending_recurrent_state(layer_idx)
