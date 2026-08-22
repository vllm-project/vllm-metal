# SPDX-License-Identifier: Apache-2.0
"""Per-request conv state cache for ShortConv (LFM2-style) layers.

LFM2/LFM2.5 conv layers keep a fixed-size causal-conv tail per request: the
last ``conv_kernel_dim - 1`` rows of the gated ``B*x`` activation.

Layout per conv layer:
  - conv_state: [allocated_seqs, conv_kernel_dim - 1, conv_dim]

Each request occupies one slot for its lifetime, assigned by
``HybridGDNStateManager`` (the same per-request slot lifecycle GDN hybrids
use in ``mamba_cache_mode="none"``).  ``max_seqs`` is the scheduler-visible
hard cap; ``allocated_seqs`` grows on request admission and never exceeds it.

Unlike ``GDNPagedStateCache`` there is no pending-compact handoff: the
ShortConv wrapper writes each request's updated tail straight into the stable
pool, so ``apply_pending_states`` is a no-op kept for interface parity.  The
block-id-keyed pooling layout and ``copy_blocks`` that cache grew for
align-mode prefix caching are likewise absent — conv hybrids run none mode
only, and would gain them alongside align support.
"""

from __future__ import annotations

import mlx.core as mx


class ShortConvStateCache:
    """Per-layer MLX arrays for ShortConv causal-conv state."""

    def __init__(
        self,
        *,
        num_layers: int,
        max_seqs: int,
        conv_kernel_dim: int,
        conv_dim: int,
        initial_seqs: int | None = None,
        dtype: mx.Dtype = mx.float16,
    ) -> None:
        if dtype not in (mx.float16, mx.bfloat16, mx.float32):
            raise ValueError(f"Unsupported dtype for ShortConv state cache: {dtype}")
        if max_seqs < 0:
            raise ValueError("max_seqs must be non-negative")
        if conv_kernel_dim < 2:
            raise ValueError("conv_kernel_dim must be at least 2 to carry state")
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
        self.dtype = dtype

        self.conv_states: list[mx.array] = [
            mx.zeros(self._conv_shape(initial_seqs), dtype=dtype)
            for _ in range(num_layers)
        ]
        self._eval_state_arrays()

    def _conv_shape(self, num_seqs: int) -> tuple[int, int, int]:
        return (num_seqs, self.conv_kernel_dim - 1, self.conv_dim)

    def _eval_state_arrays(self) -> None:
        if self.conv_states:
            mx.eval(*self.conv_states)

    def ensure_capacity(self, num_seqs: int) -> None:
        """Grow stable state pools so slots ``[0, num_seqs)`` are valid."""
        if num_seqs < 0:
            raise ValueError("num_seqs must be non-negative")
        if num_seqs > self.max_seqs:
            raise RuntimeError(
                "ShortConv state cache requested more slots than max_num_seqs "
                f"({num_seqs} > {self.max_seqs})"
            )
        if num_seqs <= self.allocated_seqs:
            return

        old_allocated = self.allocated_seqs
        # Grow one pool at a time, releasing each old array before building the
        # next, so peak memory holds one extra pool rather than a full copy of
        # every layer's state.
        for layer_idx in range(self.num_layers):
            grown = mx.zeros(self._conv_shape(num_seqs), dtype=self.dtype)
            if old_allocated:
                grown[:old_allocated] = self.conv_states[layer_idx]
            mx.eval(grown)
            self.conv_states[layer_idx] = grown

        self.allocated_seqs = num_seqs

    def require_allocated_slots(self, slot_ids: list[int]) -> None:
        """Validate slots against both the scheduler cap and allocated rows."""
        if any(slot < 0 or slot >= self.max_seqs for slot in slot_ids):
            raise RuntimeError("ShortConv wrapper received out-of-range slot mapping")
        if any(slot >= self.allocated_seqs for slot in slot_ids):
            raise RuntimeError(
                "ShortConv wrapper received slot mapping beyond allocated state cache"
            )

    def reset_slot(self, slot: int) -> None:
        """Clear state for one allocated slot before it is reused."""
        self.require_allocated_slots([slot])
        for layer_idx in range(self.num_layers):
            conv = self.conv_states[layer_idx]
            conv[slot] = mx.zeros_like(conv[slot])
            self.conv_states[layer_idx] = conv

    def apply_pending_states(self) -> None:
        """No-op: ShortConv writes state into the stable pool directly."""

    def updated_state_arrays(self) -> list[mx.array]:
        """Return the state arrays a forward pass mutates (for submission)."""
        return list(self.conv_states)
