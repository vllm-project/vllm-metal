# SPDX-License-Identifier: Apache-2.0
"""Convolution-only state for LFM2-style ShortConv layers.

Each slab holds the last ``conv_kernel_dim - 1`` gated input rows, shaped
``[history, conv_dim]`` in the activation dtype. None mode assigns slabs to
requests; align mode addresses them by scheduler block id and shares physical
pools according to the scheduler's layer layout.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Sequence

import mlx.core as mx


@functools.cache
def _native_row_scatter() -> Callable[..., mx.array]:
    """Reuse the native row primitive without copying the whole state pool."""
    from vllm_metal.metal import get_ops

    return get_ops().gdn_state_scatter


class ShortConvStateCache:
    """MLX convolution tails with the shared hybrid state lifecycle API."""

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
        self.conv_states = [
            mx.zeros(self._conv_shape(initial_seqs), dtype=dtype)
            for _ in range(num_layers)
        ]
        self._layer_group_ordinals = [0] * num_layers
        self._pool_siblings = [[i] for i in range(num_layers)]
        self._canonical_layers = list(range(num_layers))
        mx.eval(*self.updated_state_arrays())

    def _conv_shape(self, num_seqs: int) -> tuple[int, int, int]:
        return (num_seqs, self.conv_kernel_dim - 1, self.conv_dim)

    @property
    def num_state_pools(self) -> int:
        """Number of distinct physical pools under the adopted layout."""
        return len(self._canonical_layers)

    def store_conv_state(self, layer_idx: int, array: mx.array) -> None:
        """Rebind every layer sharing this physical pool to its new handle."""
        for sibling in self._pool_siblings[layer_idx]:
            self.conv_states[sibling] = array

    def ensure_capacity(self, num_seqs: int) -> None:
        """Grow physical pools to cover the requested slab ids."""
        if num_seqs < 0:
            raise ValueError("num_seqs must be non-negative")
        if num_seqs > self.max_seqs:
            raise RuntimeError(
                "ShortConv state cache requested more slots than max_num_seqs "
                f"({num_seqs} > {self.max_seqs})"
            )
        if num_seqs <= self.allocated_seqs:
            return

        for layer_idx in self._canonical_layers:
            grown = mx.zeros(self._conv_shape(num_seqs), dtype=self.dtype)
            if self.allocated_seqs:
                grown[: self.allocated_seqs] = self.conv_states[layer_idx]
            # Finish one pool at a time so growth holds only one old pool
            # beyond the final allocation, as budgeted by the hybrid plan.
            mx.eval(grown)
            self.store_conv_state(layer_idx, grown)
        self.allocated_seqs = num_seqs

    def require_allocated_slots(self, slot_ids: list[int]) -> None:
        """Validate slabs against both the hard cap and allocated storage."""
        if any(slot < 0 or slot >= self.max_seqs for slot in slot_ids):
            raise RuntimeError("ShortConv wrapper received out-of-range slot mapping")
        if any(slot >= self.allocated_seqs for slot in slot_ids):
            raise RuntimeError(
                "ShortConv wrapper received slot mapping beyond allocated state cache"
            )

    def reset_slot(self, slot: int) -> None:
        """Clear a private request slot before reusing it in none mode."""
        self.zero_slots([slot], self._canonical_layers)

    def set_layer_layout(
        self, group_ordinals: list[int], pool_ordinals: list[int]
    ) -> None:
        """Adopt scheduler groups and physical pools before any state is written.

        Layers may share a pool only across different groups, whose scheduler
        block ids are disjoint. None mode retains the initial one-pool-per-layer
        layout because its slots are private request indices.
        """
        if not (len(group_ordinals) == len(pool_ordinals) == self.num_layers):
            raise ValueError(
                "expected one group and one pool ordinal per ShortConv layer "
                f"({self.num_layers}), got {len(group_ordinals)}/{len(pool_ordinals)}"
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

        self._layer_group_ordinals = list(group_ordinals)
        self._pool_siblings = [pool_members[pool] for pool in pool_ordinals]
        self._canonical_layers = sorted(members[0] for members in pool_members.values())
        for layer_idx in self._canonical_layers:
            self.store_conv_state(layer_idx, self.conv_states[layer_idx])

    def layer_group_ordinal(self, cache_idx: int) -> int:
        """Return the block-table group addressing one layer's state."""
        return self._layer_group_ordinals[cache_idx]

    def layers_for_group_ordinal(self, ordinal: int) -> list[int]:
        """Return the layer cache indices owned by one scheduler group."""
        return [
            idx
            for idx, group in enumerate(self._layer_group_ordinals)
            if group == ordinal
        ]

    def write_conv_rows(self, layer_idx: int, rows: mx.array, ids: mx.array) -> None:
        """Write distinct slabs in place and chain updates across pool siblings."""
        pool = self.conv_states[layer_idx]
        # Indexed MLX assignment casts implicitly; the native primitive needs
        # the exact pool dtype. Only the compact rows are converted here.
        updated = _native_row_scatter()(pool, rows.astype(pool.dtype), ids)
        self.store_conv_state(layer_idx, updated)

    def copy_slots(
        self, src_ids: list[int], dst_ids: list[int], layer_indices: list[int]
    ) -> None:
        """Copy slabs without advancing or overwriting the source checkpoint."""
        if not src_ids or not layer_indices:
            return
        self.require_allocated_slots(src_ids)
        self.require_allocated_slots(dst_ids)
        src = mx.array(src_ids, dtype=mx.int32)
        dst = mx.array(dst_ids, dtype=mx.int32)
        for layer_idx in layer_indices:
            # Gather all sources before the in-place write: a destination may
            # also be another pair's source. The temporary is only O(rows).
            self.write_conv_rows(layer_idx, self.conv_states[layer_idx][src], dst)

    def copy_blocks(self, block_copies: Sequence[tuple[int, int]]) -> None:
        """Apply scheduler copy-on-write operations to each physical pool once."""
        if not block_copies:
            return
        src_ids, dst_ids = zip(*block_copies, strict=True)
        self.ensure_capacity(max(*src_ids, *dst_ids) + 1)
        self.copy_slots(list(src_ids), list(dst_ids), self._canonical_layers)

    def zero_slots(self, slot_ids: list[int], layer_indices: list[int]) -> None:
        """Zero only the given slabs when their scheduler blocks are recycled."""
        if not slot_ids or not layer_indices:
            return
        self.require_allocated_slots(slot_ids)
        ids = mx.array(slot_ids, dtype=mx.int32)
        zeros = mx.zeros(self._conv_shape(len(slot_ids)), dtype=self.dtype)
        for layer_idx in layer_indices:
            self.write_conv_rows(layer_idx, zeros, ids)

    def apply_pending_states(self) -> None:
        """No-op: ShortConv writes its new tail directly to the stable pool."""

    def updated_state_arrays(self) -> list[mx.array]:
        """Submit one authoritative array per physical pool after a forward."""
        return [self.conv_states[i] for i in self._canonical_layers]
