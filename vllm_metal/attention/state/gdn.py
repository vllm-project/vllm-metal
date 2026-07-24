# SPDX-License-Identifier: Apache-2.0
"""Hybrid GDN request-state lifecycle.

The hybrid paged runtime owns two different state systems:

- SDPA KV cache, indexed by scheduler block tables
- GDN recurrent state, indexed by one stable slot per resident request

`HybridGDNStateManager` owns the second one. It keeps request-to-slot
assignment stable across request reordering, grows the recurrent cache when new
requests arrive, resets reused slots before they are handed to a new request,
and tracks when stable GDN arrays must be materialized out of the lazy graph.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import mlx.core as mx

from vllm_metal.attention.caches.gdn_cache import (
    GDNPagedStateCache,
    GDNStateSnapshot,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from vllm_metal.attention.context import PagedAttentionContext


class HybridGDNStateManager:
    """Own request-to-slot lifecycle for one hybrid runtime."""

    def __init__(
        self,
        state_cache: GDNPagedStateCache,
        *,
        block_size: int | None = None,
    ) -> None:
        self._state_cache = state_cache
        self._req_to_slot: dict[str, int] = {}
        self._free_slots: list[int] = []
        self._needs_materialize = False
        self._new_req_ids: set[str] = set()
        self._block_size = block_size
        self._num_blocks = 0
        self._mamba_group_layers: dict[int, tuple[int, ...]] = {}
        self._block_snapshots: dict[int, GDNStateSnapshot] = {}
        self._block_snapshot_hits = 0
        self._block_snapshot_misses = 0
        self._block_snapshot_stores = 0
        self._block_snapshot_replacements = 0
        self._block_snapshot_invalidations = 0
        self._block_snapshot_bytes = 0
        self._block_snapshot_peak_count = 0
        self._block_snapshot_peak_bytes = 0

    @property
    def request_slots(self) -> dict[str, int]:
        """Stable request-to-slot mapping for the active hybrid batch set."""
        return dict(self._req_to_slot)

    @property
    def free_slots(self) -> tuple[int, ...]:
        """Slots available for reuse on later scheduler steps."""
        return tuple(self._free_slots)

    @property
    def needs_materialize(self) -> bool:
        """Whether released slots are waiting for state materialization."""
        return self._needs_materialize

    def populate_step_context(
        self, *, req_ids: list[str], ctx: PagedAttentionContext
    ) -> None:
        """Attach stable GDN slot ids to one forward-pass context."""
        ctx.gdn_slot_mapping = self.assign_step_slots(req_ids)

    def assign_step_slots(self, req_ids: list[str]) -> list[int]:
        """Plan one scheduler step's stable request-to-slot mapping atomically."""
        self._new_req_ids = set()
        step_slot_ids: list[int] = []
        planned_slots_by_request: dict[str, int] = {}
        new_assignments: list[tuple[str, int, bool]] = []
        reusable_slots = list(self._free_slots)
        next_unallocated_slot = self._state_cache.allocated_seqs

        for req_id in req_ids:
            existing_slot = self._req_to_slot.get(req_id)
            if existing_slot is not None:
                step_slot_ids.append(existing_slot)
                continue

            planned_slot = planned_slots_by_request.get(req_id)
            if planned_slot is not None:
                step_slot_ids.append(planned_slot)
                continue

            if reusable_slots:
                slot_id = reusable_slots.pop()
                reuses_existing_slot = True
            else:
                slot_id = next_unallocated_slot
                next_unallocated_slot += 1
                reuses_existing_slot = False

            planned_slots_by_request[req_id] = slot_id
            new_assignments.append((req_id, slot_id, reuses_existing_slot))
            step_slot_ids.append(slot_id)

        if not new_assignments:
            return step_slot_ids

        target_capacity = max(slot_id for _, slot_id, _ in new_assignments) + 1
        self._state_cache.ensure_capacity(target_capacity)

        # Reset reused state inside the forward-pass graph so the next request
        # starts from a clean slot without adding a separate synchronization
        # point to the release path.
        for _, slot_id, reuses_existing_slot in new_assignments:
            if reuses_existing_slot:
                self._state_cache.reset_slot(slot_id)

        for req_id, slot_id, _ in new_assignments:
            self._req_to_slot[req_id] = slot_id
            self._new_req_ids.add(req_id)
        self._free_slots = reusable_slots
        return step_slot_ids

    @property
    def block_snapshot_ids(self) -> tuple[int, ...]:
        """Physical scheduler block IDs with resident GDN checkpoints."""
        return tuple(sorted(self._block_snapshots))

    @property
    def block_snapshot_stats(self) -> dict[str, int]:
        """Bounded scheduler-block checkpoint lifecycle counters."""
        return {
            "count": len(self._block_snapshots),
            "bytes": self._block_snapshot_bytes,
            "peak_count": self._block_snapshot_peak_count,
            "peak_bytes": self._block_snapshot_peak_bytes,
            "stores": self._block_snapshot_stores,
            "replacements": self._block_snapshot_replacements,
            "invalidations": self._block_snapshot_invalidations,
            "hits": self._block_snapshot_hits,
            "misses": self._block_snapshot_misses,
        }

    @property
    def cache_groups_configured(self) -> bool:
        """Whether scheduler-owned align-mode Mamba groups are installed."""
        return bool(self._mamba_group_layers)

    def configure_cache_groups(
        self,
        *,
        num_blocks: int,
        block_size: int,
        mamba_group_layers: dict[int, tuple[int, ...]],
    ) -> None:
        """Install the scheduler's Mamba-group to GDN-layer ownership map."""
        if num_blocks <= 0 or block_size <= 0:
            raise RuntimeError("Hybrid GDN APC requires positive block capacity")
        if not mamba_group_layers:
            raise RuntimeError("Hybrid GDN APC found no scheduler Mamba cache groups")
        ordered_layers = [
            layer_idx
            for layer_indices in mamba_group_layers.values()
            for layer_idx in layer_indices
        ]
        if len(set(ordered_layers)) != len(ordered_layers):
            raise RuntimeError("Scheduler Mamba cache groups overlap GDN layers")
        if set(ordered_layers) != set(range(self._state_cache.num_layers)):
            raise RuntimeError(
                "Scheduler Mamba cache groups do not cover every Metal GDN layer"
            )
        if self._block_snapshots:
            raise RuntimeError(
                "Cannot reconfigure GDN cache groups with live snapshots"
            )
        self._num_blocks = num_blocks
        self._block_size = block_size
        self._mamba_group_layers = dict(mamba_group_layers)

    def invalidate_blocks(self, block_ids: Sequence[int]) -> None:
        """Discard checkpoints exactly when scheduler physical blocks are reused."""
        for block_id in block_ids:
            snapshot = self._block_snapshots.pop(int(block_id), None)
            if snapshot is not None:
                self._block_snapshot_bytes -= snapshot.nbytes
                self._block_snapshot_invalidations += 1

    def _require_block_tables(self, block_tables: Sequence[Sequence[int]]) -> None:
        if self._block_size is None or not self._mamba_group_layers:
            raise RuntimeError("Scheduler Mamba cache-group metadata is not configured")
        max_group_idx = max(self._mamba_group_layers)
        if len(block_tables) <= max_group_idx:
            raise RuntimeError(
                "Scheduler block tables are missing a configured Mamba cache group"
            )

    def _physical_block(
        self,
        block_tables: Sequence[Sequence[int]],
        *,
        group_idx: int,
        logical_block_idx: int,
    ) -> int:
        table = block_tables[group_idx]
        if logical_block_idx < 0 or logical_block_idx >= len(table):
            raise RuntimeError(
                "Scheduler Mamba block table is too short for the state boundary"
            )
        block_id = int(table[logical_block_idx])
        if block_id < 0 or block_id >= self._num_blocks:
            raise RuntimeError(
                f"Scheduler Mamba state boundary uses invalid block id {block_id}"
            )
        return block_id

    def restore_prefix(
        self,
        req_id: str,
        block_tables: Sequence[Sequence[int]],
        num_computed_tokens: int,
    ) -> bool:
        """Restore a scheduler-approved aligned prefix into a new live slot."""
        if req_id not in self._new_req_ids:
            return False
        if num_computed_tokens <= 0:
            return False
        slot = self._req_to_slot.get(req_id)
        if slot is None:
            raise RuntimeError(f"No GDN state slot exists for request {req_id!r}")
        self._require_block_tables(block_tables)
        assert self._block_size is not None
        if num_computed_tokens % self._block_size != 0:
            raise RuntimeError(
                "Hybrid GDN prefix restore boundary is not scheduler-block aligned"
            )
        logical_block_idx = num_computed_tokens // self._block_size - 1
        snapshots: list[GDNStateSnapshot] = []
        source_blocks: list[int] = []
        for group_idx, expected_layers in self._mamba_group_layers.items():
            block_id = self._physical_block(
                block_tables,
                group_idx=group_idx,
                logical_block_idx=logical_block_idx,
            )
            snapshot = self._block_snapshots.get(block_id)
            if snapshot is None or snapshot.layer_indices != expected_layers:
                self._block_snapshot_misses += 1
                logger.error(
                    "GDN_BLOCK_SNAPSHOT_MISS req=%s group=%d block=%d "
                    "tokens=%d misses=%d",
                    req_id,
                    group_idx,
                    block_id,
                    num_computed_tokens,
                    self._block_snapshot_misses,
                )
                raise RuntimeError(
                    "Scheduler reported a hybrid prefix-cache hit but Metal has "
                    "no complete GDN state for physical Mamba block "
                    f"{block_id} (group {group_idx})"
                )
            snapshots.append(snapshot)
            source_blocks.append(block_id)

        self._state_cache.restore_slot(slot, tuple(snapshots))
        self._block_snapshot_hits += 1
        logger.info(
            "GDN_BLOCK_SNAPSHOT_HIT req=%s tokens=%d blocks=%s hits=%d",
            req_id,
            num_computed_tokens,
            source_blocks,
            self._block_snapshot_hits,
        )
        return True

    def checkpoint_blocks(
        self,
        checkpoints: Sequence[tuple[str, Sequence[Sequence[int]], int]],
    ) -> None:
        """Save post-forward state into scheduler-owned physical Mamba blocks."""
        if not checkpoints:
            return
        if not self._mamba_group_layers:
            return

        # Lazy GDN kernels may leave compact pending arrays. Scatter and force
        # them before copying any layer so every block checkpoint represents one
        # authoritative post-forward boundary.
        self._state_cache.apply_pending_states()
        mx.eval(*self._state_cache.updated_state_arrays())

        writes: list[tuple[int, GDNStateSnapshot]] = []
        seen_destination_blocks: set[int] = set()
        for req_id, block_tables, num_computed_tokens in checkpoints:
            if num_computed_tokens <= 0:
                continue
            slot = self._req_to_slot.get(req_id)
            if slot is None:
                raise RuntimeError(f"No GDN state slot exists for request {req_id!r}")
            self._require_block_tables(block_tables)
            assert self._block_size is not None
            logical_block_idx = (num_computed_tokens - 1) // self._block_size
            for group_idx, layer_indices in self._mamba_group_layers.items():
                block_id = self._physical_block(
                    block_tables,
                    group_idx=group_idx,
                    logical_block_idx=logical_block_idx,
                )
                if block_id in seen_destination_blocks:
                    raise RuntimeError(
                        "Scheduler assigned one physical Mamba block to multiple "
                        "GDN checkpoint destinations in the same step"
                    )
                seen_destination_blocks.add(block_id)
                writes.append(
                    (
                        block_id,
                        self._state_cache.snapshot_slot(slot, layer_indices),
                    )
                )

        for block_id, snapshot in writes:
            previous = self._block_snapshots.get(block_id)
            if previous is not None:
                self._block_snapshot_bytes -= previous.nbytes
                self._block_snapshot_replacements += 1
            self._block_snapshots[block_id] = snapshot
            self._block_snapshot_bytes += snapshot.nbytes
            self._block_snapshot_stores += 1
        self._block_snapshot_peak_count = max(
            self._block_snapshot_peak_count, len(self._block_snapshots)
        )
        self._block_snapshot_peak_bytes = max(
            self._block_snapshot_peak_bytes, self._block_snapshot_bytes
        )
        logger.info(
            "GDN_BLOCK_SNAPSHOT_STORE checkpoints=%d writes=%d write_bytes=%d "
            "current_blocks=%d current_bytes=%d peak_blocks=%d peak_bytes=%d "
            "stores=%d replacements=%d invalidations=%d",
            len(checkpoints),
            len(writes),
            sum(snapshot.nbytes for _, snapshot in writes),
            len(self._block_snapshots),
            self._block_snapshot_bytes,
            self._block_snapshot_peak_count,
            self._block_snapshot_peak_bytes,
            self._block_snapshot_stores,
            self._block_snapshot_replacements,
            self._block_snapshot_invalidations,
        )

    def extend_forward_eval_outputs(self, outputs: list[mx.array]) -> None:
        """Append authoritative GDN state arrays that the forward mutates."""
        outputs.extend(self._state_cache.updated_state_arrays())

    def release_requests(self, req_ids: set[str]) -> None:
        """Release slots for requests whose recurrent state is no longer valid."""
        freed_slots: list[int] = []
        for req_id in req_ids:
            slot = self._req_to_slot.pop(req_id, None)
            if slot is not None:
                freed_slots.append(slot)

        if not freed_slots:
            return

        self._state_cache.apply_pending_states()
        self._needs_materialize = True
        self._free_slots.extend(freed_slots)

    def materialize_pending_state(self) -> None:
        """Force stable GDN arrays after a slot release updated them lazily."""
        if not self._needs_materialize:
            return

        self._state_cache.apply_pending_states()
        mx.eval(*self._state_cache.updated_state_arrays())
        self._needs_materialize = False
