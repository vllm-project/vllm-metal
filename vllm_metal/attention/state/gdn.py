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

from typing import TYPE_CHECKING

import mlx.core as mx

from vllm_metal.attention.caches.gdn_cache import GDNPagedStateCache

if TYPE_CHECKING:
    from vllm_metal.attention.context import PagedAttentionContext


class HybridGDNStateManager:
    """Own request-to-slot lifecycle for one hybrid runtime."""

    def __init__(self, state_cache: GDNPagedStateCache) -> None:
        self._state_cache = state_cache
        self._req_to_slot: dict[str, int] = {}
        self._free_slots: list[int] = []
        self._needs_materialize = False

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
        self._free_slots = reusable_slots
        return step_slot_ids

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
