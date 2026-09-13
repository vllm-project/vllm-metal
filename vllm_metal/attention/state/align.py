# SPDX-License-Identifier: Apache-2.0
"""Align-mode GDN state lifecycle (hybrid prefix caching).

With ``mamba_cache_mode="align"`` the scheduler's mamba cache groups carry a
position-indexed block table per request, exactly like upstream: the state
slab for a request is the block covering its last token, and prefix caching
checkpoints a slab whenever a scheduler step ends on a block boundary.  This
manager owns the two per-step motions upstream runs in
``preprocess_state`` (vllm's Triton
``preprocess_mamba_align_fused_kernel`` + pre-copy):

- **copy-forward**: when a request's step moves it into a new block, copy its
  state from the previous block's slab into the new block's slab *before* the
  forward, leaving the previous slab untouched — that slab may be a cached
  checkpoint another request will restore from.
- **zero-init**: a fresh request (``num_computed == 0``) starts from zero
  state; its block may hold a previous life's bytes.

Restore-on-hit needs no extra motion: a prefix hit admits the request with
``num_computed = hit_length`` and a block table whose hit block holds the
checkpointed slab, so the first copy-forward reads the restored state.

Physical slots are *not* the scheduler block ids.  Mamba blocks draw their
ids from the same fungible BlockPool as full-attention KV blocks, so under
prefix caching the live mamba ids scatter across the whole ``[0, num_blocks)``
id space; a pool indexed directly by block id would have to materialize rows
up to the highest mamba id ever seen, most of them empty.  This manager keeps
an indirection ``block id → compact slot`` instead: a physical row exists for
every *distinct* mamba block currently holding state (live or cached), so the
pool grows by block count, never by id span.  A slot is reclaimed when its
block id is observed back in a full-attention group's table: BlockPool strips
a reallocated block's cached hash at allocation time, so such a checkpoint
can never be restored from again and the id is safe to forget.  Rebirth of a
reclaimed slot is always a full overwrite (zero-init or copy-forward) before
any read, so stale bytes are never observable.

Requests never own slabs here — the scheduler's block lifecycle does — so
release/materialize become no-ops apart from keeping the lazy pending-state
machinery drained.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

import mlx.core as mx

from vllm_metal.attention.caches.gdn_cache import GDNPagedStateCache

if TYPE_CHECKING:
    from vllm_metal.attention.context import PagedAttentionContext


class AlignGDNStateManager:
    """Drive block-indexed GDN state for align-mode prefix caching."""

    def __init__(self, state_cache: GDNPagedStateCache, block_size: int) -> None:
        self._state_cache = state_cache
        self._block_size = block_size
        self._needs_materialize = False
        # Compact indirection: scheduler block id → physical slab slot.  Only
        # ids that actually hold mamba state occupy rows; the pool never grows
        # to cover the id span of the shared BlockPool.
        self._slot_of: dict[int, int] = {}
        self._free_slots: list[int] = []
        self._next_slot = 0

    @property
    def needs_materialize(self) -> bool:
        return self._needs_materialize

    @property
    def occupied_slots(self) -> int:
        """Distinct scheduler blocks currently holding a physical slab."""
        return len(self._slot_of)

    @property
    def free_slots(self) -> tuple[int, ...]:
        """Slots reclaimed from retired block ids, ready for reuse."""
        return tuple(self._free_slots)

    def slot_for(self, block_id: int) -> int | None:
        """Return the compact slot backing ``block_id`` (None if unmapped)."""
        return self._slot_of.get(block_id)

    def _alloc_slot(self, block_id: int) -> int:
        """Bind ``block_id`` to a slot, preferring a reclaimed one."""
        if self._free_slots:
            slot = self._free_slots.pop()
        else:
            slot = self._next_slot
            self._next_slot += 1
        self._slot_of[block_id] = slot
        return slot

    def _ensure_slot_capacity(self) -> None:
        """Grow the pool to cover every slot handed out so far (geometric).

        The cap stays the pool's block count: distinct mamba block ids can
        never exceed it, so the scheduler-visible worst case still fits.
        """
        if self._next_slot > self._state_cache.allocated_seqs:
            target = min(
                self._state_cache.max_seqs,
                max(self._next_slot, 2 * self._state_cache.allocated_seqs),
            )
            self._state_cache.ensure_capacity(target)

    def _retire_slots(self, kv_block_ids: Iterable[int] | None) -> None:
        """Forget block ids now owned by full-attention groups.

        A block id observed in a KV group's table was reallocated by the
        BlockPool, which strips its cached hash on allocation — the mamba
        checkpoint it may have carried is unreachable from then on.  Slabs
        for cached-but-idle mamba blocks are *not* touched: only a role
        change retires a slot, so prefix-cache restores stay intact.
        """
        if not kv_block_ids or not self._slot_of:
            return
        for block_id in set(kv_block_ids).intersection(self._slot_of):
            self._free_slots.append(self._slot_of.pop(block_id))

    def apply_block_copies(
        self,
        block_copies: Sequence[tuple[int, int]],
        *,
        kv_block_ids: set[int] | None = None,
    ) -> None:
        """Apply scheduler copy-on-write to GDN slabs through the indirection.

        Pairs outside the mamba groups (full-attention CoW — the common case
        in hybrid models) have an unmapped source and are skipped: they hold
        no state, and copying them would both waste bandwidth and grow the
        pool toward the shared id-span worst case.

        Retirement (when the runner forwards this step's KV-group ids) runs
        *before* the CoW allocation: capacity never shrinks, so allocating
        first would grow the pool to cover dst slots that the about-to-be-
        freed ones could have served, stranding the difference.
        """
        pairs = [(src, dst) for src, dst in block_copies if src in self._slot_of]
        if not pairs and not kv_block_ids:
            return
        self._state_cache.apply_pending_states()
        self._retire_slots(kv_block_ids)
        if not pairs:
            return
        src_slots: list[int] = []
        dst_slots: list[int] = []
        for src, dst in pairs:
            dst_slot = self._slot_of.get(dst)
            if dst_slot is None:
                dst_slot = self._alloc_slot(dst)
            src_slots.append(self._slot_of[src])
            dst_slots.append(dst_slot)
        self._ensure_slot_capacity()
        self._state_cache.copy_slots(
            src_slots, dst_slots, self._state_cache.canonical_layers
        )

    def populate_step_context(
        self,
        *,
        req_ids: list[str],
        ctx: PagedAttentionContext,
        state_block_ids: list[list[list[int]]] | None = None,
        step_positions: list[tuple[int, int]] | None = None,
        kv_block_ids: Iterable[int] | None = None,
    ) -> None:
        """Plan one step's state motion and attach per-group slab mappings.

        ``state_block_ids[i][g]`` is request *i*'s block-id row for mamba
        group *g* (position-indexed); ``step_positions[i]`` is its
        ``(num_computed, num_scheduled)`` for this step; ``kv_block_ids``
        is the set of block ids the scheduled requests hold in their
        full-attention groups this step (slot-reclamation input).  The
        optional arguments are optional only so the state managers share
        one signature; align mode cannot plan without the first two.
        """
        if state_block_ids is None or step_positions is None:
            raise RuntimeError(
                "align-mode GDN state requires per-request mamba block ids "
                "and (num_computed, num_scheduled) step positions"
            )
        if not (len(state_block_ids) == len(step_positions) == len(req_ids)):
            raise RuntimeError(
                "align GDN state manager requires block ids and step positions "
                f"for every request (got {len(state_block_ids)} tables / "
                f"{len(step_positions)} positions for {len(req_ids)} requests)"
            )
        num_groups = len(state_block_ids[0]) if state_block_ids else 0

        # The lazy kernels' deferred compact updates are keyed by slot id;
        # slabs can move between steps here, so drain them before planning.
        # This also settles writes into slots about to be reclaimed.
        self._state_cache.apply_pending_states()
        self._retire_slots(kv_block_ids)

        # Pass 1: motions in scheduler block-id space.
        group_dst_ids: list[list[int]] = []
        group_zero_ids: list[list[int]] = []
        group_copy_src: list[list[int]] = []
        group_copy_dst: list[list[int]] = []
        for group in range(num_groups):
            dst_ids: list[int] = []
            copy_src: list[int] = []
            copy_dst: list[int] = []
            zero_ids: list[int] = []
            for req_idx, (num_computed, num_scheduled) in enumerate(step_positions):
                row = state_block_ids[req_idx][group]
                if num_scheduled <= 0:
                    raise RuntimeError(
                        f"align GDN state manager: request "
                        f"{req_ids[req_idx]!r} scheduled {num_scheduled} tokens"
                    )
                dst_idx = (num_computed + num_scheduled - 1) // self._block_size
                if dst_idx >= len(row):
                    raise RuntimeError(
                        f"align GDN state manager: request "
                        f"{req_ids[req_idx]!r} needs state block index "
                        f"{dst_idx} but its mamba block table has {len(row)} "
                        "entries"
                    )
                dst = row[dst_idx]
                dst_ids.append(dst)
                if num_computed == 0:
                    zero_ids.append(dst)
                    continue
                src_idx = (num_computed - 1) // self._block_size
                src = row[src_idx]
                if src != dst:
                    copy_src.append(src)
                    copy_dst.append(dst)
            group_dst_ids.append(dst_ids)
            group_zero_ids.append(zero_ids)
            group_copy_src.append(copy_src)
            group_copy_dst.append(copy_dst)

        # Pass 2: bind unmapped ids (dst slabs and copy-forward sources —
        # a restore's hit block appears as a src), then grow once for the
        # whole step before any row is touched.
        for group in range(num_groups):
            for block_id in (*group_dst_ids[group], *group_copy_src[group]):
                if block_id not in self._slot_of:
                    self._alloc_slot(block_id)
        self._ensure_slot_capacity()

        # Pass 3: motions in compact slot space.
        group_mappings: list[list[int]] = []
        for group in range(num_groups):
            dst_slots = [self._slot_of[b] for b in group_dst_ids[group]]
            zero_slots = [self._slot_of[b] for b in group_zero_ids[group]]
            src_slots = [self._slot_of[b] for b in group_copy_src[group]]
            copy_dst_slots = [self._slot_of[b] for b in group_copy_dst[group]]

            layer_indices = self._state_cache.layers_for_group_ordinal(group)
            self._state_cache.zero_slots(zero_slots, layer_indices)
            self._state_cache.copy_slots(src_slots, copy_dst_slots, layer_indices)
            group_mappings.append(dst_slots)

        ctx.gdn_group_slot_mappings = tuple(group_mappings)
        self._needs_materialize = True

    def extend_forward_eval_outputs(self, outputs: list[mx.array]) -> None:
        """Append authoritative GDN state arrays that the forward mutates."""
        outputs.extend(self._state_cache.updated_state_arrays())

    def release_requests(self, req_ids: set[str]) -> None:
        """Slabs belong to scheduler blocks, not requests — nothing to free.

        Preempted/finished requests leave their last written slab in place;
        the scheduler either caches that block (checkpoint) or frees and
        reuses it, in which case the next owner zero-inits or copies over it.
        """
        del req_ids

    def materialize_pending_state(self) -> None:
        """Force stable GDN arrays out of the lazy graph between steps."""
        if not self._needs_materialize:
            return
        self._state_cache.apply_pending_states()
        mx.eval(*self._state_cache.updated_state_arrays())
        self._needs_materialize = False
