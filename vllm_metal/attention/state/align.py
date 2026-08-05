# SPDX-License-Identifier: Apache-2.0
"""Align-mode GDN state lifecycle (hybrid prefix caching).

With ``mamba_cache_mode="align"`` the scheduler's mamba cache groups carry a
position-indexed block table per request, exactly like upstream: the state
slab for a request is the block covering its last token, and prefix caching
checkpoints a slab whenever a scheduler step ends on a block boundary.  The
state pool is therefore indexed directly by scheduler block id (one slab id
per pool block, materialized lazily to the high-water block id), and this
manager owns the two per-step motions
upstream runs in ``preprocess_state`` (vllm's Triton
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

Requests never own slabs here — the scheduler's block lifecycle does — so
release/materialize become no-ops apart from keeping the lazy pending-state
machinery drained.
"""

from __future__ import annotations

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

    @property
    def needs_materialize(self) -> bool:
        return self._needs_materialize

    def populate_step_context(
        self,
        *,
        req_ids: list[str],
        ctx: PagedAttentionContext,
        state_block_ids: list[list[list[int]]] | None = None,
        step_positions: list[tuple[int, int]] | None = None,
    ) -> None:
        """Plan one step's state motion and attach per-group slab mappings.

        ``state_block_ids[i][g]`` is request *i*'s block-id row for mamba
        group *g* (position-indexed); ``step_positions[i]`` is its
        ``(num_computed, num_scheduled)`` for this step.  Both arguments are
        optional only so the state managers share one signature; align mode
        cannot plan without them.
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

        # The lazy kernels' deferred compact updates are keyed by slab id;
        # slabs can move between steps here, so drain them before planning.
        self._state_cache.apply_pending_states()

        # Lazily grow the pool to the highest block id this step touches
        # (geometric, capped at the pool size the plan budgeted).  BlockPool
        # allocates low ids first, so the high-water mark tracks the live +
        # cached set rather than the worst case.
        high_water = 0
        for tables in state_block_ids:
            for row in tables:
                if row:
                    high_water = max(high_water, max(row) + 1)
        if high_water > self._state_cache.allocated_seqs:
            target = min(
                self._state_cache.max_seqs,
                max(high_water, 2 * self._state_cache.allocated_seqs),
            )
            self._state_cache.ensure_capacity(target)

        group_mappings: list[list[int]] = []
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

            layer_indices = self._state_cache.layers_for_group_ordinal(group)
            self._state_cache.zero_slots(zero_ids, layer_indices)
            self._state_cache.copy_slots(copy_src, copy_dst, layer_indices)
            group_mappings.append(dst_ids)

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
