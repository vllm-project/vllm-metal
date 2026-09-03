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

    def __init__(
        self,
        state_cache: GDNPagedStateCache,
        block_size: int,
        num_speculative_blocks: int = 0,
    ) -> None:
        if num_speculative_blocks < 0:
            raise ValueError("num_speculative_blocks must be non-negative")
        self._state_cache = state_cache
        self._block_size = block_size
        self._num_speculative_blocks = num_speculative_blocks
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
        group_state_chains: list[list[list[int]]] = []
        has_speculative_chains = False
        for group in range(num_groups):
            dst_ids: list[int] = []
            request_chains: list[list[int]] = []
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

                is_speculative_verify = (
                    self._num_speculative_blocks > 0
                    and req_idx < ctx.num_decode_requests
                    and num_scheduled > 1
                )
                if is_speculative_verify:
                    if num_computed <= 0:
                        raise RuntimeError(
                            "speculative GDN verification requires an existing "
                            "committed state"
                        )
                    num_drafts = num_scheduled - 1
                    if num_drafts > self._num_speculative_blocks:
                        raise RuntimeError(
                            f"request {req_ids[req_idx]!r} scheduled {num_drafts} "
                            "draft tokens but its MambaSpec reserves only "
                            f"{self._num_speculative_blocks} speculative blocks"
                        )
                    state_idx = (num_computed - 1) // self._block_size
                    end = state_idx + num_scheduled
                    if end > len(row):
                        raise RuntimeError(
                            f"request {req_ids[req_idx]!r} needs GDN state "
                            f"columns [{state_idx}, {end}) but its mamba block "
                            f"table has {len(row)} entries"
                        )
                    # This mirrors upstream mamba_get_block_table_tensor():
                    # current state block followed by K speculative blocks.
                    output_slots = list(row[state_idx:end])
                    # Token zero reads and overwrites the current private state
                    # block; each draft writes the following speculative block.
                    request_chains.append([output_slots[0], *output_slots])
                    dst_ids.append(output_slots[-1])
                    has_speculative_chains = True
                    continue

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
                request_chains.append([])
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
            group_state_chains.append(request_chains)

        ctx.gdn_group_slot_mappings = tuple(group_mappings)
        ctx.gdn_group_state_chains = (
            tuple(group_state_chains) if has_speculative_chains else None
        )
        self._needs_materialize = True

    def commit_speculative_state(
        self,
        *,
        req_ids: list[str],
        state_block_ids: list[list[list[int]]],
        step_positions: list[tuple[int, int]],
        num_sampled_tokens: list[int],
    ) -> None:
        """Promote the verifier-selected GDN checkpoint.

        A verification window with ``K`` drafts writes state after each of its
        ``K + 1`` input tokens into consecutive Mamba block-table columns.
        The verifier emits ``num_sampled`` tokens; upstream's invariant is that
        the committed recurrent checkpoint is column ``num_sampled - 1``.
        """
        if not (
            len(req_ids)
            == len(state_block_ids)
            == len(step_positions)
            == len(num_sampled_tokens)
        ):
            raise RuntimeError("speculative GDN commit metadata length mismatch")

        self._state_cache.apply_pending_states()
        num_groups = len(state_block_ids[0]) if state_block_ids else 0
        for group in range(num_groups):
            copy_src: list[int] = []
            copy_dst: list[int] = []
            for req_idx, (num_computed, num_scheduled) in enumerate(step_positions):
                if num_scheduled <= 1:
                    continue
                sampled = num_sampled_tokens[req_idx]
                if sampled < 1 or sampled > num_scheduled:
                    raise RuntimeError(
                        f"request {req_ids[req_idx]!r} sampled {sampled} tokens "
                        f"for a {num_scheduled}-token verification window"
                    )
                row = state_block_ids[req_idx][group]
                state_idx = (num_computed - 1) // self._block_size
                selected_idx = state_idx + sampled - 1
                committed_idx = (num_computed + sampled - 1) // self._block_size
                if max(selected_idx, committed_idx) >= len(row):
                    raise RuntimeError(
                        f"request {req_ids[req_idx]!r} speculative commit "
                        "exceeds its mamba block table"
                    )
                selected = row[selected_idx]
                committed = row[committed_idx]
                if selected != committed:
                    copy_src.append(selected)
                    copy_dst.append(committed)

            layer_indices = self._state_cache.layers_for_group_ordinal(group)
            self._state_cache.copy_slots(copy_src, copy_dst, layer_indices)

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
