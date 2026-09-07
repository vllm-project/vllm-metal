# SPDX-License-Identifier: Apache-2.0
"""Draft-model speculative decoding proposer for the Metal paged path.

A :class:`DraftModelProposer` drafts with a *separate* full model (vLLM
``method="draft_model"``). Its KV is split into two parts with different
ownership, matching where cross-request cache *reuse* actually happens
(#482):

- The **committed** portion (``[0, committed_len)`` per request) is a real
  scheduler-owned KV-cache group (``cache_policy.ModelCachePolicy.
  _draft_layer_specs``), registered and sized exactly like the target's own
  groups. The scheduler hashes, matches, admits, and evicts it -- so
  ``cache_salt``, per-request cache-read rules (``skip_reading_prefix_cache``),
  and the real admission/allocation limit all apply to it automatically,
  the same way they already apply to the target. This is where content-based
  reuse across requests lives.
- The **speculative lookahead** tail (positions drafted ahead of a request's
  committed length, not yet verified) has no scheduler concept -- no
  registered group is ever "ahead" of a request's committed tokens -- so it
  stays a small, proposer-local scratch reservation
  (``committed_num_blocks + scratch_reserve_blocks``, see
  ``DraftModelProposer.build`` and ``ModelCachePolicy.
  draft_scratch_reserve_blocks``). It never claims cross-request reuse, so
  none of the scheduler-owned group's guarantees are needed there.

**Block-alignment note.** When ``committed_len`` is not a multiple of
``block_size``, the last committed block is only partially filled with
scheduler-owned KV.  The lookahead draft steps write their speculative KV
into the remaining slots of that same scheduler-owned block (and into scratch
blocks beyond it, if needed).  This is safe because ingest rewrites every
position in ``[draft_seq_len, committed_len)`` *before* the block can be
hashed: the scheduler only content-hashes a block once it is fully filled,
and by that point a future ingest step will have overwritten the speculative
slots with the correct committed KV.  No stale draft KV can leak into a
prefix-cache match.

**Why every active row ingests, not just drafting-eligible ones.** The
scheduler advances a request's ``num_computed_tokens`` and marks the
committed group's blocks "cached" based on its own per-step bookkeeping,
trusting that whatever it scheduled, the worker actually computed -- for
*every* registered group, uniformly, every step. But
``SpeculativeDecodeController.draft_eligible_requests`` intentionally
excludes non-greedy requests and intermediate prefill chunks from
*drafting*. If those rows also skipped *ingest*, the scheduler would believe
their committed-group blocks hold real KV when they never did, and a later,
unrelated request whose prompt happens to hash-match that prefix could be
handed those blocks as a "cache hit" -- silent data corruption, not just a
missed optimization. So ``propose()`` ingests every decode and prefill row
(chunked or not, greedy or not) to keep the committed group's physical KV
genuinely in sync with what the scheduler believes, and only *drafts*
(produces returned token ids, runs the extra lookahead decode steps) for the
eligible subset.

Each step: ingest advances every row's committed KV to ``committed_len``,
then ``num_speculative_tokens - 1`` single-token decode steps run for
drafting rows only. Drafts are handed back via ``take_draft_token_ids`` and
verified next step by ``SpeculativeDecodeController.verify_greedy``.

**Skipping already-valid speculative KV (#482, direction 2).** The
lookahead draft steps write KV for drafts ``d1..d(K-1)`` at positions
``[committed_len, committed_len + K - 1)``. When the next round's committed
tokens match those drafts (greedy acceptance is a prefix, so accepted drafts
keep their identities), that KV is exactly what the ingest would recompute
-- so the ingest skips it (``_speculative_kv_valid_through``), shrinking the
steady-state K+1-token ingest to 2 rows on full acceptance. A position is
skipped only when its committed token equals the recorded drafted token AND
the scheduler's committed-group block table still maps the position to the
same physical block the speculative write landed in (speculative KV that
crossed into a scratch block is never reused -- the scheduler allocates its
own block there). The last committed token is always re-ingested: its
logits are needed to predict this round's first draft token anyway.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import mlx.core as mx
from mlx_lm import load as mlx_lm_load
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.outputs import DraftTokenIds

from vllm_metal import envs
from vllm_metal.attention.context import (
    OffsetCache,
    clear_context,
    prepare_unified,
)
from vllm_metal.attention.runtime.mha import MHAPagedAttentionRuntime
from vllm_metal.metal.constants import PA_WINDOW_MAX_HEAD_SIZE
from vllm_metal.utils import get_model_download_path
from vllm_metal.v1.mlx_lm_paths import mlx_lm_compatible_model_path

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from vllm.config import ParallelConfig
    from vllm.config.speculative import SpeculativeConfig

    from vllm_metal.v1.model_runner import PrefillRequest, RequestState
    from vllm_metal.v1.proposer import ProposeContext
    from vllm_metal.v1.spec_decode import SpeculativeDecodeController

logger = init_logger(__name__)


@dataclass(frozen=True, slots=True)
class DraftDims:
    num_layers: int
    num_kv_heads: int
    head_dim: int


@dataclass(frozen=True, slots=True)
class _DraftPlan:
    """One request's drafting plan for the current step."""

    req_id: str
    block_ids: list[int]
    committed_len: int
    # Effective start of this step's ingest forward: the position whose KV
    # is not already valid in the draft cache (either never written, or
    # written speculatively last round with a token/block that no longer
    # matches -- see _speculative_kv_valid_through).
    draft_seq_len: int
    ingest_tokens: list[int]
    is_drafting: bool


# Ingests at or below this size are submitted as expanded decode rows instead
# of a prefill segment (see _ingest_and_draft_first). Covers the steady-state
# K+1-token ingest for any practical num_speculative_tokens while keeping
# full-prompt catch-up ingests on the tiled prefill kernel.
_DECODE_INGEST_MAX_TOKENS = 16


class DraftModelProposer:
    """:class:`vllm_metal.v1.proposer.MetalProposer` backed by a separate model."""

    def __init__(
        self,
        *,
        model: Any,
        block_size: int,
        committed_num_blocks: int,
        scratch_reserve_blocks: int,
        num_layers: int,
        controller: SpeculativeDecodeController,
        extract_logits: Callable[[Any], mx.array],
        merge_ingest_windows: bool = False,
    ) -> None:
        self._model = model
        self._block_size = block_size
        self._controller = controller
        self._extract_logits = extract_logits
        # Structural half of the ingest window gate (see `build`); the
        # operator half (VLLM_METAL_SPEC_VERIFY_WINDOW) is read per call
        # like the runner's `merge_verify_windows` property.  The same
        # env governs the target's verify layout and this ingest layout;
        # each side still applies its own structural gate, so an
        # ineligible target does not block an eligible draft (or vice
        # versa).
        self._merge_ingest_windows = merge_ingest_windows
        # Stateless RoPE/mask shims for the draft forward (one per layer). The
        # real per-request offsets come from the paged context, so these carry
        # no state — allocate once and reuse across steps, not per propose().
        self._offset_caches = [OffsetCache(0) for _ in range(num_layers)]
        # Committed positions actually written into the draft cache, per
        # request. Self-tracked so it only advances when we actually ran a
        # forward pass; seeded from the scheduler's cache-hit boundary the
        # first time a request is seen (see _make_decode_plan), so a
        # resubmitted/shared prefix skips re-ingest (#482).
        self._draft_seq_lens: dict[str, int] = {}
        # Speculative KV written by past lookahead draft steps, per request:
        # position -> (block_id, drafted token). Lets the next round's ingest
        # skip re-ingesting accepted drafts whose KV the lookahead already
        # wrote (#482 direction 2). Entries are replaced wholesale each time
        # a request drafts, and survive non-drafting rounds in between --
        # each position is re-validated against the current committed tokens
        # and block table before use, so staleness can only cost performance,
        # never correctness.
        self._spec_kv_writes: dict[str, dict[int, tuple[int, int]]] = {}
        # Scheduler-owned KV-cache group index for the committed portion.
        # Unknown at construction time -- the physical backend below is
        # built before the scheduler has decided kv_cache_config (see
        # ModelCachePolicy._adopt_draft_scheduler_group) -- so this is set
        # later via adopt_committed_group().
        self._committed_group_index: int | None = None
        self._committed_num_blocks = committed_num_blocks
        # The speculative lookahead tail draws from block ids the scheduler
        # never assigns: [committed_num_blocks, committed_num_blocks +
        # scratch_reserve_blocks). The physical backend (see `build`) is
        # over-provisioned to cover this range.
        self._scratch_free_blocks: list[int] = list(
            range(
                committed_num_blocks,
                committed_num_blocks + scratch_reserve_blocks,
            )
        )
        self._scratch_req_blocks: dict[str, list[int]] = {}

    # -- construction --------------------------------------------------------

    @classmethod
    def build(
        cls,
        *,
        speculative_config: SpeculativeConfig,
        parallel_config: ParallelConfig,
        controller: SpeculativeDecodeController,
        extract_logits: Callable[[Any], mx.array],
        committed_num_blocks: int,
        scratch_reserve_blocks: int,
        block_size: int,
        dtype: mx.Dtype,
    ) -> DraftModelProposer:
        model, dims = _load_draft_model(speculative_config, parallel_config)
        total_blocks = committed_num_blocks + scratch_reserve_blocks
        backend = MHAPagedAttentionRuntime(
            num_layers=dims.num_layers,
            num_kv_heads=dims.num_kv_heads,
            head_dim=dims.head_dim,
            block_size=block_size,
            dtype=dtype,
        )
        backend.initialize(total_blocks)
        n_patched = backend.patch_model(model)
        logger.info(
            "Draft model loaded for speculative decoding: %s "
            "(layers=%d, kv_heads=%d, head_dim=%d, patched=%d, "
            "committed_blocks=%d, scratch_blocks=%d)",
            speculative_config.draft_model_config.model,
            dims.num_layers,
            dims.num_kv_heads,
            dims.head_dim,
            n_patched,
            committed_num_blocks,
            scratch_reserve_blocks,
        )
        return cls(
            model=model,
            block_size=block_size,
            committed_num_blocks=committed_num_blocks,
            scratch_reserve_blocks=scratch_reserve_blocks,
            num_layers=dims.num_layers,
            controller=controller,
            extract_logits=extract_logits,
            # Mirror of the runner's `merge_verify_windows` structural
            # conditions, reduced to what can arise here: this proposer
            # patches drafts through `MHAPagedAttentionRuntime`, so the
            # runner's MLA-native-decode and GDN-pure-decode arms are
            # vacuous, and `_load_draft_model` resolves one uniform
            # head_dim.  Only the decode kernel's head bound remains.
            merge_ingest_windows=dims.head_dim <= PA_WINDOW_MAX_HEAD_SIZE,
        )

    def adopt_committed_group(self, group_index: int) -> None:
        """Record which scheduler KV-cache group owns the committed portion.

        Called from ``ModelCachePolicy._adopt_draft_scheduler_group`` once
        ``kv_cache_config`` exists -- after this proposer is built, since the
        physical backend above is sized before the scheduler has decided
        groups.
        """
        self._committed_group_index = group_index

    # -- MetalProposer protocol ---------------------------------------------

    def needs_target_hidden_states(
        self,
        decode_segments: Any,
        *,
        has_final_prefill: bool,
    ) -> bool:
        # A standalone draft model consumes only token ids; it never reads the
        # target's hidden states (mirrors upstream pass_hidden_states_to_model=False).
        return False

    def propose(self, ctx: ProposeContext) -> DraftTokenIds | None:
        num_speculative_tokens = ctx.num_speculative_tokens
        if num_speculative_tokens <= 0:
            return None
        if self._committed_group_index is None:
            raise RuntimeError(
                "DraftModelProposer.propose() called before "
                "adopt_committed_group() -- initialize_kv_cache() must run "
                "before the first speculative decode step"
            )

        self._prune_finished(ctx.request_states)
        plans = self._collect_draft_plans(ctx, num_speculative_tokens)
        if not plans:
            return None

        # Ingest every active row (see module docstring); predicts a first
        # token per row, used below only for drafting rows.
        first_tokens = self._ingest_and_draft_first(plans, self._offset_caches)

        # The committed group now holds KV through committed_len for every
        # row that ingested this step, drafting or not.
        for plan in plans:
            self._draft_seq_lens[plan.req_id] = plan.committed_len

        drafting_plans = [plan for plan in plans if plan.is_drafting]
        if not drafting_plans:
            return None

        drafting_indices = mx.array(
            [i for i, plan in enumerate(plans) if plan.is_drafting],
            dtype=mx.int32,
        )
        draft_cols: list[mx.array] = [mx.take(first_tokens, drafting_indices, axis=0)]
        # Steps 2..K: single-token decode per drafting row.
        for draft_index in range(1, num_speculative_tokens):
            draft_cols.append(
                self._draft_step(
                    drafting_plans,
                    draft_cols[-1],
                    draft_index,
                    self._offset_caches,
                )
            )

        drafts = mx.stack(draft_cols, axis=1)  # [num_drafting, K]
        mx.eval(drafts)
        rows: list[list[int]] = drafts.tolist()  # type: ignore[assignment]

        # Record the speculative KV this round's lookahead steps wrote, for
        # next round's ingest skip (see module docstring). Draft step i fed
        # row[i-1] at position committed_len + i - 1 for i in 1..K-1, writing
        # its KV there; the K-th draft's KV is never written (no step feeds
        # it before verification).
        for plan, row in zip(drafting_plans, rows, strict=True):
            writes: dict[int, tuple[int, int]] = {}
            for i in range(num_speculative_tokens - 1):
                position = plan.committed_len + i
                writes[position] = (
                    plan.block_ids[position // self._block_size],
                    int(row[i]),
                )
            self._spec_kv_writes[plan.req_id] = writes

        return DraftTokenIds(
            req_ids=[plan.req_id for plan in drafting_plans],
            draft_token_ids=[[int(token) for token in row] for row in rows],
        )

    def release_requests(self, req_ids: set[str]) -> None:
        # Only the speculative lookahead tail is proposer-owned; the
        # committed portion's lifecycle (finish/evict/refcount) belongs to
        # the scheduler now, like any other KV-cache group. Mirrors
        # _prune_finished for an explicit lifecycle set.
        for req_id in req_ids:
            blocks = self._scratch_req_blocks.pop(req_id, None)
            if blocks is not None:
                self._scratch_free_blocks.extend(blocks)
            self._draft_seq_lens.pop(req_id, None)
            self._spec_kv_writes.pop(req_id, None)

    # -- internals -----------------------------------------------------------

    def _prune_finished(self, request_states: Mapping[str, RequestState]) -> None:
        for req_id in list(self._scratch_req_blocks.keys()):
            if req_id not in request_states:
                self._scratch_free_blocks.extend(self._scratch_req_blocks.pop(req_id))
        for req_id in list(self._draft_seq_lens.keys()):
            if req_id not in request_states:
                del self._draft_seq_lens[req_id]
        for req_id in list(self._spec_kv_writes.keys()):
            if req_id not in request_states:
                del self._spec_kv_writes[req_id]

    def _collect_draft_plans(
        self, ctx: ProposeContext, num_speculative_tokens: int
    ) -> list[_DraftPlan]:
        # can_draft_greedy/draft_eligible_requests decide *drafting*
        # eligibility only here (greedy + non-intermediate prefill +
        # greedy-only sampling); ingest itself runs for every active row
        # regardless -- see module docstring.
        drafting_req_ids = {
            req_id
            for req_id, _ in self._controller.draft_eligible_requests(
                ctx.decode_reqs,
                ctx.decode_token_ids,
                ctx.prefill_reqs,
                ctx.prefill_result_modes,
                ctx.request_states,
            )
        }
        plans: list[_DraftPlan] = []
        for req_id, state in ctx.decode_reqs:
            plan = self._make_decode_plan(
                req_id, state, num_speculative_tokens, drafting_req_ids
            )
            if plan is not None:
                plans.append(plan)

        seen = {plan.req_id for plan in plans}
        for prefill_index, (prefill, result_mode) in enumerate(
            zip(ctx.prefill_reqs, ctx.prefill_result_modes, strict=True)
        ):
            if prefill.req_id in seen:
                continue
            seen.add(prefill.req_id)
            if prefill.req_id in drafting_req_ids:
                # Target prefill already committed its first output token.
                # Ingest it so the first draft predicts the following token.
                prefill = prefill._replace(
                    token_ids=[*prefill.token_ids, ctx.prefill_token_ids[prefill_index]]
                )
            plan = self._make_prefill_plan(
                prefill, result_mode, num_speculative_tokens, drafting_req_ids
            )
            if plan is not None:
                plans.append(plan)
        return plans

    def _make_decode_plan(
        self,
        req_id: str,
        state: RequestState,
        num_speculative_tokens: int,
        drafting_req_ids: set[str],
    ) -> _DraftPlan | None:
        # state.token_ids already reflects this step's sampled token(s) by
        # the time propose() runs (mirrors the runner's own decode-state
        # update), so this is valid for decode -- unlike prefill, where
        # token_ids is pre-populated with the whole future prompt and
        # committed_len must come from the scheduled chunk instead (see
        # _make_prefill_plan).
        committed_len = len(state.token_ids)
        draft_seq_len = self._draft_seq_lens.setdefault(
            req_id, state.num_computed_tokens
        )
        if draft_seq_len >= committed_len:
            # No newly committed tokens to ingest (should not happen for an
            # accepted decode step); skip rather than emit an empty forward.
            return None
        is_drafting = req_id in drafting_req_ids
        assert self._committed_group_index is not None
        committed_group_block_ids = state.block_ids[self._committed_group_index]
        # Skip the leading accepted drafts whose KV the previous round's
        # lookahead steps already wrote (#482 direction 2); the walk stops at
        # the first position whose recorded token or block no longer matches
        # what is now committed. Capped one short of committed_len: the last
        # committed token's row must still run (its logits predict this
        # round's first draft token), so the ingest is never empty.
        draft_seq_len = min(
            self._speculative_kv_valid_through(
                req_id,
                state.token_ids,
                committed_group_block_ids,
                draft_seq_len,
                committed_len,
            ),
            committed_len - 1,
        )
        block_ids = self._ensure_blocks(
            req_id,
            committed_group_block_ids=committed_group_block_ids,
            total_positions=committed_len
            + (num_speculative_tokens if is_drafting else 0),
        )
        return _DraftPlan(
            req_id=req_id,
            block_ids=block_ids,
            committed_len=committed_len,
            draft_seq_len=draft_seq_len,
            ingest_tokens=list(state.token_ids[draft_seq_len:committed_len]),
            is_drafting=is_drafting,
        )

    def _make_prefill_plan(
        self,
        prefill: PrefillRequest,
        result_mode: str,
        num_speculative_tokens: int,
        drafting_req_ids: set[str],
    ) -> _DraftPlan | None:
        if not prefill.token_ids:
            return None
        req_id = prefill.req_id
        committed_len = prefill.start_pos + len(prefill.token_ids)
        is_drafting = result_mode != "intermediate" and req_id in drafting_req_ids
        assert self._committed_group_index is not None
        # The ingest includes the target's sampled token. Producing K drafts
        # then feeds only K-1 more tokens through the draft model.
        block_ids = self._ensure_blocks(
            req_id,
            committed_group_block_ids=prefill.block_ids[self._committed_group_index],
            total_positions=committed_len
            + (max(num_speculative_tokens - 1, 0) if is_drafting else 0),
        )
        plan = _DraftPlan(
            req_id=req_id,
            block_ids=block_ids,
            committed_len=committed_len,
            draft_seq_len=prefill.start_pos,
            ingest_tokens=list(prefill.token_ids),
            is_drafting=is_drafting,
        )
        return plan

    def _ensure_blocks(
        self,
        req_id: str,
        *,
        committed_group_block_ids: list[int],
        total_positions: int,
    ) -> list[int]:
        """Scheduler-assigned committed blocks + this request's scratch tail.

        The committed prefix comes straight from the scheduler's own
        allocation for the draft KV-cache group -- never grown or freed
        here. Only the tail beyond it (the speculative lookahead, when
        drafting) is drawn from the local scratch pool, grown or shrunk to
        exactly what this step needs.

        When ``committed_len`` is not block-aligned, ``total_positions``
        extends into the last committed block's unfilled tail.  The
        lookahead draft steps write speculative KV into those slots of a
        scheduler-owned block.  This is safe: ingest always rewrites the
        committed range before the scheduler can hash the block (see
        module docstring, "Block-alignment note").
        """
        needed_total = cdiv(total_positions, self._block_size)
        scratch_needed = max(0, needed_total - len(committed_group_block_ids))

        scratch_blocks = self._scratch_req_blocks.setdefault(req_id, [])
        if len(scratch_blocks) > scratch_needed:
            self._scratch_free_blocks.extend(scratch_blocks[scratch_needed:])
            del scratch_blocks[scratch_needed:]
        while len(scratch_blocks) < scratch_needed:
            if not self._scratch_free_blocks:
                raise RuntimeError(
                    f"Draft KV scratch pool exhausted: request {req_id!r} "
                    f"needs {scratch_needed} lookahead block(s) but none are "
                    "free. Lower --max-num-seqs or raise "
                    "VLLM_METAL_MEMORY_FRACTION."
                )
            scratch_blocks.append(self._scratch_free_blocks.pop())
        if not scratch_blocks:
            self._scratch_req_blocks.pop(req_id, None)

        return list(committed_group_block_ids) + scratch_blocks

    def _speculative_kv_valid_through(
        self,
        req_id: str,
        token_ids: list[int],
        committed_group_block_ids: list[int],
        draft_seq_len: int,
        committed_len: int,
    ) -> int:
        """First position in ``[draft_seq_len, committed_len)`` whose committed
        KV is not already valid in the draft cache.

        Consults the speculative-write ledger from the last round that
        drafted this request. A position is skippable only when its committed
        token equals the recorded drafted token AND the position still maps
        to the same scheduler-owned committed block the speculative write
        landed in; anything else (rejected draft, scratch-block write,
        re-allocated block table, no ledger) stops the walk, because the
        ingest forward must be one contiguous range of positions whose KV is
        valid up to its start.
        """
        writes = self._spec_kv_writes.get(req_id)
        if not writes:
            return draft_seq_len
        position = draft_seq_len
        while position < committed_len:
            write = writes.get(position)
            if write is None:
                break
            block_id, drafted_token = write
            block_index = position // self._block_size
            if (
                block_index >= len(committed_group_block_ids)
                or committed_group_block_ids[block_index] != block_id
                or token_ids[position] != drafted_token
            ):
                break
            position += 1
        return position

    def _ingest_and_draft_first(
        self, plans: list[_DraftPlan], offset_caches: list[OffsetCache]
    ) -> mx.array:
        # The steady-state ingest (K+1 committed tokens per accepted round)
        # rides the decode path as one single-query segment per token — the
        # same shape the target's verify rows use. Submitting it as a prefill
        # segment routes the whole forward to the tiled prefill kernel, which
        # reads the entire context regardless of query size: ~70 ms vs ~11 ms
        # per pass at an 8k prefix (#482, Problem 2). Large ingests (the
        # first-propose catch-up) stay on the tiled path, where it wins.
        if max(len(plan.ingest_tokens) for plan in plans) <= _DECODE_INGEST_MAX_TOKENS:
            packed: list[int] = []
            last_rows: list[int] = []
            for plan in plans:
                packed.extend(plan.ingest_tokens)
                last_rows.append(len(packed) - 1)
            input_ids = mx.array([packed], dtype=mx.int32)
            decode_specs = [
                (plan.block_ids, plan.draft_seq_len, len(plan.ingest_tokens))
                for plan in plans
            ]
            # Same opt-in as the target's verify path (#534): when the
            # operator sets VLLM_METAL_SPEC_VERIFY_WINDOW and the draft
            # fits the window kernel, keep each plan's ingest as ONE
            # merged segment so K/V block loads are shared across its
            # rows; otherwise the expanded per-token layout below is
            # bit-for-bit the pre-window behavior.
            prepare_unified(
                decode_specs,
                [],
                self._block_size,
                merge_verify_windows=self._merge_ingest_windows
                and envs.VLLM_METAL_SPEC_VERIFY_WINDOW,
            )
            try:
                logits = self._extract_logits(
                    self._model(input_ids, cache=offset_caches)
                )
            finally:
                clear_context()

            last = mx.take(logits[0], mx.array(last_rows, dtype=mx.int32), axis=0)
            return mx.argmax(last, axis=-1)

        # Cold ingest: a fresh prefix re-ingests the whole prompt into the
        # draft cache in one tiled prefill forward, stalling the engine for
        # the full prompt length (#482, direction 3). Run it in chunks so
        # each dispatch is bounded (default 1024 tokens, ~2 ms of draft-model
        # work) and the logits peak allocation scales with the chunk, not the
        # prompt. The chunk start maps to draft position
        # ``draft_seq_len + start``, and the final row of each plan's last
        # chunk is the last ingested token, whose logits predict the plan's
        # first draft token — identical to the single-forward path.
        max_len = max(len(plan.ingest_tokens) for plan in plans)
        chunk_size = envs.VLLM_METAL_SPEC_INGEST_CHUNK
        if chunk_size <= 0:
            chunk_size = max_len  # "0" restores the single-forward behavior
        final_rows: dict[int, mx.array] = {}
        for start in range(0, max_len, chunk_size):
            round_packed: list[int] = []
            prefill_specs: list[tuple[list[int], int, int]] = []
            final_row_indices: list[tuple[int, int]] = []
            for plan_index, plan in enumerate(plans):
                ingest_len = len(plan.ingest_tokens)
                end = min(start + chunk_size, ingest_len)
                if end <= start:
                    continue
                round_packed.extend(plan.ingest_tokens[start:end])
                prefill_specs.append(
                    (plan.block_ids, end - start, plan.draft_seq_len + start)
                )
                final_row_indices.append((plan_index, len(round_packed) - 1))
            input_ids = mx.array([round_packed], dtype=mx.int32)
            prepare_unified([], prefill_specs, self._block_size)
            try:
                logits = self._extract_logits(
                    self._model(input_ids, cache=offset_caches)
                )
            finally:
                clear_context()
            for plan_index, row in final_row_indices:
                final_rows[plan_index] = logits[0][row]

        last = mx.stack(
            [final_rows[plan_index] for plan_index in range(len(plans))], axis=0
        )
        return mx.argmax(last, axis=-1)

    def _draft_step(
        self,
        plans: list[_DraftPlan],
        prev_tokens: mx.array,
        draft_index: int,
        offset_caches: list[OffsetCache],
    ) -> mx.array:
        # prev_tokens[i] sits at position committed_len_i + (draft_index - 1).
        decode_specs = [
            (plan.block_ids, plan.committed_len + draft_index - 1, 1) for plan in plans
        ]
        input_ids = prev_tokens[None, :].astype(mx.int32)

        prepare_unified(decode_specs, [], self._block_size)
        try:
            logits = self._extract_logits(self._model(input_ids, cache=offset_caches))
        finally:
            clear_context()

        return mx.argmax(logits[0], axis=-1)


def resolve_draft_dims(
    speculative_config: SpeculativeConfig,
    parallel_config: ParallelConfig,
) -> DraftDims:
    """Resolve the draft model's cache shape from its HF config alone.

    Config-only and weight-free, so this can run early in the runner
    lifecycle (before ``determine_available_memory()``/``get_kv_cache_spec()``)
    to size a scheduler-visible KV-cache group for the draft model, well
    before its MLX weights are actually loaded in ``_load_draft_model``.

    Delegates to ``ModelConfig``'s own accessors rather than re-deriving GQA
    head-count and head-size fallbacks from raw ``hf_config`` fields --
    ``get_head_size()``/``get_num_kv_heads()`` already handle those (and the
    MLA-becomes-MQA case), the same way upstream's own
    ``SpecDecodeBaseProposer`` reads them off its ``draft_model_config``.

    Raises :class:`NotImplementedError` if the draft model uses sliding-window
    or hybrid attention.  ``_draft_layer_specs`` unconditionally emits
    ``FullAttentionSpec`` for every draft layer, so a sliding-window draft
    would silently get the wrong spec and corrupt KV cache sizing.
    """
    draft_model_config = speculative_config.draft_model_config
    if draft_model_config is None:
        raise ValueError(
            "draft_model speculative decoding requires a draft_model_config"
        )
    _require_full_attention_draft(draft_model_config)
    return DraftDims(
        num_layers=draft_model_config.get_total_num_hidden_layers(),
        num_kv_heads=draft_model_config.get_num_kv_heads(parallel_config),
        head_dim=draft_model_config.get_head_size(),
    )


def _require_full_attention_draft(draft_model_config: Any) -> None:
    """Reject draft models that are not plain full-attention transformers.

    The scheduler-managed draft KV cache (``_draft_layer_specs``) registers
    every draft layer as a ``FullAttentionSpec``.  A draft model whose HF
    config declares sliding-window attention or mixed layer types (e.g.
    Gemma4's ``sliding_attention`` / ``full_attention`` hybrid) would be
    silently misrepresented, leading to incorrect KV cache sizing and
    potential data corruption.  Fail fast instead.
    """
    # 1) Global or per-model sliding window
    sliding_window = draft_model_config.get_sliding_window()
    if sliding_window is not None:
        raise NotImplementedError(
            f"Draft model has sliding_window={sliding_window}, but the "
            "draft KV cache only supports full-attention transformers. "
            "Sliding-window draft models are not supported."
        )
    # 2) Hybrid layer types (e.g. Gemma4 sliding_attention + full_attention)
    hf_text_config = getattr(draft_model_config, "hf_text_config", None)
    layer_types = (
        getattr(hf_text_config, "layer_types", None) if hf_text_config else None
    )
    if layer_types is not None:
        non_full = [lt for lt in layer_types if lt != "full_attention"]
        if non_full:
            raise NotImplementedError(
                f"Draft model has non-full-attention layer types "
                f"{set(non_full)}, but the draft KV cache only supports "
                "plain full-attention transformers."
            )


def _load_draft_model(
    speculative_config: SpeculativeConfig,
    parallel_config: ParallelConfig,
) -> tuple[Any, DraftDims]:
    dims = resolve_draft_dims(speculative_config, parallel_config)
    draft_model_config = speculative_config.draft_model_config
    assert draft_model_config is not None  # resolve_draft_dims already checked

    # Its own instance (patched to its own draft KV cache, so it must not alias
    # the target). AWQ / variable-head-dim drafts aren't handled here yet
    # (canonical loader: ModelLifecycle._load_generation_model).
    model_path = get_model_download_path(draft_model_config.model)
    with mlx_lm_compatible_model_path(model_path) as compatible_path:
        model, _ = mlx_lm_load(str(compatible_path))

    return model, dims
