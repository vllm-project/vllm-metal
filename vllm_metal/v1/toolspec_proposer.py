# SPDX-License-Identifier: Apache-2.0
"""ToolSpec: grammar-forced drafting plus retrieval over past invocations.

This is the Metal port of ToolSpec (arXiv 2604.13519, Xia et al.), which pairs
schema-aware drafting with retrieval-augmented drafting. We already have the
first half and ours is more general: :class:`~vllm_metal.v1.grammar_proposer
.GrammarProposer` walks real xgrammar ``find_jump_forward_string()`` over any
JSON schema or structural tag, where the paper's ``schema_fsm.py`` is a
three-state machine over schemas scraped out of the system prompt. So this class
adds only the second half, and composes the first.

What retrieval is for
---------------------
The grammar drafts structure and stops at every value. Measured by
``tools/grammar_determinism.py`` that is 37% of emitted tokens over API-Bank and
50% over BFCL, and it is a *ceiling*: no grammar-aware drafter can do better,
because the rest is genuinely the model's choice. The rest is argument *content*
-- and ToolSpec's
observation is that content repeats across requests, so a value some earlier
invocation already produced is a decent guess for the one being generated now.
:class:`~vllm_metal.v1.retrieval_store.RetrievalStore` holds that memory.

Composition: grammar first, retrieval fills the gaps
----------------------------------------------------
The paper's fallback chain is retrieval -> schema FSM -> logits top-k. This
inverts the first two: the grammar proposes first, and retrieval is consulted
only for requests it left undrafted.

That is deliberate. A grammar draft is *legal by construction* -- the matcher
checked it -- while a retrieval draft is a guess that the target may reject. When
both have something to say, the certain one should win. And the requests the
grammar leaves undrafted are exactly the free-text spans retrieval exists to
cover, so the two partition the work rather than compete for it. The cost is
that a long retrieval match cannot pre-empt a short grammar one; that trade buys
a composition where :class:`GrammarProposer` needs no modification and its
accounting stays exactly as reviewed.

The third rung of the paper's chain (logits top-k) is dropped: it drafts on pure
model confidence with nothing forcing or matching it, which is the fixed-K
behaviour that *loses* at batch 1 on Metal, where step cost is governed by query
row count (1 row = 23.0 ms, 2 rows = 50.4 ms on an M4 Pro).

Linear drafts, not trees
------------------------
ToolSpec drafts a *tree* of candidates and verifies it with tree attention,
scoring paths by ``cumprod(...).sum()``. Metal's verify half is linear
(``SpeculativeDecodeController.verify_greedy``) and shared with the MTP and
draft-model proposers, so this port takes the single best candidate instead.
Part of the paper's headline speedup comes from tree breadth, so this should not
be expected to reproduce it -- an honest gap, not an oversight.

Question vectors are prefill-only
---------------------------------
Retrieval keys on the target's hidden state at the *last prompt token*. That row
exists only on a final-prefill forward, so ``needs_target_hidden_states``
returns True only when the step has one. Decode steps never pay hidden-state
collection, which is what keeps the no-grammar fast path exactly as fast as it
is without this proposer -- the property the sonnet benchmark arm exists to
verify.

Cross-request drafting
----------------------
The memory is shared across requests: one request's output can seed another's
draft. Verification is unchanged and lossless, so this can only ever change what
is *guessed*, never what is *emitted*. It is a drafting cache, and
``RetrievalStore.clear()`` empties it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import mlx.core as mx
import numpy as np
from vllm.logger import init_logger
from vllm.v1.outputs import DraftTokenIds
from vllm.v1.structured_output.request import get_structured_output_key

from vllm_metal import envs
from vllm_metal.v1.grammar_proposer import GrammarProposer
from vllm_metal.v1.retrieval_store import RetrievalStore

if TYPE_CHECKING:
    from collections.abc import Sequence

    from vllm.config import VllmConfig

    from vllm_metal.v1.proposer import ProposeContext
    from vllm_metal.v1.spec_decode import (
        PagedDecodeSegment,
        SpeculativeDecodeController,
    )

logger = init_logger(__name__)

# Retrieval searches whether or not it finds anything: a cosine ranking over the
# memory plus an n-gram scan of the top-k traces, paid every eligible step. On
# traffic with nothing to match -- ordinary prose -- that is pure overhead.
# Measured on the sonnet arm before this throttle existed: 1977 match calls, 0
# hits, 0.15s of search across a 16s run, about 1% slower than the grammar half
# alone, which drafts nothing there and costs exactly nothing.
#
# The same shape of tax the n-gram proposer pays, so the same fix, with its
# constants: back off after a streak of misses, but retry periodically rather
# than giving up, since a response can turn matchable partway through.
# See ``NgramProposer._on_cooldown`` / ``_record_miss``.
_MAX_CONSECUTIVE_MISSES = 8
_COOLDOWN_STEPS = 8


@dataclass
class ToolSpecProposerStats:
    """Counters the benchmark harness reads. Cheap to maintain, never reset."""

    propose_calls: int = 0
    propose_seconds: float = 0.0
    # Time inside retrieval specifically (similarity + n-gram match), so the
    # added cost can be separated from the grammar half's.
    retrieval_seconds: float = 0.0
    # Question vectors captured off final-prefill rows.
    questions_captured: int = 0
    # Finished traces committed to the memory.
    traces_stored: int = 0
    # (request, step) pairs where the grammar drafted nothing and retrieval was
    # therefore consulted.
    steps_eligible: int = 0
    steps_drafted: int = 0
    # Steps where retrieval was skipped because the request was on a miss
    # cooldown. This is the search cost *avoided* on unmatchable traffic.
    steps_throttled: int = 0
    drafts_offered: int = 0
    drafts_accepted: int = 0
    rejected_drafts: int = 0
    truncated_drafts: int = 0
    altered_drafts: int = 0
    hits_by_ngram: dict[int, int] = field(default_factory=dict)


class ToolSpecProposer:
    """:class:`~vllm_metal.v1.proposer.MetalProposer` composing grammar + retrieval."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        controller: SpeculativeDecodeController,
    ) -> None:
        self._controller = controller
        self.stats = ToolSpecProposerStats()

        # The grammar half is used unmodified. Its own constructor enforces the
        # reasoning-parser refusal and builds the worker-side xgrammar backend,
        # so nothing here needs to repeat those checks.
        self._grammar = GrammarProposer(
            vllm_config=vllm_config, controller=controller
        )

        spec = vllm_config.speculative_config
        assert spec is not None
        self._max_draft = spec.num_speculative_tokens

        self.store = RetrievalStore(
            capacity=envs.VLLM_METAL_TOOLSPEC_CAPACITY,
            top_k=envs.VLLM_METAL_TOOLSPEC_TOP_K,
            ngram_max=envs.VLLM_METAL_TOOLSPEC_NGRAM_MAX,
            ngram_min=envs.VLLM_METAL_TOOLSPEC_NGRAM_MIN,
        )

        # req_id -> unit-unnormalised question vector, captured at final prefill.
        self._questions: dict[str, np.ndarray] = {}
        # req_id -> live RequestState, retained so a request that finishes can
        # still be harvested after the runner drops it from request_states.
        self._live_states: dict[str, Any] = {}
        # req_id -> (committed position, tokens proposed), scored next step.
        self._pending: dict[str, tuple[int, tuple[int, ...]]] = {}
        # Miss-streak throttling, so a request with nothing to retrieve stops
        # paying the search. See _MAX_CONSECUTIVE_MISSES.
        self._miss_streak: dict[str, int] = {}
        self._cooldown: dict[str, int] = {}
        self._logged_altered = False

        ngram_min, ngram_max = self.store.ngram_window
        logger.info(
            "ToolSpec (grammar + retrieval) speculative decoding enabled "
            "(num_speculative_tokens=%d, memory capacity=%d, top_k=%d, "
            "ngram window=%d..%d)",
            self._max_draft,
            self.store.capacity,
            envs.VLLM_METAL_TOOLSPEC_TOP_K,
            ngram_min,
            ngram_max,
        )

    # -- construction --------------------------------------------------------

    @classmethod
    def build(
        cls,
        *,
        vllm_config: VllmConfig,
        controller: SpeculativeDecodeController,
    ) -> ToolSpecProposer:
        return cls(vllm_config=vllm_config, controller=controller)

    # -- MetalProposer protocol ---------------------------------------------

    def needs_target_hidden_states(
        self,
        decode_segments: Sequence[PagedDecodeSegment],
        *,
        has_final_prefill: bool,
    ) -> bool:
        # Only the last prompt token's row is wanted, and it exists only on a
        # final-prefill forward. Decode steps stay off this path entirely, so
        # the no-grammar fast path is untouched.
        del decode_segments
        return has_final_prefill

    def release_requests(self, req_ids: set[str]) -> None:
        """Release per-request state, harvesting the trace on the way out.

        This is where traces are actually collected. The runner drives its
        lifecycle reconcile from the same finished-id set the proposer sees, and
        it releases *before* the next ``propose`` runs -- so by the time a
        finished id shows up in ``ProposeContext.finished_req_ids`` its state is
        already gone. Harvesting only there stored 0 traces across 32 real
        requests while capturing all 32 question vectors.

        Eviction and preemption come through here too, and their traces are
        partial. They are stored anyway: a prefix of a real invocation is still
        a valid drafting hint, a resumed request is harvested again later with
        its fuller trace, and verification makes a bad hint cost a wider step
        and nothing else.
        """
        self._grammar.release_requests(req_ids)
        for req_id in req_ids:
            self._store_trace(req_id, self._live_states.get(req_id))
            self._questions.pop(req_id, None)
            self._live_states.pop(req_id, None)
            self._pending.pop(req_id, None)
            self._miss_streak.pop(req_id, None)
            self._cooldown.pop(req_id, None)

    def propose(self, ctx: ProposeContext) -> DraftTokenIds | None:
        started = time.perf_counter()
        self.stats.propose_calls += 1
        try:
            return self._propose(ctx)
        finally:
            self.stats.propose_seconds += time.perf_counter() - started

    def _propose(self, ctx: ProposeContext) -> DraftTokenIds | None:
        # Order matters. Harvesting runs before the snapshot, so a request that
        # has left request_states is still visible as departed rather than
        # being re-recorded as live; and both run before the grammar half,
        # whose _prune_finished drops the state a finishing request is
        # harvested from.
        self._capture_questions(ctx)
        self._harvest(ctx)
        self._snapshot_traces(ctx)
        self._resolve_pending(ctx)

        grammar_result = self._grammar.propose(ctx)

        if ctx.num_speculative_tokens <= 0:
            return grammar_result

        drafted: set[str] = set()
        req_ids: list[str] = []
        draft_token_ids: list[list[int]] = []
        if grammar_result is not None:
            drafted.update(grammar_result.req_ids)
            req_ids.extend(grammar_result.req_ids)
            draft_token_ids.extend(grammar_result.draft_token_ids)

        max_tokens = min(ctx.num_speculative_tokens, self._max_draft)
        started = time.perf_counter()
        try:
            for req_id, state in self._controller.draft_eligible_requests(
                ctx.decode_reqs,
                ctx.decode_token_ids,
                ctx.prefill_reqs,
                ctx.prefill_result_modes,
                ctx.request_states,
            ):
                if req_id in drafted:
                    # The grammar already has this one; its draft is legal by
                    # construction and wins.
                    continue
                if self._on_cooldown(req_id):
                    self.stats.steps_throttled += 1
                    continue
                self.stats.steps_eligible += 1
                draft = self._draft_retrieved(req_id, state, max_tokens)
                if not draft:
                    self._record_miss(req_id)
                    continue
                self.stats.steps_drafted += 1
                self._pending[req_id] = (len(state.token_ids), tuple(draft))
                req_ids.append(req_id)
                draft_token_ids.append(draft)
        finally:
            self.stats.retrieval_seconds += time.perf_counter() - started

        if not req_ids:
            return None
        return DraftTokenIds(req_ids=req_ids, draft_token_ids=draft_token_ids)

    # -- miss-streak throttling ----------------------------------------------

    def _on_cooldown(self, req_id: str) -> bool:
        remaining = self._cooldown.get(req_id, 0)
        if remaining <= 0:
            return False
        if remaining == 1:
            del self._cooldown[req_id]
        else:
            self._cooldown[req_id] = remaining - 1
        return True

    def _record_miss(self, req_id: str) -> None:
        # Capped rather than cleared, so a miss on the first retry drops the
        # request straight back into cooldown instead of buying another full
        # grace period. Only real acceptance clears it (_resolve_pending).
        streak = min(self._miss_streak.get(req_id, 0) + 1, _MAX_CONSECUTIVE_MISSES)
        self._miss_streak[req_id] = streak
        if streak >= _MAX_CONSECUTIVE_MISSES:
            self._cooldown[req_id] = _COOLDOWN_STEPS

    # -- retrieval -----------------------------------------------------------

    def _draft_retrieved(
        self, req_id: str, state: Any, max_tokens: int
    ) -> list[int]:
        """Draft the continuation of a matching span in a similar past trace."""
        if len(self.store) == 0:
            return []
        vector = self._questions.get(req_id)
        if vector is None:
            # No question vector: the request was admitted before this proposer
            # started collecting, or its prefill row was not final.
            return []
        output = state.token_ids[state.prompt_len :]
        if not output:
            return []
        records = self.store.retrieve(
            vector=vector, group_key=self._group_key(state)
        )
        if not records:
            return []
        before = self.store.stats.hits_by_ngram.copy()
        draft = self.store.match(
            context=output, records=records, max_tokens=max_tokens
        )
        if draft:
            for size, count in self.store.stats.hits_by_ngram.items():
                delta = count - before.get(size, 0)
                if delta > 0:
                    self.stats.hits_by_ngram[size] = (
                        self.stats.hits_by_ngram.get(size, 0) + delta
                    )
        return draft

    @staticmethod
    def _group_key(state: Any) -> object | None:
        """The request's grammar identity, used to prefer same-schema traces."""
        params = getattr(state.sampling_params, "structured_outputs", None)
        if params is None or params.all_constraints_none():
            return None
        try:
            return get_structured_output_key(params)
        except Exception:
            # A key that cannot be computed is not worth failing a draft over;
            # retrieval just searches the whole memory instead.
            return None

    # -- memory lifecycle ----------------------------------------------------

    def _capture_questions(self, ctx: ProposeContext) -> None:
        """Take each final-prefill row's last hidden state as a question vector."""
        hidden = ctx.target_hidden_states
        if hidden is None or not ctx.prefill_reqs:
            return
        for i, (prefill, result_mode) in enumerate(
            zip(ctx.prefill_reqs, ctx.prefill_result_modes, strict=True)
        ):
            if result_mode == "intermediate":
                # An intermediate chunk's last row is mid-prompt, not the
                # question's end.
                continue
            if prefill.req_id in self._questions:
                continue
            # Same row arithmetic the MTP proposer uses for its prefill seeds
            # (SpeculativeDecodeController.build_gemma4_mtp_draft_seeds).
            row = ctx.cu_seqlens[ctx.num_decode_segments + i + 1] - 1
            if row < 0 or row >= hidden.shape[0]:
                continue
            # bfloat16 has no numpy dtype, so cast on the MLX side first.
            vector = np.asarray(hidden[row].astype(mx.float32))
            self._questions[prefill.req_id] = vector
            self.stats.questions_captured += 1

    def _snapshot_traces(self, ctx: ProposeContext) -> None:
        """Retain each live request's state so it can be harvested on finish.

        A *reference*, not a copy: a request finishing this step may already be
        gone from ``ctx.request_states`` by the time ``_harvest_finished`` looks
        for it, but copying its token ids every step would put an O(output
        length) memcpy on the per-step path this proposer is supposed to leave
        alone. The reference is dropped as soon as the request finishes or is
        released.
        """
        for req_id, state in ctx.request_states.items():
            self._live_states[req_id] = state

    def _harvest(self, ctx: ProposeContext) -> None:
        """Commit completed requests' traces to the memory, then drop their state.

        Two triggers, because ``finished_req_ids`` alone does not fire often
        enough to be useful. The scheduler reports a finished id on the step
        *after* the request completes, so when the engine then goes idle -- one
        request in flight, which is exactly the sequential benchmark and much
        real serving traffic -- that step never happens and nothing is ever
        stored. Measured: 32 requests, 32 question vectors captured, zero
        traces stored.

        So a request is also harvested once it has *left* ``request_states``,
        which the next request's own prefill step observes. The only trace never
        collected is the last one before the engine goes idle for good, which
        nothing would have drafted from anyway.
        """
        departed = [
            req_id for req_id in self._live_states if req_id not in ctx.request_states
        ]
        for req_id in (*ctx.finished_req_ids, *departed):
            state = ctx.request_states.get(req_id) or self._live_states.get(req_id)
            self._store_trace(req_id, state)
            self._live_states.pop(req_id, None)
            self._questions.pop(req_id, None)
            self._pending.pop(req_id, None)
            self._miss_streak.pop(req_id, None)
            self._cooldown.pop(req_id, None)

    def _store_trace(self, req_id: str, state: Any) -> None:
        """Commit one request's output to the memory, if it is usable."""
        vector = self._questions.get(req_id)
        if state is None or vector is None:
            return
        trace = state.token_ids[state.prompt_len :]
        if self.store.add(
            vector=vector,
            output_ids=trace,
            group_key=self._group_key(state),
            req_id=req_id,
        ):
            self.stats.traces_stored += 1

    # -- scoring -------------------------------------------------------------

    def _resolve_pending(self, ctx: ProposeContext) -> None:
        """Score the previous step's retrieval drafts against what landed.

        Mirrors ``GrammarProposer._resolve_pending``. The one difference in
        meaning: a rejected retrieval draft is entirely ordinary -- it is a
        guess from a *similar* invocation, not a legality proof -- so the
        rejection count here is expected to be non-trivial where the grammar's
        is not.
        """
        if not self._pending:
            return
        # Unlike the grammar half this keeps segments whose draft_token_ids came
        # back *empty*. A grammar draft is legal by construction, so an empty
        # return there is the scheduler padding or dropping on batch admission
        # and is worth leaving pending. A retrieval draft is only a guess, and
        # `Scheduler.update_draft_token_ids` filters it through the engine's
        # grammar -- under a constrained request an entirely illegal
        # continuation is filtered to nothing, which is both common and real
        # information. Skipping those would also leak the pending entry, since
        # nothing would ever resolve it.
        scheduled_by_req = {
            segment.req_id: segment.draft_token_ids
            for segment in ctx.decode_segments
        }
        decode_states = dict(ctx.decode_reqs)
        for req_id in list(self._pending):
            scheduled = scheduled_by_req.get(req_id)
            if scheduled is None:
                continue  # not verified this step; leave pending
            state = decode_states.get(req_id)
            if state is None:
                continue
            position, proposed = self._pending.pop(req_id)
            scheduled = tuple(scheduled)

            if scheduled != proposed[: len(scheduled)]:
                self.stats.altered_drafts += 1
                if not self._logged_altered:
                    self._logged_altered = True
                    logger.error(
                        "ToolSpec retrieval draft was altered before "
                        "verification for request %s: proposed %s, scheduled "
                        "%s. Retrieval drafts are filtered through the "
                        "engine's grammar by Scheduler.update_draft_token_ids, "
                        "so under a constrained request this is expected when "
                        "a retrieved continuation is not grammar-legal. Output "
                        "is unaffected. Further occurrences are counted in "
                        "stats.altered_drafts and not logged.",
                        req_id,
                        list(proposed),
                        list(scheduled),
                    )
            elif len(scheduled) < len(proposed):
                self.stats.truncated_drafts += 1

            if not scheduled:
                continue
            committed = state.token_ids[position : position + len(scheduled)]
            accepted = 0
            # committed can be shorter than scheduled when the request stopped
            # inside the verification window (EOS, max_tokens), which is not a
            # rejection -- strict=False by design.
            for proposed_id, committed_id in zip(scheduled, committed, strict=False):
                if proposed_id != committed_id:
                    break
                accepted += 1
            self.stats.drafts_offered += len(scheduled)
            self.stats.drafts_accepted += accepted
            if accepted:
                # A real hit clears the streak outright: this request has
                # matchable content, so it should not be one miss away from
                # cooldown.
                self._miss_streak.pop(req_id, None)
                self._cooldown.pop(req_id, None)
            if accepted < len(committed):
                self.stats.rejected_drafts += 1
