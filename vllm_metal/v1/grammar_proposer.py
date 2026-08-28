# SPDX-License-Identifier: Apache-2.0
"""Grammar-forced speculative decoding proposer for the Metal paged path.

A :class:`GrammarProposer` drafts the text a request's grammar has *already*
determined. Inside a JSON schema or a tool-call structural tag the braces,
quotes, key names and separators are fixed before the model runs; only the
values are the model's to choose. This proposer emits the fixed part and stops
at every genuine decision point, costing no model forward at all.

Unlike the other three Metal proposers it loads no model, keeps no KV cache, and
needs no target hidden states. Its per-request state is one xgrammar matcher.

Why the obvious version does not work
-------------------------------------
The tempting rule is "draft a token whenever the grammar leaves exactly one
legal token". Measured against a real tool-call schema on Qwen3's tokenizer,
that rule fires on **zero** of 23 positions. The *string* is forced but the
*tokenization* never is: after ``{`` the grammar demands ``"name"``, and ``"``,
``"n``, ``"na`` and ``"name`` are all legal tokens. Single-legal-token drafting
is a no-op on exactly the workload it was meant for.

What works is xgrammar's ``find_jump_forward_string()``, which returns the whole
forced string rather than one token, combined with a greedy longest-match walk
over the vocabulary:

    S = matcher.find_jump_forward_string()   # e.g. '", "arguments": {"location": "'
    tok = longest vocabulary token that is a prefix of S
    matcher.accept_token(tok)                # also the legality check
    repeat until S runs out or K is reached

This reproduces canonical BPE, which is what the model emits, because that is
how its training data was tokenized. On the schema above it drafts 83 tokens
across a 23-token generation with **zero** mismatches, at 74% coverage and a
#MAT of 4.6 -- in the same range as ToolSpec (arXiv 2604.13519), without their
tree attention.

The boundary rule is what makes it safe. A token that consumes *all* of the
forced string sits on the edge of the free text that follows, and the model may
well prefer a token that merges across that edge (after ``"location": `` the
grammar forces ``"``, but the model emits ``" P`` as one token). So the walk
stops one token short of the end of the forced string -- unless consuming it
terminates the grammar, in which case nothing follows to merge with. Without
this rule the same benchmark drops from 100% to 85.6% precision, and every
single mismatch is the last token of a run.

What is and is not guaranteed
-----------------------------
Verification is lossless, so output is unaffected either way. But note the
honest limit: because several tokenizations of a forced string are legal, a
drafted token is *grammar-legal by construction* (``accept_token`` checks it)
yet only *empirically* the target's argmax. It is a very good guess -- 100% on
the cases measured -- not a proof. Acceptance is therefore a number to watch,
not an invariant to assert.

Coverage is deliberately narrow and tracks the benefit exactly. On a step where
nothing is forced the proposer returns ``None`` for that request and the step
stays an ordinary one-row decode -- which matters far more on Metal than on a
GPU, where a wrong draft is nearly free. Here decode step cost is governed by
query-row count alone (1 row = 23.0 ms, 2 rows = 50.4 ms, 8 rows = 52.4 ms on an
M4 Pro; see ``gemma4-e2b-mtp-report.md``), so a fixed-K drafter that guesses
badly at batch 1 *loses*. Drafting only inside forced spans is what makes the
trade safe here.

Nothing is drafted for plain chat, for free-form string values, or for the
model's own choice of which tool to call -- those still cost one forward each.

The two matchers
----------------
The engine keeps its own matcher (it emits the bitmask and advances it over
committed tokens); this class keeps a second, independent one in the worker,
because that is where ``propose()`` runs. They must agree. Three things keep
them in sync, and one detects it when they do not:

* the worker's backend is built from the same ``vllm_config`` (hence the same
  ``disable_any_whitespace``), the same ``cached_tokenizer_from_config``
  tokenizer, and the same ``model_config.get_vocab_size()`` the engine's
  ``StructuredOutputManager`` uses;
* the matcher is advanced over *output* tokens only, never the prompt;
* matcher state is kept a pure function of the committed output prefix, so
  rejection, preemption, resume and prefix-cache hits are all handled by the
  same resync (see ``_sync_matcher``) rather than by special cases;
* ``_resolve_pending`` compares each proposal against what the scheduler
  actually scheduled next step. This repo's synchronous draft handoff runs
  through ``Scheduler.update_draft_token_ids``, which filters drafts through the
  engine's own ``grammar.validate_tokens()`` and **silently truncates** whatever
  it rejects -- it never sets ``num_invalid_spec_tokens``, so the sentinel guard
  in ``spec_decode.py`` never fires for us. A truncation is therefore the only
  evidence that the two matchers have drifted, and it is logged loudly.

Requests whose reasoning is gated behind a reasoning parser are not supported at
all: while reasoning is unfinished the engine sets ``apply_bitmask=False`` and
does *not* advance its matcher, so a worker matcher would desynchronize
silently. Construction fails loudly instead.

The verify half is unchanged: drafts are handed back via ``take_draft_token_ids``
and verified next step by ``SpeculativeDecodeController.verify_greedy``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger
from vllm.tokenizers import cached_tokenizer_from_config
from vllm.v1.outputs import DraftTokenIds
from vllm.v1.structured_output.backend_xgrammar import XgrammarBackend
from vllm.v1.structured_output.request import get_structured_output_key

if TYPE_CHECKING:
    from collections.abc import Sequence

    from vllm.config import VllmConfig
    from vllm.v1.structured_output.backend_types import (
        StructuredOutputGrammar,
        StructuredOutputKey,
    )

    from vllm_metal.v1.proposer import ProposeContext
    from vllm_metal.v1.spec_decode import (
        PagedDecodeSegment,
        SpeculativeDecodeController,
    )

logger = init_logger(__name__)

# Compiling a grammar in the worker happens on the decode thread, unlike the
# engine's threadpool compile. The backend's GrammarCompiler caches by grammar,
# so only the first request per distinct schema pays it -- but a slow one stalls
# a decode step, which is worth saying out loud rather than hiding in a metric.
_SLOW_COMPILE_S = 0.05

# The replacement character a byte-level BPE token decodes to when it holds a
# partial UTF-8 sequence. Such a token's "text" is meaningless for prefix
# matching, so they are kept out of the lookup table.
_REPLACEMENT_CHAR = "�"


@dataclass
class _MatcherState:
    """One request's worker-side grammar matcher and its sync position."""

    grammar: StructuredOutputGrammar
    key: StructuredOutputKey
    # Output tokens (never prompt tokens) this matcher has accepted. The
    # matcher's state is a pure function of this prefix, which is what makes
    # the resync in _sync_matcher total rather than case-by-case.
    consumed: int = 0
    # Set when the matcher rejected a token the engine committed. That means
    # the two matchers have already diverged, so this request must never draft
    # again -- a rebuilt matcher would just diverge the same way.
    broken: bool = False


@dataclass
class GrammarProposerStats:
    """Counters the benchmark harness reads. Cheap to maintain, never reset."""

    propose_calls: int = 0
    propose_seconds: float = 0.0
    # Time inside the forced-string walk specifically, so the drafting cost can
    # be separated from the per-step bookkeeping around it.
    draft_seconds: float = 0.0
    compiles: int = 0
    compile_seconds: float = 0.0
    vocab_table_seconds: float = 0.0
    # Coverage: how often there was anything to draft at all. steps_eligible
    # counts (request, step) pairs that reached the draft loop.
    steps_eligible: int = 0
    steps_drafted: int = 0
    drafts_offered: int = 0
    drafts_accepted: int = 0
    # A drafted token is grammar-legal by construction but only empirically the
    # target's argmax, so this is a statistic, not an invariant.
    rejected_drafts: int = 0
    # This one IS an invariant: the engine's own grammar refusing a draft our
    # matcher accepted means the two matchers disagree.
    truncated_drafts: int = 0
    # Requests skipped because they were routed to a non-xgrammar backend.
    skipped_backends: dict[str, int] = field(default_factory=dict)


class _VocabPrefixTable:
    """Longest-token-first lookup of vocabulary text against a forced string.

    Built once per process. ``longest_prefix`` answers "what is the longest
    vocabulary token whose text starts this string", which is the greedy BPE
    step -- the tokenization the model was trained to emit.
    """

    def __init__(self, tokenizer: Any, vocab_size: int) -> None:
        text_to_id: dict[str, int] = {}
        max_len = 0
        special = set(getattr(tokenizer, "all_special_ids", ()) or ())
        for token, token_id in tokenizer.get_vocab().items():
            if token_id >= vocab_size or token_id in special:
                continue
            text = tokenizer.convert_tokens_to_string([token])
            # Partial UTF-8 tokens have no usable text, and an empty decode
            # would match every string.
            if not text or _REPLACEMENT_CHAR in text:
                continue
            # Two tokens can decode to the same text; prefer the lower id so
            # the choice is deterministic across processes.
            current = text_to_id.get(text)
            if current is None or token_id < current:
                text_to_id[text] = token_id
            max_len = max(max_len, len(text))
        self._text_to_id = text_to_id
        self._max_len = max_len

    def __len__(self) -> int:
        return len(self._text_to_id)

    @property
    def max_token_len(self) -> int:
        return self._max_len

    def longest_prefix(self, text: str) -> tuple[int, int]:
        """Return ``(token_id, length)`` for the longest token prefixing ``text``.

        Returns ``(-1, 0)`` when no vocabulary token matches.
        """
        for length in range(min(self._max_len, len(text)), 0, -1):
            token_id = self._text_to_id.get(text[:length])
            if token_id is not None:
                return token_id, length
        return -1, 0


class GrammarProposer:
    """:class:`vllm_metal.v1.proposer.MetalProposer` drafting grammar-forced text."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        controller: SpeculativeDecodeController,
    ) -> None:
        self._controller = controller
        self.stats = GrammarProposerStats()

        structured_config = vllm_config.structured_outputs_config
        if structured_config.reasoning_parser:
            # While reasoning is unfinished the engine sets apply_bitmask=False
            # and does not advance its matcher (StructuredOutputManager
            # .should_fill_bitmask). A worker matcher has no way to observe that
            # boundary, so it would advance over reasoning content and drift.
            # Fail loud rather than draft tokens that quietly stop being forced.
            raise NotImplementedError(
                "Grammar-forced speculative decoding does not support a "
                f"reasoning parser (got {structured_config.reasoning_parser!r}). "
                "The engine suspends grammar advancement during reasoning, which "
                "the worker cannot observe. Remove --reasoning-parser or use a "
                "different speculative method."
            )

        spec = vllm_config.speculative_config
        assert spec is not None
        self._max_draft = spec.num_speculative_tokens

        model_config = vllm_config.model_config
        self._vocab_size = model_config.get_vocab_size()
        tokenizer = cached_tokenizer_from_config(model_config)

        started = time.perf_counter()
        # Built exactly the way StructuredOutputManager.grammar_init builds the
        # engine's: same vllm_config (hence the same disable_any_whitespace),
        # same cached tokenizer, same vocab size. Anything hand-rolled here
        # would be a way for the two matchers to disagree.
        self._backend = XgrammarBackend(
            vllm_config,
            tokenizer=tokenizer,
            vocab_size=self._vocab_size,
        )
        # compile_grammar() passes this straight to GrammarMatcher as
        # max_rollback_tokens. The walk rolls back its whole draft, plus one
        # more for the tentative boundary token it may take back, so the
        # engine's own K would be one short. (xgrammar 0.2.3 deprecates the
        # parameter and treats rollback as unlimited, but upstream still passes
        # it, so this keeps us correct if that ever changes back.)
        self._backend.num_speculative_tokens = self._max_draft + 1
        backend_seconds = time.perf_counter() - started

        started = time.perf_counter()
        self._vocab = _VocabPrefixTable(tokenizer, self._vocab_size)
        self.stats.vocab_table_seconds = time.perf_counter() - started

        self._matchers: dict[str, _MatcherState] = {}
        # Each non-empty draft is stashed here (committed position it starts at,
        # tokens proposed) and checked against the *next* step's scheduled
        # drafts and committed history in _resolve_pending.
        self._pending: dict[str, tuple[int, tuple[int, ...]]] = {}
        # One-shot log guards: a drift bug would otherwise flood every step.
        self._logged_truncation = False
        self._logged_broken = False

        logger.info(
            "Grammar-forced speculative decoding enabled "
            "(num_speculative_tokens=%d, vocab_size=%d, prefix table %d entries "
            "/ max %d chars in %.2fs, backend in %.2fs)",
            self._max_draft,
            self._vocab_size,
            len(self._vocab),
            self._vocab.max_token_len,
            self.stats.vocab_table_seconds,
            backend_seconds,
        )

    # -- construction --------------------------------------------------------

    @classmethod
    def build(
        cls,
        *,
        vllm_config: VllmConfig,
        controller: SpeculativeDecodeController,
    ) -> GrammarProposer:
        return cls(vllm_config=vllm_config, controller=controller)

    # -- MetalProposer protocol ---------------------------------------------

    def needs_target_hidden_states(
        self,
        decode_segments: Sequence[PagedDecodeSegment],
        *,
        has_final_prefill: bool,
    ) -> bool:
        # The grammar drafts in token space; it never reads the target's
        # hidden states.
        return False

    def release_requests(self, req_ids: set[str]) -> None:
        # Unlike n-gram's throttle counters these matchers hold pushdown
        # automaton state, so eviction and preemption must really free them
        # rather than leave them pinned while the request waits. A resumed
        # request rebuilds by replaying its committed output in _sync_matcher.
        for req_id in req_ids:
            self._matchers.pop(req_id, None)
            self._pending.pop(req_id, None)

    def propose(self, ctx: ProposeContext) -> DraftTokenIds | None:
        started = time.perf_counter()
        self.stats.propose_calls += 1
        try:
            return self._propose(ctx)
        finally:
            self.stats.propose_seconds += time.perf_counter() - started

    def _propose(self, ctx: ProposeContext) -> DraftTokenIds | None:
        # Bookkeeping runs unconditionally, before the num_speculative_tokens
        # check: a step with drafting disabled still has to prune finished ids
        # and score outstanding proposals, or a request id reused in a disabled
        # step would inherit the previous request's matcher.
        self._prune_finished(ctx.finished_req_ids)
        self._resolve_pending(ctx)

        if ctx.num_speculative_tokens <= 0:
            return None

        drafting = self._controller.draft_eligible_requests(
            ctx.decode_reqs,
            ctx.decode_token_ids,
            ctx.prefill_reqs,
            ctx.prefill_result_modes,
            ctx.request_states,
        )
        if not drafting:
            return None

        max_tokens = min(ctx.num_speculative_tokens, self._max_draft)
        req_ids: list[str] = []
        draft_token_ids: list[list[int]] = []
        for req_id, state in drafting:
            entry = self._sync_matcher(req_id, state)
            if entry is None:
                continue
            self.stats.steps_eligible += 1
            draft = self._draft_forced(entry, max_tokens)
            if not draft:
                continue
            self.stats.steps_drafted += 1
            # Not scored as accepted yet -- the target has not verified it.
            # _resolve_pending scores it next step against what actually landed.
            self._pending[req_id] = (len(state.token_ids), tuple(draft))
            req_ids.append(req_id)
            draft_token_ids.append(draft)

        if not req_ids:
            return None

        return DraftTokenIds(req_ids=req_ids, draft_token_ids=draft_token_ids)

    # -- drafting ------------------------------------------------------------

    def _draft_forced(self, entry: _MatcherState, max_tokens: int) -> list[int]:
        """Walk the grammar's forced string with greedy longest-token matching.

        The matcher is left exactly where it started: this is a lookahead over
        the committed head, not an advance of it.
        """
        grammar = entry.grammar
        if grammar.is_terminated():
            return []

        # The walk uses the raw xgrammar matcher rather than the XgrammarGrammar
        # wrapper: the wrapper has no find_jump_forward_string, and its
        # accept_token/rollback would churn num_processed_tokens bookkeeping for
        # tokens that are about to be rolled back anyway.
        matcher = grammar.matcher
        draft: list[int] = []
        started = time.perf_counter()
        while len(draft) < max_tokens:
            forced = matcher.find_jump_forward_string()
            if not forced:
                # Nothing is determined here -- a free-text span or a genuine
                # decision point. This is the early-out that keeps the proposer
                # off unstructured content entirely.
                break
            token_id, length = self._vocab.longest_prefix(forced)
            if token_id < 0:
                break
            if not matcher.accept_token(token_id):
                # The grammar refused a token spelling its own forced string.
                # Possible for a token the tokenizer can produce but the grammar
                # excludes; stop rather than propose something it will reject.
                break
            if length >= len(forced) and not matcher.is_terminated():
                # This token ends the forced region, so the model may instead
                # emit one that merges into the free text that follows. Only
                # safe when nothing follows, i.e. the grammar has terminated.
                matcher.rollback(1)
                break
            draft.append(token_id)
            if matcher.is_terminated():
                break
        if draft:
            matcher.rollback(len(draft))
        self.stats.draft_seconds += time.perf_counter() - started
        return draft

    # -- per-request matcher lifecycle ---------------------------------------

    def _sync_matcher(self, req_id: str, state: Any) -> _MatcherState | None:
        """Return this request's matcher, advanced to its committed output head.

        Returns ``None`` when the request has no usable grammar, so the caller
        drafts nothing for it and the step stays narrow.
        """
        params = getattr(state.sampling_params, "structured_outputs", None)
        if params is None or params.all_constraints_none():
            return None
        backend = getattr(params, "_backend", None)
        if backend != "xgrammar":
            # outlines / guidance / lm-format-enforcer requests have no
            # xgrammar matcher to fork from. Counted rather than silently
            # dropped so "it never drafts" is diagnosable.
            name = str(backend)
            self.stats.skipped_backends[name] = (
                self.stats.skipped_backends.get(name, 0) + 1
            )
            return None

        key = get_structured_output_key(params)
        entry = self._matchers.get(req_id)
        if entry is not None and entry.broken:
            return None
        # A finished request id can be handed straight back out to a new
        # request, so a matching id is not enough -- the grammar must match too.
        if entry is not None and entry.key != key:
            entry = None

        output_len = len(state.token_ids) - state.prompt_len
        if output_len < 0:
            return None
        if entry is not None and output_len < entry.consumed:
            # The committed output shrank (preemption, or a rolled-back draft
            # window). xgrammar can only roll back a bounded number of tokens,
            # so rebuild and replay instead: matcher state is a pure function
            # of the output prefix, which makes replay always correct.
            entry = None

        if entry is None:
            entry = self._new_matcher(key)
            if entry is None:
                return None
            self._matchers[req_id] = entry

        if output_len > entry.consumed:
            delta = state.token_ids[state.prompt_len + entry.consumed :]
            # accept_tokens takes output tokens only. Feeding it the prompt
            # would desynchronize it from the engine's matcher immediately.
            if not entry.grammar.accept_tokens(req_id, list(delta)):
                entry.broken = True
                if not self._logged_broken:
                    self._logged_broken = True
                    logger.error(
                        "Worker grammar matcher rejected committed output for "
                        "request %s at offset %d (tokens %s). The worker and "
                        "engine matchers have diverged; this request will not "
                        "draft again. Output is unaffected.",
                        req_id,
                        entry.consumed,
                        list(delta[:16]),
                    )
                return None
            entry.consumed = output_len

        return entry

    def _new_matcher(self, key: StructuredOutputKey) -> _MatcherState | None:
        request_type, grammar_spec = key
        started = time.perf_counter()
        try:
            grammar = self._backend.compile_grammar(request_type, grammar_spec)
        except Exception:
            # The engine compiled this same grammar successfully or the request
            # would not be running, so a failure here is worth shouting about --
            # but it must not fail the request, which is decoding correctly
            # without us.
            logger.exception(
                "Worker-side grammar compile failed for %s; this request will "
                "decode without drafting.",
                request_type,
            )
            return None
        elapsed = time.perf_counter() - started
        self.stats.compiles += 1
        self.stats.compile_seconds += elapsed
        if elapsed > _SLOW_COMPILE_S:
            logger.info(
                "Grammar compiled in the drafter in %.3fs (on the decode "
                "thread; subsequent requests reuse the compiler cache).",
                elapsed,
            )
        return _MatcherState(grammar=grammar, key=key)

    # -- scoring and invariant checking --------------------------------------

    def _resolve_pending(self, ctx: ProposeContext) -> None:
        """Score the previous step's proposals against what actually happened.

        Two different things are measured here, and only one of them is an
        invariant:

        * the scheduler *truncating* a draft is a hard invariant violation.
          ``Scheduler.update_draft_token_ids`` filters our proposal through the
          engine's ``validate_tokens()`` and drops the tail it rejects without
          recording an invalid-token count, so comparing against
          ``decode_segments[i].draft_token_ids`` -- what was actually scheduled
          and verified -- is the only way to see the two matchers disagree.
        * the target *rejecting* a scheduled draft token is an ordinary miss.
          Several tokenizations of a forced string are legal, so a drafted token
          is grammar-legal by construction but only empirically the argmax.

        Only requests whose draft was really verified this step are scored:
        being in ``decode_reqs`` is not enough, because a request can sit there
        with an empty ``draft_token_ids`` when the scheduler padded or dropped
        the proposal on batch admission.
        """
        if not self._pending:
            return
        scheduled_by_req = {
            segment.req_id: segment.draft_token_ids
            for segment in ctx.decode_segments
            if segment.draft_token_ids
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
                self._report_truncation(req_id, proposed, scheduled, altered=True)
            elif len(scheduled) < len(proposed):
                self._report_truncation(req_id, proposed, scheduled, altered=False)

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
            if accepted < len(committed):
                self.stats.rejected_drafts += 1

    def _report_truncation(
        self,
        req_id: str,
        proposed: tuple[int, ...],
        scheduled: tuple[int, ...],
        *,
        altered: bool,
    ) -> None:
        self.stats.truncated_drafts += 1
        if self._logged_truncation:
            return
        self._logged_truncation = True
        logger.error(
            "Grammar-forced draft was %s by the engine's own grammar for "
            "request %s: proposed %s, scheduled %s. The worker and engine "
            "matchers disagree -- output is unaffected (the engine's grammar "
            "wins) but the speedup is not. Further occurrences are counted in "
            "stats.truncated_drafts and not logged.",
            "altered" if altered else "truncated",
            req_id,
            list(proposed),
            list(scheduled),
        )

    def _prune_finished(self, finished_req_ids: set[str]) -> None:
        if not finished_req_ids:
            return
        for req_id in finished_req_ids:
            self._matchers.pop(req_id, None)
            self._pending.pop(req_id, None)
