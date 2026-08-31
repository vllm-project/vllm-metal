# SPDX-License-Identifier: Apache-2.0
"""Tests for the Metal grammar-forced speculative-decode proposer.

These need no model and no engine, but they do use a *real* tokenizer and a
*real* compiled xgrammar grammar: the whole point of the proposer is which
tokenization of a forced string it picks, and a mocked matcher would test
nothing. A ``SimpleNamespace`` stands in for the parts of ``VllmConfig`` the
xgrammar backend actually reads (``disable_any_whitespace`` and
``num_speculative_tokens``), and a hand-built ``ProposeContext`` drives
``propose``.
"""

from __future__ import annotations

import functools
import json
import string
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from vllm.sampling_params import SamplingParams, StructuredOutputsParams

from vllm_metal.v1 import grammar_proposer as grammar_mod
from vllm_metal.v1.grammar_proposer import (
    GrammarProposer,
    _is_byte_fallback,
    _VocabPrefixTable,
)
from vllm_metal.v1.proposer import ProposeContext
from vllm_metal.v1.spec_decode import PagedDecodeSegment, SpeculativeDecodeController

_TOKENIZER_ID = "Qwen/Qwen3-0.6B"
# A SentencePiece vocabulary, which carries byte-fallback tokens (`<0x29>`)
# alongside the real ones. Qwen3 is byte-level BPE and has none, so the
# token-id-selection tests below are vacuous without a second tokenizer.
_BYTE_FALLBACK_TOKENIZER_ID = "google/gemma-4-E2B-it"


@functools.cache
def _load_tokenizer(name: str = _TOKENIZER_ID):
    """The locally cached tokenizer, or ``None`` when it is not available."""
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(name)
    except Exception:  # pragma: no cover - environment dependent
        return None


def _require_tokenizer(name: str):
    tokenizer = _load_tokenizer(name)
    if tokenizer is None:  # pragma: no cover - environment dependent
        pytest.skip(f"{name} tokenizer is not available locally")
    return tokenizer


_TOKENIZER = _load_tokenizer()
_needs_tokenizer = pytest.mark.skipif(
    _TOKENIZER is None,
    reason=f"{_TOKENIZER_ID} tokenizer is not available locally",
)

# Every character a JSON skeleton is built from. These are exactly the
# positions a forced string walks through, so a vocabulary table that picks the
# wrong id for any of them drafts tokens the model will never emit.
_JSON_CHARS = string.ascii_letters + string.digits + '{}[]",:_-. '

# A tool-call-shaped schema: a fixed skeleton, one enum the model picks from,
# and one free string it writes itself. Exercises forced runs, decision points
# and free spans in a single grammar.
_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "enum": ["get_weather"]},
        "arguments": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location", "unit"],
        },
    },
    "required": ["name", "arguments"],
}
_SCHEMA_JSON = json.dumps(_SCHEMA)
_TARGET_TEXT = (
    '{"name": "get_weather", "arguments": {"location": "Paris", "unit": "celsius"}}'
)


def _proposer(
    *,
    num_speculative_tokens: int = 8,
    reasoning_parser: str = "",
    disable_any_whitespace: bool = True,
) -> GrammarProposer:
    assert _TOKENIZER is not None
    vocab_size = len(_TOKENIZER.get_vocab())
    vllm_config = SimpleNamespace(
        structured_outputs_config=SimpleNamespace(
            disable_any_whitespace=disable_any_whitespace,
            reasoning_parser=reasoning_parser,
        ),
        speculative_config=SimpleNamespace(
            num_speculative_tokens=num_speculative_tokens
        ),
        model_config=SimpleNamespace(get_vocab_size=lambda: vocab_size),
    )
    with patch.object(
        grammar_mod, "cached_tokenizer_from_config", return_value=_TOKENIZER
    ):
        return GrammarProposer.build(
            vllm_config=vllm_config,
            controller=SpeculativeDecodeController(),
        )


def _structured_params(
    *, backend: str | None = "xgrammar", schema: str | None = _SCHEMA_JSON
) -> SamplingParams:
    structured = None
    if schema is not None:
        structured = StructuredOutputsParams(json=schema)
        structured._backend = backend
    return SamplingParams(temperature=0.0, structured_outputs=structured)


def _request_state(
    output_token_ids: list[int],
    *,
    prompt_len: int = 3,
    params: SamplingParams | None = None,
) -> SimpleNamespace:
    # The prompt is never fed to the matcher; only what follows prompt_len is.
    return SimpleNamespace(
        token_ids=list(range(prompt_len)) + list(output_token_ids),
        prompt_len=prompt_len,
        sampling_params=params if params is not None else _structured_params(),
        generated_tokens=len(output_token_ids),
    )


def _segment(req_id: str, draft_token_ids: tuple[int, ...]) -> PagedDecodeSegment:
    num_query_tokens = len(draft_token_ids) + 1
    return PagedDecodeSegment(
        req_id=req_id,
        input_token_ids=tuple(range(num_query_tokens)),
        start_row=0,
        num_query_tokens=num_query_tokens,
        draft_token_ids=draft_token_ids,
        cache_start_pos=0,
        block_ids=(0,),
    )


def _context(
    *,
    decode_reqs: list[tuple[str, SimpleNamespace]] | None = None,
    decode_token_ids: list[list[int]] | None = None,
    decode_segments: list[PagedDecodeSegment] | None = None,
    prefill_reqs: list[SimpleNamespace] | None = None,
    prefill_result_modes: list[str] | None = None,
    request_states: dict[str, SimpleNamespace] | None = None,
    num_speculative_tokens: int = 8,
    finished_req_ids: set[str] | None = None,
) -> ProposeContext:
    decode_reqs = decode_reqs or []
    prefill_reqs = prefill_reqs or []
    if decode_token_ids is None:
        decode_token_ids = [[state.token_ids[-1]] for _, state in decode_reqs]
    if prefill_result_modes is None:
        prefill_result_modes = ["new_final"] * len(prefill_reqs)
    if request_states is None:
        request_states = dict(decode_reqs)
    return ProposeContext(
        target_hidden_states=None,
        decode_reqs=decode_reqs,
        decode_segments=decode_segments or [],
        decode_token_ids=decode_token_ids,
        prefill_reqs=prefill_reqs,
        prefill_token_ids=[0] * len(prefill_reqs),
        prefill_result_modes=prefill_result_modes,
        request_states=request_states,
        cu_seqlens=[],
        num_decode_segments=len(decode_reqs),
        num_speculative_tokens=num_speculative_tokens,
        finished_req_ids=finished_req_ids or set(),
    )


def _target_ids() -> list[int]:
    assert _TOKENIZER is not None
    return _TOKENIZER.encode(_TARGET_TEXT, add_special_tokens=False)


@_needs_tokenizer
class TestVocabPrefixTable:
    def test_longest_prefix_is_greedy(self) -> None:
        assert _TOKENIZER is not None
        table = _VocabPrefixTable(_TOKENIZER, len(_TOKENIZER.get_vocab()))
        token_id, length = table.longest_prefix('{"name": "')
        assert length > 0
        text = _TOKENIZER.convert_ids_to_tokens(token_id)
        decoded = _TOKENIZER.convert_tokens_to_string([text])
        assert '{"name": "'.startswith(decoded)
        assert len(decoded) == length
        # Greedy means it must not settle for a shorter token when a longer one
        # also matches -- `{"` exists in this vocabulary, so `{` is not enough.
        assert length > 1

    def test_no_match_returns_sentinel(self) -> None:
        assert _TOKENIZER is not None
        table = _VocabPrefixTable(_TOKENIZER, len(_TOKENIZER.get_vocab()))
        assert table.longest_prefix("") == (-1, 0)


# Not nested under the Qwen-only skip: the point of the byte-fallback case is
# that Qwen cannot exercise it.
@pytest.mark.parametrize("tokenizer_id", [_TOKENIZER_ID, _BYTE_FALLBACK_TOKENIZER_ID])
class TestVocabPrefixTableTokenSelection:
    """The table must pick the id the *tokenizer itself* uses for a text.

    Several vocabulary entries can decode to the same text. Picking the wrong
    one still drafts a grammar-legal token -- it spells the right characters, so
    ``accept_token`` accepts it -- but the model emits the other id, so the
    target rejects it and the speedup quietly halves.

    SentencePiece vocabularies make this concrete: Gemma-4 has ``<0x29>`` (a
    byte-fallback token) and ``)`` both decoding to ``)``, with the fallback at
    the lower id. A "lowest id wins" tiebreak picks the fallback for every ASCII
    character. Qwen3 is byte-level BPE with no fallback tokens at all, which is
    why a Qwen-only suite cannot catch this.
    """

    def test_single_characters_resolve_to_the_tokenizers_own_id(
        self, tokenizer_id: str
    ) -> None:
        tokenizer = _require_tokenizer(tokenizer_id)
        table = _VocabPrefixTable(tokenizer, len(tokenizer.get_vocab()))

        mismatched = []
        for char in _JSON_CHARS:
            expected = tokenizer.encode(char, add_special_tokens=False)
            if len(expected) != 1:
                # The tokenizer does not spell this character with one token,
                # so there is no single right answer to compare against.
                continue
            token_id, length = table.longest_prefix(char)
            if token_id != expected[0] or length != 1:
                mismatched.append((char, token_id, expected[0]))

        assert not mismatched, (
            f"{len(mismatched)} of {len(_JSON_CHARS)} JSON characters resolve to "
            f"a token {tokenizer_id} would never emit: {mismatched[:8]}"
        )

    def test_byte_fallback_tokens_do_not_displace_real_ones(
        self, tokenizer_id: str
    ) -> None:
        tokenizer = _require_tokenizer(tokenizer_id)
        vocab = tokenizer.get_vocab()
        fallbacks = {
            token: token_id
            for token, token_id in vocab.items()
            if _is_byte_fallback(token)
        }
        if not fallbacks:
            pytest.skip(f"{tokenizer_id} has no byte-fallback tokens")

        table = _VocabPrefixTable(tokenizer, len(vocab))
        # One pass over the vocabulary: which real tokens spell each text.
        real_ids_by_text: dict[str, list[int]] = {}
        for token, token_id in vocab.items():
            if _is_byte_fallback(token):
                continue
            text = tokenizer.convert_tokens_to_string([token])
            if text:
                real_ids_by_text.setdefault(text, []).append(token_id)

        # A byte-fallback token must lose to a real token spelling the same
        # text -- but the table must not shrink, because a fallback is the only
        # way to spell a character that has no real token at all.
        displaced = []
        fallback_only = 0
        for token, token_id in fallbacks.items():
            text = tokenizer.convert_tokens_to_string([token])
            if not text:
                continue
            alternatives = real_ids_by_text.get(text)
            if not alternatives:
                fallback_only += 1
                assert table.longest_prefix(text)[0] == token_id, (
                    f"{text!r} is spelled only by byte-fallback token {token_id} "
                    "but the table dropped it, losing coverage"
                )
                continue
            if table.longest_prefix(text)[0] == token_id:
                displaced.append((text, token_id, alternatives[:2]))

        assert not displaced, (
            f"byte-fallback tokens won the tiebreak for {len(displaced)} texts "
            f"that have a real token: {displaced[:8]}"
        )


@_needs_tokenizer
class TestGrammarProposerProtocol:
    def test_never_needs_target_hidden_states(self) -> None:
        proposer = _proposer()
        assert proposer.needs_target_hidden_states([], has_final_prefill=False) is False
        assert proposer.needs_target_hidden_states([], has_final_prefill=True) is False

    def test_reasoning_parser_is_refused_loudly(self) -> None:
        with pytest.raises(NotImplementedError, match="reasoning parser"):
            _proposer(reasoning_parser="deepseek_r1")


@_needs_tokenizer
class TestGrammarProposerDrafting:
    def test_drafts_the_canonical_tokenization_of_the_forced_prefix(self) -> None:
        # From an empty output the whole tool-call skeleton up to the free
        # `location` value is determined, so the draft must be exactly the
        # canonical tokenization of that span.
        proposer = _proposer()
        state = _request_state([])
        drafts = proposer.propose(_context(decode_reqs=[("r0", state)]))

        assert drafts is not None
        assert drafts.req_ids == ["r0"]
        assert (
            drafts.draft_token_ids[0] == _target_ids()[: len(drafts.draft_token_ids[0])]
        )

    def test_respects_the_scheduler_draft_width(self) -> None:
        proposer = _proposer()
        state = _request_state([])
        drafts = proposer.propose(
            _context(decode_reqs=[("r0", state)], num_speculative_tokens=3)
        )

        assert drafts is not None
        assert len(drafts.draft_token_ids[0]) == 3

    def test_zero_draft_width_returns_none(self) -> None:
        proposer = _proposer()
        state = _request_state([])
        assert (
            proposer.propose(
                _context(decode_reqs=[("r0", state)], num_speculative_tokens=0)
            )
            is None
        )

    def test_stops_before_a_free_text_span(self) -> None:
        # Position the request just before the free `location` value. The
        # grammar forces nothing there, so nothing may be drafted.
        assert _TOKENIZER is not None
        ids = _target_ids()
        prefix_text = '{"name": "get_weather", "arguments": {"location": "'
        n = len(_TOKENIZER.encode(prefix_text, add_special_tokens=False))
        proposer = _proposer()
        state = _request_state(ids[:n])

        assert proposer.propose(_context(decode_reqs=[("r0", state)])) is None

    def test_never_drafts_across_the_forced_boundary(self) -> None:
        # Whatever it drafts, the last token must leave the forced region
        # unfinished: a token that consumes the whole forced string may merge
        # with the free text that follows, which is the one case the model's
        # own tokenization can disagree.
        proposer = _proposer()
        state = _request_state([])
        drafts = proposer.propose(_context(decode_reqs=[("r0", state)]))
        assert drafts is not None

        entry = proposer._matchers["r0"]
        matcher = entry.grammar.matcher
        for token_id in drafts.draft_token_ids[0]:
            assert matcher.find_jump_forward_string() != ""
            assert matcher.accept_token(token_id)
        # Still inside a forced span after the whole draft.
        assert matcher.find_jump_forward_string() != ""
        matcher.rollback(len(drafts.draft_token_ids[0]))

    def test_propose_leaves_the_matcher_where_it_found_it(self) -> None:
        # The lookahead must roll back: two identical steps must draft
        # identically, which they cannot if the matcher advanced.
        proposer = _proposer()
        state = _request_state([])
        first = proposer.propose(_context(decode_reqs=[("r0", state)]))
        second = proposer.propose(_context(decode_reqs=[("r0", state)]))

        assert first is not None and second is not None
        assert first.draft_token_ids == second.draft_token_ids

    def test_offline_replay_of_the_golden_accepts_every_draft(self) -> None:
        """Score the drafter against a real generation, with no model involved.

        ``_TARGET_TEXT`` is the pinned greedy output of Qwen3-0.6B under
        ``_SCHEMA`` -- its token ids are exactly the ``GREEDY_GOLDEN`` the e2e
        gate asserts on, minus the trailing EOS. Replaying it position by
        position and scoring each draft against what the model actually emitted
        reproduces the benchmark's acceptance number offline, in a second.

        This is the cheap way to make a change to the walk decidable rather than
        speculative: a strategy that lowers acceptance fails here long before
        anyone re-runs the benchmark.
        """
        proposer = _proposer()
        golden = _target_ids()

        offered = accepted = 0
        for position in range(len(golden)):
            state = _request_state(golden[:position])
            drafts = proposer.propose(_context(decode_reqs=[("r0", state)]))
            if drafts is None:
                continue
            draft = drafts.draft_token_ids[0]
            offered += len(draft)
            for drafted_id, actual_id in zip(draft, golden[position:], strict=False):
                if drafted_id != actual_id:
                    break
                accepted += 1

        assert offered > 0, "drafted nothing across the whole generation"
        assert accepted == offered, (
            f"{offered - accepted} of {offered} drafted tokens did not match "
            "what the model actually emitted"
        )

    def test_advances_over_newly_committed_output(self) -> None:
        proposer = _proposer()
        ids = _target_ids()
        state = _request_state([])
        first = proposer.propose(_context(decode_reqs=[("r0", state)]))
        assert first is not None

        # Commit the first three tokens and draft again from there.
        state.token_ids.extend(ids[:3])
        second = proposer.propose(_context(decode_reqs=[("r0", state)]))

        assert second is not None
        assert proposer._matchers["r0"].consumed == 3
        assert second.draft_token_ids[0] == ids[3 : 3 + len(second.draft_token_ids[0])]


@_needs_tokenizer
class TestGrammarProposerFiltering:
    def test_request_without_structured_outputs_drafts_nothing(self) -> None:
        proposer = _proposer()
        state = _request_state([], params=_structured_params(schema=None))

        assert proposer.propose(_context(decode_reqs=[("r0", state)])) is None

    def test_non_xgrammar_backend_is_skipped_and_counted(self) -> None:
        proposer = _proposer()
        state = _request_state([], params=_structured_params(backend="outlines"))

        assert proposer.propose(_context(decode_reqs=[("r0", state)])) is None
        assert proposer.stats.skipped_backends == {"outlines": 1}

    def test_non_greedy_request_is_filtered_before_any_matcher_work(self) -> None:
        proposer = _proposer()
        params = _structured_params()
        params.temperature = 0.8
        state = _request_state([], params=params)

        assert proposer.propose(_context(decode_reqs=[("r0", state)])) is None
        # draft_eligible_requests rejected it, so no matcher was ever built.
        assert proposer._matchers == {}

    def test_skips_intermediate_prefill(self) -> None:
        proposer = _proposer()
        state = _request_state([])
        ctx = _context(
            prefill_reqs=[SimpleNamespace(req_id="p0")],
            prefill_result_modes=["intermediate"],
            request_states={"p0": state},
        )

        assert proposer.propose(ctx) is None

    def test_finalized_prefill_participates(self) -> None:
        proposer = _proposer()
        state = _request_state([])
        ctx = _context(
            prefill_reqs=[SimpleNamespace(req_id="p0")],
            prefill_result_modes=["new_final"],
            request_states={"p0": state},
        )

        drafts = proposer.propose(ctx)

        assert drafts is not None
        assert drafts.req_ids == ["p0"]


@_needs_tokenizer
class TestGrammarProposerLifecycle:
    def test_release_requests_frees_the_matcher(self) -> None:
        proposer = _proposer()
        state = _request_state([])
        proposer.propose(_context(decode_reqs=[("r0", state)]))
        assert "r0" in proposer._matchers

        proposer.release_requests({"r0"})

        assert proposer._matchers == {}
        assert proposer._pending == {}

    def test_finished_id_reused_in_the_same_step_gets_a_fresh_matcher(self) -> None:
        proposer = _proposer()
        ids = _target_ids()
        first = _request_state(ids[:5])
        proposer.propose(_context(decode_reqs=[("r0", first)]))
        assert proposer._matchers["r0"].consumed == 5

        # vLLM can hand a finished id straight back out to a new request in the
        # same step; the new request must not inherit the old matcher.
        fresh = _request_state([])
        proposer.propose(_context(decode_reqs=[("r0", fresh)], finished_req_ids={"r0"}))

        assert proposer._matchers["r0"].consumed == 0

    def test_shrinking_output_rebuilds_by_replay(self) -> None:
        proposer = _proposer()
        ids = _target_ids()
        state = _request_state(ids[:6])
        proposer.propose(_context(decode_reqs=[("r0", state)]))
        assert proposer._matchers["r0"].consumed == 6

        # Preemption or a rolled-back window: the committed output shrank.
        del state.token_ids[-4:]
        drafts = proposer.propose(_context(decode_reqs=[("r0", state)]))

        assert proposer._matchers["r0"].consumed == 2
        assert drafts is not None
        assert drafts.draft_token_ids[0] == ids[2 : 2 + len(drafts.draft_token_ids[0])]

    def test_matcher_rejecting_committed_output_disables_the_request(self) -> None:
        proposer = _proposer()
        # Token ids that cannot spell a valid prefix of the schema.
        state = _request_state([999, 998, 997])

        assert proposer.propose(_context(decode_reqs=[("r0", state)])) is None
        assert proposer._matchers["r0"].broken is True
        # A broken matcher must stay broken rather than be rebuilt and re-diverge.
        assert proposer.propose(_context(decode_reqs=[("r0", state)])) is None


@_needs_tokenizer
class TestGrammarProposerScoring:
    def test_full_acceptance_is_scored(self) -> None:
        proposer = _proposer()
        state = _request_state([])
        drafts = proposer.propose(_context(decode_reqs=[("r0", state)]))
        assert drafts is not None
        draft = drafts.draft_token_ids[0]

        # Next step: the target accepted the whole draft.
        state.token_ids.extend(draft)
        proposer.propose(
            _context(
                decode_reqs=[("r0", state)],
                decode_segments=[_segment("r0", tuple(draft))],
            )
        )

        assert proposer.stats.drafts_offered == len(draft)
        assert proposer.stats.drafts_accepted == len(draft)
        assert proposer.stats.truncated_drafts == 0
        assert proposer.stats.rejected_drafts == 0

    def test_engine_truncating_a_draft_is_counted_but_not_an_invariant(self) -> None:
        # A shorter draft with its prefix intact is ambiguous: the engine's
        # grammar may have rejected the tail, or the scheduler may have clipped
        # it against the token budget. Counted, never asserted on.
        proposer = _proposer()
        state = _request_state([])
        drafts = proposer.propose(_context(decode_reqs=[("r0", state)]))
        assert drafts is not None
        draft = drafts.draft_token_ids[0]
        assert len(draft) > 2

        kept = tuple(draft[:2])
        state.token_ids.extend(kept)
        proposer.propose(
            _context(
                decode_reqs=[("r0", state)],
                decode_segments=[_segment("r0", kept)],
            )
        )

        assert proposer.stats.truncated_drafts == 1
        assert proposer.stats.altered_drafts == 0

    def test_engine_altering_a_draft_violates_the_invariant(self) -> None:
        # Nothing benign rewrites a draft's prefix, so this is the unambiguous
        # signal that the worker's matcher and the engine's have diverged.
        proposer = _proposer()
        state = _request_state([])
        drafts = proposer.propose(_context(decode_reqs=[("r0", state)]))
        assert drafts is not None
        draft = drafts.draft_token_ids[0]
        assert len(draft) > 2

        altered = (draft[0], draft[1] + 1, *draft[2:])
        state.token_ids.extend(altered)
        proposer.propose(
            _context(
                decode_reqs=[("r0", state)],
                decode_segments=[_segment("r0", tuple(altered))],
            )
        )

        assert proposer.stats.altered_drafts == 1
        assert proposer.stats.truncated_drafts == 0

    def test_target_rejection_is_counted_but_not_a_truncation(self) -> None:
        proposer = _proposer()
        state = _request_state([])
        drafts = proposer.propose(_context(decode_reqs=[("r0", state)]))
        assert drafts is not None
        draft = drafts.draft_token_ids[0]

        # Scheduled in full, but the target diverged on the second token.
        committed = [draft[0], draft[1] + 1]
        state.token_ids.extend(committed)
        proposer.propose(
            _context(
                decode_reqs=[("r0", state)],
                decode_segments=[_segment("r0", tuple(draft))],
            )
        )

        assert proposer.stats.truncated_drafts == 0
        assert proposer.stats.rejected_drafts == 1
        assert proposer.stats.drafts_accepted == 1

    def test_unverified_draft_stays_pending(self) -> None:
        proposer = _proposer()
        state = _request_state([])
        proposer.propose(_context(decode_reqs=[("r0", state)]))

        # A step where the request decoded without its draft being scheduled:
        # this step's token is not the verification outcome of that draft.
        proposer.propose(
            _context(
                decode_reqs=[("r0", state)],
                decode_segments=[_segment("r0", ())],
            )
        )

        assert proposer.stats.drafts_offered == 0
        assert "r0" in proposer._pending
