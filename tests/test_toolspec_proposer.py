# SPDX-License-Identifier: Apache-2.0
"""Tests for retrieval-augmented drafting (the ToolSpec port).

Two halves, tested separately: :class:`RetrievalStore` is pure data structure
and needs no model, while :class:`ToolSpecProposer` composes it with a real
:class:`GrammarProposer` and therefore needs a tokenizer.

The property that matters most here is the *negative* one: a request with no
grammar and an empty memory must produce no draft at all, so unstructured
traffic keeps Metal's one-row decode fast path. That is what the sonnet
benchmark arm measures end to end, and what these tests pin down cheaply.
"""

from __future__ import annotations

import functools
import json
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import numpy as np
import pytest
from vllm.sampling_params import SamplingParams, StructuredOutputsParams

from vllm_metal.v1 import grammar_proposer as grammar_mod
from vllm_metal.v1.proposer import ProposeContext
from vllm_metal.v1.retrieval_store import RetrievalStore, _find_last
from vllm_metal.v1.spec_decode import PagedDecodeSegment, SpeculativeDecodeController
from vllm_metal.v1.toolspec_proposer import ToolSpecProposer

_TOKENIZER_ID = "Qwen/Qwen3-0.6B"


@functools.cache
def _load_tokenizer(name: str = _TOKENIZER_ID):
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(name)
    except Exception:  # pragma: no cover - environment dependent
        return None


_TOKENIZER = _load_tokenizer()
_needs_tokenizer = pytest.mark.skipif(
    _TOKENIZER is None,
    reason=f"{_TOKENIZER_ID} tokenizer is not available locally",
)

_SCHEMA_JSON = json.dumps(
    {
        "type": "object",
        "properties": {
            "name": {"type": "string", "enum": ["get_weather"]},
            "arguments": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        },
        "required": ["name", "arguments"],
    }
)

_HIDDEN = 8


def _vec(*values: float) -> np.ndarray:
    """A question vector padded out to the fixed test hidden width."""
    out = np.zeros(_HIDDEN, dtype=np.float32)
    out[: len(values)] = values
    return out


# ---------------------------------------------------------------- store ----


class TestRetrievalStoreWriting:
    def test_add_and_len(self) -> None:
        store = RetrievalStore(capacity=4, ngram_min=2, ngram_max=4)
        assert store.add(vector=_vec(1, 0), output_ids=[1, 2, 3, 4]) is True
        assert len(store) == 1

    def test_trace_shorter_than_needle_is_dropped(self) -> None:
        # A trace at or below the smallest needle can never be matched, so it
        # must not consume a slot.
        store = RetrievalStore(capacity=4, ngram_min=5, ngram_max=7)
        assert store.add(vector=_vec(1, 0), output_ids=[1, 2, 3, 4, 5]) is False
        assert len(store) == 0

    def test_zero_vector_is_rejected(self) -> None:
        # A zero vector has no direction, and normalising it would put NaNs
        # into every later ranking.
        store = RetrievalStore(capacity=4, ngram_min=2, ngram_max=4)
        assert store.add(vector=_vec(0, 0), output_ids=[1, 2, 3, 4]) is False
        assert len(store) == 0

    def test_non_finite_vector_is_rejected(self) -> None:
        store = RetrievalStore(capacity=4, ngram_min=2, ngram_max=4)
        bad = _vec(1.0)
        bad[1] = np.inf
        assert store.add(vector=bad, output_ids=[1, 2, 3, 4]) is False

    def test_fifo_eviction_at_capacity(self) -> None:
        store = RetrievalStore(capacity=2, ngram_min=2, ngram_max=4)
        store.add(vector=_vec(1, 0), output_ids=[1, 1, 1, 1], req_id="a")
        store.add(vector=_vec(0, 1), output_ids=[2, 2, 2, 2], req_id="b")
        store.add(vector=_vec(1, 1), output_ids=[3, 3, 3, 3], req_id="c")
        assert len(store) == 2
        assert store.stats.evictions == 1
        # The oldest ("a") is the one that went.
        kept = {r.req_id for r in store.retrieve(vector=_vec(1, 1), top_k=2)}
        assert kept == {"b", "c"}

    def test_clear_empties_the_memory(self) -> None:
        store = RetrievalStore(capacity=4, ngram_min=2, ngram_max=4)
        store.add(vector=_vec(1, 0), output_ids=[1, 2, 3, 4])
        store.clear()
        assert len(store) == 0
        assert store.retrieve(vector=_vec(1, 0)) == []

    def test_rejects_bad_construction(self) -> None:
        with pytest.raises(ValueError):
            RetrievalStore(capacity=0)
        with pytest.raises(ValueError):
            RetrievalStore(ngram_min=7, ngram_max=5)


class TestRetrievalStoreRetrieval:
    def test_ranks_by_cosine_similarity(self) -> None:
        store = RetrievalStore(capacity=8, ngram_min=2, ngram_max=4)
        store.add(vector=_vec(1, 0), output_ids=[1, 1, 1, 1], req_id="x")
        store.add(vector=_vec(0, 1), output_ids=[2, 2, 2, 2], req_id="y")
        got = store.retrieve(vector=_vec(0.9, 0.1), top_k=2)
        assert [r.req_id for r in got] == ["x", "y"]

    def test_magnitude_does_not_change_ranking(self) -> None:
        # Both sides are unit-normalised, so only direction should matter.
        store = RetrievalStore(capacity=8, ngram_min=2, ngram_max=4)
        store.add(vector=_vec(10, 0), output_ids=[1, 1, 1, 1], req_id="x")
        store.add(vector=_vec(0, 0.01), output_ids=[2, 2, 2, 2], req_id="y")
        got = store.retrieve(vector=_vec(0.5, 0), top_k=1)
        assert [r.req_id for r in got] == ["x"]

    def test_empty_store_retrieves_nothing(self) -> None:
        store = RetrievalStore(capacity=4, ngram_min=2, ngram_max=4)
        assert store.retrieve(vector=_vec(1, 0)) == []

    def test_group_key_filters(self) -> None:
        store = RetrievalStore(capacity=8, ngram_min=2, ngram_max=4)
        store.add(vector=_vec(1, 0), output_ids=[1, 1, 1, 1], group_key="A", req_id="a")
        store.add(vector=_vec(1, 0), output_ids=[2, 2, 2, 2], group_key="B", req_id="b")
        got = store.retrieve(vector=_vec(1, 0), group_key="B", top_k=4)
        assert [r.req_id for r in got] == ["b"]

    def test_group_key_falls_back_to_whole_memory(self) -> None:
        # An unseen schema should still get candidates rather than nothing.
        store = RetrievalStore(capacity=8, ngram_min=2, ngram_max=4)
        store.add(vector=_vec(1, 0), output_ids=[1, 1, 1, 1], group_key="A", req_id="a")
        got = store.retrieve(vector=_vec(1, 0), group_key="Z", top_k=4)
        assert [r.req_id for r in got] == ["a"]
        assert store.stats.group_fallbacks == 1

    def test_mismatched_hidden_width_is_refused(self) -> None:
        store = RetrievalStore(capacity=4, ngram_min=2, ngram_max=4)
        store.add(vector=_vec(1, 0), output_ids=[1, 2, 3, 4])
        assert store.retrieve(vector=np.ones(3, dtype=np.float32)) == []
        assert store.add(vector=np.ones(3, dtype=np.float32), output_ids=[1, 2, 3, 4]) is False


class TestRetrievalStoreMatching:
    def test_matches_suffix_and_returns_continuation(self) -> None:
        store = RetrievalStore(capacity=4, ngram_min=2, ngram_max=4)
        store.add(vector=_vec(1, 0), output_ids=[9, 1, 2, 3, 7, 7, 7])
        records = store.retrieve(vector=_vec(1, 0))
        assert store.match(context=[1, 2, 3], records=records, max_tokens=8) == [7, 7, 7]

    def test_respects_max_tokens(self) -> None:
        store = RetrievalStore(capacity=4, ngram_min=2, ngram_max=4)
        store.add(vector=_vec(1, 0), output_ids=[1, 2, 3, 7, 7, 7])
        records = store.retrieve(vector=_vec(1, 0))
        assert store.match(context=[1, 2, 3], records=records, max_tokens=2) == [7, 7]

    def test_prefers_the_longest_needle(self) -> None:
        # Two traces match, one on 4 tokens and one on 2. The longer, more
        # confident match must win.
        store = RetrievalStore(capacity=4, ngram_min=2, ngram_max=4)
        store.add(vector=_vec(1, 0), output_ids=[3, 4, 55, 55], req_id="short")
        store.add(vector=_vec(1, 0), output_ids=[1, 2, 3, 4, 99, 99], req_id="long")
        records = store.retrieve(vector=_vec(1, 0), top_k=4)
        assert store.match(context=[1, 2, 3, 4], records=records, max_tokens=4) == [
            99,
            99,
        ]
        assert store.stats.hits_by_ngram == {4: 1}

    def test_no_match_returns_empty(self) -> None:
        store = RetrievalStore(capacity=4, ngram_min=2, ngram_max=4)
        store.add(vector=_vec(1, 0), output_ids=[1, 2, 3, 4])
        records = store.retrieve(vector=_vec(1, 0))
        assert store.match(context=[41, 42], records=records, max_tokens=4) == []

    def test_match_at_trace_end_yields_nothing(self) -> None:
        # The needle matches but nothing follows it, so there is no draft.
        store = RetrievalStore(capacity=4, ngram_min=2, ngram_max=4)
        store.add(vector=_vec(1, 0), output_ids=[1, 2, 3, 4])
        records = store.retrieve(vector=_vec(1, 0))
        assert store.match(context=[3, 4], records=records, max_tokens=4) == []

    def test_empty_context_or_zero_budget(self) -> None:
        store = RetrievalStore(capacity=4, ngram_min=2, ngram_max=4)
        store.add(vector=_vec(1, 0), output_ids=[1, 2, 3, 4, 5])
        records = store.retrieve(vector=_vec(1, 0))
        assert store.match(context=[], records=records, max_tokens=4) == []
        assert store.match(context=[1, 2], records=records, max_tokens=0) == []

    def test_find_last_takes_the_most_recent_occurrence(self) -> None:
        assert _find_last((1, 2, 9, 1, 2, 8), (1, 2)) == 3
        assert _find_last((1, 2), (5,)) == -1
        assert _find_last((1,), (1, 2)) == -1


# ------------------------------------------------------------- proposer ----


def _proposer(*, num_speculative_tokens: int = 8) -> ToolSpecProposer:
    assert _TOKENIZER is not None
    vocab_size = len(_TOKENIZER.get_vocab())
    vllm_config = SimpleNamespace(
        structured_outputs_config=SimpleNamespace(
            disable_any_whitespace=True,
            reasoning_parser="",
            backend="xgrammar",
        ),
        speculative_config=SimpleNamespace(
            num_speculative_tokens=num_speculative_tokens
        ),
        model_config=SimpleNamespace(get_vocab_size=lambda: vocab_size),
    )
    with patch.object(
        grammar_mod, "cached_tokenizer_from_config", return_value=_TOKENIZER
    ):
        return ToolSpecProposer.build(
            vllm_config=vllm_config,
            controller=SpeculativeDecodeController(),
        )


def _params(*, schema: str | None = _SCHEMA_JSON) -> SamplingParams:
    structured = None
    if schema is not None:
        structured = StructuredOutputsParams(json=schema)
        structured._backend = "xgrammar"
    return SamplingParams(temperature=0.0, structured_outputs=structured)


def _state(
    output_token_ids: list[int],
    *,
    prompt_len: int = 3,
    params: SamplingParams | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        token_ids=list(range(prompt_len)) + list(output_token_ids),
        prompt_len=prompt_len,
        sampling_params=params if params is not None else _params(),
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
    decode_reqs=None,
    decode_segments=None,
    prefill_reqs=None,
    prefill_result_modes=None,
    request_states=None,
    target_hidden_states=None,
    cu_seqlens=None,
    num_speculative_tokens: int = 8,
    finished_req_ids=None,
) -> ProposeContext:
    decode_reqs = decode_reqs or []
    prefill_reqs = prefill_reqs or []
    if prefill_result_modes is None:
        prefill_result_modes = ["new_final"] * len(prefill_reqs)
    if request_states is None:
        request_states = dict(decode_reqs)
    return ProposeContext(
        target_hidden_states=target_hidden_states,
        decode_reqs=decode_reqs,
        decode_segments=decode_segments or [],
        decode_token_ids=[[state.token_ids[-1]] for _, state in decode_reqs],
        prefill_reqs=prefill_reqs,
        prefill_token_ids=[0] * len(prefill_reqs),
        prefill_result_modes=prefill_result_modes,
        request_states=request_states,
        cu_seqlens=cu_seqlens if cu_seqlens is not None else [],
        num_decode_segments=len(decode_reqs),
        num_speculative_tokens=num_speculative_tokens,
        finished_req_ids=finished_req_ids or set(),
    )


@_needs_tokenizer
class TestToolSpecProposerProtocol:
    def test_hidden_states_only_wanted_at_final_prefill(self) -> None:
        proposer = _proposer()
        # Decode steps must not pay hidden-state collection: that is what keeps
        # the no-grammar fast path exactly as fast as without this proposer.
        assert proposer.needs_target_hidden_states([], has_final_prefill=False) is False
        assert proposer.needs_target_hidden_states([], has_final_prefill=True) is True

    def test_no_grammar_and_empty_memory_drafts_nothing(self) -> None:
        # The sonnet case: unstructured request, nothing stored. This must be a
        # complete no-op, not a short or empty draft.
        proposer = _proposer()
        state = _state([10, 11, 12], params=SamplingParams(temperature=0.0))
        ctx = _context(decode_reqs=[("r1", state)])
        assert proposer.propose(ctx) is None
        assert proposer.stats.steps_drafted == 0

    def test_drafting_disabled_returns_nothing(self) -> None:
        proposer = _proposer()
        state = _state([10, 11, 12], params=SamplingParams(temperature=0.0))
        ctx = _context(decode_reqs=[("r1", state)], num_speculative_tokens=0)
        assert proposer.propose(ctx) is None


@_needs_tokenizer
class TestToolSpecQuestionCapture:
    def test_captures_last_prompt_row_of_a_final_prefill(self) -> None:
        proposer = _proposer()
        prefill = SimpleNamespace(req_id="r1", token_ids=[1, 2, 3])
        # One prefill of 3 rows, no decode rows: cu_seqlens = [0, 3].
        hidden = mx.arange(3 * _HIDDEN, dtype=mx.float32).reshape(3, _HIDDEN)
        ctx = _context(
            prefill_reqs=[prefill],
            request_states={"r1": _state([])},
            target_hidden_states=hidden,
            cu_seqlens=[0, 3],
        )
        proposer.propose(ctx)
        assert proposer.stats.questions_captured == 1
        # Row 2 is the last prompt token's hidden state.
        assert np.allclose(proposer._questions["r1"], np.asarray(hidden[2]))

    def test_intermediate_chunks_are_skipped(self) -> None:
        proposer = _proposer()
        prefill = SimpleNamespace(req_id="r1", token_ids=[1, 2, 3])
        hidden = mx.zeros((3, _HIDDEN), dtype=mx.float32)
        ctx = _context(
            prefill_reqs=[prefill],
            prefill_result_modes=["intermediate"],
            request_states={"r1": _state([])},
            target_hidden_states=hidden,
            cu_seqlens=[0, 3],
        )
        proposer.propose(ctx)
        assert proposer.stats.questions_captured == 0

    def test_no_hidden_states_captures_nothing(self) -> None:
        proposer = _proposer()
        prefill = SimpleNamespace(req_id="r1", token_ids=[1, 2, 3])
        ctx = _context(
            prefill_reqs=[prefill],
            request_states={"r1": _state([])},
            target_hidden_states=None,
            cu_seqlens=[0, 3],
        )
        proposer.propose(ctx)
        assert proposer.stats.questions_captured == 0


@_needs_tokenizer
class TestToolSpecHarvest:
    def _capture(self, proposer: ToolSpecProposer, req_id: str, state) -> None:
        prefill = SimpleNamespace(req_id=req_id, token_ids=[1, 2, 3])
        hidden = mx.ones((3, _HIDDEN), dtype=mx.float32)
        proposer.propose(
            _context(
                prefill_reqs=[prefill],
                request_states={req_id: state},
                target_hidden_states=hidden,
                cu_seqlens=[0, 3],
            )
        )

    def test_finished_request_is_stored(self) -> None:
        proposer = _proposer()
        state = _state(list(range(20, 40)))
        self._capture(proposer, "r1", state)
        proposer.propose(
            _context(request_states={"r1": state}, finished_req_ids={"r1"})
        )
        assert proposer.stats.traces_stored == 1
        assert len(proposer.store) == 1

    def test_finished_request_without_a_question_vector_is_skipped(self) -> None:
        proposer = _proposer()
        state = _state(list(range(20, 40)))
        proposer.propose(
            _context(request_states={"r1": state}, finished_req_ids={"r1"})
        )
        assert proposer.stats.traces_stored == 0

    def test_harvest_works_after_state_is_torn_down(self) -> None:
        # The runner may already have dropped the request from request_states
        # by the time it appears in finished_req_ids.
        proposer = _proposer()
        state = _state(list(range(20, 40)))
        self._capture(proposer, "r1", state)
        proposer.propose(_context(request_states={}, finished_req_ids={"r1"}))
        assert proposer.stats.traces_stored == 1

    def test_departed_request_is_stored_without_a_finished_id(self) -> None:
        # The scheduler reports a finished id on the step *after* completion,
        # so when the engine goes idle that step never comes. Harvesting only
        # on finished_req_ids stored 0 traces across 32 real requests; the
        # request leaving request_states is the trigger that actually fires.
        proposer = _proposer()
        state = _state(list(range(20, 40)))
        self._capture(proposer, "r1", state)
        # A later request's prefill step, with r1 gone and never announced.
        proposer.propose(
            _context(request_states={"r2": _state([])}, finished_req_ids=set())
        )
        assert proposer.stats.traces_stored == 1
        assert len(proposer.store) == 1

    def test_a_request_is_not_harvested_twice(self) -> None:
        proposer = _proposer()
        state = _state(list(range(20, 40)))
        self._capture(proposer, "r1", state)
        proposer.propose(_context(request_states={}, finished_req_ids={"r1"}))
        proposer.propose(_context(request_states={}, finished_req_ids={"r1"}))
        assert proposer.stats.traces_stored == 1

    def test_release_harvests_and_clears(self) -> None:
        # release_requests is the signal that actually fires: the runner drives
        # its lifecycle reconcile from the finished-id set and releases before
        # the next propose(), so a trace harvested only from
        # ProposeContext.finished_req_ids is always already gone.
        proposer = _proposer()
        state = _state(list(range(20, 40)))
        self._capture(proposer, "r1", state)
        proposer.release_requests({"r1"})
        assert proposer.stats.traces_stored == 1
        assert "r1" not in proposer._questions
        assert "r1" not in proposer._live_states

    def test_release_without_a_question_vector_stores_nothing(self) -> None:
        proposer = _proposer()
        proposer.release_requests({"never-seen"})
        assert len(proposer.store) == 0


@_needs_tokenizer
class TestToolSpecRetrievalDrafting:
    def test_drafts_the_continuation_of_a_similar_trace(self) -> None:
        proposer = _proposer()
        # Seed the memory directly: an unstructured request whose output
        # repeats a span the next request is about to walk into.
        trace = [50, 51, 52, 53, 54, 55, 56, 57, 58, 59]
        proposer.store.add(vector=np.ones(_HIDDEN, dtype=np.float32), output_ids=trace)

        state = _state([50, 51, 52, 53, 54], params=SamplingParams(temperature=0.0))
        proposer._questions["r1"] = np.ones(_HIDDEN, dtype=np.float32)
        result = proposer.propose(_context(decode_reqs=[("r1", state)]))

        assert result is not None
        assert result.req_ids == ["r1"]
        assert result.draft_token_ids == [[55, 56, 57, 58, 59]]
        assert proposer.stats.steps_drafted == 1

    def test_no_question_vector_means_no_retrieval(self) -> None:
        proposer = _proposer()
        proposer.store.add(
            vector=np.ones(_HIDDEN, dtype=np.float32),
            output_ids=[50, 51, 52, 53, 54, 55, 56],
        )
        state = _state([50, 51, 52, 53, 54], params=SamplingParams(temperature=0.0))
        assert proposer.propose(_context(decode_reqs=[("r1", state)])) is None

    def test_grammar_draft_wins_over_retrieval(self) -> None:
        # Both halves have something to say for the same request. The grammar's
        # draft is legal by construction, so it must be the one sent, and
        # retrieval must not also queue a proposal for that request.
        proposer = _proposer()
        proposer.store.add(
            vector=np.ones(_HIDDEN, dtype=np.float32),
            output_ids=[50, 51, 52, 53, 54, 55, 56, 57],
        )
        proposer._questions["r1"] = np.ones(_HIDDEN, dtype=np.float32)
        state = _state([], params=_params())  # fresh constrained request
        result = proposer.propose(_context(decode_reqs=[("r1", state)]))

        assert result is not None
        assert result.req_ids == ["r1"]
        # Came from the grammar half, which drafts the schema's opening.
        assert proposer._grammar.stats.steps_drafted == 1
        assert "r1" not in proposer._pending


@_needs_tokenizer
class TestToolSpecThrottling:
    def _unmatchable(self, proposer: ToolSpecProposer) -> None:
        # A populated memory that this request's output will never match, which
        # is the prose case: retrieval searches every step and finds nothing.
        proposer.store.add(
            vector=np.ones(_HIDDEN, dtype=np.float32),
            output_ids=list(range(900, 940)),
        )
        proposer._questions["r1"] = np.ones(_HIDDEN, dtype=np.float32)

    def test_backs_off_after_a_miss_streak(self) -> None:
        proposer = _proposer()
        self._unmatchable(proposer)
        for step in range(12):
            state = _state(
                list(range(10, 10 + step + 6)),
                params=SamplingParams(temperature=0.0),
            )
            proposer.propose(_context(decode_reqs=[("r1", state)]))
        # Eight misses earn a cooldown, so the remaining steps must not search.
        assert proposer.stats.steps_eligible == 8
        assert proposer.stats.steps_throttled == 4

    def test_acceptance_clears_the_streak(self) -> None:
        proposer = _proposer()
        trace = [50, 51, 52, 53, 54, 55, 56, 57, 58, 59]
        proposer.store.add(vector=np.ones(_HIDDEN, dtype=np.float32), output_ids=trace)
        proposer._questions["r1"] = np.ones(_HIDDEN, dtype=np.float32)
        proposer._miss_streak["r1"] = 7
        state = _state([50, 51, 52, 53, 54], params=SamplingParams(temperature=0.0))
        proposer.propose(_context(decode_reqs=[("r1", state)]))
        after = _state(
            [50, 51, 52, 53, 54, 55, 56], params=SamplingParams(temperature=0.0)
        )
        proposer.propose(
            _context(
                decode_reqs=[("r1", after)],
                decode_segments=[_segment("r1", (55, 56))],
            )
        )
        assert "r1" not in proposer._miss_streak


@_needs_tokenizer
class TestToolSpecScoring:
    def _drafted(self, proposer: ToolSpecProposer):
        trace = [50, 51, 52, 53, 54, 55, 56, 57, 58, 59]
        proposer.store.add(vector=np.ones(_HIDDEN, dtype=np.float32), output_ids=trace)
        proposer._questions["r1"] = np.ones(_HIDDEN, dtype=np.float32)
        state = _state([50, 51, 52, 53, 54], params=SamplingParams(temperature=0.0))
        proposer.propose(_context(decode_reqs=[("r1", state)]))
        return state

    def test_full_acceptance_is_scored(self) -> None:
        proposer = _proposer()
        state = self._drafted(proposer)
        # The target committed exactly what was drafted.
        after = _state(
            [50, 51, 52, 53, 54, 55, 56],
            params=SamplingParams(temperature=0.0),
        )
        proposer.propose(
            _context(
                decode_reqs=[("r1", after)],
                decode_segments=[_segment("r1", (55, 56))],
            )
        )
        assert proposer.stats.drafts_offered == 2
        assert proposer.stats.drafts_accepted == 2
        assert proposer.stats.rejected_drafts == 0
        del state

    def test_rejection_is_scored(self) -> None:
        proposer = _proposer()
        self._drafted(proposer)
        # The target went somewhere else at the first drafted position.
        after = _state(
            [50, 51, 52, 53, 54, 99, 98],
            params=SamplingParams(temperature=0.0),
        )
        proposer.propose(
            _context(
                decode_reqs=[("r1", after)],
                decode_segments=[_segment("r1", (55, 56))],
            )
        )
        assert proposer.stats.drafts_accepted == 0
        assert proposer.stats.rejected_drafts == 1

    def test_fully_filtered_draft_resolves_rather_than_leaking(self) -> None:
        # A retrieval draft the engine's grammar rejects outright comes back
        # empty. It must still resolve, or the pending entry leaks forever.
        proposer = _proposer()
        self._drafted(proposer)
        assert "r1" in proposer._pending
        after = _state([50, 51, 52, 53, 54, 99], params=SamplingParams(temperature=0.0))
        proposer.propose(
            _context(
                decode_reqs=[("r1", after)],
                decode_segments=[_segment("r1", ())],
            )
        )
        assert "r1" not in proposer._pending
        assert proposer.stats.truncated_drafts == 1

    def test_finished_request_clears_pending(self) -> None:
        proposer = _proposer()
        self._drafted(proposer)
        proposer.propose(_context(request_states={}, finished_req_ids={"r1"}))
        assert "r1" not in proposer._pending
