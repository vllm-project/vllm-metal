# SPDX-License-Identifier: Apache-2.0
"""DSpark context invariants with real tiny drafter projections and KV buffers."""

from __future__ import annotations

from dataclasses import replace

import mlx.core as mx
import numpy as np
import pytest
from vllm.sampling_params import SamplingParams

from tests.stub_runner import make_stub_runner
from tests.test_dspark_contracts import draft_hf_config
from vllm_metal.v1.dspark.calibration import ConfidenceRecorder
from vllm_metal.v1.dspark.config import DSparkConfig
from vllm_metal.v1.dspark.memory import DSparkMemoryPlan
from vllm_metal.v1.dspark.model import ArenaCache, CtxCache, DSparkDrafter
from vllm_metal.v1.dspark.sampling import (
    RequestRandomStreams,
    SamplingTransforms,
    sample_from_distribution,
)
from vllm_metal.v1.dspark_proposer import (
    DSparkProposer,
    _DraftPlan,
    _RequestContext,
    _SpanWrite,
)
from vllm_metal.v1.model_runner import PrefillRequest, RequestState
from vllm_metal.v1.proposer import ProposeContext
from vllm_metal.v1.spec_decode import PagedDecodeSegment, SpeculativeDecodeController


def _proposer(**plan_overrides):
    config = DSparkConfig.from_dict(draft_hf_config().to_dict())
    plan = {
        "max_num_seqs": 32,
        "max_model_len": 2048,
        "max_num_batched_tokens": 8192,
        **plan_overrides,
    }
    return DSparkProposer(
        drafter=DSparkDrafter(config),
        config=config,
        runner=make_stub_runner(),
        controller=SpeculativeDecodeController(),
        memory_plan=DSparkMemoryPlan.build(config, itemsize=4, **plan),
    )


def _arena_caches(proposer):
    """Acquire one context slot in every layer's arena (as the proposer does)."""
    slot = proposer._arena[0].acquire()
    for layer in proposer._arena[1:]:
        assert layer.acquire() == slot
    return [ArenaCache(layer, slot) for layer in proposer._arena]


def _state(tokens, *, prompt_len=None, **params):
    prompt_len = len(tokens) - 1 if prompt_len is None else prompt_len
    sampling = {"temperature": 0.0, "max_tokens": 64, **params}
    return RequestState(
        token_ids=list(tokens),
        prompt_len=prompt_len,
        sampling_params=SamplingParams(**sampling),
        generated_tokens=len(tokens) - prompt_len,
    )


def _features(positions):
    # Distinct values per absolute position AND feature; wrong slices/offsets
    # cannot pass by appending equal zero rows to a length counter.
    return mx.sin(
        mx.array(positions, dtype=mx.float32)[:, None] * 0.37
        + mx.arange(64, dtype=mx.float32)[None] * 0.13
    )


def _context(*, prefill=(), decode=(), k=2, hidden=True, finished=()):
    """Prefill: (id, state, start, end, final); decode: (id, state, start, inputs, outputs)."""
    boundaries, positions, segments, decode_reqs, emitted = [0], [], [], [], []
    states, prefills, modes, prefill_tokens = {}, [], [], []
    for req_id, state, start, inputs, outputs in decode:
        segments.append(
            PagedDecodeSegment(
                req_id,
                tuple(inputs),
                boundaries[-1],
                len(inputs),
                tuple(inputs[1:]),
                start,
                ((0,),),
            )
        )
        boundaries.append(boundaries[-1] + len(inputs))
        positions.extend(range(start, start + len(inputs)))
        decode_reqs.append((req_id, state))
        emitted.append(outputs)
        states[req_id] = state
    for req_id, state, start, end, final in prefill:
        prefills.append(
            PrefillRequest(
                req_id,
                state.token_ids[start:end],
                state.sampling_params,
                [[0]],
                None,
                state.prompt_len if final else None,
                start,
                state.token_ids[: state.prompt_len],
            )
        )
        modes.append("cached_final" if final else "intermediate")
        prefill_tokens.append(state.token_ids[-1] if final else 0)
        boundaries.append(boundaries[-1] + end - start)
        positions.extend(range(start, end))
        states[req_id] = state
    return ProposeContext(
        _features(positions) if hidden else None,
        decode_reqs,
        segments,
        emitted,
        prefills,
        prefill_tokens,
        modes,
        states,
        boundaries,
        len(segments),
        k,
        set(finished),
    )


def _seed(proposer, state, req_id="r", *, k=2):
    return proposer.propose(
        _context(prefill=[(req_id, state, 0, len(state.token_ids) - 1, True)], k=k)
    )


def _assert_context(proposer, req_id, positions):
    record = proposer._contexts[req_id]
    assert record.disabled_reason is None
    assert record.covered_end == len(positions)
    reference = proposer._drafter.make_ctx_cache()
    if positions:
        proposer._drafter.update_context(_features(positions)[None], 0, reference)
    for actual, expected in zip(record.caches, reference, strict=True):
        assert actual.length == len(positions)
        if positions:
            assert bool(mx.allclose(actual.k, expected.k, atol=2e-5, rtol=2e-5).item())
            assert bool(mx.allclose(actual.v, expected.v, atol=2e-5, rtol=2e-5).item())


@pytest.mark.parametrize("accepted", range(8))
def test_commits_only_accepted_verification_inputs(accepted):
    proposer = _proposer()
    state = _state([1, 2, 3, 4, 5])
    assert _seed(proposer, state) is not None
    inputs = [5, 6, 7, 8, 9, 10, 11, 12]
    outputs = inputs[1 : accepted + 1] + [30]
    state.token_ids.extend(outputs)
    state.generated_tokens += len(outputs)
    result = proposer.propose(_context(decode=[("r", state, 4, inputs, outputs)], k=7))
    assert result is not None
    _assert_context(proposer, "r", list(range(5 + accepted)))
    assert proposer._contexts["r"].covered_end == len(state.token_ids) - 1


def test_intermediate_chunks_and_k_zero_ingest_without_prompt_replay():
    proposer = _proposer()
    state = _state(list(range(1, 10)), prompt_len=9)
    for start, end in ((0, 3), (3, 7)):
        assert (
            proposer.propose(_context(prefill=[("r", state, start, end, False)], k=0))
            is None
        )
        _assert_context(proposer, "r", list(range(end)))
    state.token_ids.append(15)
    state.generated_tokens = 1
    assert proposer.propose(_context(prefill=[("r", state, 7, 9, True)])) is not None
    _assert_context(proposer, "r", list(range(9)))
    for k, pending in ((0, 16), (2, 17)):
        previous = state.token_ids[-1]
        start = len(state.token_ids) - 1
        state.token_ids.append(pending)
        state.generated_tokens += 1
        result = proposer.propose(
            _context(decode=[("r", state, start, [previous], [pending])], k=k)
        )
        assert (result is not None) is (k > 0)
        _assert_context(proposer, "r", list(range(start + 1)))


@pytest.mark.parametrize("cached,start,end", [(5, 2, 3), (5, 3, 5), (5, 0, 4)])
def test_overlap_and_rollback_trim_every_physical_layer(cached, start, end):
    proposer = _proposer()
    state = _state(list(range(cached + 1)))
    _seed(proposer, state, k=0)
    state.token_ids = list(range(end + 1))
    state.generated_tokens = 1
    # A recomputed prefix/overlap replaces the suffix using absolute positions.
    proposer.propose(_context(prefill=[("r", state, start, end, True)], k=0))
    _assert_context(proposer, "r", list(range(end)))


@pytest.mark.parametrize("initial", [0, 2])
def test_missing_prefix_falls_back_without_partial_context(initial):
    proposer = _proposer()
    state = _state(list(range(8)))
    if initial:
        proposer.propose(_context(prefill=[("r", state, 0, initial, False)], k=0))
    result = proposer.propose(_context(prefill=[("r", state, 6, 7, True)]))
    assert result is None
    record = proposer._contexts["r"]
    assert record.disabled_reason == "missing target feature prefix"
    assert record.caches == []
    # Subsequent decode must not silently renumber the missing prefix.
    state.token_ids.append(9)
    assert proposer.propose(_context(decode=[("r", state, 7, [7], [9])])) is None
    # A complete scheduler recompute is sufficient to recover.
    assert _seed(proposer, state) is not None
    _assert_context(proposer, "r", list(range(8)))


def test_missing_features_releases_context_and_requires_recompute():
    proposer = _proposer()
    state = _state([1, 2, 3])
    _seed(proposer, state)
    state.token_ids.append(4)
    assert (
        proposer.propose(_context(decode=[("r", state, 2, [3], [4])], hidden=False))
        is None
    )
    assert proposer._contexts["r"].caches == []
    assert proposer._contexts["r"].disabled_reason == "target features are unavailable"


def test_mixed_spans_and_admission_skip_still_ingest_every_request():
    proposer = _proposer()
    proposer._max_drafts_per_step = 1
    first, second = _state([1, 2, 3]), _state([10, 11, 12, 13, 14])
    _seed(proposer, first, "a", k=0)
    first.token_ids.extend([20, 21])
    first.generated_tokens += 2
    result = proposer.propose(
        _context(
            decode=[("a", first, 2, [3, 20, 30], [20, 21])],
            prefill=[("b", second, 0, 4, True)],
        )
    )
    assert result.req_ids == ["a"]
    _assert_context(proposer, "a", list(range(4)))
    _assert_context(proposer, "b", list(range(4)))
    second.token_ids.append(15)
    second.generated_tokens += 1
    result = proposer.propose(_context(decode=[("b", second, 4, [14], [15])]))
    assert result.req_ids == ["b"]
    _assert_context(proposer, "b", list(range(5)))


def test_decode_spans_of_one_step_ingest_in_one_batched_pass(monkeypatch):
    proposer = _proposer()
    a, b, c = _state([1, 2, 3]), _state([10, 11, 12, 13, 14]), _state([30, 31, 32, 33])
    for req_id, state in (("a", a), ("b", b), ("c", c)):
        _seed(proposer, state, req_id, k=0)
    a.token_ids.extend([20, 21])
    b.token_ids.append(15)
    c.token_ids.extend([40, 41, 42])
    for state, count in ((a, 2), (b, 1), (c, 3)):
        state.generated_tokens += count
    real_spans = proposer._drafter.update_context_spans
    real_single = proposer._drafter.update_context
    calls: list = []
    monkeypatch.setattr(
        proposer._drafter,
        "update_context_spans",
        lambda hidden, spans, arenas: (
            calls.append(spans),
            real_spans(hidden, spans, arenas),
        ),
    )
    monkeypatch.setattr(
        proposer._drafter,
        "update_context",
        lambda *args, **kwargs: (calls.append("single"), real_single(*args, **kwargs)),
    )
    result = proposer.propose(
        _context(
            decode=[
                ("a", a, 2, [3, 20, 30], [20, 21]),
                ("b", b, 4, [14, 50, 51], [15]),
                ("c", c, 3, [33, 40, 41, 50], [40, 41, 42]),
            ]
        )
    )
    assert result.req_ids == ["a", "b", "c"]
    # One batched write for the three spans (counts 2, 1 and 3), no single-row writes.
    assert len(calls) == 1 and [span[2] for span in calls[0]] == [2, 1, 3]
    _assert_context(proposer, "a", list(range(4)))
    _assert_context(proposer, "b", list(range(5)))
    _assert_context(proposer, "c", list(range(6)))
    # The padded columns of the shorter spans never became context.
    for req_id, length in (("a", 4), ("b", 5), ("c", 6)):
        assert all(
            cache.length == length for cache in proposer._contexts[req_id].caches
        )


def test_deferred_step_allowed_follows_the_mode_and_the_planner():
    from vllm_metal.v1.dspark.planner import DraftDecision

    proposer = _proposer()
    state = _state([1, 2, 3])
    _seed(proposer, state, k=0)
    decode = [("r", state)]
    # Fixed mode drafts every eligible request: the step keeps its sync.
    assert proposer.deferred_step_allowed(decode, 2) is False
    # No speculative tokens scheduled or the bypass mode: nothing to draft.
    assert proposer.deferred_step_allowed(decode, 0) is True
    proposer.bypass_only = True
    assert proposer.deferred_step_allowed(decode, 2) is True
    proposer.bypass_only = False
    # Adaptive mode: the planner's own decision, offered every draftable
    # request at its full cap and the batch's request count and context.
    seen = []

    class Planner:
        def __init__(self, draft):
            self.draft = draft

        def decide(self, candidates, *, active_requests, context):
            seen.append((candidates, active_requests, context))
            return DraftDecision(draft=self.draft, reason="stub")

    proposer.adaptive = Planner(draft=False)
    assert proposer.deferred_step_allowed(decode, 2) is True
    ((candidates, active, context),) = seen
    assert [(c[0], c[2], c[3]) for c in candidates] == [("r", "greedy", 2)]
    assert active == 1 and context == 3
    proposer.adaptive = Planner(draft=True)
    assert proposer.deferred_step_allowed(decode, 2) is False
    # A request without a usable context is not a candidate; a batch of such
    # requests has nothing to draft and may defer.
    other = _state([5, 6])
    proposer.adaptive = Planner(draft=True)
    assert proposer.deferred_step_allowed([("other", other)], 2) is True
    # A request at its output budget contributes no candidate either.
    capped = _state([1, 2, 3], max_tokens=1)
    _seed(proposer, capped, "c", k=0)
    assert proposer.deferred_step_allowed([("c", capped)], 2) is True


def test_ingest_deferred_step_advances_context_without_drafting(monkeypatch, caplog):
    proposer = _proposer()
    state = _state([1, 2, 3])
    _seed(proposer, state, k=0)
    # The runner appends the pending placeholder before the deferred ingest.
    state.token_ids.append(-1)
    state.generated_tokens += 1
    real_backbone = proposer._drafter.backbone
    backbone_calls: list = []
    monkeypatch.setattr(
        proposer._drafter,
        "backbone",
        lambda *a, **k: (backbone_calls.append(1), real_backbone(*a, **k))[1],
    )
    ctx = replace(_context(decode=[("r", state, 2, [3], [])]), decode_token_ids=[()])
    with caplog.at_level("INFO", logger="vllm_metal.v1.dspark_proposer"):
        proposer.ingest_deferred_step(ctx)
    assert backbone_calls == []
    assert any("first deferred step" in record.message for record in caplog.records)
    _assert_context(proposer, "r", list(range(3)))
    assert proposer._contexts["r"].covered_end == 3
    assert proposer.counters.steps == 2
    assert proposer.counters.bypass_reasons["deferred-step"] == 1
    assert proposer.proposals == {}
    # The next synchronous step continues from the resolved token.
    state.token_ids[-1] = 9
    result = proposer.propose(_context(decode=[("r", state, 3, [9], [9])]))
    assert result is not None and result.req_ids == ["r"]
    assert backbone_calls == [1]


def test_write_spans_pads_past_the_committed_length_only():
    proposer = _proposer()
    short, long = _arena_caches(proposer), _arena_caches(proposer)
    hidden = _features([0, 0, 1, 2])
    proposer._drafter.update_context_spans(
        hidden,
        [(0, 0, 1, short[0].slot), (1, 0, 3, long[0].slot)],
        proposer._arena,
    )
    for caches, length in ((short, 1), (long, 3)):
        for cache in caches:
            cache.extend_to(length)
        expected = proposer._drafter.make_ctx_cache()
        proposer._drafter.update_context(
            _features(list(range(length)))[None], 0, expected
        )
        for actual, reference in zip(caches, expected, strict=True):
            assert actual.k.shape[2] == length
            assert bool(mx.allclose(actual.k, reference.k, atol=2e-5, rtol=2e-5).item())
            assert bool(mx.allclose(actual.v, reference.v, atol=2e-5, rtol=2e-5).item())
    # A later single-row append lands right after the committed length.
    proposer._drafter.update_context(_features([1])[None], 1, short)
    expected = proposer._drafter.make_ctx_cache()
    proposer._drafter.update_context(_features([0, 1])[None], 0, expected)
    assert bool(mx.allclose(short[0].k, expected[0].k, atol=2e-5, rtol=2e-5).item())
    arena = proposer._arena[0]
    with pytest.raises(ValueError, match="exceeds reserved capacity"):
        arena.write_spans(
            [short[0].slot],
            [arena.capacity - 1],
            [2],
            mx.zeros((1, arena.keys.shape[1], 2, arena.keys.shape[-1])),
            mx.zeros((1, arena.keys.shape[1], 2, arena.keys.shape[-1])),
        )
    with pytest.raises(ValueError, match="within reserved capacity"):
        short[0].extend_to(0)


@pytest.mark.parametrize(
    "params",
    [
        {"repetition_penalty": 1.1},
        {"logprobs": 2},
        {"temperature": 0.8, "presence_penalty": 0.5},
        {"temperature": 0.8, "min_p": 0.1},
        {"temperature": 0.8, "logit_bias": {1: 1.0}},
    ],
)
def test_undraftable_requests_keep_context_without_drafting(params):
    proposer = _proposer()
    state = _state([1, 2, 3], **params)
    assert _seed(proposer, state) is None
    _assert_context(proposer, "r", [0, 1])
    assert "r" not in proposer.proposals


def _stochastic_state(tokens, **params):
    return _state(tokens, temperature=0.8, top_p=0.9, **params)


def test_stochastic_requests_draft_from_recorded_distributions():
    proposer = _proposer()
    proposer._runner.model_config.seed = 0
    state = _stochastic_state([1, 2, 3], seed=4)
    drafts = _seed(proposer, state, k=2)
    assert drafts.req_ids == ["r"] and len(drafts.draft_token_ids[0]) == 2
    record = proposer.proposals["r"]
    assert record.owner is state
    assert record.anchor_position == 2 and record.anchor_token == 3
    assert record.token_ids == drafts.draft_token_ids[0]
    assert record.transforms == SamplingTransforms(0.8, 0, 0.9)
    rows = np.array(record.distributions.tolist())
    assert rows.shape == (2, 64)
    assert np.allclose(rows.sum(axis=1), 1.0, atol=1e-6)
    assert all(rows[i][t] > 0 for i, t in enumerate(record.token_ids))
    assert (rows == 0).any()  # top-p truncation leaves zero-mass tokens
    # Replaying the request's own proposal stream reproduces every drafted
    # token from the recorded rows: the tokens came from exactly these q.
    replay = RequestRandomStreams.for_request(4, engine_seed=0, ordinal=1)
    expected = sample_from_distribution(
        record.distributions, mx.array(replay.proposal.random(2), dtype=mx.float32)
    )
    assert expected.tolist() == record.token_ids


def test_records_are_spent_on_the_next_schedule_and_released_with_the_request():
    proposer = _proposer()
    proposer._runner.model_config.seed = 0
    state = _stochastic_state([1, 2, 3])
    _seed(proposer, state, k=2)
    drafted = proposer.proposals["r"].token_ids
    # The scheduler verified one draft and committed a correction token.
    state.token_ids.extend([drafted[0], 9])
    proposer.propose(
        _context(decode=[("r", state, 2, [3, *drafted], [drafted[0], 9])], k=2)
    )
    record = proposer.proposals["r"]
    assert record.anchor_position == 4 and record.anchor_token == 9
    assert proposer._streams["r"] is record.streams
    # A scheduled request without a new draft (K=0 step) leaves no stale record.
    state.token_ids.extend(record.token_ids[:1] + [8])
    proposer.propose(
        _context(
            decode=[("r", state, 4, [9, *record.token_ids], [record.token_ids[0], 8])],
            k=0,
        )
    )
    assert "r" not in proposer.proposals
    proposer.release_requests({"r"})
    assert "r" not in proposer._streams


def test_mixed_greedy_and_stochastic_rows_match_independent_rows():
    proposer = _proposer()
    specs = [(6, 3, None), (0, 2, 5), (2, 3, 6), (4, 1, None), (2, 2, 7)]

    def plans():
        built = []
        for ordinal, (length, cap, seed) in enumerate(specs, start=1):
            owner = (
                _stochastic_state([2] * (length + 1))
                if seed
                else _state([2] * (length + 1))
            )
            caches = _arena_caches(proposer)
            if length:
                proposer._drafter.update_context(
                    _features(list(range(length)))[None], 0, caches
                )
            transforms = streams = None
            if seed:
                transforms = SamplingTransforms.from_params(owner.sampling_params)
                streams = RequestRandomStreams.for_request(
                    seed, engine_seed=0, ordinal=ordinal
                )
            built.append(
                _DraftPlan(
                    f"{ordinal}",
                    2,
                    _RequestContext(owner, caches, length),
                    cap,
                    transforms,
                    streams,
                )
            )
        return built

    alone = [proposer._batch_draft([plan]) for plan in plans()]
    _, batched_rows, batched_records = proposer._batch_draft(plans())
    for (_length, cap, seed), (_, rows, records), row in zip(
        specs, alone, batched_rows, strict=True
    ):
        assert row == rows[0] and len(row) == cap
        if seed:
            (key,) = records
            record, batched = records[key], batched_records[key]
            assert batched.token_ids == record.token_ids
            # The padded multi-row backbone and the single-row backbone differ
            # by summation order only; the recorded rows are the ones the
            # tokens were sampled from in either case.
            assert np.allclose(
                np.array(batched.distributions.tolist()),
                np.array(record.distributions.tolist()),
                atol=1e-6,
                rtol=1e-5,
            )
            assert batched.distributions.shape == (cap, 64)
        else:
            (key,) = records
            assert records[key].distributions is None
            assert batched_records[key].distributions is None
        # Every drafted row carries one raw confidence logit per drafted position.
        (key,) = records
        assert len(records[key].confidence) == cap
        assert np.allclose(
            records[key].confidence, batched_records[key].confidence, atol=1e-4
        )
    assert set(batched_records) == {"1", "2", "3", "4", "5"}


def test_request_streams_survive_other_requests_and_reuse_ordinals_deterministically():
    def run(with_other: bool):
        mx.random.seed(11)  # identical drafter weights in both runs
        proposer = _proposer()
        proposer._runner.model_config.seed = 0
        state = _stochastic_state([1, 2, 3])
        _seed(proposer, state, k=2)
        first = proposer.proposals["r"].token_ids
        if with_other:
            other = _stochastic_state([4, 5, 6])
            _seed(proposer, other, "other", k=2)
            proposer.release_requests({"other"})
        state.token_ids.extend([first[0], 9])
        proposer.propose(
            _context(decode=[("r", state, 2, [3, *first], [first[0], 9])], k=2)
        )
        return first, proposer.proposals["r"].token_ids

    assert run(False) == run(True)


@pytest.mark.parametrize("event", ["finish", "cancel", "preempt", "resume"])
def test_runner_lifecycle_invalidates_only_affected_context(event):
    proposer = _proposer()
    old, keep = _state([1, 2, 3]), _state([4, 5, 6])
    _seed(proposer, old, "r")
    _seed(proposer, keep, "keep")
    runner = proposer._runner
    runner._drafter = proposer
    runner._request_states = {"r": old, "keep": keep}
    runner._reconcile_request_lifecycle(
        {"r"} if event in ("finish", "cancel") else set(),
        preempted_req_ids={"r"} if event == "preempt" else set(),
        resumed_req_ids={"r"} if event == "resume" else set(),
    )
    assert set(proposer._contexts) == {"keep"}
    _assert_context(proposer, "keep", [0, 1])
    # Reuse in the same step: finished_req_ids describes OLD state already
    # released by the runner. It must not delete the newly ingested generation.
    new = _state([20, 21, 22, 23])
    ctx = _context(prefill=[("r", new, 0, 3, True)], finished={"r"})
    assert proposer.propose(ctx) is not None
    assert proposer._contexts["r"].owner is new
    _assert_context(proposer, "r", [0, 1, 2])


def test_changed_request_identity_cannot_reuse_old_prefix():
    proposer = _proposer()
    _seed(proposer, _state([1, 2, 3, 4]))
    replacement = _state([9, 10, 11, 12])
    assert proposer.propose(_context(prefill=[("r", replacement, 2, 3, True)])) is None
    assert proposer._contexts["r"].owner is replacement
    assert proposer._contexts["r"].caches == []


@pytest.mark.parametrize("phase", ["ingest", "draft"])
def test_errors_drop_partial_context_before_retry(monkeypatch, phase):
    proposer = _proposer()
    state = _state([1, 2, 3])
    real_update = proposer._drafter.update_context

    def fail(*args, **kwargs):
        if phase == "ingest":
            real_update(*args, **kwargs)
        raise RuntimeError("injected failure")

    monkeypatch.setattr(
        proposer._drafter, "update_context" if phase == "ingest" else "backbone", fail
    )
    with pytest.raises(RuntimeError, match="injected failure"):
        _seed(proposer, state)
    assert proposer._contexts == {}


@pytest.mark.parametrize("fault", ["rows", "boundaries", "tokens", "physical"])
def test_inconsistent_handoff_fails_closed(fault):
    proposer = _proposer()
    state = _state([1, 2, 3])
    _seed(proposer, state)
    state.token_ids.append(4)
    ctx = _context(decode=[("r", state, 2, [3], [4])])
    if fault == "rows":
        ctx = replace(ctx, target_hidden_states=mx.zeros((0, 64)))
    elif fault == "boundaries":
        ctx = replace(ctx, cu_seqlens=[0, 2])
    elif fault == "tokens":
        state.token_ids[2] = 50
    else:
        proposer._contexts["r"].caches[0].trim_to(1)
    with pytest.raises((ValueError, RuntimeError), match="DSpark"):
        proposer.propose(ctx)
    assert "r" not in proposer._contexts


@pytest.mark.parametrize(
    "order,caps",
    [
        ((0, 2, 6), (1, 2, 3)),
        ((6, 0, 2), (1, 2, 3)),
        ((0, 2, 6), (7, 1, 3)),
        ((6, 0, 2), (3, 7, 1)),
    ],
)
def test_ragged_and_empty_context_drafts_match_independent_rows(order, caps):
    proposer = _proposer()
    plans, expected = [], []
    for length, cap in zip(order, caps, strict=True):
        owner = _state([2] * (length + 1))
        caches = _arena_caches(proposer)
        if length:
            proposer._drafter.update_context(
                _features(list(range(length)))[None], 0, caches
            )
        plan = _DraftPlan(str(length), 2, _RequestContext(owner, caches, length), cap)
        plans.append(plan)
        expected.append(proposer._batch_draft([plan])[1][0])

    def snapshot(plan):
        return [
            (cache.length, None if cache.k is None else mx.array(cache.k))
            for cache in plan.context.caches
        ]

    before = [snapshot(plan) for plan in plans]
    batched = proposer._batch_draft(plans)[1]
    assert batched == expected
    assert [len(row) for row in batched] == list(caps)
    # Draft block/scratch positions must never enter persistent context: the
    # committed length and every committed position are unchanged.
    for plan, original in zip(plans, before, strict=True):
        for cache, (length, keys) in zip(plan.context.caches, original, strict=True):
            assert cache.length == length
            if keys is not None:
                assert bool(mx.array_equal(cache.k, keys).item())


@pytest.mark.parametrize("remaining,expected", [(1, 0), (2, 1), (4, 3)])
def test_per_row_output_and_model_caps_reserve_bonus(remaining, expected):
    proposer = _proposer()
    state = _state([1, 2, 3], max_tokens=remaining + 1)
    assert proposer._draft_cap(state, 7) == expected
    proposer._runner.model_config.max_model_len = len(state.token_ids) + 2
    assert proposer._draft_cap(state, 7) == min(expected, 1)


@pytest.mark.parametrize("ignore_eos", [False, True])
def test_terminal_token_respects_ignore_eos(ignore_eos):
    proposer = _proposer()
    state = _state([1, 2, 3], ignore_eos=ignore_eos)
    state.sampling_params.update_from_generation_config({}, eos_token_id=3)
    assert proposer._draft_cap(state, 7) == (7 if ignore_eos else 0)


def test_cache_trim_rejects_negative_and_growth():
    cache = CtxCache()
    for length in (-1, 1):
        with pytest.raises(ValueError, match="existing prefix"):
            cache.trim_to(length)


def test_no_sample_and_clipped_schedule_use_only_actual_input_span():
    proposer = _proposer()
    state = _state([1, 2, 3])
    _seed(proposer, state)
    ctx = _context(decode=[("r", state, 2, [3, 4, 5], [])])
    assert proposer.propose(ctx) is None
    _assert_context(proposer, "r", [0, 1])
    # The scheduler verifies only one of the possible seven draft positions.
    state.token_ids.extend([4, 20])
    state.generated_tokens += 2
    assert (
        proposer.propose(_context(decode=[("r", state, 2, [3, 4], [4, 20])], k=7))
        is not None
    )
    _assert_context(proposer, "r", [0, 1, 2, 3])


def test_request_stop_token_prevents_further_drafting():
    proposer = _proposer()
    state = _state([1, 2, 3], stop_token_ids=[3])
    assert proposer._draft_cap(state, 7) == 0


def test_context_checks_both_keys_and_values():
    cache = CtxCache()
    with pytest.raises(ValueError, match="matching batch, head and token axes"):
        cache.append(mx.zeros((1, 2, 3, 8)), mx.zeros((1, 2, 2, 8)))
    cache.append(mx.zeros((1, 2, 3, 8)), mx.zeros((1, 2, 3, 8)))
    cache._values = cache._values[:, :, :2]
    with pytest.raises(RuntimeError, match="K/V coverage disagrees"):
        cache.trim_to(1)


def test_per_step_cap_rotates_least_recently_drafted():
    proposer = _proposer()
    proposer._max_drafts_per_step = 1
    states = {}
    for req_id, tokens in (("a", [1, 2, 3]), ("b", [4, 5, 6]), ("c", [7, 8, 9])):
        states[req_id] = _state(tokens)
        _seed(proposer, states[req_id], req_id, k=0)
    drafted = []
    for step in range(6):
        decode = []
        for req_id, state in states.items():
            previous = state.token_ids[-1]
            start = len(state.token_ids) - 1
            state.token_ids.append(20 + step)
            state.generated_tokens += 1
            decode.append((req_id, state, start, [previous], [20 + step]))
        result = proposer.propose(_context(decode=decode, k=2))
        drafted.append(tuple(result.req_ids))
        # Requests that wait for the cap still have their context advanced.
        for req_id, state in states.items():
            record = proposer._contexts[req_id]
            assert record.disabled_reason is None
            assert record.covered_end == len(state.token_ids) - 1
    assert drafted == [("a",), ("b",), ("c",), ("a",), ("b",), ("c",)]
    proposer.release_requests({"b"})
    assert "b" not in proposer._last_drafted
    with pytest.raises(ValueError, match="positive"):
        DSparkProposer(
            drafter=proposer._drafter,
            config=proposer._config,
            runner=proposer._runner,
            controller=proposer._controller,
            memory_plan=proposer.memory_plan,
            max_drafts_per_step=0,
        )


def test_per_step_cap_above_eligible_drafts_everyone_in_batch_order():
    proposer = _proposer()
    proposer._max_drafts_per_step = 2
    states = {
        req_id: _state(tokens) for req_id, tokens in (("x", [1, 2]), ("y", [3, 4]))
    }
    for req_id, state in states.items():
        _seed(proposer, state, req_id, k=0)
    decode = []
    for req_id, state in states.items():
        previous = state.token_ids[-1]
        start = len(state.token_ids) - 1
        state.token_ids.append(9)
        state.generated_tokens += 1
        decode.append((req_id, state, start, [previous], [9]))
    assert proposer.propose(_context(decode=decode, k=2)).req_ids == ["x", "y"]


@pytest.mark.parametrize("accepted", [0, 1, 2])
def test_recorder_observes_verified_outcomes_with_censoring(accepted):
    proposer = _proposer()
    proposer._runner.model_config.seed = 0
    recorder = ConfidenceRecorder()
    proposer.recorder = recorder
    state = _state([1, 2, 3])
    drafts = _seed(proposer, state, k=3)
    record = proposer.proposals["r"]
    assert len(record.confidence) == 3
    assert record.distributions is None  # greedy rows keep no distributions
    drafted = list(drafts.draft_token_ids[0])
    # The scheduler clipped the proposal to two positions and verification
    # accepted ``accepted`` of them; the runner committed the prefix plus one.
    scheduled = drafted[:2]
    outputs = scheduled[:accepted] + [9]
    state.token_ids.extend(outputs)
    proposer.propose(_context(decode=[("r", state, 2, [3, *scheduled], outputs)], k=3))
    (sample,) = recorder.samples
    assert sample.mode == "greedy" and sample.scheduled == 2
    assert sample.accepted == accepted
    assert sample.logits == tuple(record.confidence[:2])
    assert sample.survival == tuple(1 if i < accepted else 0 for i in range(2))
    # A scheduled step without drafts (nothing to verify) records nothing.
    proposer.recorder = ConfidenceRecorder()
    later = proposer.proposals["r"]
    state.token_ids.extend([later.token_ids[0], 8])
    proposer.propose(
        _context(
            decode=[
                (
                    "r",
                    state,
                    len(state.token_ids) - 3,
                    [9, later.token_ids[0]],
                    [later.token_ids[0], 8],
                )
            ],
            k=0,
        )
    )
    assert proposer.recorder.samples[0].scheduled == 1
    proposer.propose(
        _context(decode=[("r", state, len(state.token_ids) - 1, [8], [7])], k=0)
    )
    assert len(proposer.recorder.samples) == 1


def test_stochastic_outcomes_record_the_sampling_mode():
    proposer = _proposer()
    proposer._runner.model_config.seed = 0
    proposer.recorder = ConfidenceRecorder()
    state = _stochastic_state([1, 2, 3], seed=4)
    drafts = _seed(proposer, state, k=2)
    drafted = list(drafts.draft_token_ids[0])
    state.token_ids.extend([drafted[0], 9])
    proposer.propose(
        _context(decode=[("r", state, 2, [3, *drafted], [drafted[0], 9])], k=2)
    )
    (sample,) = proposer.recorder.samples
    assert sample.mode == "stochastic" and sample.temperature == 0.8
    assert sample.accepted == 1 and sample.scheduled == 2


# ---- load regime (lapse) ------------------------------------------------------


class _Planner:
    def __init__(self, draft: bool) -> None:
        self.draft = draft
        self.calls = 0

    def decide(self, candidates, *, active_requests, context):
        from vllm_metal.v1.dspark.planner import DraftDecision

        self.calls += 1
        return DraftDecision(draft=self.draft, reason="stub")

    def forget(self, req_ids):
        pass

    def forget_idle(self, scheduled, drafted):
        pass


def _deferred_ctx(state, req_id="r", k=2):
    """A deferred step after the placeholder append: the query row is the previous anchor."""
    anchor = len(state.token_ids) - 2
    return replace(
        _context(decode=[(req_id, state, anchor, [state.token_ids[anchor]], [])], k=k),
        decode_token_ids=[()],
    )


def test_lapse_enters_after_sustained_declines_and_resumes(monkeypatch, caplog):
    from vllm_metal.v1 import dspark_proposer as module

    proposer = _proposer()
    proposer.adaptive = _Planner(draft=False)
    state = _state([1, 2, 3])
    _seed(proposer, state, k=0)
    assert proposer.needs_target_hidden_states((), has_final_prefill=False) is True
    assert "r" in proposer._contexts
    with caplog.at_level("INFO", logger="vllm_metal.v1.dspark_proposer"):
        for _ in range(module.LAPSE_ENTER_STEPS - 1):
            state.token_ids.append(-1)
            proposer.ingest_deferred_step(_deferred_ctx(state))
            state.token_ids[-1] = 7
        assert proposer._lapsed is False
        state.token_ids.append(-1)
        proposer.ingest_deferred_step(_deferred_ctx(state))
        state.token_ids[-1] = 7
    assert proposer._lapsed is True
    assert proposer._contexts == {}  # every context released with its slot
    assert proposer._arena[0].free_slots == proposer._arena[0].slots
    assert proposer.counters.lapse_entries == 1
    assert any("load regime lapse" in r.message for r in caplog.records)
    assert proposer.needs_target_hidden_states((), has_final_prefill=True) is False
    assert proposer.deferred_step_allowed([("r", state)], 2) is True
    # Lapsed steps neither ingest nor draft. A lapse entered under load short-circuits
    # on the cheap load test and never asks the planner; this one began at a single
    # request, where there is no lower load to wait for, so the verdict is the only
    # thing that can end it and the planner is consulted once per step.
    calls = proposer.adaptive.calls
    result = proposer.propose(
        _context(
            decode=[("r", state, len(state.token_ids) - 1, [7], [7])], hidden=False
        )
    )
    assert result is None and proposer._contexts == {}
    assert proposer.counters.bypass_reasons["lapse"] >= 1
    assert proposer.adaptive.calls == calls + 1
    # A prefill during the lapse creates no context either.
    fresh = _state([4, 5, 6, 7])
    proposer.propose(_context(prefill=[("p", fresh, 0, 3, True)], hidden=False))
    assert "p" not in proposer._contexts
    # This lapse began while a single request was decoding, and this branch only
    # runs with at least one decode request, so there is no lower load to wait
    # for. The planner's verdict alone must be able to end it, on LAPSE_EXIT_STEPS
    # consecutive steps -- otherwise a single-client server never drafts again.
    proposer.adaptive = _Planner(draft=True)
    for _ in range(module.LAPSE_EXIT_STEPS):
        assert proposer._lapsed is True
        state.token_ids.append(-1)
        proposer.ingest_deferred_step(_deferred_ctx(state))
        state.token_ids[-1] = 7
    assert proposer._lapsed is False
    assert proposer.counters.lapse_exits == 1
    assert proposer.needs_target_hidden_states((), has_final_prefill=False) is True
    # New requests are primed again on the synchronous path.
    later = _state([4, 5, 6, 7])
    proposer.propose(_context(prefill=[("q", later, 0, 3, True)], k=0))
    assert "q" in proposer._contexts


def test_drafting_resets_the_decline_streak(monkeypatch):
    from vllm_metal.v1 import dspark_proposer as module

    proposer = _proposer()
    proposer.adaptive = _Planner(draft=False)
    state = _state([1, 2, 3])
    _seed(proposer, state, k=0)
    for _ in range(module.LAPSE_ENTER_STEPS - 1):
        state.token_ids.append(-1)
        proposer.ingest_deferred_step(_deferred_ctx(state))
        state.token_ids[-1] = 7
    assert proposer._decline_streak == module.LAPSE_ENTER_STEPS - 1
    # A synchronous step that drafts (fixed-style planner verdict) resets the streak.
    proposer.adaptive = None
    result = proposer.propose(
        _context(decode=[("r", state, len(state.token_ids) - 1, [7], [7])])
    )
    assert result is not None and result.req_ids == ["r"]
    proposer.adaptive = _Planner(draft=False)
    state.token_ids.append(-1)
    proposer.ingest_deferred_step(_deferred_ctx(state))
    assert proposer._decline_streak == 1 and proposer._lapsed is False


def test_fixed_mode_and_disabled_knob_never_lapse(monkeypatch):
    from vllm_metal.v1 import dspark_proposer as module

    proposer = _proposer()  # fixed mode: no planner
    state = _state([1, 2, 3])
    _seed(proposer, state, k=0)
    for _ in range(module.LAPSE_ENTER_STEPS + 1):
        state.token_ids.append(-1)
        proposer.ingest_deferred_step(_deferred_ctx(state, k=0))
        state.token_ids[-1] = 7
    assert proposer._lapsed is False and "r" in proposer._contexts
    disabled = _proposer()
    disabled.lapse_enabled = False
    disabled.adaptive = _Planner(draft=False)
    other = _state([1, 2, 3])
    _seed(disabled, other, k=0)
    for _ in range(module.LAPSE_ENTER_STEPS + 1):
        other.token_ids.append(-1)
        disabled.ingest_deferred_step(_deferred_ctx(other))
        other.token_ids[-1] = 7
    assert disabled._lapsed is False and "r" in disabled._contexts
    disabled.bypass_only = True
    disabled.lapse_for_bypass()
    assert disabled._lapsed is False


def test_bypass_mode_lapses_from_the_start(caplog):
    proposer = _proposer()
    proposer.bypass_only = True
    with caplog.at_level("INFO", logger="vllm_metal.v1.dspark_proposer"):
        proposer.lapse_for_bypass()
    assert proposer._lapsed is True
    assert proposer.needs_target_hidden_states((), has_final_prefill=True) is False
    fresh = _state([4, 5, 6, 7])
    assert (
        proposer.propose(_context(prefill=[("p", fresh, 0, 3, True)], hidden=False))
        is None
    )
    assert proposer._contexts == {}
    assert proposer.counters.bypass_reasons["lapse"] == 1
    # never resumes
    state = _state([1, 2, 3])
    for _ in range(10):
        state.token_ids.append(-1)
        proposer.ingest_deferred_step(_deferred_ctx(state))
        state.token_ids[-1] = 7
    assert proposer._lapsed is True


def test_a_lapse_entered_under_load_waits_for_the_load_to_drop(monkeypatch):
    """Entry at one request exits on the verdict; entry under load needs a drop.

    Without the second half, the fix for the unreachable single-request exit would
    turn the load regime into "resume whenever the planner says draft", which is the
    flapping the entry streak exists to prevent.
    """
    from vllm_metal.v1 import dspark_proposer as module

    proposer = _proposer()
    proposer.adaptive = _Planner(draft=False)
    state = _state([1, 2, 3])
    _seed(proposer, state, k=0)
    for _ in range(module.LAPSE_ENTER_STEPS):
        state.token_ids.append(-1)
        proposer.ingest_deferred_step(_deferred_ctx(state))
        state.token_ids[-1] = 7
    assert proposer._lapsed is True

    # pretend the lapse began while four requests were decoding
    proposer._lapse_entry_active = 4
    proposer.adaptive = _Planner(draft=True)
    for _ in range(module.LAPSE_EXIT_STEPS + 2):
        state.token_ids.append(-1)
        proposer.ingest_deferred_step(_deferred_ctx(state))
        state.token_ids[-1] = 7
    # one decode request is at or below 4 - 1, so this DOES resume; the guard is
    # that a batch still at the entry load would not.
    assert proposer._lapsed is False


class TestPagedIngestSurvivesAnExhaustedPool:
    """A full context pool degrades to target-only; it does not abort the ingest.

    `_flush_span_writes` used to grow the pool inside its own loop with nothing to
    catch `PagedContextFullError`, so a pool that emptied on the third row raised
    through the engine step with the first two rows already grown and neither their
    span written nor their context committed. The pool cannot run out while it is
    sized for the worst case, which is exactly why the path was never exercised.
    """

    def _exhausted_pool(self, proposer):
        from vllm_metal.v1.dspark.paged_context import DSparkPagedContext

        # One block, which is the padding sink, so nothing at all can be handed out.
        return DSparkPagedContext(
            num_layers=1,
            kv_heads=1,
            head_dim=64,
            block_size=16,
            num_blocks=1,
            draft_block=proposer._config.block_size,
            dtype=mx.float32,
        )

    def test_every_row_falls_back_instead_of_raising(self):
        proposer = _proposer()
        proposer._pool = self._exhausted_pool(proposer)
        records = [_RequestContext(owner=_state([1, 2, 3])) for _ in range(3)]
        pending = [
            _SpanWrite(
                req_id=f"r{i}",
                record=record,
                start_row=i,
                start_pos=0,
                count=1,
                end_pos=8,
            )
            for i, record in enumerate(records)
        ]
        # Must return, not raise: the engine step cannot recover from an exception here.
        proposer._flush_span_writes(mx.zeros((3, 1)), pending)
        assert [r.disabled_reason for r in records] == [
            "context capacity exhausted"
        ] * 3
        assert all(r.covered_end == 0 for r in records)

    def test_the_pool_is_left_whole_when_a_batch_is_refused(self):
        proposer = _proposer()
        pool = self._exhausted_pool(proposer)
        proposer._pool = pool
        free_before = pool.free_blocks
        record = _RequestContext(owner=_state([1, 2, 3]))
        proposer._flush_span_writes(
            mx.zeros((1, 1)),
            [
                _SpanWrite(
                    req_id="r",
                    record=record,
                    start_row=0,
                    start_pos=0,
                    count=1,
                    end_pos=8,
                )
            ],
        )
        assert pool.free_blocks == free_before
        assert pool.table_for("r") == []


class TestTheProposerBuildsTheBackendThePlannerReservedFor:
    """One decision, made once.

    `cache_policy` subtracts the plan's reserve from the target model's KV cache before
    the proposer exists. If the proposer then chose its backend independently, a planner
    that sized for the arena could be paired with a pool, or the reverse, and the target
    cache would be wrong in whichever direction. The plan's `paged_blocks` IS the choice.
    """

    def _build(self, monkeypatch, *, env_at_plan: bool, env_at_build: bool):
        config = replace(
            DSparkConfig.from_dict(draft_hf_config().to_dict()), head_dim=64
        )
        if env_at_plan:
            monkeypatch.setenv("VLLM_METAL_DSPARK_PAGED_CONTEXT", "1")
        else:
            monkeypatch.delenv("VLLM_METAL_DSPARK_PAGED_CONTEXT", raising=False)
        plan = DSparkMemoryPlan.build(
            config,
            itemsize=4,
            max_num_seqs=4,
            max_model_len=256,
            max_num_batched_tokens=512,
        )
        if env_at_build:
            monkeypatch.setenv("VLLM_METAL_DSPARK_PAGED_CONTEXT", "1")
        else:
            monkeypatch.delenv("VLLM_METAL_DSPARK_PAGED_CONTEXT", raising=False)
        proposer = DSparkProposer(
            drafter=DSparkDrafter(config),
            config=config,
            runner=make_stub_runner(),
            controller=SpeculativeDecodeController(),
            memory_plan=plan,
        )
        return plan, proposer

    def test_the_plan_and_the_backend_agree(self, monkeypatch):
        for enabled in (False, True):
            plan, proposer = self._build(
                monkeypatch, env_at_plan=enabled, env_at_build=enabled
            )
            assert bool(plan.paged_blocks) is enabled
            assert (proposer._pool is not None) is enabled

    def test_the_env_flipping_after_planning_does_not_change_the_backend(
        self, monkeypatch
    ):
        # The plan reserved for the arena. Whatever the environment says now, building a
        # pool here would use memory the target model was already given.
        plan, proposer = self._build(monkeypatch, env_at_plan=False, env_at_build=True)
        assert plan.paged_blocks == 0
        assert proposer._pool is None

    def test_the_pool_gets_exactly_the_pages_the_plan_reserved(self, monkeypatch):
        plan, proposer = self._build(monkeypatch, env_at_plan=True, env_at_build=True)
        assert proposer._pool is not None
        assert proposer._pool.total_blocks == plan.paged_blocks
