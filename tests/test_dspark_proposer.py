# SPDX-License-Identifier: Apache-2.0
"""DSpark context invariants with real tiny drafter projections and KV buffers."""

from __future__ import annotations

from dataclasses import replace

import mlx.core as mx
import pytest
from vllm.sampling_params import SamplingParams

from tests.stub_runner import make_stub_runner
from tests.test_dspark_contracts import draft_hf_config
from vllm_metal.v1.dspark.config import DSparkConfig
from vllm_metal.v1.dspark.memory import DSparkMemoryPlan
from vllm_metal.v1.dspark.model import CtxCache, DSparkDrafter
from vllm_metal.v1.dspark_proposer import DSparkProposer, _DraftPlan, _RequestContext
from vllm_metal.v1.model_runner import PrefillRequest, RequestState
from vllm_metal.v1.proposer import ProposeContext
from vllm_metal.v1.spec_decode import PagedDecodeSegment, SpeculativeDecodeController


def _proposer():
    config = DSparkConfig.from_dict(draft_hf_config().to_dict())
    return DSparkProposer(
        drafter=DSparkDrafter(config),
        config=config,
        runner=make_stub_runner(),
        controller=SpeculativeDecodeController(),
        memory_plan=DSparkMemoryPlan.build(
            config,
            itemsize=4,
            max_num_seqs=32,
            max_model_len=2048,
            max_num_batched_tokens=8192,
        ),
    )


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


@pytest.mark.parametrize(
    "params", [{"temperature": 0.8}, {"repetition_penalty": 1.1}, {"logprobs": 2}]
)
def test_non_greedy_requests_keep_context_without_drafting(params):
    proposer = _proposer()
    state = _state([1, 2, 3], **params)
    assert _seed(proposer, state) is None
    _assert_context(proposer, "r", [0, 1])


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


@pytest.mark.parametrize("order", [(0, 2, 6), (6, 0, 2)])
def test_ragged_and_empty_context_drafts_match_independent_rows(order):
    proposer = _proposer()
    plans, expected = [], []
    for index, length in enumerate(order):
        owner = _state([2] * (length + 1))
        caches = proposer._drafter.make_ctx_cache()
        if length:
            proposer._drafter.update_context(
                _features(list(range(length)))[None], 0, caches
            )
        cap = index + 1
        plan = _DraftPlan(str(length), 2, _RequestContext(owner, caches, length), cap)
        plans.append(plan)
        expected.append(proposer._batch_draft([plan])[1][0])
    before = [
        [(cache._keys, cache._values) for cache in plan.context.caches]
        for plan in plans
    ]
    assert proposer._batch_draft(plans)[1] == expected
    # Draft block/scratch positions must never enter persistent context.
    for plan, original in zip(plans, before, strict=True):
        for cache, (key, value) in zip(plan.context.caches, original, strict=True):
            assert cache._keys is key and cache._values is value


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
