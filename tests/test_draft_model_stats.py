# SPDX-License-Identifier: Apache-2.0
"""Draft fallback counters exposed through the worker RPC boundary."""

from types import SimpleNamespace

import pytest

from tests.stub_draft_model import StubDraftModel
from tests.stub_runner import adopt_draft_group, make_stub_runner
from tests.test_draft_model_proposer import (
    _context,
    _proposer,
    _request_state,
)
from vllm_metal.v1.worker import MetalWorker


def _worker(runner):
    worker = MetalWorker.__new__(MetalWorker)
    worker.model_runner = runner
    return worker


@pytest.mark.parametrize("min_draft_tokens", [0, 1, 3])
def test_worker_reports_effective_limits_and_detached_snapshots(min_draft_tokens):
    proposer = _proposer(
        StubDraftModel(),
        max_model_len=4096,
        min_speculative_tokens=min_draft_tokens,
    )
    adopt_draft_group(proposer, 0, 32)
    worker = _worker(make_stub_runner(_drafter=proposer))
    expected = {
        "num_context_limit_fallback_requests": 0,
        "min_draft_tokens": min_draft_tokens,
        "max_model_len": 32,
    }
    snapshot = worker.get_draft_model_stats()
    assert snapshot == expected
    snapshot["num_context_limit_fallback_requests"] = -1
    snapshot["max_model_len"] = -1
    assert worker.get_draft_model_stats() == expected


def test_counter_is_cumulative_after_cleanup_and_independent_of_log_level(caplog):
    proposer = _proposer(StubDraftModel(), max_model_len=32, min_speculative_tokens=3)
    runner = make_stub_runner(_drafter=proposer)
    worker = _worker(runner)
    caplog.set_level("ERROR", logger="vllm_metal.v1.draft_model_proposer")
    for expected_count in (1, 2):
        state = _request_state(scheduler_block_ids=[0, 1], token_ids=list(range(31)))
        runner._request_states["r"] = state
        ctx = _context("r", state, runner._request_states, num_speculative_tokens=3)
        assert proposer.propose(ctx) is None
        assert proposer.propose(ctx) is None
        runner._reconcile_request_lifecycle({"r"})
        # Reads after completion preserve the total and do not reset it.
        first = worker.get_draft_model_stats()
        assert first["num_context_limit_fallback_requests"] == expected_count
        assert worker.get_draft_model_stats() == first
    assert not caplog.records
    fresh = _proposer(StubDraftModel())
    assert fresh.get_stats()["num_context_limit_fallback_requests"] == 0


@pytest.mark.parametrize("drafter", [None, SimpleNamespace()])
def test_other_decoding_methods_have_no_draft_model_stats(drafter):
    assert _worker(make_stub_runner(_drafter=drafter)).get_draft_model_stats() is None


def test_non_generation_runner_has_no_draft_model_stats():
    from vllm_metal.v1.stt_model_runner import STTModelRunner

    assert (
        _worker(STTModelRunner.__new__(STTModelRunner)).get_draft_model_stats() is None
    )
