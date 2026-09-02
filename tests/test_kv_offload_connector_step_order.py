# SPDX-License-Identifier: Apache-2.0
"""Step-to-step ordering of the KV connector context on the pipelined path.

Upstream's store fencing depends on an ordering across step boundaries. Step
k's context close runs ``prepare_store_kv``, which queues that step's store
jobs; step k+1's ``handle_preemptions`` then submits them, and ``submit_store``
performs the copy synchronously. If step k's context is still open at that
point its jobs are not queued yet, so the copy runs a step later, against
blocks the intervening forward has overwritten. That is a silent wrong-KV
write into the offload pool, so the ordering is a correctness contract.

Two levels are covered. ``execute_model`` is driven for real, far enough to
record that it handles preemptions before opening the step, then stopped with
a deliberate raise; that pins the call order itself. The cross-step tests then
drive the runner's helpers in that same order to pin the lifecycle across a
step boundary, which no single ``execute_model`` call can show.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import mlx.core as mx
import pytest
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.outputs import KVConnectorOutput

import vllm_metal.v1.model_runner as mr
from tests.stub_runner import make_stub_runner


class _ConnectorSpy:
    """Records the connector lifecycle and enforces upstream's invariants.

    Mirrors the two things upstream does that this test is about: metadata is
    bound on enter and cleared on the no-forward path, and ``get_finished``
    asserts the metadata is still bound when the context closes.
    """

    def __init__(self) -> None:
        self.events: list[str] = []
        self.metadata_bound = False
        self.step = "?"

    # -- the runner's collaborators -------------------------------------
    def handle_preemptions(self, metadata) -> None:
        del metadata
        self.events.append(f"preempt:{self.step}")

    def no_forward(self, *args, **kwargs):
        del args, kwargs
        self.events.append(f"no_forward:{self.step}")
        self.metadata_bound = False  # upstream clears on this path
        return mr.EMPTY_MODEL_RUNNER_OUTPUT

    @contextmanager
    def step_context(self, scheduler_output, *args, **kwargs):
        del scheduler_output, args, kwargs
        self.metadata_bound = True
        self.events.append(f"open:{self.step}")
        output = KVConnectorOutput()
        try:
            yield output
        finally:
            if not self.metadata_bound:
                raise AssertionError(
                    "connector step closed after its metadata was cleared"
                )
            self.events.append(f"close:{self.step}")
            output.finished_recving = {f"r-{self.step}"}


@pytest.fixture
def spy(monkeypatch) -> _ConnectorSpy:
    connector = _ConnectorSpy()
    monkeypatch.setattr(mr, "has_kv_transfer_group", lambda: True)
    monkeypatch.setattr(mr, "get_kv_transfer_group", lambda: connector)
    monkeypatch.setattr(mr, "set_forward_context", lambda *a, **k: nullcontext())
    monkeypatch.setattr(
        mr.KVConnectorModelRunnerMixin,
        "_get_kv_connector_output",
        staticmethod(connector.step_context),
    )
    monkeypatch.setattr(
        mr.KVConnectorModelRunnerMixin,
        "kv_connector_no_forward",
        staticmethod(connector.no_forward),
    )
    return connector


def _scheduler_output(req_ids: list[str]) -> SchedulerOutput:
    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens=dict.fromkeys(req_ids, 1),
        total_num_scheduled_tokens=len(req_ids),
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
        num_invalid_spec_tokens=None,
        num_spec_tokens_to_schedule=0,
    )
    scheduler_output.kv_connector_metadata = SimpleNamespace()
    return scheduler_output


def _drive_execute_model_prologue(runner, scheduler_output) -> None:
    """Run the real execute_model until just past the connector step opening.

    _handle_new_requests is the first statement after the step opens, so
    raising there stops the drive without a forward, a device, or a live
    connector, while still exercising execute_model's own ordering.
    """

    def stop(*args, **kwargs):
        raise RuntimeError("stop after the connector step opened")

    runner._handle_new_requests = stop
    with pytest.raises(RuntimeError, match="stop after"):
        runner.execute_model(scheduler_output)


def test_execute_model_handles_preemptions_before_opening_the_step(spy):
    """The store fence depends on this order inside execute_model itself."""
    runner = make_stub_runner(model=SimpleNamespace())
    spy.step = "1"

    _drive_execute_model_prologue(runner, _scheduler_output(["r0"]))

    assert spy.events == ["preempt:1", "open:1"]


def test_zero_token_step_still_handles_preemptions(spy):
    """A step with no forward must still let KV transfers progress."""
    runner = make_stub_runner(model=SimpleNamespace())
    spy.step = "1"

    runner.execute_model(_scheduler_output([]))

    assert spy.events == ["preempt:1", "no_forward:1"]


def _pipelined_decode_step(runner, spy, tag: str) -> None:
    """One pipeline-eligible decode step, in execute_model's own order."""
    spy.step = tag
    scheduler_output = _scheduler_output(["r0"])

    # execute_model: preemptions are handled BEFORE the step context opens.
    spy.handle_preemptions(None)
    runner._kv_connector_start_step(scheduler_output)

    state = mr.RequestState(
        token_ids=[3, 9],
        prompt_len=1,
        cache=[],
        sampling_params=SamplingParams(temperature=0.0),
        generator=None,
        generated_tokens=1,
    )
    runner._paged_request_seq_lens = {"r0": 1}
    runner._execute_model_state = mr._PagedForwardState(
        batch=mr._ExecutionBatch(),
        prefill_reqs=[],
        decode_reqs=[("r0", state)],
        scheduler_output=scheduler_output,
        logits=mx.array([[[0.0, 10.0, 0.0, 0.0]]]),
        target_hidden_states=None,
        pooling_hidden_states=None,
        cu_seqlens=[0, 1],
        logits_cu_seqlens=[0, 1],
        decode_segments=[],
        num_decode_tokens=1,
        mm_prefill_deltas={},
    )
    runner._decode_pipeline.begin_step(
        mr.PipelineGateDecision(eligible=True, reason="eligible")
    )
    runner.sample_tokens(grammar_output=None)


def test_step_context_closes_before_the_next_step_handles_preemptions(spy):
    """close(k) must precede preempt(k+1), or the flush fence is a no-op."""
    runner = make_stub_runner(model=SimpleNamespace())

    _pipelined_decode_step(runner, spy, "1")
    _pipelined_decode_step(runner, spy, "2")

    assert spy.events == [
        "preempt:1",
        "open:1",
        "close:1",
        "preempt:2",
        "open:2",
        "close:2",
    ]


def test_zero_token_step_after_a_pipelined_step_does_not_crash(spy):
    """A leaked context plus a zero-token step kills engine core upstream.

    The no-forward path clears the connector metadata, so a later close of the
    still-open previous context trips get_finished's metadata assertion."""
    runner = make_stub_runner(model=SimpleNamespace())

    _pipelined_decode_step(runner, spy, "1")

    # Step 2 schedules nothing: execute_model takes the no-forward path and
    # sample_tokens is never called.
    spy.step = "2"
    spy.handle_preemptions(None)
    spy.no_forward(_scheduler_output([]), runner.vllm_config)

    # Step 3 opens its own context. With step 1 leaked, the recovery close
    # would run against cleared metadata and raise.
    spy.step = "3"
    runner._kv_connector_start_step(_scheduler_output(["r0"]))

    assert "close:1" in spy.events
    assert spy.events.index("close:1") < spy.events.index("no_forward:2")
