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
a deliberate raise. That pins the call order itself, and leaves the step open
as a real failure would. The cross-step tests chain steps to pin the lifecycle
across a step boundary, which no single ``execute_model`` call can show. The
real ``MetalKVConnector`` runs throughout, over a spy transfer group.
"""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import pytest
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput

import vllm_metal.v1.model_runner as mr
from tests.stub_runner import make_stub_runner


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


def test_execute_model_handles_preemptions_before_opening_the_step(spy_group):
    """The store fence depends on this order inside execute_model itself."""
    runner = make_stub_runner(model=SimpleNamespace())
    spy_group.step = "1"

    _drive_execute_model_prologue(runner, _scheduler_output(["r0"]))

    assert spy_group.events == ["preempt:1", "open:1"]


def test_zero_token_step_still_handles_preemptions(spy_group):
    """A step with no forward must still let KV transfers progress."""
    runner = make_stub_runner(model=SimpleNamespace())
    spy_group.step = "1"

    runner.execute_model(_scheduler_output([]))

    # no_forward runs a whole step: preempt, open and close.
    assert spy_group.events == ["preempt:1", "open:1", "close:1"]
    assert not spy_group.metadata_bound


def _pipelined_decode_step(runner, spy_group, tag: str) -> None:
    """One pipeline-eligible decode step, in execute_model's own order."""
    spy_group.step = tag
    scheduler_output = _scheduler_output(["r0"])

    # As execute_model: the step start handles preemptions, then opens.
    runner._kv_connector_start_step(scheduler_output)

    state = mr.RequestState(
        token_ids=[3, 9],
        prompt_len=1,
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
        decode_segments=(),
        num_decode_tokens=1,
        mm_prefill_deltas={},
    )
    runner._decode_pipeline.begin_step(
        mr.PipelineGateDecision(eligible=True, reason="eligible")
    )
    runner.sample_tokens(grammar_output=None)


def test_step_closes_before_the_next_step_handles_preemptions(spy_group):
    """close(k) must precede preempt(k+1), or the flush fence is a no-op."""
    runner = make_stub_runner(model=SimpleNamespace())

    _pipelined_decode_step(runner, spy_group, "1")
    _pipelined_decode_step(runner, spy_group, "2")

    assert spy_group.events == [
        "preempt:1",
        "open:1",
        "close:1",
        "preempt:2",
        "open:2",
        "close:2",
    ]


def test_zero_token_step_after_a_pipelined_step_does_not_crash(spy_group):
    """A pipelined step closes at submit, so a zero-token step after it runs
    against no open step. See the leaked-step tests below for a step that
    was never closed."""
    runner = make_stub_runner(model=SimpleNamespace())

    _pipelined_decode_step(runner, spy_group, "1")

    # Step 2 schedules nothing: execute_model takes the no-forward path and
    # sample_tokens is never called.
    spy_group.step = "2"
    runner.execute_model(_scheduler_output([]))

    # Step 3 opens its own step; nothing is left open to recover.
    spy_group.step = "3"
    runner._kv_connector_start_step(_scheduler_output(["r0"]))

    assert "close:1" in spy_group.events
    assert spy_group.events.index("close:1") < spy_group.events.index("preempt:2")


def test_sample_tokens_without_pending_state_closes_the_step(spy_group):
    """execute_model can fail after opening the step; sample_tokens ends it."""
    runner = make_stub_runner(model=SimpleNamespace())
    spy_group.step = "1"
    _drive_execute_model_prologue(runner, _scheduler_output(["r0"]))
    runner._execute_model_state = None

    assert runner.sample_tokens(grammar_output=None) is None

    assert spy_group.events == ["preempt:1", "open:1", "close:1"]
    # Worker shutdown closes again; the step must not close twice.
    assert runner.finish_kv_connector_step() is None
    assert spy_group.events.count("close:1") == 1


def test_leaked_step_closes_before_the_next_step_handles_preemptions(spy_group):
    """execute_model raised after opening step 1, so nothing closed it."""
    runner = make_stub_runner(model=SimpleNamespace())
    spy_group.step = "1"
    _drive_execute_model_prologue(runner, _scheduler_output(["r0"]))

    spy_group.step = "2"
    _drive_execute_model_prologue(runner, _scheduler_output(["r0"]))

    assert spy_group.events == ["preempt:1", "open:1", "close:1", "preempt:2", "open:2"]


def test_leaked_step_closes_before_a_zero_token_step(spy_group):
    """A zero-token step after a leaked one must not handle its preemptions
    first, and must not clear the leaked step's metadata before its close."""
    runner = make_stub_runner(model=SimpleNamespace())
    spy_group.step = "1"
    _drive_execute_model_prologue(runner, _scheduler_output(["r0"]))

    spy_group.step = "2"
    runner.execute_model(_scheduler_output([]))
    spy_group.step = "3"
    _drive_execute_model_prologue(runner, _scheduler_output(["r0"]))

    assert spy_group.events == [
        "preempt:1",
        "open:1",
        "close:1",
        "preempt:2",
        "open:2",
        "close:2",
        "preempt:3",
        "open:3",
    ]


def test_each_close_tells_the_connector_the_forward_is_done(spy_group):
    """finish_forward runs once per closed step."""
    runner = make_stub_runner(model=SimpleNamespace())
    spy_group.step = "1"
    _drive_execute_model_prologue(runner, _scheduler_output(["r0"]))
    assert spy_group.finish_forward_calls == 0

    runner.finish_kv_connector_step()
    runner.finish_kv_connector_step()

    assert spy_group.finish_forward_calls == 1


@pytest.fixture
def no_connector(monkeypatch) -> None:
    """No KV connector configured: any connector call fails the test."""

    def forbidden(*args, **kwargs):
        raise AssertionError("connector touched without a KV connector")

    import vllm_metal.v1.kv_connector as metal_kv_connector

    monkeypatch.setattr(mr, "has_kv_transfer_group", lambda: False)
    monkeypatch.setattr(mr, "get_kv_transfer_group", forbidden)
    monkeypatch.setattr(metal_kv_connector, "get_kv_transfer_group", forbidden)


def test_zero_token_step_without_connector_is_unchanged(no_connector):
    runner = make_stub_runner(model=SimpleNamespace())

    output = runner.execute_model(_scheduler_output([]))

    assert output is not None
    assert output.kv_connector_output is None


def test_pipelined_step_without_connector_carries_no_connector_output(
    no_connector,
):
    runner = make_stub_runner(model=SimpleNamespace())
    scheduler_output = _scheduler_output(["r0"])
    state = mr.RequestState(
        token_ids=[3, 9],
        prompt_len=1,
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
        decode_segments=(),
        num_decode_tokens=1,
        mm_prefill_deltas={},
    )
    runner._decode_pipeline.begin_step(
        mr.PipelineGateDecision(eligible=True, reason="eligible")
    )

    async_output = runner.sample_tokens(grammar_output=None)

    assert isinstance(async_output, mr.MetalAsyncModelRunnerOutput)
    output = async_output.get_output()
    assert output.sampled_token_ids == [[1]]
    assert output.kv_connector_output is None


def _sdpa_runtime():
    from vllm_metal.attention.runtime.sdpa import SDPAPagedAttentionRuntime

    return SDPAPagedAttentionRuntime(
        num_layers=1, num_kv_heads=1, head_dim=4, block_size=4, dtype=mx.float32
    )


def test_register_hands_the_runtime_storage_to_the_connector(monkeypatch):
    registered = []
    monkeypatch.setattr(
        mr,
        "get_kv_transfer_group",
        lambda: SimpleNamespace(register_kv_caches=registered.append),
    )
    runtime = _sdpa_runtime()
    storage = object()
    runtime._storage = storage
    runner = make_stub_runner(model=SimpleNamespace())
    runner._paged_attention_runtime = runtime

    runner.register_kv_connector_caches()

    assert registered == [storage]


@pytest.mark.parametrize(
    "runtime_factory",
    [
        pytest.param(lambda: None, id="no-runtime"),
        pytest.param(SimpleNamespace, id="non-sdpa-runtime"),
        # Count-initialized (draft-model) caches never bind storage.
        pytest.param(_sdpa_runtime, id="sdpa-without-storage"),
    ],
)
def test_register_rejects_runtimes_without_storage(monkeypatch, runtime_factory):
    def forbidden():
        raise AssertionError("registered a runtime without storage")

    monkeypatch.setattr(mr, "get_kv_transfer_group", forbidden)
    runner = make_stub_runner(model=SimpleNamespace())
    runner._paged_attention_runtime = runtime_factory()

    with pytest.raises(NotImplementedError, match="KV offloading on Metal"):
        runner.register_kv_connector_caches()
