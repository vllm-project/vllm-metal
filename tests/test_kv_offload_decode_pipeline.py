# SPDX-License-Identifier: Apache-2.0
"""KV connector step lifecycle on the pipelined (deferred) decode path.

``sample_tokens`` has three exits. The two synchronous ones close the KV
connector step and stamp its ``KVConnectorOutput`` onto the returned
``ModelRunnerOutput``. The deferred exit returns a
``MetalAsyncModelRunnerOutput`` and the real output is built one step later
inside ``DecodePipeline._resolve_pending``. These tests pin the two things
the scheduler depends on for that exit: the resolved output carries the
connector output, and the step context does not leak into the next step.
"""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import pytest
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput

import vllm_metal.v1.model_runner as mr
from tests.kv_connector_spy import SpyTransferGroup
from tests.stub_runner import make_stub_runner

FINISHED_RECVING = {"r1"}
INVALID_BLOCK_IDS = {7, 11}


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
    # The scheduler sets this whenever a KV connector is configured.
    scheduler_output.kv_connector_metadata = SimpleNamespace()
    return scheduler_output


def _decode_state() -> mr.RequestState:
    return mr.RequestState(
        token_ids=[3, 9],
        prompt_len=1,
        sampling_params=SamplingParams(temperature=0.0),
        generator=None,
        generated_tokens=1,
    )


def _paged_state(decode_reqs, scheduler_output) -> mr._PagedForwardState:
    return mr._PagedForwardState(
        batch=mr._ExecutionBatch(),
        prefill_reqs=[],
        decode_reqs=list(decode_reqs),
        scheduler_output=scheduler_output,
        logits=mx.array([[[0.0, 10.0, 0.0, 0.0]] * len(decode_reqs)]),
        target_hidden_states=None,
        pooling_hidden_states=None,
        cu_seqlens=list(range(len(decode_reqs) + 1)),
        # Pure decode projects every row, so the logits boundaries are the
        # packed ones (#590).
        logits_cu_seqlens=list(range(len(decode_reqs) + 1)),
        decode_segments=(),
        num_decode_tokens=len(decode_reqs),
        mm_prefill_deltas={},
    )


class _ConnectorEnv:
    """Spy connector whose step output is tagged per step."""

    def __init__(self, group: SpyTransferGroup) -> None:
        self.group = group
        group.finished_recving = set(FINISHED_RECVING)
        group.invalid_block_ids = set(INVALID_BLOCK_IDS)
        self.tag = "s0"

    @property
    def tag(self) -> str:
        return self._tag

    @tag.setter
    def tag(self, value: str) -> None:
        self._tag = value
        self.group.finished_sending = {value}


@pytest.fixture
def connector_env(spy_group: SpyTransferGroup) -> _ConnectorEnv:
    """Make the runner's connector step runnable off-device.

    ``_kv_connector_start_step`` and ``MetalKVConnector`` run for real, since
    the step's lifetime is under test. Only the transfer group is a spy.
    """
    return _ConnectorEnv(spy_group)


def _arm_eligible_step(runner, scheduler_output) -> None:
    """Put an existing runner on a pipeline-eligible pure-decode step."""
    state = _decode_state()
    runner._paged_request_seq_lens = {"r0": 1}
    runner._execute_model_state = _paged_state([("r0", state)], scheduler_output)
    runner._decode_pipeline.begin_step(
        mr.PipelineGateDecision(eligible=True, reason="eligible")
    )


def _runner_on_eligible_step(scheduler_output) -> mr.MetalModelRunner:
    runner = make_stub_runner(model=SimpleNamespace())
    _arm_eligible_step(runner, scheduler_output)
    return runner


class TestDeferredDecodeCarriesConnectorOutput:
    """The deferred exit must not drop the connector's step results."""

    def test_resolved_output_carries_kv_connector_output(self, connector_env):
        # Arrange: a KV connector step is open for a pipeline-eligible
        # pure-decode step, exactly as execute_model leaves it.
        scheduler_output = _scheduler_output(["r0"])
        runner = _runner_on_eligible_step(scheduler_output)
        runner._kv_connector_start_step(scheduler_output)

        # Act: the engine takes the async output and resolves it a step later.
        async_output = runner.sample_tokens(grammar_output=None)
        assert isinstance(async_output, mr.MetalAsyncModelRunnerOutput)
        output = async_output.get_output()

        # Assert: the scheduler reads finished/invalid sets off this output.
        assert output.kv_connector_output is not None
        assert output.kv_connector_output.finished_sending == {connector_env.tag}
        assert output.kv_connector_output.finished_recving == FINISHED_RECVING
        assert output.kv_connector_output.invalid_block_ids == INVALID_BLOCK_IDS

    def test_step_context_is_closed_once_the_deferred_step_is_submitted(
        self, connector_env
    ):
        # Arrange
        scheduler_output = _scheduler_output(["r0"])
        runner = _runner_on_eligible_step(scheduler_output)
        runner._kv_connector_start_step(scheduler_output)

        # Act
        runner.sample_tokens(grammar_output=None)

        # Assert - the deferred step left no context open for the next step
        # to find. Asserted directly rather than via the leak log, which is
        # wording that can change.
        assert runner.finish_kv_connector_step() is None

    def test_each_pipelined_step_carries_its_own_connector_output(self, connector_env):
        """Two steps in flight must not swap or share their connector output."""
        runner = make_stub_runner(model=SimpleNamespace())

        outputs = []
        for step in ("a", "b"):
            connector_env.tag = step
            scheduler_output = _scheduler_output(["r0"])
            _arm_eligible_step(runner, scheduler_output)
            runner._kv_connector_start_step(scheduler_output)
            outputs.append(runner.sample_tokens(grammar_output=None))

        resolved = [o.get_output().kv_connector_output for o in outputs]
        assert [r.finished_sending for r in resolved] == [{"a"}, {"b"}]
