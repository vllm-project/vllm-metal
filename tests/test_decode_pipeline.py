# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the one-step-ahead decode pipeline owner.

Covers the gate decision table, the submit/resolve token backfill through
direct RequestState references, the barrier-on-ineligible transition, the
device gather permutation, and the fail-fast rejections.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mlx.core as mx
import pytest
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput

from vllm_metal.v1.decode_pipeline import (
    PENDING_TOKEN_PLACEHOLDER,
    DecodePipeline,
    MetalAsyncModelRunnerOutput,
    PendingBackfillEntry,
    PendingSampleStep,
    RunnerCapabilities,
    SamplingShape,
    SchedulerStepShape,
)

_CAPABLE_KWARGS = {
    "pipeline_enabled": True,
    "use_async_scheduling": True,
    "paged_runtime_active": True,
    "is_pooling": False,
    "pp_active": False,
    "hybrid_without_lazy_gdn": False,
    "spec_decode_configured": False,
    "uniproc_executor": True,
}

_CLEAN_STEP_KWARGS = {
    "has_new_requests": False,
    "has_prefill_phase_requests": False,
    "has_resumed_requests": False,
    "has_preempted_requests": False,
    "has_encoder_inputs": False,
    "has_spec_tokens": False,
    "has_structured_output": False,
    "has_mm_decode": False,
    "decode_req_ids": (),
}

_GREEDY_SAMPLING_KWARGS = {
    "native_greedy": True,
    "has_prompt_logprobs": False,
}


def _gate(
    pipeline: DecodePipeline,
    capability_overrides: dict | None = None,
    step_overrides: dict | None = None,
    sampling_overrides: dict | None = None,
):
    return pipeline.evaluate_gate(
        RunnerCapabilities(**{**_CAPABLE_KWARGS, **(capability_overrides or {})}),
        SchedulerStepShape(**{**_CLEAN_STEP_KWARGS, **(step_overrides or {})}),
        SamplingShape(**{**_GREEDY_SAMPLING_KWARGS, **(sampling_overrides or {})}),
    )


@dataclass
class _State:
    """Minimal RequestState stand-in: only token_ids is touched by resolve."""

    token_ids: list[int] = field(default_factory=list)


@dataclass
class _Batch:
    """Minimal _ExecutionBatch stand-in for output-slot bookkeeping."""

    req_ids: list[str] = field(default_factory=list)
    sampled_tokens: list[list[int]] = field(default_factory=list)

    def add_output(self, req_id: str, token_ids: list[int]) -> int:
        self.req_ids.append(req_id)
        self.sampled_tokens.append(token_ids)
        return len(self.req_ids) - 1

    def set_output(self, output_idx: int, token_ids: list[int]) -> None:
        self.sampled_tokens[output_idx] = token_ids


@dataclass(frozen=True)
class _Segment:
    """Minimal PagedDecodeSegment stand-in for input assembly."""

    req_id: str
    num_query_tokens: int = 1


def _make_scheduler_output(req_ids: list[str]) -> SchedulerOutput:
    return SchedulerOutput(
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


def _make_pipeline() -> tuple[DecodePipeline, list]:
    resolved_batches: list = []

    def build_output(batch):
        resolved_batches.append(batch)
        return batch

    def validate(batch, scheduler_output):
        del batch, scheduler_output

    pipeline = DecodePipeline(build_output=build_output, validate=validate)
    return pipeline, resolved_batches


def _eligible(pipeline: DecodePipeline) -> None:
    pipeline.begin_step(_gate(pipeline))


def _submit_step(
    pipeline: DecodePipeline,
    token_values: list[int],
    req_ids: list[str],
) -> tuple[MetalAsyncModelRunnerOutput, list[_State], _Batch]:
    states = []
    batch = _Batch()
    entries = []
    for row, req_id in enumerate(req_ids):
        state = _State(token_ids=[7, 8, PENDING_TOKEN_PLACEHOLDER])
        states.append(state)
        output_idx = batch.add_output(req_id, [])
        entries.append(
            PendingBackfillEntry(
                req_id=req_id,
                state=state,
                row=row,
                token_index=2,
                output_idx=output_idx,
            )
        )
    tokens = mx.array(token_values, dtype=mx.uint32)
    handle = pipeline.submit(
        PendingSampleStep(
            tokens=tokens,
            entries=tuple(entries),
            batch=batch,
            scheduler_output=_make_scheduler_output(req_ids),
        )
    )
    return handle, states, batch


class TestGate:
    """Decision table for the pipeline eligibility gate."""

    def test_pure_decode_greedy_step_is_eligible(self):
        pipeline, _ = _make_pipeline()
        decision = _gate(pipeline)
        assert decision.eligible
        assert decision.reason == "eligible"

    @pytest.mark.parametrize(
        ("flag", "value", "reason"),
        [
            ("pipeline_enabled", False, "pipeline disabled"),
            ("use_async_scheduling", False, "async scheduling off"),
            ("paged_runtime_active", False, "non-paged path"),
            ("is_pooling", True, "pooling model"),
            ("pp_active", True, "pipeline parallelism"),
            (
                "hybrid_without_lazy_gdn",
                True,
                "hybrid model without lazy GDN kernels",
            ),
            ("spec_decode_configured", True, "speculative decode"),
            ("uniproc_executor", False, "non-uniproc executor"),
        ],
    )
    def test_each_capability_blocker_rejects_with_its_reason(self, flag, value, reason):
        pipeline, _ = _make_pipeline()
        decision = _gate(pipeline, capability_overrides={flag: value})
        assert not decision.eligible
        assert decision.reason == reason

    @pytest.mark.parametrize(
        ("flag", "value", "reason"),
        [
            ("has_new_requests", True, "new requests scheduled"),
            ("has_prefill_phase_requests", True, "prefill-phase requests"),
            ("has_resumed_requests", True, "resumed requests"),
            ("has_preempted_requests", True, "preempted requests"),
            ("has_encoder_inputs", True, "encoder inputs scheduled"),
            ("has_spec_tokens", True, "speculative decode"),
            ("has_structured_output", True, "structured output"),
            ("has_mm_decode", True, "multimodal decode state"),
        ],
    )
    def test_each_step_blocker_rejects_with_its_reason(self, flag, value, reason):
        pipeline, _ = _make_pipeline()
        decision = _gate(pipeline, step_overrides={flag: value})
        assert not decision.eligible
        assert decision.reason == reason

    @pytest.mark.parametrize(
        ("flag", "value", "reason"),
        [
            ("native_greedy", False, "non-native-greedy sampling"),
            ("has_prompt_logprobs", True, "prompt logprobs requested"),
        ],
    )
    def test_each_sampling_blocker_rejects_with_its_reason(self, flag, value, reason):
        pipeline, _ = _make_pipeline()
        decision = _gate(pipeline, sampling_overrides={flag: value})
        assert not decision.eligible
        assert decision.reason == reason

    def test_reentrant_decode_request_without_pending_row_is_blocked(self):
        # Arrange — a pending step covers only "a"; a cached request "b"
        # re-enters after being absent, so it has no pending lazy token.
        pipeline, _ = _make_pipeline()
        _eligible(pipeline)
        _submit_step(pipeline, [11], ["a"])

        # Act
        decision = _gate(pipeline, step_overrides={"decode_req_ids": ("a", "b")})

        # Assert
        assert not decision.eligible
        assert decision.reason == "cached request without a pending row"

    def test_shrunk_decode_membership_stays_eligible(self):
        # Arrange — "b" finished between steps; a subset of the pending rows
        # is fine (the assembly gathers the surviving rows).
        pipeline, _ = _make_pipeline()
        _eligible(pipeline)
        _submit_step(pipeline, [11, 22], ["a", "b"])

        # Act
        decision = _gate(pipeline, step_overrides={"decode_req_ids": ("a",)})

        # Assert
        assert decision.eligible

    def test_decode_membership_is_unchecked_without_pending_step(self):
        pipeline, _ = _make_pipeline()
        decision = _gate(pipeline, step_overrides={"decode_req_ids": ("a", "b")})
        assert decision.eligible


class TestSubmitResolve:
    """Deferred token backfill and output pairing."""

    def test_resolve_backfills_tokens_through_state_references(self):
        # Arrange
        pipeline, _ = _make_pipeline()
        _eligible(pipeline)

        # Act
        handle, states, batch = _submit_step(pipeline, [11, 22], ["a", "b"])
        output = handle.get_output()

        # Assert
        assert states[0].token_ids == [7, 8, 11]
        assert states[1].token_ids == [7, 8, 22]
        assert batch.sampled_tokens == [[11], [22]]
        assert output is batch

    def test_submit_resolves_previous_step_before_registering_next(self):
        # Arrange
        pipeline, resolved = _make_pipeline()
        _eligible(pipeline)
        first_handle, first_states, first_batch = _submit_step(pipeline, [11], ["a"])

        # Act: engine submits step k+1, then pops step k's output.
        _eligible(pipeline)
        _submit_step(pipeline, [33], ["a"])
        first_output = first_handle.get_output()

        # Assert
        assert first_states[0].token_ids == [7, 8, 11]
        assert resolved == [first_batch]
        assert first_output is first_batch

    def test_get_output_twice_raises(self):
        # Arrange
        pipeline, _ = _make_pipeline()
        _eligible(pipeline)
        handle, _, _ = _submit_step(pipeline, [11], ["a"])
        handle.get_output()

        # Act / Assert
        with pytest.raises(RuntimeError, match="no output for serial"):
            handle.get_output()

    def test_resolve_rejects_overwritten_placeholder(self):
        # Arrange
        pipeline, _ = _make_pipeline()
        _eligible(pipeline)
        handle, states, _ = _submit_step(pipeline, [11], ["a"])
        states[0].token_ids[2] = 99  # desync: something wrote the slot

        # Act / Assert
        with pytest.raises(RuntimeError, match="overwritten before resolve"):
            handle.get_output()

    def test_third_submit_without_consuming_output_raises(self):
        # The engine (queue depth 2) pops step k's output before submitting
        # step k+2; a third unconsumed submit must fail fast instead of
        # silently overwriting the cached resolved output.
        # Arrange
        pipeline, _ = _make_pipeline()
        _eligible(pipeline)
        _submit_step(pipeline, [11], ["a"])
        _eligible(pipeline)
        _submit_step(pipeline, [22], ["a"])

        # Act / Assert
        _eligible(pipeline)
        with pytest.raises(RuntimeError, match="slot still occupied"):
            _submit_step(pipeline, [33], ["a"])

    def test_submit_on_ineligible_step_raises(self):
        # Arrange
        pipeline, _ = _make_pipeline()
        pipeline.begin_step(
            _gate(pipeline, step_overrides={"has_structured_output": True})
        )

        # Act / Assert
        with pytest.raises(RuntimeError, match="gate-ineligible"):
            _submit_step(pipeline, [11], ["a"])


class TestBarrier:
    """Ineligible steps resolve the pending step before any state mutation."""

    def test_begin_step_barriers_pending_on_ineligible_transition(self):
        # Arrange
        pipeline, resolved = _make_pipeline()
        _eligible(pipeline)
        handle, states, batch = _submit_step(pipeline, [11], ["a"])

        # Act: next step brings a prefill — gate ineligible; then the engine
        # pops the deferred output.
        pipeline.begin_step(
            _gate(pipeline, step_overrides={"has_prefill_phase_requests": True})
        )
        output = handle.get_output()

        # Assert: token already backfilled, pending drained, output cached.
        assert states[0].token_ids == [7, 8, 11]
        assert not pipeline.has_pending
        assert resolved == [batch]
        assert output is batch


class TestAssembleInputIds:
    """Device-side decode input assembly from the pending lazy tokens."""

    def test_returns_none_without_pending_step(self):
        pipeline, _ = _make_pipeline()
        _eligible(pipeline)
        assert pipeline.assemble_decode_input_ids([_Segment("a")]) is None

    def test_identity_order_feeds_tokens_directly(self):
        # Arrange
        pipeline, _ = _make_pipeline()
        _eligible(pipeline)
        _submit_step(pipeline, [11, 22], ["a", "b"])
        _eligible(pipeline)

        # Act
        input_ids = pipeline.assemble_decode_input_ids([_Segment("a"), _Segment("b")])

        # Assert
        assert input_ids is not None
        assert input_ids.dtype == mx.int32
        assert input_ids.tolist() == [[11, 22]]

    def test_permuted_membership_gathers_previous_rows(self):
        # Arrange: request "b" finished; "c" is not in the pending step.
        pipeline, _ = _make_pipeline()
        _eligible(pipeline)
        _submit_step(pipeline, [11, 22, 33], ["a", "b", "c"])
        _eligible(pipeline)

        # Act: next step decodes (c, a) — a permuted subset.
        input_ids = pipeline.assemble_decode_input_ids([_Segment("c"), _Segment("a")])

        # Assert: values equal the reference gather, not the identity order.
        assert input_ids is not None
        assert input_ids.tolist() == [[33, 11]]
        assert input_ids.tolist() != [[11, 22]]

    def test_unmapped_request_raises(self):
        # Arrange
        pipeline, _ = _make_pipeline()
        _eligible(pipeline)
        _submit_step(pipeline, [11], ["a"])
        _eligible(pipeline)

        # Act / Assert
        with pytest.raises(RuntimeError, match="no pending-token row"):
            pipeline.assemble_decode_input_ids([_Segment("zzz")])

    def test_multi_token_segment_raises(self):
        # Arrange
        pipeline, _ = _make_pipeline()
        _eligible(pipeline)
        _submit_step(pipeline, [11], ["a"])
        _eligible(pipeline)

        # Act / Assert
        with pytest.raises(RuntimeError, match="single-token segments"):
            pipeline.assemble_decode_input_ids([_Segment("a", num_query_tokens=2)])
