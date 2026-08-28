# SPDX-License-Identifier: Apache-2.0
"""One-step-ahead decode pipelining for the Metal model runner.

Fully serial decode pays graph build + ``mx.async_eval`` encode while the GPU
idles, then blocks on the sampling eval while the CPU idles. This module
defers the sampling sync one step, the same overlap mlx_lm's generate loop
uses: step ``k`` submits a lazy sample over its logits (greedy argmax, or a
native temperature/top-k/top-p graph when ``VLLM_METAL_NATIVE_SAMPLING``
admits the batch) and returns an
:class:`AsyncModelRunnerOutput`; step ``k+1`` builds its graph with the lazy
sampled tokens as decode input (a device gather, no host round-trip) while
the GPU still executes step ``k``; the engine's deferred ``get_output()``
then resolves step ``k`` with a pure wait.

Invariants the runner must keep (pinned by tests):

- ``submit()`` happens only on gate-eligible steps: pure decode, batch-wide
  native greedy, no spec/structured-output/pooling/PP/mm work in flight.
- All value-independent bookkeeping (placeholder append, ``generated_tokens``,
  seq-len advance) is done by the runner BEFORE ``submit()``; ``resolve()``
  only backfills token values through direct ``RequestState`` references and
  never touches runner-owned dicts (a request evicted between submit and
  resolve must not be re-inserted).
- Any ineligible step calls ``begin_step()`` first, which resolves the
  pending step before the runner mutates request state, so the synchronous
  path never observes a ``PENDING_TOKEN_PLACEHOLDER``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import mlx.core as mx
from vllm.logger import init_logger
from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

    from vllm_metal.v1.model_runner import RequestState, _ExecutionBatch
    from vllm_metal.v1.spec_decode import PagedDecodeSegment

logger = init_logger(__name__)

# Appended to ``RequestState.token_ids`` at submit time and backfilled at
# resolve time. Legal token ids are non-negative, so a leaked placeholder is
# detectable everywhere it could surface.
PENDING_TOKEN_PLACEHOLDER: Final[int] = -1


@dataclass(frozen=True)
class RunnerCapabilities:
    """Serve-level facts about the runner; fixed for a server's lifetime."""

    pipeline_enabled: bool
    use_async_scheduling: bool
    paged_runtime_active: bool
    is_pooling: bool
    pp_active: bool
    hybrid_without_lazy_gdn: bool
    spec_decode_configured: bool
    uniproc_executor: bool

    def block_reason(self) -> str | None:
        """First capability that rules the pipeline out, or ``None``."""
        blockers = (
            (not self.pipeline_enabled, "pipeline disabled"),
            (not self.use_async_scheduling, "async scheduling off"),
            (not self.paged_runtime_active, "non-paged path"),
            (self.is_pooling, "pooling model"),
            (self.pp_active, "pipeline parallelism"),
            (
                self.hybrid_without_lazy_gdn,
                "hybrid model without lazy GDN kernels",
            ),
            (self.spec_decode_configured, "speculative decode"),
            # Positive check: the pipeline's pending-step state assumes the
            # in-process uniproc executor's single-threaded output ordering;
            # mp/ray/custom executors do not share that contract.
            (not self.uniproc_executor, "non-uniproc executor"),
        )
        for blocked, reason in blockers:
            if blocked:
                return reason
        return None


@dataclass(frozen=True)
class SchedulerStepShape:
    """Shape of one scheduled step as handed over by the scheduler."""

    has_new_requests: bool
    has_prefill_phase_requests: bool
    has_resumed_requests: bool
    has_preempted_requests: bool
    has_encoder_inputs: bool
    has_spec_tokens: bool
    has_structured_output: bool
    has_mm_decode: bool
    # Decode-phase membership of this step: the pipeline cross-checks it
    # against its pending rows (a cached request can re-enter after being
    # absent from the previous step and then has no pending row).
    decode_req_ids: tuple[str, ...]

    def block_reason(self) -> str | None:
        """First step-shape fact that rules the pipeline out, or ``None``."""
        blockers = (
            (self.has_new_requests, "new requests scheduled"),
            (self.has_prefill_phase_requests, "prefill-phase requests"),
            (self.has_resumed_requests, "resumed requests"),
            (self.has_preempted_requests, "preempted requests"),
            (self.has_encoder_inputs, "encoder inputs scheduled"),
            (self.has_spec_tokens, "speculative decode"),
            (self.has_structured_output, "structured output"),
            (self.has_mm_decode, "multimodal decode state"),
        )
        for blocked, reason in blockers:
            if blocked:
                return reason
        return None


@dataclass(frozen=True)
class SamplingShape:
    """Sampling-side facts for the decode requests of one step."""

    native_greedy: bool
    native_random: bool
    has_prompt_logprobs: bool

    def block_reason(self) -> str | None:
        """First sampling fact that rules the pipeline out, or ``None``.

        Checked in order: native sampling mode first, then prompt logprobs.
        Either native mode (greedy argmax or eligible random sampling with
        ``VLLM_METAL_NATIVE_SAMPLING``) keeps the step deferrable.
        """
        if not (self.native_greedy or self.native_random):
            return "non-native sampling"
        # Prompt logprobs need eager logits on the sampling step.
        if self.has_prompt_logprobs:
            return "prompt logprobs requested"
        return None


@dataclass(frozen=True)
class PipelineGateDecision:
    """Outcome of the eligibility gate for one step."""

    eligible: bool
    reason: str


@dataclass(frozen=True)
class PendingBackfillEntry:
    """Where one deferred token value lands once resolved."""

    req_id: str
    # Direct reference: backfill must work even after the runner evicted the
    # request from its dicts (resolve never re-inserts runner state).
    state: RequestState
    row: int
    token_index: int
    output_idx: int


@dataclass(frozen=True)
class PendingSampleStep:
    """One submitted-but-unresolved decode step."""

    tokens: mx.array
    entries: tuple[PendingBackfillEntry, ...]
    batch: _ExecutionBatch
    scheduler_output: SchedulerOutput


class MetalAsyncModelRunnerOutput(AsyncModelRunnerOutput):
    """Deferred step output; ``get_output()`` blocks until tokens resolve."""

    def __init__(self, pipeline: DecodePipeline, serial: int) -> None:
        self._pipeline = pipeline
        self._serial = serial

    def get_output(self) -> ModelRunnerOutput:
        return self._pipeline.resolve(self._serial)


class DecodePipeline:
    """Owner of one-step-ahead decode pipeline state and policy.

    Holds at most one pending step: ``submit()`` resolves the previous
    pending step first (the engine pops its ``AsyncModelRunnerOutput`` only
    after the next step was submitted, so resolution is cached until then).
    """

    def __init__(
        self,
        *,
        build_output: Callable[[_ExecutionBatch], ModelRunnerOutput],
        validate: Callable[[_ExecutionBatch, SchedulerOutput], None],
    ) -> None:
        self._build_output = build_output
        self._validate = validate
        self._pending: PendingSampleStep | None = None
        self._pending_serial = 0
        self._resolved: tuple[int, ModelRunnerOutput] | None = None
        self._rows: dict[str, int] = {}
        self._step_eligible = False

    def evaluate_gate(
        self,
        capabilities: RunnerCapabilities,
        step: SchedulerStepShape,
        sampling: SamplingShape,
    ) -> PipelineGateDecision:
        """Decide whether the upcoming step may defer its sampling sync.

        The runner only collects facts; each shape owns its block reasons,
        and the pipeline adds the one check against its own state: every
        decode request of the step must map to a pending row. Shapes are
        checked capabilities, then step, then sampling, then pending
        coverage; the reported reason is the first match.
        """
        for reason in (
            capabilities.block_reason(),
            step.block_reason(),
            sampling.block_reason(),
            self._pending_coverage_reason(step),
        ):
            if reason is not None:
                return PipelineGateDecision(eligible=False, reason=reason)
        return PipelineGateDecision(eligible=True, reason="eligible")

    def _pending_coverage_reason(self, step: SchedulerStepShape) -> str | None:
        """Block a step whose decode membership exceeds the pending rows.

        A cached request can re-enter a step after being absent from the
        previous one (the scheduler re-syncs it via ``all_token_ids``); it
        then has no pending lazy token to feed the next forward, so the step
        must resolve the pending sample and take the synchronous path.
        """
        if self._pending is None:
            return None
        if any(req_id not in self._rows for req_id in step.decode_req_ids):
            return "cached request without a pending row"
        return None

    def begin_step(self, decision: PipelineGateDecision) -> None:
        """Record the gate decision; barrier the pending step if ineligible.

        Must run before the runner mutates any request state for the new
        step, so the synchronous path never sees a pending placeholder.
        """
        self._step_eligible = decision.eligible
        if not decision.eligible:
            self.barrier()

    def submit(self, step: PendingSampleStep) -> MetalAsyncModelRunnerOutput:
        """Register a deferred step and return its async output wrapper."""
        if not self._step_eligible:
            raise RuntimeError(
                "DecodePipeline.submit called on a gate-ineligible step; "
                "the runner must route ineligible steps through the "
                "synchronous sampling path."
            )
        # Single-slot cache: the engine (queue depth 2) pops step k's async
        # output before submitting step k+2, so the slot must be free here —
        # occupied means outputs were consumed out of order and resolving the
        # pending step would silently overwrite an unconsumed result.
        if self._resolved is not None:
            raise RuntimeError(
                "DecodePipeline resolved-output slot still occupied by serial "
                f"{self._resolved[0]}; the engine must consume the previous "
                "async output before another step is submitted."
            )
        # The engine pops the previous step's async output only after this
        # submit, so resolve it now into the single-slot cache.
        self.barrier()
        self._pending = step
        self._pending_serial += 1
        self._rows = {entry.req_id: entry.row for entry in step.entries}
        return MetalAsyncModelRunnerOutput(self, self._pending_serial)

    def assemble_decode_input_ids(
        self, decode_segments: Sequence[PagedDecodeSegment]
    ) -> mx.array | None:
        """Build the decode input row from the pending lazy tokens.

        Returns ``None`` when there is no pending step or the current step is
        ineligible — the caller then assembles host token ids as before. On an
        eligible step with a pending predecessor, every decode segment must map
        to a pending row (the gate guarantees membership can only shrink).
        """
        pending = self._pending
        if pending is None or not self._step_eligible or not decode_segments:
            return None

        perm: list[int] = []
        for segment in decode_segments:
            if segment.num_query_tokens != 1:
                raise RuntimeError(
                    "Pipelined decode expects single-token segments; got "
                    f"{segment.num_query_tokens} query tokens for request "
                    f"{segment.req_id!r} (gate should have blocked this step)."
                )
            row = self._rows.get(segment.req_id)
            if row is None:
                raise RuntimeError(
                    f"Decode request {segment.req_id!r} has no pending-token "
                    "row; runner/pipeline membership desynced (its last token "
                    "would be a placeholder)."
                )
            perm.append(row)

        tokens = pending.tokens
        if perm == list(range(tokens.shape[0])):
            gathered = tokens
        else:
            gathered = tokens[mx.array(perm, dtype=mx.int32)]
        return gathered.astype(mx.int32)[None, :]

    def barrier(self) -> None:
        """Resolve the pending step (if any) into the single-slot cache."""
        if self._pending is not None:
            self._resolve_pending()

    def resolve(self, serial: int) -> ModelRunnerOutput:
        """Return the resolved output for *serial* (engine ``get_output``)."""
        if self._resolved is not None and self._resolved[0] == serial:
            output = self._resolved[1]
            self._resolved = None
            return output
        if self._pending is not None and self._pending_serial == serial:
            self._resolve_pending()
            assert self._resolved is not None
            output = self._resolved[1]
            self._resolved = None
            return output
        raise RuntimeError(
            f"DecodePipeline has no output for serial {serial} "
            f"(pending serial {self._pending_serial}); get_output was called "
            "more than once or outputs were consumed out of order."
        )

    @property
    def has_pending(self) -> bool:
        return self._pending is not None

    @property
    def step_eligible(self) -> bool:
        return self._step_eligible

    def _resolve_pending(self) -> None:
        """Wait for the pending tokens and materialize the step output."""
        pending = self._pending
        assert pending is not None
        self._pending = None
        self._rows = {}

        mx.eval(pending.tokens)
        token_values = pending.tokens.tolist()
        for entry in pending.entries:
            value = int(token_values[entry.row])  # type: ignore[index]
            if entry.state.token_ids[entry.token_index] != PENDING_TOKEN_PLACEHOLDER:
                raise RuntimeError(
                    f"Pending token slot for request {entry.req_id!r} was "
                    "overwritten before resolve; submit-time bookkeeping "
                    "desynced."
                )
            entry.state.token_ids[entry.token_index] = value
            pending.batch.set_output(entry.output_idx, [value])

        self._validate(pending.batch, pending.scheduler_output)
        self._resolved = (self._pending_serial, self._build_output(pending.batch))
