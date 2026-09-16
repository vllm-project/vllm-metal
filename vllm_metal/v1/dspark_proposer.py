# SPDX-License-Identifier: Apache-2.0
"""Standalone DSpark drafting over exact, request-owned target feature context.

Every scheduled prefill/decode span is ingested before draft eligibility. A
request may draft only when every physical KV layer covers [0, anchor_position).
Missing features disable speculation for that request; target KV alone is not
sufficient to reconstruct them. The runner owns invalidation on finish,
cancellation, preemption and resume, before any same-step request-ID reuse.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import mlx.core as mx
import numpy as np
from vllm.logger import init_logger
from vllm.v1.outputs import DraftTokenIds

from vllm_metal import envs
from vllm_metal.v1.dspark.adaptive import AdaptivePlanner, DSparkCounters
from vllm_metal.v1.dspark.calibration import ConfidenceRecorder
from vllm_metal.v1.dspark.config import DSparkConfig
from vllm_metal.v1.dspark.memory import CONTEXT_ALIGNMENT, DSparkMemoryPlan
from vllm_metal.v1.dspark.model import (
    ArenaBatch,
    ArenaCache,
    ContextArena,
    CtxCache,
    DSparkDrafter,
)
from vllm_metal.v1.dspark.paged_context import (
    KERNEL_HEAD_SIZES,
    DSparkPagedContext,
    PagedContextFullError,
    PagedCtxCache,
    PagedLayerBatch,
)
from vllm_metal.v1.dspark.sampling import (
    DSparkProposal,
    RequestRandomStreams,
    SamplingTransforms,
    batched_transformed_distribution,
    sample_from_distribution,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from vllm_metal.v1.model_runner import MetalModelRunner, RequestState
    from vllm_metal.v1.proposer import ProposeContext
    from vllm_metal.v1.spec_decode import (
        PagedDecodeSegment,
        SpeculativeDecodeController,
    )

logger = init_logger(__name__)

# One informational counters snapshot per this many drafting steps.
COUNTER_LOG_EVERY = 2000

# Load regime. When the planner would decline to draft a batch of the step's
# size on this many consecutive steps, the proposer lapses: it stops capturing
# and ingesting target features and releases every context, so the server
# costs what target-only serving costs while drafting cannot pay; it resumes
# priming new requests after the load has dropped below the count it lapsed
# at and the planner would draft, on this many consecutive steps. Requests
# that ran through a lapse keep target-only generation. (A verdict alone is
# not enough to leave: at a steady load near the planner's threshold it flips
# between arrivals, and every flip re-primed every new prompt for nothing.)
PAGED_BLOCK_SIZE = 16  # kernel page sizes are 8, 16 and 32


def _pool_blocks(memory_plan, config) -> int:
    """Blocks to give the drafter's context pool.

    The arena reserved `max_contexts * max_context_tokens` up front because every
    slot was sized for the whole model length. The pool holds the same worst case
    but hands blocks out as contexts grow, so a server that never reaches that
    length never touches most of them, and one request's unused tail is available
    to another. The extra block is the padding sink.
    """
    per_context = (
        memory_plan.max_context_tokens + config.block_size + PAGED_BLOCK_SIZE - 1
    ) // PAGED_BLOCK_SIZE
    return memory_plan.max_contexts * per_context + 1


LAPSE_ENTER_STEPS = 32
LAPSE_EXIT_STEPS = 8


def _pad_uniforms(drawn: np.ndarray, width: int) -> np.ndarray:
    """Right-pad one request's proposal uniforms to the batch's drafting width.

    The padded positions sit past the request's own cap and are clipped from its
    draft, so their values are never used; they exist only so the batch stacks.
    """
    if drawn.shape[0] >= width:
        return drawn
    return np.concatenate([drawn, np.zeros(width - drawn.shape[0], dtype=drawn.dtype)])


@dataclass
class _RequestContext:
    # The actual RequestState object identifies this request generation. Keeping
    # it alive also prevents Python object-id reuse from aliasing old context.
    owner: RequestState
    caches: list[CtxCache] = field(default_factory=list)
    covered_end: int = 0
    disabled_reason: str | None = None


@dataclass(frozen=True)
class _SpanWrite:
    """A validated feature span whose context write is deferred to one batch."""

    req_id: str
    record: _RequestContext
    start_row: int
    start_pos: int
    count: int
    end_pos: int


@dataclass(frozen=True)
class _DraftPlan:
    req_id: str
    pending: int
    context: _RequestContext
    cap: int
    # ``None`` drafts greedily (argmax chain); otherwise the request's target
    # transforms shape the proposal distributions and ``streams`` draws them.
    transforms: SamplingTransforms | None = None
    streams: RequestRandomStreams | None = None


def _plan_mode(plan: _DraftPlan) -> str:
    return "stochastic" if plan.transforms is not None else "greedy"


class DSparkProposer:
    """DSpark proposer with contiguous per-request feature coverage.

    Greedy requests draft the argmax Markov chain and are verified exactly;
    plain temperature/top-k/top-p requests draft by sampling from exact
    float32 proposal distributions that stay attached to the scheduled
    proposal (:class:`DSparkProposal`) until the next step verifies them.
    """

    def __init__(
        self,
        *,
        drafter: DSparkDrafter,
        config: DSparkConfig,
        runner: MetalModelRunner,
        controller: SpeculativeDecodeController,
        memory_plan: DSparkMemoryPlan,
        memory_budget_bytes: int | None = None,
        max_drafts_per_step: int | None = None,
    ) -> None:
        self._drafter = drafter
        self._config = config
        self._runner = runner
        self._controller = controller
        self.capture_layer_ids = list(config.target_layer_ids)
        self._block_size = config.block_size
        self._mask_token_id = config.mask_token_id
        self._contexts: dict[str, _RequestContext] = {}
        self.memory_plan = memory_plan
        # One shared K/V arena per draft layer: a slot per reserved context,
        # the committed context first and the block's scratch positions after
        # it, so a drafting batch gathers its rows with one op per layer.
        # The drafter's context K/V is a KV cache: one entry per layer per committed
        # token, a pure function of the prefix. The pool keeps it in the target's own
        # paged layout; the arena is the private per-request store it replaces.
        self._pool: DSparkPagedContext | None = None
        if (
            envs.VLLM_METAL_DSPARK_PAGED_CONTEXT
            and config.attn_head_dim in KERNEL_HEAD_SIZES
        ):
            self._pool = DSparkPagedContext(
                num_layers=len(drafter.layers),
                kv_heads=config.n_kv_heads,
                head_dim=config.attn_head_dim,
                block_size=PAGED_BLOCK_SIZE,
                num_blocks=_pool_blocks(memory_plan, config),
                draft_block=config.block_size,
                dtype=drafter.hidden_norm.weight.dtype,
            )
            mx.eval(self._pool.key_caches, self._pool.value_caches)
            logger.info(
                "DSpark draft context: paged pool, %d blocks of %d tokens "
                "(%d usable, 1 padding sink), %.2f GB reserved across %d layers",
                self._pool.total_blocks,
                PAGED_BLOCK_SIZE,
                self._pool.usable_blocks,
                self._pool.bytes_reserved() / 1e9,
                len(drafter.layers),
            )
        elif envs.VLLM_METAL_DSPARK_PAGED_CONTEXT:
            logger.warning(
                "DSpark paged context requested but the attention kernel is not "
                "instantiated for head size %d (supported: %s); keeping the private "
                "context arena for this drafter",
                config.attn_head_dim,
                ", ".join(str(h) for h in KERNEL_HEAD_SIZES),
            )
        self._arena = [
            ContextArena(
                slots=memory_plan.max_contexts,
                kv_heads=config.n_kv_heads,
                capacity=memory_plan.max_context_tokens,
                block_size=config.block_size,
                head_dim=config.attn_head_dim,
                dtype=drafter.hidden_norm.weight.dtype,
            )
            for _ in ([] if self._pool is not None else drafter.layers)
        ]
        if self._arena:
            mx.eval([(layer.keys, layer.values) for layer in self._arena])
            logger.info(
                "DSpark draft context: private arena, %d slots of %d tokens "
                "across %d layers",
                memory_plan.max_contexts,
                memory_plan.max_context_tokens,
                len(drafter.layers),
            )
        self._memory_budget_bytes = memory_budget_bytes
        limit = memory_plan.max_contexts
        if max_drafts_per_step is not None:
            if max_drafts_per_step < 1:
                raise ValueError("DSpark max_drafts_per_step must be positive")
            limit = min(limit, max_drafts_per_step)
        self._max_drafts_per_step = limit
        # Per-step admission bookkeeping: the step a request was last drafted
        # in, so a binding cap rotates least-recently-drafted requests first.
        self._last_drafted: dict[str, int] = {}
        self._draft_step = 0
        # Proposal records of the most recent draft per request (tokens,
        # confidence logits, and for stochastic rows the exact distributions),
        # consumed at the request's next scheduled step, and the request-owned
        # random streams behind stochastic rows.
        self._proposals: dict[str, DSparkProposal] = {}
        self._streams: dict[str, RequestRandomStreams] = {}
        self._stream_ordinal = 0
        # Optional confidence recorder (calibration tooling attaches it); it
        # observes every verified proposal's raw logits and survival labels.
        self.recorder: ConfidenceRecorder | None = None
        # Adaptive mode (runner attaches it when configured): calibrated
        # planning of each draft prefix and the draft-or-not decision.
        self.adaptive: AdaptivePlanner | None = None
        # Bypass mode: the drafter stays loaded and contexts advance, but no
        # request is drafted (the planner's alternative step, made explicit).
        self.bypass_only = False
        self.counters = DSparkCounters()
        self.lapse_enabled = envs.VLLM_METAL_DSPARK_LAPSE
        self._lapsed = False
        self._decline_streak = 0
        self._draft_streak = 0
        self._lapse_entry_active = 0

    @property
    def proposals(self) -> Mapping[str, DSparkProposal]:
        """Proposal records of the drafts handed to the scheduler last step."""
        return self._proposals

    def _observe_outcomes(self, ctx: ProposeContext) -> None:
        """Feed the recorder with each scheduled proposal's verification outcome.

        The runner has already committed the accepted prefix and the
        correction or bonus token, so the accepted draft count is the distance
        from the anchor to the new pending token minus one. Positions the
        scheduler clipped are censored (the record is cut to the scheduled
        width); a request that finished during verification is not scheduled
        again and records nothing.
        """
        recorder = self.recorder
        for (req_id, state), segment in zip(
            ctx.decode_reqs, ctx.decode_segments, strict=True
        ):
            record = self._proposals.get(req_id)
            width = len(segment.draft_token_ids)
            if (
                record is None
                or record.confidence is None
                or width == 0
                or not record.matches(
                    state, segment.cache_start_pos, segment.input_token_ids[0]
                )
                or list(record.token_ids[:width]) != list(segment.draft_token_ids)
            ):
                continue
            accepted = len(state.token_ids) - 2 - segment.cache_start_pos
            if not 0 <= accepted <= width:
                continue
            self.counters.observe_outcome(width, accepted)
            if recorder is None:
                continue
            params = state.sampling_params
            recorder.observe(
                mode="stochastic" if record.stochastic else "greedy",
                logits=record.confidence,
                scheduled=width,
                accepted=accepted,
                temperature=float(params.temperature),
            )

    def needs_target_hidden_states(
        self,
        decode_segments: Sequence[PagedDecodeSegment],
        *,
        has_final_prefill: bool,
    ) -> bool:
        # Context must advance even for intermediate chunks and K=0 steps,
        # except in a lapse, where no context is kept at all.
        return not self._lapsed

    def release_requests(self, req_ids: set[str]) -> None:
        for req_id in req_ids:
            record = self._contexts.pop(req_id, None)
            if record is not None:
                self._release_slot(record)
            self._last_drafted.pop(req_id, None)
            self._proposals.pop(req_id, None)
            self._streams.pop(req_id, None)
        if self.adaptive is not None:
            self.adaptive.forget(list(req_ids))

    def _streams_for(self, req_id: str, state: RequestState) -> RequestRandomStreams:
        streams = self._streams.get(req_id)
        record = self._contexts.get(req_id)
        if streams is None or record is None or record.owner is not state:
            self._stream_ordinal += 1
            streams = RequestRandomStreams.for_request(
                state.sampling_params.seed,
                engine_seed=getattr(self._runner.model_config, "seed", None),
                ordinal=self._stream_ordinal,
            )
            self._streams[req_id] = streams
        return streams

    @staticmethod
    def _scheduled(ctx: ProposeContext) -> set[str]:
        scheduled = {seg.req_id for seg in ctx.decode_segments}
        scheduled.update(req_id for req_id, _ in ctx.decode_reqs)
        scheduled.update(pr.req_id for pr in ctx.prefill_reqs)
        return scheduled

    def deferred_step_allowed(
        self,
        decode_reqs: Sequence[tuple[str, RequestState]],
        num_speculative_tokens: int,
    ) -> bool:
        """Whether a pure-decode step may defer its sampling sync.

        True when this proposer will not draft at the end of the step: no
        speculative tokens are scheduled, the bypass mode is on, or the
        adaptive planner declines the batch for its request count, context
        and calibrated expectations. Every draftable request is offered at
        its full cap, so a batch the planner declines here it would decline
        after sampling as well; the fixed mode drafts every eligible request
        and always keeps the sync.
        """
        if num_speculative_tokens <= 0 or self.bypass_only or self._lapsed:
            return True
        if self.adaptive is None:
            return False
        candidates: list[tuple[str, Any, str, int]] = []
        lengths: list[int] = []
        for req_id, state in decode_reqs:
            lengths.append(len(state.token_ids))
            record = self._contexts.get(req_id)
            if (
                record is None
                or record.owner is not state
                or record.disabled_reason is not None
            ):
                continue
            mode = self._controller.draft_mode(state)
            if mode is None:
                continue
            cap = self._draft_cap(state, num_speculative_tokens)
            if cap <= 0:
                continue
            candidates.append((req_id, state, mode, cap))
        if not candidates:
            return True
        decision = self.adaptive.decide(
            candidates,
            active_requests=len(decode_reqs),
            context=max(1, round(sum(lengths) / len(lengths))),
        )
        return not decision.draft

    def ingest_deferred_step(self, ctx: ProposeContext) -> None:
        """Advance every scheduled context on a step whose sync is deferred.

        The step's sampled tokens are still on the device, so nothing here
        reads token values: the scheduled requests' proposal records are
        spent, the target features are ingested and the arenas are queued
        for evaluation behind the step's own work. No request is drafted.
        """
        self.counters.steps += 1
        self.counters.bypass_reasons["deferred-step"] += 1
        if self.counters.bypass_reasons["deferred-step"] == 1:
            logger.info(
                "DSpark: first deferred step ingested; the decode pipeline is "
                "running on non-drafting steps"
            )
        scheduled = self._scheduled(ctx)
        self._observe_outcomes(ctx)
        for req_id in scheduled:
            self._proposals.pop(req_id, None)
        if self._lapsed:
            self.counters.bypass_reasons["lapse"] += 1
            self._forget_idle(scheduled, ())
            self._regime_step(ctx, drafted=False)
            return
        try:
            self._ingest_step(ctx)
            mx.async_eval([(arena.keys, arena.values) for arena in self._arena])
        except Exception as error:
            self._recover(error, scheduled)
            return
        self._forget_idle(scheduled, ())
        self._regime_step(ctx, drafted=False)

    def propose(self, ctx: ProposeContext) -> DraftTokenIds | None:
        self.counters.steps += 1
        scheduled = self._scheduled(ctx)
        if self._lapsed:
            # No feature was captured for this step and no context is kept:
            # spend the records, count the step and re-evaluate the regime.
            self._observe_outcomes(ctx)
            for req_id in scheduled:
                self._proposals.pop(req_id, None)
            self.counters.bypass_reasons["lapse"] += 1
            self._forget_idle(scheduled, ())
            self._regime_step(ctx, drafted=False)
            return None
        result = self._propose_synchronous(ctx, scheduled)
        self._regime_step(ctx, drafted=result is not None and bool(result.req_ids))
        return result

    def _propose_synchronous(
        self, ctx: ProposeContext, scheduled: set[str]
    ) -> DraftTokenIds | None:
        # The scheduler consumes a request's drafts the next time it schedules
        # the request (verified this step, clipped, or dropped for a prefill
        # chunk), so a record of a scheduled request is spent either way.
        self._observe_outcomes(ctx)
        for req_id in scheduled:
            self._proposals.pop(req_id, None)
        try:
            self._ingest_step(ctx)
            # Materialize once per step, including K=0 and intermediate chunks.
            # Persistent KV must not retain the target activation graph from
            # every earlier chunk. There is no full-prompt feature stash/replay.
            # Every context lives in the per-layer arenas, so one call settles
            # all of this step's writes; it is queued behind the step's GPU
            # work without a host wait (a drafting step's own evaluation
            # waits for it, a non-drafting step overlaps it with the next
            # step's preparation).
            if self._pool is not None:
                # The pool's per-layer caches hold every request's context; a
                # PagedCtxCache is a view on them and owns no tensors itself.
                mx.async_eval(self._pool.key_caches + self._pool.value_caches)
            else:
                mx.async_eval(
                    [(arena.keys, arena.values) for arena in self._arena]
                    + [
                        (cache.k, cache.v)
                        for req_id in scheduled
                        if (record := self._contexts.get(req_id)) is not None
                        for cache in record.caches
                        if not isinstance(cache, ArenaCache)
                    ]
                )
            if ctx.num_speculative_tokens <= 0:
                return None
            eligible = self._controller.draft_eligible_requests(
                ctx.decode_reqs,
                ctx.decode_token_ids,
                ctx.prefill_reqs,
                ctx.prefill_result_modes,
                ctx.request_states,
                allow_stochastic=True,
            )
            plans = []
            for req_id, state in eligible:
                record = self._contexts.get(req_id)
                if (
                    req_id not in scheduled
                    or record is None
                    or record.owner is not state
                    or record.disabled_reason is not None
                    or record.covered_end != len(state.token_ids) - 1
                ):
                    continue
                cap = self._draft_cap(state, ctx.num_speculative_tokens)
                if cap <= 0:
                    continue
                if self._controller.draft_mode(state) == "stochastic":
                    plan = _DraftPlan(
                        req_id,
                        state.token_ids[-1],
                        record,
                        cap,
                        SamplingTransforms.from_params(state.sampling_params),
                        self._streams_for(req_id, state),
                    )
                else:
                    plan = _DraftPlan(req_id, state.token_ids[-1], record, cap)
                plans.append(plan)
            if not plans:
                self._forget_idle(scheduled, ())
                return None
            if not self._memory_available():
                logger.debug("DSpark draft skipped: workspace memory budget exhausted")
                self._forget_idle(scheduled, ())
                return None
            if self.bypass_only:
                self.counters.bypass_reasons["mode-bypass"] += 1
                return None
            selected = self._select_plans(plans)
            active = self._active_requests(ctx)
            context = self._batch_context(ctx)
            if self.adaptive is not None:
                decision = self.adaptive.decide(
                    [
                        (plan.req_id, plan.context.owner, _plan_mode(plan), plan.cap)
                        for plan in selected
                    ],
                    active_requests=active,
                    context=context,
                )
                if not decision.draft:
                    self.counters.bypass_reasons[decision.reason] += 1
                    self._forget_idle(scheduled, ())
                    return None
            req_ids, rows, proposals = self._batch_draft(selected)
            if self.adaptive is not None:
                req_ids, rows = self._allocate(
                    selected, req_ids, rows, proposals, active, context
                )
            self._forget_idle(scheduled, req_ids)
            self._proposals.update(proposals)
            self.counters.drafting_steps += 1
            self.counters.proposed_tokens += sum(len(row) for row in rows)
            if self.counters.drafting_steps % COUNTER_LOG_EVERY == 0:
                logger.info("DSpark counters: %s", self.counters.snapshot())
            if not req_ids:
                return None
            return DraftTokenIds(req_ids=req_ids, draft_token_ids=rows)
        except Exception as error:
            self._recover(error, scheduled)
            return None

    def _would_draft(self, ctx: ProposeContext) -> bool:
        """The planner's verdict for a batch of this step's size with prior expectations.

        Every draftable decode request is offered at its full cap whether or
        not it holds a context, so the verdict follows the load (request
        count and context) rather than the contexts a lapse released.
        """
        if self.adaptive is None or ctx.num_speculative_tokens <= 0:
            return False
        candidates: list[tuple[str, Any, str, int]] = []
        lengths: list[int] = []
        for req_id, state in ctx.decode_reqs:
            lengths.append(len(state.token_ids))
            mode = self._controller.draft_mode(state)
            if mode is None:
                continue
            cap = self._draft_cap(state, ctx.num_speculative_tokens)
            if cap <= 0:
                continue
            candidates.append((req_id, state, mode, cap))
        if not candidates:
            return False
        decision = self.adaptive.decide(
            candidates,
            active_requests=len(ctx.decode_reqs),
            context=max(1, round(sum(lengths) / len(lengths))),
        )
        return bool(decision.draft)

    def _regime_step(self, ctx: ProposeContext, *, drafted: bool) -> None:
        """Enter or leave the lapse from this step's load; fixed mode never lapses."""
        if drafted:
            self._decline_streak = 0
        if not self.lapse_enabled or self.adaptive is None or self.bypass_only:
            return
        if not ctx.decode_reqs:
            return
        if self._lapsed:
            # Leave only on a real drop in load: at or below the load the lapse
            # began with, less one, and the planner's verdict for that batch is to
            # draft. The floor of one matters: a lapse entered while a single
            # request was decoding has no lower load to wait for -- this branch
            # only runs with at least one decode request -- so a strict "fewer
            # than the entry load" test could never be satisfied and the server
            # stayed lapsed for the rest of its life.
            dropped = len(ctx.decode_reqs) <= max(1, self._lapse_entry_active - 1)
            resume = dropped and self._would_draft(ctx)
            self._draft_streak = self._draft_streak + 1 if resume else 0
            if self._draft_streak >= LAPSE_EXIT_STEPS:
                self._lapsed = False
                self._draft_streak = 0
                self._decline_streak = 0
                self.counters.lapse_exits += 1
                logger.info(
                    "DSpark: load regime resumed drafting at %d requests "
                    "(lapsed at %d; new requests are primed again)",
                    len(ctx.decode_reqs),
                    self._lapse_entry_active,
                )
            return
        would_draft = drafted or self._would_draft(ctx)
        self._decline_streak = 0 if would_draft else self._decline_streak + 1
        if self._decline_streak >= LAPSE_ENTER_STEPS:
            self._enter_lapse(len(ctx.decode_reqs))

    def lapse_for_bypass(self) -> None:
        """The bypass mode never drafts: lapse from the start (no capture, no context)."""
        if self.lapse_enabled and not self._lapsed:
            self._lapsed = True
            logger.info(
                "DSpark: bypass mode keeps no draft context; target features are not captured"
            )

    def _enter_lapse(self, active: int) -> None:
        self._lapsed = True
        self._lapse_entry_active = active
        self._decline_streak = 0
        self._draft_streak = 0
        self.counters.lapse_entries += 1
        released = len(self._contexts)
        self.release_requests(set(self._contexts))
        logger.info(
            "DSpark: load regime lapse at %d requests after %d declined steps; "
            "%d draft contexts released, features no longer captured",
            active,
            LAPSE_ENTER_STEPS,
            released,
        )

    def _recover(self, error: Exception, scheduled: set[str]) -> None:
        """Release the step's contexts; swallow allocation failures, re-raise the rest."""
        # A partially written layer set cannot survive an ingest/draft
        # failure, even if the engine subsequently retries these requests.
        self.release_requests(scheduled)
        if isinstance(error, MemoryError) or (
            isinstance(error, RuntimeError)
            and str(error).startswith(
                (
                    "[metal::malloc] Resource limit (",
                    "[metal::malloc] Attempting to allocate ",
                    "[malloc] Unable to allocate ",
                )
            )
        ):
            # Drafting has no target KV side effects. Discard all private
            # context under allocation pressure and keep the already
            # sampled target output. Later requests/recomputation can
            # obtain fresh complete context; other failures remain fatal.
            self.release_requests(set(self._contexts))
            mx.clear_cache()
            logger.warning(
                "DSpark allocation failed; released draft context and using target-only output"
            )
            return
        raise error

    @staticmethod
    def _active_requests(ctx: ProposeContext) -> int:
        """Requests the next target step decodes or verifies (one row each at least)."""
        decoding = sum(1 for ids in ctx.decode_token_ids if ids)
        finishing = sum(
            1 for mode in ctx.prefill_result_modes if mode != "intermediate"
        )
        return max(1, decoding + finishing)

    @staticmethod
    def _batch_context(ctx: ProposeContext) -> int:
        """Mean committed length of the requests the next step decodes."""
        lengths = [
            len(state.token_ids)
            for (_, state), ids in zip(
                ctx.decode_reqs, ctx.decode_token_ids, strict=True
            )
            if ids
        ]
        for prefill, mode in zip(
            ctx.prefill_reqs, ctx.prefill_result_modes, strict=True
        ):
            if mode != "intermediate":
                state = ctx.request_states.get(prefill.req_id)
                if state is not None:
                    lengths.append(len(state.token_ids))
        return max(1, round(sum(lengths) / len(lengths))) if lengths else 1

    def _forget_idle(self, scheduled: set[str], drafted: Sequence[str]) -> None:
        if self.adaptive is not None:
            self.adaptive.forget_idle(sorted(scheduled), drafted)

    def _allocate(
        self,
        plans: list[_DraftPlan],
        req_ids: list[str],
        rows: list[list[int]],
        proposals: dict[str, DSparkProposal],
        active: int,
        context: int,
    ) -> tuple[list[str], list[list[int]]]:
        """Cut every drafted row to the planner's prefix; drop empty ones."""
        assert self.adaptive is not None
        lengths = self.adaptive.allocate(
            [
                (
                    plan.req_id,
                    plan.context.owner,
                    _plan_mode(plan),
                    plan.cap,
                    proposals[plan.req_id].confidence or [],
                )
                for plan in plans
            ],
            active_requests=active,
            context=context,
        )
        plan_result = self.adaptive.last_plan
        self.counters.planner_predicted_ratio = plan_result.ratio
        self.counters.planner_target_only_ratio = plan_result.target_only_ratio
        kept_ids: list[str] = []
        kept_rows: list[list[int]] = []
        for req_id, row, length in zip(req_ids, rows, lengths, strict=True):
            self.counters.planner_lengths[length] += 1
            if length <= 0:
                proposals.pop(req_id, None)
                continue
            record = proposals[req_id]
            record.token_ids = row[:length]
            if record.distributions is not None:
                record.distributions = record.distributions[:length]
            if record.confidence is not None:
                record.confidence = record.confidence[:length]
            kept_ids.append(req_id)
            kept_rows.append(row[:length])
        return kept_ids, kept_rows

    def _select_plans(self, plans: list[_DraftPlan]) -> list[_DraftPlan]:
        """Apply the per-step draft cap fairly.

        When more requests are eligible than the cap allows, the ones drafted
        least recently go first (never-drafted requests before all others),
        ties broken by their position in the packed batch. The selected plans
        keep their batch order. Every request's context was already ingested
        this step, so a request that waits loses nothing but this step's draft.
        """
        self._draft_step += 1
        if len(plans) > self._max_drafts_per_step:
            ranked = sorted(
                range(len(plans)),
                key=lambda index: (
                    self._last_drafted.get(plans[index].req_id, -1),
                    index,
                ),
            )
            plans = [
                plans[index] for index in sorted(ranked[: self._max_drafts_per_step])
            ]
        for plan in plans:
            self._last_drafted[plan.req_id] = self._draft_step
        return plans

    def _memory_available(self, extra_bytes: int = 0) -> bool:
        return self._memory_budget_bytes is None or (
            mx.get_active_memory() + extra_bytes + self.memory_plan.workspace_bytes
            <= self._memory_budget_bytes
        )

    def _draft_cap(self, state: RequestState, requested: int) -> int:
        params = state.sampling_params
        if state.token_ids[-1] == params.eos_token_id or state.token_ids[-1] in (
            params.stop_token_ids or ()
        ):
            return 0
        # Reserve a correction/bonus output in both token and model budgets.
        remaining = (
            params.max_tokens - state.generated_tokens
            if params.max_tokens is not None
            else requested + 1
        )
        return max(
            0,
            min(
                requested,
                self._block_size,
                remaining - 1,
                self._runner.model_config.max_model_len - len(state.token_ids) - 1,
            ),
        )

    def _release_slot(self, record: _RequestContext) -> None:
        for cache in record.caches:
            if isinstance(cache, ArenaCache):
                cache.arena.release(cache.slot)
            elif isinstance(cache, PagedCtxCache) and self._pool is not None:
                self._pool.release(cache.request_id)
                break  # one block table serves every layer of the request
        record.caches = []

    def _disable(self, req_id: str, record: _RequestContext, reason: str) -> None:
        if record.disabled_reason is None:
            logger.debug("DSpark target-only fallback for %s: %s", req_id, reason)
        self._release_slot(record)
        record.covered_end = 0
        record.disabled_reason = reason

    def _ingest_step(self, ctx: ProposeContext) -> None:
        # These are existing runner DTOs, not a second interpretation of packed
        # tensor shapes. Validate the seam before accepting any feature rows.
        boundaries = [0]
        for segment in ctx.decode_segments:
            if segment.start_row != boundaries[-1]:
                raise ValueError("DSpark decode feature rows are not contiguous")
            boundaries.append(boundaries[-1] + segment.num_query_tokens)
        for prefill in ctx.prefill_reqs:
            boundaries.append(boundaries[-1] + len(prefill.token_ids))
        if (
            ctx.num_decode_segments != len(ctx.decode_segments)
            or list(ctx.cu_seqlens) != boundaries
        ):
            raise ValueError("DSpark feature boundaries disagree with packed requests")
        if boundaries[-1] > self.memory_plan.max_step_tokens:
            raise ValueError(
                "DSpark packed features exceed the reserved scheduler token bound"
            )
        hidden = ctx.target_hidden_states
        if hidden is not None and hidden.shape != (
            boundaries[-1],
            self._config.hidden_size * len(self.capture_layer_ids),
        ):
            raise ValueError(
                "DSpark target feature shape disagrees with packed requests"
            )
        seen = set()
        # Decode spans are at most one verification window each; their
        # context writes go out as one batched pass per layer.
        pending: list[_SpanWrite] = []
        for (req_id, state), segment in zip(
            ctx.decode_reqs, ctx.decode_segments, strict=True
        ):
            if segment.req_id != req_id or req_id in seen:
                raise ValueError(
                    "DSpark decode feature identity disagrees with requests"
                )
            seen.add(req_id)
            end = len(state.token_ids) - 1
            count = end - segment.cache_start_pos
            if count >= 0 and state.token_ids[segment.cache_start_pos : end] != list(
                segment.input_token_ids[:count]
            ):
                raise ValueError(
                    "DSpark verification features do not match committed inputs"
                )
            self._ingest_span(
                req_id,
                state,
                hidden,
                segment.start_row,
                segment.cache_start_pos,
                segment.num_query_tokens,
                end,
                pending,
            )
        self._flush_span_writes(hidden, pending)
        for index, (prefill, mode) in enumerate(
            zip(ctx.prefill_reqs, ctx.prefill_result_modes, strict=True)
        ):
            req_id = prefill.req_id
            if req_id in seen:
                raise ValueError("DSpark request appears in multiple feature spans")
            seen.add(req_id)
            state = ctx.request_states.get(req_id)
            if state is None:
                self.release_requests({req_id})
                continue
            end = prefill.start_pos + len(prefill.token_ids)
            if state.token_ids[prefill.start_pos : end] != prefill.token_ids:
                raise ValueError(
                    "DSpark prefill features do not match committed inputs"
                )
            if mode != "intermediate" and end != len(state.token_ids) - 1:
                raise ValueError(
                    "DSpark final prefill does not end at the pending anchor"
                )
            self._ingest_span(
                req_id,
                state,
                hidden,
                boundaries[ctx.num_decode_segments + index],
                prefill.start_pos,
                len(prefill.token_ids),
                end,
            )

    def _ingest_span(
        self,
        req_id: str,
        owner: RequestState,
        hidden: mx.array | None,
        start_row: int,
        start_pos: int,
        available_rows: int,
        end_pos: int,
        pending: list[_SpanWrite] | None = None,
    ) -> None:
        record = self._contexts.get(req_id)
        if record is None or record.owner is not owner:
            if record is not None:
                self._release_slot(record)
            record = _RequestContext(owner)
            self._contexts[req_id] = record
        if record.disabled_reason is not None:
            if start_pos != 0:
                return
            # A scheduler recompute from zero supplies a complete new prefix.
            self._release_slot(record)
            record = _RequestContext(owner)
            self._contexts[req_id] = record
        if not 0 <= start_pos <= end_pos <= start_pos + available_rows:
            self._disable(req_id, record, "accepted input span is unavailable")
            return
        if start_pos > record.covered_end:
            self._disable(req_id, record, "missing target feature prefix")
            return
        if end_pos > self.memory_plan.max_context_tokens:
            self._disable(req_id, record, "context length exceeds reserved capacity")
            return
        if not record.caches:
            if self._pool is not None:
                try:
                    self._pool.reserve(req_id, end_pos)
                except PagedContextFullError:
                    self._disable(req_id, record, "context capacity exhausted")
                    return
                record.caches = [
                    PagedCtxCache(
                        self._pool,
                        layer,
                        req_id,
                        self.memory_plan.max_context_tokens,
                    )
                    for layer in range(len(self._drafter.layers))
                ]
            else:
                if self._arena[0].free_slots == 0:
                    self._disable(req_id, record, "context capacity exhausted")
                    return
                slot = self._arena[0].acquire()
                for layer in self._arena[1:]:
                    assert layer.acquire() == slot
                record.caches = [ArenaCache(layer, slot) for layer in self._arena]
        required = (
            min(
                self.memory_plan.max_context_tokens,
                (end_pos + CONTEXT_ALIGNMENT - 1)
                // CONTEXT_ALIGNMENT
                * CONTEXT_ALIGNMENT,
            )
            * self.memory_plan.kv_bytes_per_token
        )
        allocated = sum(cache.allocated_bytes for cache in record.caches)
        if not self._memory_available(max(0, required - allocated)):
            self._disable(req_id, record, "context memory budget exhausted")
            return
        if len(record.caches) != len(self._drafter.layers) or any(
            cache.length != record.covered_end for cache in record.caches
        ):
            raise RuntimeError("DSpark physical context disagrees with its coverage")
        # Recomputed/overlapping spans replace the old suffix at its true
        # absolute position. Every physical layer is trimmed before appending.
        for cache in record.caches:
            cache.trim_to(start_pos)
        record.covered_end = start_pos
        count = end_pos - start_pos
        if count:
            if hidden is None:
                self._disable(req_id, record, "target features are unavailable")
                return
            if pending is not None and count <= self._block_size + 1:
                pending.append(
                    _SpanWrite(req_id, record, start_row, start_pos, count, end_pos)
                )
                return
            self._drafter.update_context(
                hidden[start_row : start_row + count][None],
                ctx_offset=start_pos,
                ctx_caches=record.caches,
            )
        self._commit_span(record, end_pos)

    @staticmethod
    def _commit_span(record: _RequestContext, end_pos: int) -> None:
        if any(cache.length != end_pos for cache in record.caches):
            raise RuntimeError("DSpark context update wrote an incomplete layer span")
        record.covered_end = end_pos

    def _flush_span_writes(
        self, hidden: mx.array | None, pending: list[_SpanWrite]
    ) -> None:
        """Write every deferred span: one request keeps the single-row path."""
        if not pending:
            return
        if hidden is None:
            raise RuntimeError("deferred DSpark spans need target features")
        if len(pending) == 1:
            write = pending[0]
            self._drafter.update_context(
                hidden[write.start_row : write.start_row + write.count][None],
                ctx_offset=write.start_pos,
                ctx_caches=write.record.caches,
            )
            self._commit_span(write.record, write.end_pos)
            return
        spans = []
        for write in pending:
            caches = write.record.caches
            if not caches:
                raise RuntimeError("batched context ingest needs a reserved context")
            if self._pool is not None:
                spans.append(
                    (write.start_row, write.start_pos, write.count, write.req_id)
                )
            else:
                if not all(isinstance(cache, ArenaCache) for cache in caches):
                    raise RuntimeError(
                        "batched context ingest needs arena-backed contexts"
                    )
                slot = cast("ArenaCache", caches[0]).slot
                spans.append((write.start_row, write.start_pos, write.count, slot))
        if self._pool is not None:
            for write in pending:
                self._pool.reserve(write.req_id, write.end_pos)
            self._drafter.update_context_spans_paged(hidden, spans, self._pool)
        else:
            self._drafter.update_context_spans(hidden, spans, self._arena)
        for write in pending:
            for cache in write.record.caches:
                cast("ArenaCache | PagedCtxCache", cache).extend_to(write.end_pos)
            self._commit_span(write.record, write.end_pos)

    def _batch_draft(
        self, plans: list[_DraftPlan]
    ) -> tuple[list[str], list[list[int]], dict[str, DSparkProposal]]:
        """Draft every plan in one backbone pass.

        Returns the request ids, one draft row per plan (clipped to its cap)
        and the proposal records of the stochastic rows. Greedy rows take the
        argmax of each block position's logits plus the Markov step bias of
        the previous token; stochastic rows sample the same corrected logits
        through the request's transforms with their own proposal stream, and
        keep the exact distribution of every sampled position.
        """
        # Keep all trained block positions even when a row requests fewer heads.
        noise = self._drafter.embed(
            mx.array(
                [
                    [plan.pending] + [self._mask_token_id] * (self._block_size - 1)
                    for plan in plans
                ]
            )
        )
        lengths = [plan.context.covered_end for plan in plans]
        batched_ctx: list = []
        for layer_index in range(len(self._drafter.layers)):
            caches = [plan.context.caches[layer_index] for plan in plans]
            if any(
                cache.length != length
                for cache, length in zip(caches, lengths, strict=True)
            ):
                raise RuntimeError(
                    "DSpark draft context has inconsistent physical lengths"
                )
            if self._pool is not None:
                if layer_index == 0:
                    paged_batch = self._pool.plan(
                        list(zip([plan.req_id for plan in plans], lengths, strict=True))
                    )
                batched_ctx.append(
                    PagedLayerBatch(self._pool, layer_index, paged_batch)
                )
            elif all(isinstance(cache, ArenaCache) for cache in caches):
                batched_ctx.append(
                    ArenaBatch(
                        self._arena[layer_index],
                        [cache.slot for cache in caches],
                        lengths,
                    )
                )
            elif len(plans) == 1:
                batched_ctx.append(caches[0])
            else:
                raise RuntimeError("batched drafting needs arena-backed contexts")
        hidden = self._drafter.backbone(
            noise, mx.array(lengths, dtype=mx.int32), batched_ctx
        )
        cap = max(plan.cap for plan in plans)
        logits = self._drafter.compute_logits(hidden[:, :cap])
        stochastic = [
            index for index, plan in enumerate(plans) if plan.transforms is not None
        ]
        transforms = [
            cast("SamplingTransforms", plans[index].transforms) for index in stochastic
        ]
        # Draw every proposal uniform up front from each request's own stream, and
        # draw only that request's own cap: a co-scheduled request with a wider cap
        # must never advance this stream, or the same seed stops reproducing when
        # the batch changes. Positions past a request's cap are clipped from its
        # draft below, so the padding is never consumed.
        uniforms = (
            np.stack(
                [
                    _pad_uniforms(
                        cast(
                            "RequestRandomStreams", plans[index].streams
                        ).proposal.random(plans[index].cap),
                        cap,
                    )
                    for index in stochastic
                ]
            )
            if stochastic
            else None
        )
        rows_index = mx.array(stochastic, dtype=mx.int32)
        markov = self._drafter.markov_head
        previous = mx.array([plan.pending for plan in plans], dtype=mx.int32)
        drafts, distributions = [], []
        for index in range(cap):
            step = logits[:, index]
            if markov is not None:
                step = step + markov.step_bias(previous)
            tokens = mx.argmax(step, axis=-1).astype(mx.int32)
            if uniforms is not None:
                q = batched_transformed_distribution(
                    step[rows_index], transforms, vocab_size=self._config.vocab_size
                )
                tokens[rows_index] = sample_from_distribution(
                    q, mx.array(uniforms[:, index], dtype=mx.float32)
                )
                distributions.append(q)
            drafts.append(tokens)
            previous = tokens
        draft_array = mx.stack(drafts, axis=1)
        # Raw confidence logit per block position: the head reads the block
        # hidden state and the embedding of the token preceding each position
        # (the anchor, then the drafted tokens), as in the reference evaluator.
        confidence = None
        if self._drafter.confidence_head is not None:
            previous_tokens = mx.concatenate(
                [
                    mx.array([[plan.pending] for plan in plans], dtype=mx.int32),
                    draft_array[:, : cap - 1],
                ],
                axis=1,
            )
            confidence = self._drafter.confidence_logits(
                hidden[:, :cap], previous_tokens
            ).astype(mx.float32)
        slices = []
        if distributions:
            stacked = mx.stack(distributions, axis=1)
            slices = [
                stacked[position, : plans[index].cap]
                for position, index in enumerate(stochastic)
            ]
        mx.eval(draft_array, *slices, *([confidence] if confidence is not None else []))
        rows = cast("list[list[int]]", draft_array.tolist())
        clipped = [row[: plan.cap] for row, plan in zip(rows, plans, strict=True)]
        confidence_rows = (
            cast("list[list[float]]", confidence.tolist())
            if confidence is not None
            else None
        )
        proposals = {}
        stochastic_slot = {index: position for position, index in enumerate(stochastic)}
        for index, plan in enumerate(plans):
            slot = stochastic_slot.get(index)
            proposals[plan.req_id] = DSparkProposal(
                owner=plan.context.owner,
                anchor_position=plan.context.covered_end,
                anchor_token=plan.pending,
                token_ids=clipped[index],
                distributions=slices[slot] if slot is not None else None,
                transforms=plan.transforms,
                streams=plan.streams,
                confidence=(
                    confidence_rows[index][: plan.cap]
                    if confidence_rows is not None
                    else None
                ),
            )
        return [plan.req_id for plan in plans], clipped, proposals
