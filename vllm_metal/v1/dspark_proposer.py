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
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import mlx.core as mx
from vllm.logger import init_logger
from vllm.v1.outputs import DraftTokenIds

from vllm_metal.v1.dspark.config import DSparkConfig
from vllm_metal.v1.dspark.memory import CONTEXT_ALIGNMENT, DSparkMemoryPlan
from vllm_metal.v1.dspark.model import CtxCache, DSparkDrafter

if TYPE_CHECKING:
    from collections.abc import Sequence

    from vllm_metal.v1.model_runner import MetalModelRunner, RequestState
    from vllm_metal.v1.proposer import ProposeContext
    from vllm_metal.v1.spec_decode import (
        PagedDecodeSegment,
        SpeculativeDecodeController,
    )

logger = init_logger(__name__)


def _ctx_block_mask(ctx_lens: list[int], n_block: int) -> mx.array:
    """Each row attends its valid context and the entire bidirectional block."""
    max_ctx = max(ctx_lens)
    lens = mx.array(ctx_lens, dtype=mx.int32)[:, None, None, None]
    columns = mx.arange(max_ctx + n_block)[None, None, None, :]
    return mx.broadcast_to(
        (columns >= max_ctx) | (columns < lens),
        (len(ctx_lens), 1, n_block, max_ctx + n_block),
    )


@dataclass
class _RequestContext:
    # The actual RequestState object identifies this request generation. Keeping
    # it alive also prevents Python object-id reuse from aliasing old context.
    owner: RequestState
    caches: list[CtxCache] = field(default_factory=list)
    covered_end: int = 0
    disabled_reason: str | None = None


@dataclass(frozen=True)
class _DraftPlan:
    req_id: str
    pending: int
    context: _RequestContext
    cap: int


class DSparkProposer:
    """Greedy DSpark proposer with contiguous per-request feature coverage."""

    def __init__(
        self,
        *,
        drafter: DSparkDrafter,
        config: DSparkConfig,
        runner: MetalModelRunner,
        controller: SpeculativeDecodeController,
        memory_plan: DSparkMemoryPlan,
        memory_budget_bytes: int | None = None,
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
        self._memory_budget_bytes = memory_budget_bytes
        self._max_drafts_per_step = memory_plan.max_contexts

    def needs_target_hidden_states(
        self,
        decode_segments: Sequence[PagedDecodeSegment],
        *,
        has_final_prefill: bool,
    ) -> bool:
        # Context must advance even for intermediate chunks and K=0 steps.
        return True

    def release_requests(self, req_ids: set[str]) -> None:
        for req_id in req_ids:
            self._contexts.pop(req_id, None)

    def propose(self, ctx: ProposeContext) -> DraftTokenIds | None:
        scheduled = {seg.req_id for seg in ctx.decode_segments}
        scheduled.update(req_id for req_id, _ in ctx.decode_reqs)
        scheduled.update(pr.req_id for pr in ctx.prefill_reqs)
        try:
            self._ingest_step(ctx)
            # Materialize once per step, including K=0 and intermediate chunks.
            # Persistent KV must not retain the target activation graph from
            # every earlier chunk. There is no full-prompt feature stash/replay.
            mx.eval(
                [
                    (cache.k, cache.v)
                    for req_id in scheduled
                    if (record := self._contexts.get(req_id)) is not None
                    for cache in record.caches
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
                if cap > 0:
                    plans.append(_DraftPlan(req_id, state.token_ids[-1], record, cap))
            if not plans:
                return None
            if not self._memory_available():
                logger.debug("DSpark draft skipped: workspace memory budget exhausted")
                return None
            req_ids, rows = self._batch_draft(plans[: self._max_drafts_per_step])
            return DraftTokenIds(req_ids=req_ids, draft_token_ids=rows)
        except Exception as error:
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
                return None
            raise

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

    def _disable(self, req_id: str, record: _RequestContext, reason: str) -> None:
        if record.disabled_reason is None:
            logger.debug("DSpark target-only fallback for %s: %s", req_id, reason)
        record.caches = []
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
            )
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
    ) -> None:
        record = self._contexts.get(req_id)
        if record is None or record.owner is not owner:
            record = _RequestContext(owner)
            self._contexts[req_id] = record
        if record.disabled_reason is not None:
            if start_pos != 0:
                return
            # A scheduler recompute from zero supplies a complete new prefix.
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
            if (
                sum(bool(context.caches) for context in self._contexts.values())
                >= self.memory_plan.max_contexts
            ):
                self._disable(req_id, record, "context capacity exhausted")
                return
            record.caches = self._drafter.make_ctx_cache(
                self.memory_plan.max_context_tokens
            )
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
            self._drafter.update_context(
                hidden[start_row : start_row + count][None],
                ctx_offset=start_pos,
                ctx_caches=record.caches,
            )
        if any(cache.length != end_pos for cache in record.caches):
            raise RuntimeError("DSpark context update wrote an incomplete layer span")
        record.covered_end = end_pos

    def _batch_draft(
        self, plans: list[_DraftPlan]
    ) -> tuple[list[str], list[list[int]]]:
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
        max_len = max(lengths)
        batched_ctx = []
        for layer_index in range(len(self._drafter.layers)):
            caches = [plan.context.caches[layer_index] for plan in plans]
            if any(
                cache.length != length
                for cache, length in zip(caches, lengths, strict=True)
            ):
                raise RuntimeError(
                    "DSpark draft context has inconsistent physical lengths"
                )
            if len(plans) == 1 and max_len:
                # Common latency path: no padded copy of the whole context.
                batched_ctx.append(caches[0])
                continue
            reference = next((cache for cache in caches if cache.k is not None), None)
            dtype = reference.k.dtype if reference is not None else noise.dtype
            empty_shape = (1, self._config.n_kv_heads, 0, self._config.attn_head_dim)
            keys, values = [], []
            for cache, length in zip(caches, lengths, strict=True):
                padding = [(0, 0), (0, 0), (0, max_len - length), (0, 0)]
                key = (
                    cache.k
                    if cache.k is not None
                    else mx.zeros(empty_shape, dtype=dtype)
                )
                value = (
                    cache.v
                    if cache.v is not None
                    else mx.zeros(empty_shape, dtype=dtype)
                )
                keys.append(mx.pad(key, padding))
                values.append(mx.pad(value, padding))
            batched_ctx.append(
                SimpleNamespace(
                    k=mx.concatenate(keys, axis=0), v=mx.concatenate(values, axis=0)
                )
            )
        hidden = self._drafter.backbone(
            noise,
            mx.array(lengths, dtype=mx.int32),
            batched_ctx,
            mask=(
                _ctx_block_mask(lengths, self._block_size) if len(plans) > 1 else None
            ),
        )
        cap = max(plan.cap for plan in plans)
        logits = self._drafter.compute_logits(hidden[:, :cap])
        if self._drafter.markov_head is not None:
            previous = mx.array([plan.pending for plan in plans])
            drafts = []
            for index in range(cap):
                previous = mx.argmax(
                    logits[:, index] + self._drafter.markov_head.step_bias(previous),
                    axis=-1,
                )
                drafts.append(previous)
            draft_array = mx.stack(drafts, axis=1)
        else:
            draft_array = mx.argmax(logits, axis=-1)
        mx.eval(draft_array)
        rows = cast("list[list[int]]", draft_array.tolist())
        return [plan.req_id for plan in plans], [
            row[: plan.cap] for row, plan in zip(rows, plans, strict=True)
        ]
