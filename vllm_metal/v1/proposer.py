# SPDX-License-Identifier: Apache-2.0
"""Proposer seam for Metal speculative decoding.

The model runner owns a single :class:`MetalProposer` and drives drafting
through its uniform :meth:`MetalProposer.propose` call, mirroring vLLM's
polymorphic ``self.drafter``. Gemma4 MTP and draft-model speculative decoding
are interchangeable implementations; the runner holds no per-method knowledge.

The shared *verify* half stays in
:class:`vllm_metal.v1.spec_decode.SpeculativeDecodeController`
(``build_decode_segments`` + ``verify_greedy``); only the *propose* half is
polymorphic here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import mlx.core as mx
from vllm.v1.outputs import DraftTokenIds

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from vllm_metal.v1.model_runner import (
        MetalModelRunner,
        PrefillRequest,
        RequestState,
    )
    from vllm_metal.v1.spec_decode import PagedDecodeSegment


@dataclass(frozen=True, slots=True)
class ProposeContext:
    """Per-step state a proposer may consume to draft the next tokens.

    Carries everything computed during target sampling that a drafter needs.
    Long-lived collaborators (models, caches, the assistant runtime) are held
    by the proposer implementation itself, not here.
    """

    target_hidden_states: mx.array | None
    decode_reqs: Sequence[tuple[str, RequestState]]
    decode_segments: Sequence[PagedDecodeSegment]
    decode_token_ids: Sequence[Sequence[int]]
    prefill_reqs: Sequence[PrefillRequest]
    prefill_token_ids: Sequence[int]
    prefill_result_modes: Sequence[str]
    request_states: Mapping[str, RequestState]
    cu_seqlens: Sequence[int]
    num_decode_segments: int
    num_speculative_tokens: int
    # Request ids the scheduler finished this step. vLLM can hand a finished
    # id straight back out to a new request in the same step, so a proposer
    # that keeps its own per-request state must clear against this, not
    # against absence from request_states (which the new request repopulates
    # under the same id).
    finished_req_ids: set[str]


class MetalProposer(Protocol):
    """Uniform drafting seam."""

    def needs_target_hidden_states(
        self,
        decode_segments: Sequence[PagedDecodeSegment],
        *,
        has_final_prefill: bool,
    ) -> bool:
        """Whether the runner must collect target hidden states for this drafter."""
        ...

    def propose(self, ctx: ProposeContext) -> DraftTokenIds | None:
        """Return per-request draft tokens for the next step, or ``None``."""
        ...

    def release_requests(self, req_ids: set[str]) -> None:
        """Release any per-request drafter state for these evicted/preempted ids.

        Called from the runner's lifecycle reconcile on eviction, preemption, and
        resume. A proposer that pins a bounded per-request resource (draft cache
        blocks) must release it here rather than hold it while the request waits;
        a stateless proposer is a no-op.
        """
        ...


class Gemma4MTPProposer:
    """:class:`MetalProposer` backed by the in-model Gemma4 MTP assistant.

    The assistant is read lazily from the runner: cache setup replaces it with
    a KV-sharing-bound instance (see ``cache_policy.install_gemma4_mtp_kv_sharing``)
    after model load, so capturing it at construction time would pin the
    pre-sharing object.
    """

    def __init__(self, runner: MetalModelRunner) -> None:
        self._runner = runner

    def needs_target_hidden_states(
        self,
        decode_segments: Sequence[PagedDecodeSegment],
        *,
        has_final_prefill: bool,
    ) -> bool:
        # The assistant consumes the previous target step's hidden states for
        # decode and final-prefill rows; intermediate prefill chunks never
        # sample, so they cannot seed a draft.
        return bool(decode_segments) or has_final_prefill

    def release_requests(self, req_ids: set[str]) -> None:
        # The assistant reads the target's paged KV (released by the runtime);
        # the proposer holds no per-request state of its own.
        del req_ids

    def propose(self, ctx: ProposeContext) -> DraftTokenIds | None:
        if ctx.num_speculative_tokens <= 0:
            return None

        runner = self._runner
        assistant = runner._gemma4_mtp_assistant
        if (
            assistant is None
            or not assistant.forward_ready
            or ctx.target_hidden_states is None
        ):
            return None

        seeds = runner._spec_decode_controller.build_gemma4_mtp_draft_seeds(
            decode_reqs=ctx.decode_reqs,
            decode_segments=ctx.decode_segments,
            decode_token_ids=ctx.decode_token_ids,
            prefill_reqs=ctx.prefill_reqs,
            prefill_token_ids=ctx.prefill_token_ids,
            prefill_result_modes=ctx.prefill_result_modes,
            request_states=ctx.request_states,
            cu_seqlens=ctx.cu_seqlens,
            num_decode_segments=ctx.num_decode_segments,
        )
        if not seeds:
            return None

        input_ids = mx.array([[seed.token_id for seed in seeds]], dtype=mx.int32)
        target_input_embeddings = runner._target_input_embeddings(input_ids)
        draft_token_ids = assistant.propose_draft_token_ids(
            seeds=seeds,
            target_hidden_states=ctx.target_hidden_states,
            target_input_embeddings=target_input_embeddings,
        )
        if not draft_token_ids:
            return None

        return DraftTokenIds(
            req_ids=[seed.req_id for seed in seeds],
            draft_token_ids=draft_token_ids,
        )


@dataclass(slots=True)
class _QwenMTPRequestState:
    pending_hidden: mx.array | None
    next_mtp_position: int


class QwenNativeMTPProposer:
    """One-token native Qwen MTP proposer backed by scheduler-owned KV.

    The target hybrid runtime owns both the MTP-head paged KV and a target
    boundary-hidden shadow. A new request may therefore adopt a scheduler
    prefix hit without creating a fresh request-local MTP cache.

    vLLM's EAGLE cache group drops one hash unit from a warm hit, so the target
    recomputes a non-empty suffix. The retained MTP cache is valid through the
    recompute boundary; drafting resumes *at* ``prefill.start_pos`` rather than
    overwriting the retained shared tail at ``start_pos - 1``.
    """

    def __init__(self, runner: MetalModelRunner) -> None:
        self._runner = runner
        spec = runner.vllm_config.speculative_config
        if spec is None or spec.method != "mtp":
            raise ValueError("QwenNativeMTPProposer requires method='mtp'")
        width = int(spec.num_speculative_tokens or 0)
        if width != 1:
            raise ValueError(
                "Qwen native MTP on Metal currently supports exactly one "
                f"speculative token; got {width}."
            )
        model = runner._forward_model
        if not bool(getattr(model, "supports_mtp", False)):
            raise ValueError(
                "method='mtp' requires native MTP weights and supports_mtp=True"
            )
        if not callable(getattr(model, "mtp_forward", None)):
            raise ValueError("native MTP model must expose mtp_forward()")
        self._states: dict[str, _QwenMTPRequestState] = {}
        self._prefix_hit_blocked: set[str] = set()

    def needs_target_hidden_states(
        self,
        decode_segments: Sequence[PagedDecodeSegment],
        *,
        has_final_prefill: bool,
    ) -> bool:
        del decode_segments, has_final_prefill
        # Intermediate chunks populate both the MTP KV and the reusable
        # boundary-hidden shadow, so every Qwen MTP target forward retains them.
        return True

    def release_requests(self, req_ids: set[str]) -> None:
        for req_id in req_ids:
            self._states.pop(req_id, None)
            self._prefix_hit_blocked.discard(req_id)

    def _runtime(self):
        runtime = self._runner._paged_attention_runtime
        if runtime is None or not bool(getattr(runtime, "qwen_mtp_ready", False)):
            raise RuntimeError(
                "native Qwen MTP requires the scheduler-owned paged MTP runtime"
            )
        return runtime

    def _new_state(self, prefill) -> _QwenMTPRequestState | None:
        if prefill.start_pos == 0:
            return _QwenMTPRequestState(
                pending_hidden=None,
                next_mtp_position=0,
            )
        try:
            # Validate that the target prefix's durable boundary metadata exists
            # in the same scheduler-owned block lineage. EAGLE has already
            # dropped one hash unit, so the MTP KV itself resumes at start_pos.
            self._runtime().qwen_mtp_boundary_hidden(
                prefill.block_ids,
                prefill.start_pos - 1,
            )
        except RuntimeError:
            # Correctness-preserving fallback: target generation continues but
            # this request drafts nothing. Never combine the restored target
            # prefix with a fresh or incomplete MTP cache.
            self._prefix_hit_blocked.add(prefill.req_id)
            return None
        return _QwenMTPRequestState(
            pending_hidden=None,
            next_mtp_position=prefill.start_pos,
        )

    def _run_pairs_batch(
        self,
        items: Sequence[
            tuple[
                _QwenMTPRequestState,
                mx.array,
                Sequence[int],
                Sequence[Sequence[int]],
                int,
            ]
        ],
        *,
        draft_request_indices: Sequence[int] | None = None,
    ) -> list[int]:
        if not items:
            return []
        for state, hidden_rows, next_token_ids, _, start_pos in items:
            if hidden_rows.shape[0] != len(next_token_ids):
                raise RuntimeError(
                    "Qwen MTP hidden/token pair count mismatch: "
                    f"{hidden_rows.shape[0]} != {len(next_token_ids)}"
                )
            if state.next_mtp_position != start_pos:
                raise RuntimeError(
                    "Qwen MTP logical position mismatch: "
                    f"cache expects {state.next_mtp_position}, "
                    f"caller supplied {start_pos}"
                )

        drafts = self._runtime().qwen_mtp_run_pairs_batch(
            hidden_rows_batch=[item[1] for item in items],
            next_token_ids_batch=[item[2] for item in items],
            block_ids_by_group_batch=[item[3] for item in items],
            start_positions=[item[4] for item in items],
            draft_request_indices=draft_request_indices,
        )
        expected_drafts = (
            len(items) if draft_request_indices is None else len(draft_request_indices)
        )
        if len(drafts) != expected_drafts:
            raise RuntimeError(
                "Qwen MTP batched draft result count mismatch: "
                f"{len(drafts)} != {expected_drafts}"
            )
        for state, _, next_token_ids, _, _ in items:
            state.next_mtp_position += len(next_token_ids)
        return drafts

    def _run_pairs(
        self,
        state: _QwenMTPRequestState,
        hidden_rows: mx.array,
        next_token_ids: Sequence[int],
        block_ids_by_group: Sequence[Sequence[int]],
        *,
        start_pos: int,
    ) -> int:
        drafts = self._run_pairs_batch(
            [
                (
                    state,
                    hidden_rows,
                    next_token_ids,
                    block_ids_by_group,
                    start_pos,
                )
            ]
        )
        return drafts[0]

    def _advance_prefill(
        self,
        state: _QwenMTPRequestState,
        hidden_rows: mx.array,
        input_token_ids: Sequence[int],
        block_ids_by_group: Sequence[Sequence[int]],
    ) -> None:
        if hidden_rows.shape[0] != len(input_token_ids):
            raise RuntimeError("Qwen MTP prefill hidden/token length mismatch")
        if not input_token_ids:
            return
        pair_hidden: list[mx.array] = []
        pair_tokens: list[int] = []
        if state.pending_hidden is not None:
            pair_hidden.append(state.pending_hidden)
            pair_tokens.append(int(input_token_ids[0]))
        if len(input_token_ids) > 1:
            pair_hidden.append(hidden_rows[:-1])
            pair_tokens.extend(int(token) for token in input_token_ids[1:])
        if pair_tokens:
            self._run_pairs(
                state,
                mx.concatenate(pair_hidden, axis=0),
                pair_tokens,
                block_ids_by_group,
                start_pos=state.next_mtp_position,
            )
        state.pending_hidden = hidden_rows[-1:]

    def _draft_after_prefill_sample(
        self,
        state: _QwenMTPRequestState,
        sampled_token_id: int,
        block_ids_by_group: Sequence[Sequence[int]],
    ) -> int:
        if state.pending_hidden is None:
            raise RuntimeError("Qwen MTP final prefill has no boundary hidden state")
        draft = self._run_pairs(
            state,
            state.pending_hidden,
            [sampled_token_id],
            block_ids_by_group,
            start_pos=state.next_mtp_position,
        )
        state.pending_hidden = None
        return draft

    def propose(self, ctx: ProposeContext) -> DraftTokenIds | None:
        if ctx.num_speculative_tokens != 1:
            raise RuntimeError(
                "Qwen native MTP proposer received a non-one-token runtime width"
            )
        self.release_requests(ctx.finished_req_ids)
        hidden = ctx.target_hidden_states
        if hidden is None:
            return None

        draft_req_ids: list[str] = []
        drafts: list[list[int]] = []
        first_stage: list[
            tuple[
                _QwenMTPRequestState,
                mx.array,
                Sequence[int],
                Sequence[Sequence[int]],
                int,
            ]
        ] = []
        decode_indices: list[int] = []
        decode_req_ids: list[str] = []

        for (req_id, state), segment, sampled_ids in zip(
            ctx.decode_reqs,
            ctx.decode_segments,
            ctx.decode_token_ids,
            strict=True,
        ):
            if (
                not sampled_ids
                or not self._runner._spec_decode_controller.can_draft_greedy(
                    req_id, state
                )
            ):
                continue
            request_state = self._states.get(req_id)
            if request_state is None or req_id in self._prefix_hit_blocked:
                continue
            if request_state.next_mtp_position != segment.cache_start_pos:
                raise RuntimeError(
                    f"Qwen MTP request {req_id!r} expected target position "
                    f"{request_state.next_mtp_position}, "
                    f"got {segment.cache_start_pos}"
                )
            count = len(sampled_ids)
            hidden_rows = hidden[segment.start_row : segment.start_row + count]
            decode_indices.append(len(first_stage))
            decode_req_ids.append(req_id)
            first_stage.append(
                (
                    request_state,
                    hidden_rows,
                    sampled_ids,
                    segment.block_ids,
                    segment.cache_start_pos,
                )
            )

        pending_hidden_updates: list[tuple[_QwenMTPRequestState, mx.array]] = []
        final_prefills: list[
            tuple[str, _QwenMTPRequestState, int, Sequence[Sequence[int]]]
        ] = []

        for index, (prefill, sampled_token_id, result_mode) in enumerate(
            zip(
                ctx.prefill_reqs,
                ctx.prefill_token_ids,
                ctx.prefill_result_modes,
                strict=True,
            )
        ):
            req_id = prefill.req_id
            request_state = self._states.get(req_id)
            if request_state is None:
                request_state = self._new_state(prefill)
                if request_state is None:
                    continue
                self._states[req_id] = request_state
            if req_id in self._prefix_hit_blocked:
                continue

            expected_position = prefill.start_pos - int(
                request_state.pending_hidden is not None
            )
            if request_state.next_mtp_position != expected_position:
                raise RuntimeError(
                    f"Qwen MTP prefill {req_id!r} expected MTP position "
                    f"{expected_position}, found "
                    f"{request_state.next_mtp_position}"
                )

            start = ctx.cu_seqlens[ctx.num_decode_segments + index]
            end = ctx.cu_seqlens[ctx.num_decode_segments + index + 1]
            hidden_rows = hidden[start:end]
            input_token_ids = prefill.token_ids
            if hidden_rows.shape[0] != len(input_token_ids):
                raise RuntimeError("Qwen MTP prefill hidden/token length mismatch")
            if not input_token_ids:
                continue

            pair_hidden: list[mx.array] = []
            pair_tokens: list[int] = []
            if request_state.pending_hidden is not None:
                pair_hidden.append(request_state.pending_hidden)
                pair_tokens.append(int(input_token_ids[0]))
            if len(input_token_ids) > 1:
                pair_hidden.append(hidden_rows[:-1])
                pair_tokens.extend(int(token) for token in input_token_ids[1:])
            if pair_tokens:
                first_stage.append(
                    (
                        request_state,
                        mx.concatenate(pair_hidden, axis=0),
                        pair_tokens,
                        prefill.block_ids,
                        request_state.next_mtp_position,
                    )
                )
            pending_hidden_updates.append((request_state, hidden_rows[-1:]))

            if result_mode == "intermediate":
                continue
            state = ctx.request_states.get(req_id)
            if (
                state is None
                or not self._runner._spec_decode_controller.can_draft_greedy(
                    req_id, state
                )
            ):
                continue
            final_prefills.append(
                (
                    req_id,
                    request_state,
                    int(sampled_token_id),
                    prefill.block_ids,
                )
            )

        decode_drafts = self._run_pairs_batch(
            first_stage,
            draft_request_indices=decode_indices,
        )
        for request_state, pending_hidden in pending_hidden_updates:
            request_state.pending_hidden = pending_hidden
        for req_id, draft in zip(
            decode_req_ids,
            decode_drafts,
            strict=True,
        ):
            draft_req_ids.append(req_id)
            drafts.append([draft])

        final_stage: list[
            tuple[
                _QwenMTPRequestState,
                mx.array,
                Sequence[int],
                Sequence[Sequence[int]],
                int,
            ]
        ] = []
        final_req_ids: list[str] = []
        final_states: list[_QwenMTPRequestState] = []
        for req_id, request_state, sampled_token_id, block_ids in final_prefills:
            if request_state.pending_hidden is None:
                raise RuntimeError(
                    "Qwen MTP final prefill has no boundary hidden state"
                )
            final_req_ids.append(req_id)
            final_states.append(request_state)
            final_stage.append(
                (
                    request_state,
                    request_state.pending_hidden,
                    [sampled_token_id],
                    block_ids,
                    request_state.next_mtp_position,
                )
            )

        final_drafts = self._run_pairs_batch(final_stage)
        for request_state in final_states:
            request_state.pending_hidden = None
        for req_id, draft in zip(
            final_req_ids,
            final_drafts,
            strict=True,
        ):
            draft_req_ids.append(req_id)
            drafts.append([draft])

        if not drafts:
            return None
        return DraftTokenIds(
            req_ids=draft_req_ids,
            draft_token_ids=drafts,
        )
