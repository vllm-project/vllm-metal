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
from vllm.utils.math_utils import cdiv
from vllm.v1.outputs import DraftTokenIds

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from vllm_metal.v1.model_runner import (
        MetalModelRunner,
        PrefillRequest,
        RequestState,
    )
    from vllm_metal.v1.spec_decode import PagedDecodeSegment


# Ingests at or below this size are submitted as expanded decode rows instead
# of a prefill segment (see _ingest_and_draft_first). Covers the steady-state
# K+1-token ingest for any practical num_speculative_tokens while keeping
# full-prompt catch-up ingests on the tiled prefill kernel.
_DECODE_INGEST_MAX_TOKENS = 16


def validate_scheduler_blocks(
    req_id: str,
    block_ids: Sequence[int],
    block_size: int,
    *,
    total_positions: int,
) -> None:
    """Check capacity without allocating or changing the scheduler's block table."""
    needed = cdiv(total_positions, block_size)
    if needed > len(block_ids):
        raise RuntimeError(
            f"Draft KV allocation for request {req_id!r} needs "
            f"{needed} blocks for {total_positions} positions, but the "
            f"scheduler supplied {len(block_ids)}."
        )


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
    target_aux_hidden_states: tuple[mx.array, ...] = ()


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

        draft_token_ids = assistant.propose_draft_token_ids(
            seeds=seeds,
            target_hidden_states=ctx.target_hidden_states,
            # Each recurrence step embeds its own drafted token, so the
            # runtime needs the target's backbone-width table, not one
            # pre-computed row block.
            embed_target_tokens=runner._target_input_embeddings,
            num_speculative_tokens=ctx.num_speculative_tokens,
        )
        if not draft_token_ids:
            return None

        return DraftTokenIds(
            req_ids=[seed.req_id for seed in seeds],
            draft_token_ids=draft_token_ids,
        )
