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

    from vllm.v1.sample.logits_processor import LogitsProcessors

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
    logitsprocs: LogitsProcessors | None


class MetalProposer(Protocol):
    """Uniform drafting seam.

    Implementations: :class:`Gemma4MTPProposer`, and (draft-model SD) a
    ``DraftModelProposer``.
    """

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
        # Delegate to the controller (single source of truth, method-keyed):
        # the assistant consumes the previous target step's hidden states.
        runner = self._runner
        return runner._spec_decode_controller.needs_target_hidden_states(
            decode_segments,
            has_final_prefill=has_final_prefill,
            speculative_config=runner.vllm_config.speculative_config,
        )

    def propose(self, ctx: ProposeContext) -> DraftTokenIds | None:
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
            logitsprocs=ctx.logitsprocs,
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
