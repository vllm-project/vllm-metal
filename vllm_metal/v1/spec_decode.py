# SPDX-License-Identifier: Apache-2.0
"""Speculative decode ownership for the Metal paged target path."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import mlx.core as mx
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.logits_processor.builtin import MinTokensLogitsProcessor

from vllm_metal.v1.sampling_batch import GREEDY_TEMPERATURE_EPS


class _PagedDecodeStateLike(Protocol):
    token_ids: Sequence[int]
    block_ids: Sequence[int]


class _SpecDecodeRequestStateLike(Protocol):
    sampling_params: Any
    generated_tokens: int


@dataclass(frozen=True, slots=True)
class PagedDecodeSegment:
    """Logits-row metadata for one paged decode request.

    ``input_token_ids`` is the target-model verification span:
    ``[last_token, *draft_token_ids]``. Each input token produces one logits
    row; draft rows verify scheduled draft tokens, and the final row is the
    bonus token if all drafts are accepted.
    """

    req_id: str
    input_token_ids: tuple[int, ...]
    start_row: int
    num_query_tokens: int
    draft_token_ids: tuple[int, ...]
    cache_start_pos: int
    block_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.start_row < 0:
            raise ValueError("start_row must be non-negative")
        if self.num_query_tokens != len(self.input_token_ids):
            raise ValueError("num_query_tokens must match input_token_ids")
        if self.num_query_tokens != len(self.draft_token_ids) + 1:
            raise ValueError(
                "num_query_tokens must include one bonus row after draft tokens"
            )

    @property
    def draft_verification_rows(self) -> tuple[int, ...]:
        return tuple(range(self.start_row, self.start_row + len(self.draft_token_ids)))

    @property
    def bonus_row(self) -> int:
        return self.start_row + len(self.draft_token_ids)


class SpeculativeDecodeController:
    """Owns Metal's current target-side speculative decode contract."""

    def validate_supported(
        self,
        scheduler_output: SchedulerOutput,
        decode_reqs: Sequence[tuple[str, _SpecDecodeRequestStateLike]],
        *,
        paged_attention_enabled: bool,
        is_hybrid: bool,
        logitsprocs: LogitsProcessors | None,
    ) -> None:
        """Fail fast for unsupported or inconsistent scheduler handoffs."""
        spec_tokens = scheduler_output.scheduled_spec_decode_tokens
        invalid_counts = scheduler_output.num_invalid_spec_tokens or {}
        if not spec_tokens and not invalid_counts:
            return

        active_spec_tokens = {
            req_id: tuple(tokens) for req_id, tokens in spec_tokens.items() if tokens
        }
        has_invalid_spec_tokens = any(count > 0 for count in invalid_counts.values())

        if (
            active_spec_tokens or has_invalid_spec_tokens
        ) and not paged_attention_enabled:
            raise NotImplementedError(
                "Speculative decode verification on Metal requires paged "
                "attention so draft-token rows can share scheduler-assigned "
                "KV slots."
            )
        if (active_spec_tokens or has_invalid_spec_tokens) and is_hybrid:
            raise NotImplementedError(
                "Speculative decode verification is not supported for hybrid "
                "GDN models on Metal yet."
            )

        decode_req_ids = {req_id for req_id, _ in decode_reqs}
        unexpected_req_ids = sorted(set(spec_tokens) - decode_req_ids)
        if unexpected_req_ids:
            raise ValueError(
                "Speculative decode scheduler handoff referenced requests "
                f"outside the current decode set: {unexpected_req_ids}"
            )

        if has_invalid_spec_tokens:
            raise NotImplementedError(
                "Speculative decode verification on Metal does not support "
                "scheduler-invalid draft-token sentinels yet."
            )

        if not active_spec_tokens:
            return

        request_by_id = dict(decode_reqs)
        draft_reqs: list[tuple[str, _SpecDecodeRequestStateLike]] = []
        for req_id, draft_token_ids in active_spec_tokens.items():
            if any(token_id < 0 for token_id in draft_token_ids):
                raise NotImplementedError(
                    "Speculative decode verification on Metal does not support "
                    "invalid draft-token sentinels yet."
                )

            expected_num_scheduled = len(draft_token_ids) + 1
            actual_num_scheduled = scheduler_output.num_scheduled_tokens.get(req_id)
            if actual_num_scheduled != expected_num_scheduled:
                raise ValueError(
                    "Speculative decode scheduler handoff has inconsistent "
                    f"token accounting for {req_id!r}: expected "
                    f"{expected_num_scheduled}, got {actual_num_scheduled}"
                )
            draft_reqs.append((req_id, request_by_id[req_id]))

        self._validate_greedy_sampling(draft_reqs, logitsprocs=logitsprocs)

    def build_decode_segments(
        self,
        decode_reqs: Sequence[tuple[str, _PagedDecodeStateLike]],
        scheduled_spec_decode_tokens: Mapping[str, Sequence[int]] | None,
        paged_request_seq_lens: Mapping[str, int],
    ) -> tuple[PagedDecodeSegment, ...]:
        """Build row-span metadata for paged decode requests."""
        spec_tokens = scheduled_spec_decode_tokens or {}
        decode_req_ids = {req_id for req_id, _ in decode_reqs}
        unexpected_req_ids = sorted(set(spec_tokens) - decode_req_ids)
        if unexpected_req_ids:
            raise ValueError(
                "Speculative decode scheduler handoff referenced requests "
                f"outside the current decode set: {unexpected_req_ids}"
            )

        segments: list[PagedDecodeSegment] = []
        start_row = 0

        for req_id, state in decode_reqs:
            token_ids = tuple(state.token_ids)
            block_ids = tuple(state.block_ids)
            draft_token_ids = tuple(spec_tokens.get(req_id, ()))
            if any(token_id < 0 for token_id in draft_token_ids):
                raise NotImplementedError(
                    "Speculative decode verification on Metal does not support "
                    "invalid draft-token sentinels yet."
                )
            last_token = token_ids[-1] if token_ids else 0
            input_token_ids = (last_token, *draft_token_ids)
            cache_start_pos = paged_request_seq_lens.get(req_id, len(token_ids) - 1)

            segment = PagedDecodeSegment(
                req_id=req_id,
                input_token_ids=input_token_ids,
                start_row=start_row,
                num_query_tokens=len(input_token_ids),
                draft_token_ids=draft_token_ids,
                cache_start_pos=cache_start_pos,
                block_ids=block_ids,
            )
            segments.append(segment)
            start_row += segment.num_query_tokens

        return tuple(segments)

    def verify_greedy(
        self,
        logits: mx.array,
        decode_reqs: Sequence[tuple[str, _SpecDecodeRequestStateLike]],
        decode_segments: Sequence[PagedDecodeSegment],
        *,
        logitsprocs: LogitsProcessors | None,
    ) -> list[list[int]]:
        """Verify scheduled draft tokens with greedy target logits."""
        if len(decode_reqs) != len(decode_segments):
            raise ValueError("decode_reqs and decode_segments must have equal length")

        if not decode_segments:
            return []

        for (req_id, _), segment in zip(decode_reqs, decode_segments, strict=True):
            if req_id != segment.req_id:
                raise ValueError(
                    "Speculative decode verification received mismatched request "
                    f"metadata: {req_id!r} != {segment.req_id!r}"
                )

        self._validate_greedy_sampling(decode_reqs, logitsprocs=logitsprocs)

        num_decode_rows = max(
            segment.start_row + segment.num_query_tokens for segment in decode_segments
        )
        target_tokens = mx.argmax(logits[0, :num_decode_rows, :], axis=-1)
        mx.eval(target_tokens)
        flat_target_tokens: list[int] = target_tokens.tolist()  # type: ignore[assignment]

        sampled_token_ids: list[list[int]] = []
        for segment in decode_segments:
            row_tokens = flat_target_tokens[
                segment.start_row : segment.start_row + segment.num_query_tokens
            ]
            output_ids: list[int] = []
            rejected = False

            for draft_index, draft_token_id in enumerate(segment.draft_token_ids):
                target_token_id = int(row_tokens[draft_index])
                if target_token_id == draft_token_id:
                    output_ids.append(draft_token_id)
                    continue

                output_ids.append(target_token_id)
                rejected = True
                break

            if not rejected:
                output_ids.append(int(row_tokens[len(segment.draft_token_ids)]))
            sampled_token_ids.append(output_ids)

        return sampled_token_ids

    def _validate_greedy_sampling(
        self,
        decode_reqs: Sequence[tuple[str, _SpecDecodeRequestStateLike]],
        *,
        logitsprocs: LogitsProcessors | None,
    ) -> None:
        for _, request_state in decode_reqs:
            sampling_params = request_state.sampling_params
            unsupported = (
                sampling_params.temperature >= GREEDY_TEMPERATURE_EPS
                or sampling_params.top_k > 0
                or sampling_params.top_p != 1.0
                or sampling_params.frequency_penalty != 0.0
                or sampling_params.presence_penalty != 0.0
                or sampling_params.repetition_penalty != 1.0
                or sampling_params.logprobs is not None
                or bool(sampling_params.allowed_token_ids)
                or bool(sampling_params.bad_words_token_ids)
            )
            if unsupported:
                raise NotImplementedError(
                    "Speculative decode verification on Metal currently "
                    "supports greedy sampling only (temperature=0, no "
                    "penalties, constraints, or logprobs)."
                )

            if (
                sampling_params.min_tokens
                and request_state.generated_tokens < sampling_params.min_tokens
                and sampling_params.all_stop_token_ids
            ):
                raise NotImplementedError(
                    "Speculative decode verification on Metal does not support "
                    "active min_tokens constraints yet."
                )

        if logitsprocs is None:
            return

        unsupported_processors = [
            processor
            for processor in logitsprocs.non_argmax_invariant
            if not isinstance(processor, MinTokensLogitsProcessor)
        ]
        if unsupported_processors:
            raise NotImplementedError(
                "Speculative decode verification on Metal currently supports "
                "greedy sampling only; non-argmax logits processors are not "
                "supported."
            )


__all__ = [
    "PagedDecodeSegment",
    "SpeculativeDecodeController",
]
