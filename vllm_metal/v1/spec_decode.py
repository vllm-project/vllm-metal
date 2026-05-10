# SPDX-License-Identifier: Apache-2.0
"""Paged decode metadata helpers for speculative token row spans."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import mlx.core as mx

from vllm_metal.v1.sampling_batch import GREEDY_TEMPERATURE_EPS


class _PagedDecodeStateLike(Protocol):
    token_ids: Sequence[int]
    block_ids: Sequence[int]


class _SpecDecodeRequestStateLike(Protocol):
    sampling_params: Any


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


def build_paged_decode_segments(
    decode_reqs: Sequence[tuple[str, _PagedDecodeStateLike]],
    scheduled_spec_decode_tokens: Mapping[str, Sequence[int]] | None,
    paged_request_seq_lens: Mapping[str, int],
) -> tuple[PagedDecodeSegment, ...]:
    """Build row-span metadata for paged decode requests.

    The returned metadata mirrors the current no-draft decode shape when a
    request has no scheduled draft tokens.
    """
    spec_tokens = scheduled_spec_decode_tokens or {}
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


def has_scheduled_spec_decode_tokens(
    scheduled_spec_decode_tokens: Mapping[str, Sequence[int]] | None,
) -> bool:
    """Return whether this scheduler output contains draft tokens."""
    if not scheduled_spec_decode_tokens:
        return False
    return any(bool(tokens) for tokens in scheduled_spec_decode_tokens.values())


def unsupported_spec_decode_reason(
    scheduled_spec_decode_tokens: Mapping[str, Sequence[int]] | None,
    *,
    paged_attention_enabled: bool,
    is_hybrid: bool,
) -> str | None:
    """Return why this scheduled spec-decode step is unsupported, if any."""
    if not has_scheduled_spec_decode_tokens(scheduled_spec_decode_tokens):
        return None

    if not paged_attention_enabled:
        return (
            "Speculative decode verification on Metal requires paged "
            "attention so draft-token rows can share scheduler-assigned "
            "KV slots."
        )
    if is_hybrid:
        return (
            "Speculative decode verification is not supported for hybrid "
            "GDN models on Metal yet."
        )
    return None


def validate_greedy_spec_decode_sampling(
    decode_reqs: Sequence[tuple[str, _SpecDecodeRequestStateLike]],
    *,
    logitsprocs: Any | None = None,
) -> None:
    """Raise if this batch needs sampling behavior beyond greedy argmax."""
    if getattr(logitsprocs, "non_argmax_invariant", ()):
        raise NotImplementedError(
            "Speculative decode verification on Metal currently supports "
            "greedy sampling only; logits processors that can change argmax "
            "are not supported."
        )

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
                "Speculative decode verification on Metal currently supports "
                "greedy sampling only (temperature=0, no penalties, "
                "constraints, or logprobs)."
            )


def verify_greedy_spec_decode(
    logits: mx.array,
    decode_reqs: Sequence[tuple[str, _SpecDecodeRequestStateLike]],
    decode_segments: Sequence[PagedDecodeSegment],
    num_decode_tokens: int,
    *,
    logitsprocs: Any | None = None,
) -> list[list[int]]:
    """Verify scheduled draft tokens with greedy target logits.

    The target rows verify each draft token in order. If every draft token is
    accepted, the final target row becomes the bonus token.
    """
    validate_greedy_spec_decode_sampling(decode_reqs, logitsprocs=logitsprocs)

    target_tokens = mx.argmax(logits[0, :num_decode_tokens, :], axis=-1)
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


__all__ = [
    "PagedDecodeSegment",
    "build_paged_decode_segments",
    "has_scheduled_spec_decode_tokens",
    "unsupported_spec_decode_reason",
    "validate_greedy_spec_decode_sampling",
    "verify_greedy_spec_decode",
]
