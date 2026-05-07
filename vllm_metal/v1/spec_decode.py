# SPDX-License-Identifier: Apache-2.0
"""Paged decode metadata helpers for speculative token row spans."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol


class _PagedDecodeStateLike(Protocol):
    token_ids: Sequence[int]
    block_ids: Sequence[int]


@dataclass(frozen=True, slots=True)
class PagedDecodeSegment:
    """Logits-row metadata for one paged decode request."""

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


__all__ = [
    "PagedDecodeSegment",
    "build_paged_decode_segments",
]
