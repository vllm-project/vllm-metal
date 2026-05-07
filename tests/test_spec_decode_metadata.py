# SPDX-License-Identifier: Apache-2.0
"""Tests for paged speculative decode metadata helpers."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from vllm_metal.v1.spec_decode import (
    PagedDecodeSegment,
    build_paged_decode_segments,
)


def _state(token_ids: list[int], block_ids: list[int]) -> SimpleNamespace:
    return SimpleNamespace(token_ids=token_ids, block_ids=block_ids)


class TestPagedDecodeSegment:
    def test_freezes_single_row_decode_shape(self) -> None:
        segment = PagedDecodeSegment(
            req_id="r0",
            input_token_ids=(9,),
            start_row=0,
            num_query_tokens=1,
            draft_token_ids=(),
            cache_start_pos=7,
            block_ids=(11, 12),
        )

        assert segment.req_id == "r0"
        assert segment.input_token_ids == (9,)
        assert segment.draft_token_ids == ()
        assert segment.draft_verification_rows == ()
        assert segment.bonus_row == 0

        with pytest.raises(FrozenInstanceError):
            segment.start_row = 1  # type: ignore[misc]


class TestBuildPagedDecodeSegments:
    def test_single_row_decode_matches_current_shape(self) -> None:
        segments = build_paged_decode_segments(
            [("r0", _state([5, 9], [41, 42]))],
            scheduled_spec_decode_tokens={},
            paged_request_seq_lens={"r0": 7},
        )

        assert len(segments) == 1
        segment = segments[0]
        assert segment.req_id == "r0"
        assert segment.input_token_ids == (9,)
        assert segment.start_row == 0
        assert segment.num_query_tokens == 1
        assert segment.draft_token_ids == ()
        assert segment.cache_start_pos == 7
        assert segment.block_ids == (41, 42)
        assert segment.draft_verification_rows == ()
        assert segment.bonus_row == 0

    def test_single_row_decode_uses_len_minus_one_fallback(self) -> None:
        segments = build_paged_decode_segments(
            [("r0", _state([5, 9, 17], [41, 42]))],
            scheduled_spec_decode_tokens=None,
            paged_request_seq_lens={},
        )

        assert segments[0].cache_start_pos == 2

    def test_draft_tokens_expand_the_row_span(self) -> None:
        segments = build_paged_decode_segments(
            [("r0", _state([5, 9], [41, 42]))],
            scheduled_spec_decode_tokens={"r0": [23, 24]},
            paged_request_seq_lens={"r0": 7},
        )

        segment = segments[0]
        assert segment.input_token_ids == (9, 23, 24)
        assert segment.num_query_tokens == 3
        assert segment.draft_token_ids == (23, 24)
        assert segment.start_row == 0
        assert segment.draft_verification_rows == (0, 1)
        assert segment.bonus_row == 2

    def test_mixed_batch_uses_cumulative_start_rows(self) -> None:
        segments = build_paged_decode_segments(
            [
                ("r0", _state([1, 2], [10])),
                ("r1", _state([3, 4, 5], [11])),
                ("r2", _state([6], [12])),
            ],
            scheduled_spec_decode_tokens={
                "r1": [21, 22],
                "r2": [31],
            },
            paged_request_seq_lens={"r0": 1, "r1": 2, "r2": 0},
        )

        assert [segment.start_row for segment in segments] == [0, 1, 4]
        assert [segment.num_query_tokens for segment in segments] == [1, 3, 2]
        assert segments[1].draft_verification_rows == (1, 2)
        assert segments[1].bonus_row == 3
        assert segments[2].draft_verification_rows == (4,)
        assert segments[2].bonus_row == 5
