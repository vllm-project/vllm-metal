# SPDX-License-Identifier: Apache-2.0
"""Tests for paged speculative decode metadata helpers."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import mlx.core as mx
import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor import LogitsProcessors, build_logitsprocs

from vllm_metal.v1.spec_decode import (
    PagedDecodeSegment,
    SpeculativeDecodeController,
)


def _state(token_ids: list[int], block_ids: list[int]) -> SimpleNamespace:
    return SimpleNamespace(token_ids=token_ids, block_ids=block_ids)


def _request_state(
    temperature: float = 0.0,
    *,
    generated_tokens: int = 1,
    min_tokens: int = 0,
    stop_token_ids: list[int] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        sampling_params=SamplingParams(
            temperature=temperature,
            min_tokens=min_tokens,
            stop_token_ids=stop_token_ids,
        ),
        generated_tokens=generated_tokens,
    )


def _scheduler_output(
    *,
    scheduled_spec_decode_tokens: dict[str, list[int]],
    num_scheduled_tokens: dict[str, int] | None = None,
    num_invalid_spec_tokens: dict[str, int] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
        num_scheduled_tokens=num_scheduled_tokens
        or {
            req_id: len(tokens) + 1
            for req_id, tokens in scheduled_spec_decode_tokens.items()
        },
        num_invalid_spec_tokens=num_invalid_spec_tokens,
    )


def _spec_logitsprocs():
    config = SimpleNamespace(
        speculative_config=object(),
        scheduler_config=SimpleNamespace(max_num_seqs=4),
    )
    return build_logitsprocs(
        config,
        torch.device("cpu"),
        is_pin_memory=False,
        is_pooling_model=False,
    )


def _logits(token_ids: list[int], vocab_size: int = 16) -> mx.array:
    rows = []
    for token_id in token_ids:
        row = [0.0] * vocab_size
        row[token_id] = 10.0
        rows.append(row)
    return mx.array([rows])


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
        segments = SpeculativeDecodeController().build_decode_segments(
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
        segments = SpeculativeDecodeController().build_decode_segments(
            [("r0", _state([5, 9, 17], [41, 42]))],
            scheduled_spec_decode_tokens=None,
            paged_request_seq_lens={},
        )

        assert segments[0].cache_start_pos == 2

    def test_draft_tokens_expand_the_row_span(self) -> None:
        segments = SpeculativeDecodeController().build_decode_segments(
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
        segments = SpeculativeDecodeController().build_decode_segments(
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

    def test_rejects_handoff_for_request_outside_decode_set(self) -> None:
        with pytest.raises(ValueError, match="outside the current decode set"):
            SpeculativeDecodeController().build_decode_segments(
                [("r0", _state([1, 2], [10]))],
                scheduled_spec_decode_tokens={"missing": [3]},
                paged_request_seq_lens={"r0": 1},
            )

    def test_rejects_invalid_draft_token_sentinel(self) -> None:
        with pytest.raises(NotImplementedError, match="invalid draft-token"):
            SpeculativeDecodeController().build_decode_segments(
                [("r0", _state([1, 2], [10]))],
                scheduled_spec_decode_tokens={"r0": [-1]},
                paged_request_seq_lens={"r0": 1},
            )


class TestSpecDecodePolicy:
    def test_empty_scheduled_tokens_are_supported(self) -> None:
        SpeculativeDecodeController().validate_supported(
            _scheduler_output(scheduled_spec_decode_tokens={}),
            (),
            paged_attention_enabled=False,
            is_hybrid=True,
            logitsprocs=None,
        )

    def test_non_paged_scheduled_tokens_are_rejected(self) -> None:
        with pytest.raises(NotImplementedError, match="requires paged attention"):
            SpeculativeDecodeController().validate_supported(
                _scheduler_output(scheduled_spec_decode_tokens={"r0": [1]}),
                [("r0", _request_state())],
                paged_attention_enabled=False,
                is_hybrid=False,
                logitsprocs=None,
            )

    def test_hybrid_scheduled_tokens_are_rejected(self) -> None:
        with pytest.raises(NotImplementedError, match="hybrid GDN"):
            SpeculativeDecodeController().validate_supported(
                _scheduler_output(scheduled_spec_decode_tokens={"r0": [1]}),
                [("r0", _request_state())],
                paged_attention_enabled=True,
                is_hybrid=True,
                logitsprocs=None,
            )

    def test_rejects_invalid_draft_token_sentinel(self) -> None:
        with pytest.raises(NotImplementedError, match="invalid draft-token"):
            SpeculativeDecodeController().validate_supported(
                _scheduler_output(scheduled_spec_decode_tokens={"r0": [-1]}),
                [("r0", _request_state())],
                paged_attention_enabled=True,
                is_hybrid=False,
                logitsprocs=None,
            )

    def test_rejects_scheduler_invalid_spec_tokens(self) -> None:
        with pytest.raises(NotImplementedError, match="scheduler-invalid"):
            SpeculativeDecodeController().validate_supported(
                _scheduler_output(
                    scheduled_spec_decode_tokens={"r0": [-1]},
                    num_invalid_spec_tokens={"r0": 1},
                ),
                [("r0", _request_state())],
                paged_attention_enabled=True,
                is_hybrid=False,
                logitsprocs=None,
            )

    def test_rejects_handoff_for_request_outside_decode_set(self) -> None:
        with pytest.raises(ValueError, match="outside the current decode set"):
            SpeculativeDecodeController().validate_supported(
                _scheduler_output(scheduled_spec_decode_tokens={"missing": [1]}),
                [("r0", _request_state())],
                paged_attention_enabled=True,
                is_hybrid=False,
                logitsprocs=None,
            )

    def test_rejects_empty_handoff_for_request_outside_decode_set(self) -> None:
        with pytest.raises(ValueError, match="outside the current decode set"):
            SpeculativeDecodeController().validate_supported(
                _scheduler_output(scheduled_spec_decode_tokens={"missing": []}),
                [("r0", _request_state())],
                paged_attention_enabled=True,
                is_hybrid=False,
                logitsprocs=None,
            )

    def test_rejects_mismatched_scheduler_token_accounting(self) -> None:
        with pytest.raises(ValueError, match="inconsistent token accounting"):
            SpeculativeDecodeController().validate_supported(
                _scheduler_output(
                    scheduled_spec_decode_tokens={"r0": [1, 2]},
                    num_scheduled_tokens={"r0": 2},
                ),
                [("r0", _request_state())],
                paged_attention_enabled=True,
                is_hybrid=False,
                logitsprocs=None,
            )

    def test_allows_inactive_min_tokens_processor_from_speculative_config(self) -> None:
        SpeculativeDecodeController().validate_supported(
            _scheduler_output(scheduled_spec_decode_tokens={"r0": [1]}),
            [
                (
                    "r0",
                    _request_state(
                        generated_tokens=2,
                        min_tokens=1,
                        stop_token_ids=[2],
                    ),
                )
            ],
            paged_attention_enabled=True,
            is_hybrid=False,
            logitsprocs=_spec_logitsprocs(),
        )

    def test_rejects_active_min_tokens_constraint(self) -> None:
        with pytest.raises(NotImplementedError, match="active min_tokens"):
            SpeculativeDecodeController().validate_supported(
                _scheduler_output(scheduled_spec_decode_tokens={"r0": [1]}),
                [
                    (
                        "r0",
                        _request_state(
                            generated_tokens=0,
                            min_tokens=3,
                            stop_token_ids=[2],
                        ),
                    )
                ],
                paged_attention_enabled=True,
                is_hybrid=False,
                logitsprocs=_spec_logitsprocs(),
            )


class TestVerifyGreedySpecDecode:
    def test_accepts_all_drafts_and_emits_bonus_token(self) -> None:
        segment = PagedDecodeSegment(
            req_id="r0",
            input_token_ids=(6, 7, 8),
            start_row=0,
            num_query_tokens=3,
            draft_token_ids=(7, 8),
            cache_start_pos=1,
            block_ids=(0,),
        )

        output = SpeculativeDecodeController().verify_greedy(
            _logits([7, 8, 9]),
            [("r0", _request_state())],
            (segment,),
            logitsprocs=None,
        )

        assert output == [[7, 8, 9]]

    def test_rejects_first_mismatched_draft_and_stops_before_bonus(self) -> None:
        segment = PagedDecodeSegment(
            req_id="r0",
            input_token_ids=(6, 7, 8),
            start_row=0,
            num_query_tokens=3,
            draft_token_ids=(7, 8),
            cache_start_pos=1,
            block_ids=(0,),
        )

        output = SpeculativeDecodeController().verify_greedy(
            _logits([7, 5, 9]),
            [("r0", _request_state())],
            (segment,),
            logitsprocs=None,
        )

        assert output == [[7, 5]]

    def test_rejects_non_greedy_sampling(self) -> None:
        segment = PagedDecodeSegment(
            req_id="r0",
            input_token_ids=(6, 7),
            start_row=0,
            num_query_tokens=2,
            draft_token_ids=(7,),
            cache_start_pos=1,
            block_ids=(0,),
        )

        with pytest.raises(NotImplementedError, match="greedy sampling"):
            SpeculativeDecodeController().verify_greedy(
                _logits([7, 9]),
                [("r0", _request_state(temperature=0.7))],
                (segment,),
                logitsprocs=None,
            )

    def test_rejects_logits_processors_that_can_change_argmax(self) -> None:
        class _NonArgmaxProcessor:
            def is_argmax_invariant(self) -> bool:
                return False

        with pytest.raises(NotImplementedError, match="logits processors"):
            SpeculativeDecodeController().verify_greedy(
                _logits([7, 9]),
                [("r0", _request_state())],
                (
                    PagedDecodeSegment(
                        req_id="r0",
                        input_token_ids=(6, 7),
                        start_row=0,
                        num_query_tokens=2,
                        draft_token_ids=(7,),
                        cache_start_pos=1,
                        block_ids=(0,),
                    ),
                ),
                logitsprocs=LogitsProcessors([_NonArgmaxProcessor()]),
            )
