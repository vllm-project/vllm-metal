# SPDX-License-Identifier: Apache-2.0
"""Tests for thinking_token_budget enforcement on the Metal sampler."""

from types import SimpleNamespace

import mlx.core as mx
import pytest
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.sampler import Sampler

from vllm_metal.v1.sampling_batch import SamplingBatch, sample_from_logits
from vllm_metal.v1.spec_decode import SpeculativeDecodeController
from vllm_metal.v1.thinking_budget import (
    ThinkingBudgetMarkers,
    forced_end_token,
)

START = 7
END = 8
VOCAB_SIZE = 32
# Single-token markers, as for Qwen3 <think>/</think>.
MARKERS = ThinkingBudgetMarkers(start=(START,), end=(END,), natural_end=(END,))
# A forced end sequence of three tokens whose last token is the natural end.
MULTI_END = ThinkingBudgetMarkers(start=(START,), end=(20, 21, END), natural_end=(END,))


class TestForcedEndToken:
    @pytest.mark.parametrize(
        ("token_ids", "budget", "expected"),
        [
            ([1, 2, 3], 0, None),  # never started thinking
            ([1, START, 2, 3], 3, None),  # 2 reasoning tokens < budget 3
            ([1, START, 2, 3, 4], 3, END),  # budget reached
            ([1, START], 0, END),  # budget 0 forces right after the start
            ([1, START, 2, 3, 4, END, 5], 1, None),  # thinking already ended
            # A second thinking section is budgeted from its own start.
            ([START, 2, END, 5, START, 9], 2, None),
            ([START, 2, END, 5, START, 9, 9], 2, END),
        ],
    )
    def test_single_token_markers(self, token_ids, budget, expected) -> None:
        assert forced_end_token(token_ids, budget, MARKERS) == expected

    @pytest.mark.parametrize(
        ("token_ids", "expected"),
        [
            ([START, 1, 2], 20),
            ([START, 1, 2, 20], 21),
            ([START, 1, 2, 20, 21], END),
        ],
    )
    def test_multi_token_end_continues_from_the_tail(self, token_ids, expected) -> None:
        assert forced_end_token(token_ids, 2, MULTI_END) == expected

    def test_multi_token_end_completed_stops_forcing(self) -> None:
        assert forced_end_token([START, 1, 2, 20, 21, END], 2, MULTI_END) is None


class TestMarkersFromReasoningConfig:
    def test_none_without_reasoning_config(self) -> None:
        assert ThinkingBudgetMarkers.from_reasoning_config(None) is None

    def test_requires_all_three_sequences(self) -> None:
        config = SimpleNamespace(
            reasoning_start_token_ids=[START],
            reasoning_end_token_ids=[END],
            natural_reasoning_end_token_ids=None,
        )
        assert ThinkingBudgetMarkers.from_reasoning_config(config) is None

    def test_reads_token_ids(self) -> None:
        config = SimpleNamespace(
            reasoning_start_token_ids=[START],
            reasoning_end_token_ids=[20, 21, END],
            natural_reasoning_end_token_ids=[END],
        )
        assert ThinkingBudgetMarkers.from_reasoning_config(config) == MULTI_END


def _logits_favoring(token_ids: list[int]) -> mx.array:
    """One row per entry; each row's argmax is the given token."""
    rows = []
    for token_id in token_ids:
        row = [0.0] * VOCAB_SIZE
        row[token_id] = 5.0
        rows.append(row)
    logits = mx.array(rows, dtype=mx.float32)
    mx.eval(logits)
    return logits


class TestSamplingBatchForcing:
    def test_over_budget_row_is_forced_and_others_are_not(self) -> None:
        params = [
            SamplingParams(temperature=0.0, thinking_token_budget=2),
            SamplingParams(temperature=0.0, thinking_token_budget=10),
            SamplingParams(temperature=0.0),
        ]
        batch = SamplingBatch(
            params,
            [[1, START], [1, START], [1, START]],
            [[3, 4], [3, 4], [3, 4]],
            vocab_size=VOCAB_SIZE,
            thinking_budget=MARKERS,
        )
        assert batch.forced_end_tokens == {0: END}
        assert not batch.can_use_native_greedy()

        result = sample_from_logits(_logits_favoring([5, 5, 5]), batch, Sampler())
        assert result.token_ids == [END, 5, 5]

    def test_random_sampling_is_forced_too(self) -> None:
        params = [SamplingParams(temperature=1.0, seed=0, thinking_token_budget=0)]
        batch = SamplingBatch(
            params, [[START]], [[]], vocab_size=VOCAB_SIZE, thinking_budget=MARKERS
        )
        result = sample_from_logits(_logits_favoring([5]), batch, Sampler())
        assert result.token_ids == [END]

    def test_logprobs_stay_raw(self) -> None:
        """Forcing happens after raw logprobs, as in vLLM's sampler."""
        params = [SamplingParams(temperature=0.0, logprobs=1, thinking_token_budget=0)]
        batch = SamplingBatch(
            params, [[START]], [[]], vocab_size=VOCAB_SIZE, thinking_budget=MARKERS
        )
        result = sample_from_logits(_logits_favoring([5]), batch, Sampler())
        assert result.token_ids == [END]
        assert result.logprobs is not None
        # Column 0 is the sampled token: its raw logprob, not ~0 from 1e9.
        sampled_logprob = result.logprobs.logprobs[0][0]
        assert sampled_logprob < -1.0

    def test_no_markers_leaves_budget_unenforced(self) -> None:
        params = [SamplingParams(temperature=0.0, thinking_token_budget=0)]
        batch = SamplingBatch(params, [[START]], [[]], vocab_size=VOCAB_SIZE)
        assert batch.forced_end_tokens == {}


class TestNativeAndSpecGates:
    def test_budget_disables_native_sampling(self) -> None:
        params = [SamplingParams(temperature=0.0, thinking_token_budget=4)]
        assert not SamplingBatch.params_allow_native_greedy(params)
        assert not SamplingBatch.params_allow_native_random(params)

    def test_budget_disables_drafting(self) -> None:
        state = SimpleNamespace(
            sampling_params=SamplingParams(
                temperature=0.0, max_tokens=16, thinking_token_budget=4
            ),
            generated_tokens=1,
        )
        assert not SpeculativeDecodeController().can_draft_greedy("r0", state)
