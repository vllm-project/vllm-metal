# SPDX-License-Identifier: Apache-2.0
"""Tests for the sampling a transcription request asks the STT decode for."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import mlx.core as mx
import pytest
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.sampler import Sampler

from vllm_metal.stt.sampling import STTSampling

VOCAB_SIZE = 1024
SEEDS = range(16)
DECODE_STEPS = 12
# Far enough below the scored ids that top-p and top-k never reach them.
_UNSCORED_LOGIT = -30.0
# One leading token, and a runner-up close enough for a penalty to overtake it.
LEADER, RUNNER_UP = 5, 2
_TWO_CANDIDATES = {RUNNER_UP: 1.0, LEADER: 1.5}
_FOUR_TIED = dict.fromkeys((1, 2, 5, 6), 1.0)


def _step_logits(scores: dict[int, float]) -> mx.array:
    """A fresh step of logits; the sampler penalizes the array it is given."""
    row = [scores.get(i, _UNSCORED_LOGIT) for i in range(VOCAB_SIZE)]
    return mx.array([row], dtype=mx.float32)


def _sampling(params: SamplingParams) -> STTSampling:
    return STTSampling.from_request(params, Sampler())


def _pick(
    params: SamplingParams,
    scores: dict[int, float],
    *,
    prompt_token_ids: Sequence[int] = (),
    generated: Sequence[int] = (),
) -> int:
    return _sampling(params).next_token(
        list(prompt_token_ids), list(generated), _step_logits(scores)
    )


def _decode(params: SamplingParams, scores: dict[int, float]) -> list[int]:
    sampling = _sampling(params)
    generated: list[int] = []
    for _ in range(DECODE_STEPS):
        generated.append(sampling.next_token([], generated, _step_logits(scores)))
    return generated


def _picked_over_seeds(
    make_params: Callable[[int], SamplingParams], scores: dict[int, float]
) -> set[int]:
    return {_pick(make_params(seed), scores) for seed in SEEDS}


class TestGreedyDecode:
    def test_default_request_picks_the_top_logit(self) -> None:
        assert _pick(SamplingParams(temperature=0.0), _TWO_CANDIDATES) == LEADER

    def test_request_without_sampling_params_picks_the_top_logit(self) -> None:
        sampling = STTSampling.from_request(None, Sampler())

        assert sampling.next_token([], [], _step_logits(_TWO_CANDIDATES)) == LEADER

    def test_allowed_token_ids_confine_the_pick(self) -> None:
        """vLLM runs Whisper language detection as an allowed_token_ids request."""
        params = SamplingParams(temperature=0.0, allowed_token_ids=[RUNNER_UP])

        assert _pick(params, _TWO_CANDIDATES) == RUNNER_UP


class TestSampledDecode:
    def test_temperature_reaches_below_the_top_logit(self) -> None:
        picked = _picked_over_seeds(
            lambda seed: SamplingParams(temperature=1.0, seed=seed),
            dict.fromkeys((RUNNER_UP, LEADER), 1.0),
        )

        assert picked == {RUNNER_UP, LEADER}

    @pytest.mark.parametrize(
        ("control", "scores", "expected"),
        [
            ({"top_k": 2}, {1: 3.0, 2: 3.0, 5: 2.9, 6: 2.9}, {1, 2}),
            ({"top_p": 0.5}, {1: 6.0, 2: 1.0, 5: 1.0}, {1}),
            ({"min_p": 0.5}, {1: 6.0, 2: 1.0, 5: 1.0}, {1}),
        ],
        ids=["top_k", "top_p", "min_p"],
    )
    def test_candidate_control_reaches_the_sampler(
        self, control: dict[str, float], scores: dict[int, float], expected: set[int]
    ) -> None:
        picked = _picked_over_seeds(
            lambda seed: SamplingParams(temperature=1.0, seed=seed, **control), scores
        )

        assert picked == expected


class TestSeededDecode:
    def test_same_seed_repeats_the_transcript(self) -> None:
        params = SamplingParams(temperature=1.0, seed=11)

        assert _decode(params, _FOUR_TIED) == _decode(params, _FOUR_TIED)

    def test_different_seeds_diverge(self) -> None:
        first = _decode(SamplingParams(temperature=1.0, seed=11), _FOUR_TIED)
        second = _decode(SamplingParams(temperature=1.0, seed=12), _FOUR_TIED)

        assert first != second


class TestPenalties:
    @pytest.mark.parametrize(
        "params",
        [
            SamplingParams(temperature=0.0, repetition_penalty=2.0),
            SamplingParams(temperature=0.0, presence_penalty=2.0),
            SamplingParams(temperature=0.0, frequency_penalty=2.0),
        ],
        ids=["repetition", "presence", "frequency"],
    )
    def test_penalty_demotes_an_already_generated_token(
        self, params: SamplingParams
    ) -> None:
        fresh = _pick(params, _TWO_CANDIDATES)
        repeated = _pick(params, _TWO_CANDIDATES, generated=[LEADER])

        assert (fresh, repeated) == (LEADER, RUNNER_UP)

    def test_repetition_penalty_counts_the_prompt_like_vllm(self) -> None:
        params = SamplingParams(temperature=0.0, repetition_penalty=2.0)

        in_prompt = _pick(params, _TWO_CANDIDATES, prompt_token_ids=[LEADER])
        not_in_prompt = _pick(params, _TWO_CANDIDATES, prompt_token_ids=[9])

        assert (in_prompt, not_in_prompt) == (RUNNER_UP, LEADER)

    def test_output_penalties_ignore_the_prompt_like_vllm(self) -> None:
        params = SamplingParams(temperature=0.0, presence_penalty=2.0)

        picked = _pick(params, _TWO_CANDIDATES, prompt_token_ids=[LEADER])

        assert picked == LEADER
