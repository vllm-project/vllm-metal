# SPDX-License-Identifier: Apache-2.0
"""Tests for the token lists the STT decode hands the shared sampler."""

from __future__ import annotations

from collections.abc import Sequence

import mlx.core as mx
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.sampler import Sampler

from vllm_metal.stt.sampling import STTSampling

VOCAB_SIZE = 1024
# Far enough below the scored ids that the rest of vocab never wins.
_UNSCORED_LOGIT = -30.0
# One leading token, and a runner-up close enough for a penalty to overtake it.
LEADER, RUNNER_UP = 5, 2
_TWO_CANDIDATES = {RUNNER_UP: 1.0, LEADER: 1.5}


def _step_logits(scores: dict[int, float]) -> mx.array:
    """A fresh step of logits; the sampler penalizes the array it is given."""
    row = [scores.get(i, _UNSCORED_LOGIT) for i in range(VOCAB_SIZE)]
    return mx.array([row], dtype=mx.float32)


def _pick(
    params: SamplingParams,
    *,
    prompt_token_ids: Sequence[int] = (),
    generated: Sequence[int] = (),
) -> int:
    sampling = STTSampling.from_request(params, Sampler())
    return sampling.next_token(
        list(prompt_token_ids), list(generated), _step_logits(_TWO_CANDIDATES)
    )


class TestPenaltyScope:
    """The decode owns which token lists reach the sampler."""

    def test_repetition_penalty_counts_the_prompt(self) -> None:
        params = SamplingParams(temperature=0.0, repetition_penalty=2.0)

        in_prompt = _pick(params, prompt_token_ids=[LEADER])
        not_in_prompt = _pick(params, prompt_token_ids=[9])

        assert (in_prompt, not_in_prompt) == (RUNNER_UP, LEADER)

    def test_output_penalties_count_only_the_transcript(self) -> None:
        params = SamplingParams(temperature=0.0, presence_penalty=2.0)

        in_prompt = _pick(params, prompt_token_ids=[LEADER])
        in_transcript = _pick(params, generated=[LEADER])

        assert (in_prompt, in_transcript) == (LEADER, RUNNER_UP)
