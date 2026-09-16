# SPDX-License-Identifier: Apache-2.0
"""Request sampling for the one-shot STT decode."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.sampler import Sampler

from vllm_metal.v1.sampling_batch import (
    SamplingBatch,
    create_request_generator,
    sample_from_logits,
)


@dataclass(frozen=True, slots=True)
class STTSampling:
    """How one transcription request turns decode logits into the next token id."""

    sampling_params: SamplingParams
    generator: torch.Generator | None
    sampler: Sampler

    @classmethod
    def from_request(
        cls, sampling_params: SamplingParams | None, sampler: Sampler
    ) -> STTSampling:
        """Build the sampling a request asked for, greedy when it asked for none."""
        # A decode with no request behind it has no token budget either, so the
        # stand-in must not import ``SamplingParams``' own ``max_tokens`` default.
        params = sampling_params or SamplingParams(temperature=0.0, max_tokens=None)
        return cls(
            sampling_params=params,
            generator=create_request_generator(params),
            sampler=sampler,
        )

    def decode_budget(self, model_limit: int) -> int:
        """Cap the decode at what the request and the model both allow."""
        requested = self.sampling_params.max_tokens
        return model_limit if requested is None else min(requested, model_limit)

    def next_token(
        self,
        prompt_token_ids: list[int],
        generated: list[int],
        step_logits: mx.array,
    ) -> int:
        """Pick the next transcript token from one step of decode logits."""
        batch = SamplingBatch(
            [self.sampling_params],
            [prompt_token_ids],
            [generated],
            vocab_size=int(step_logits.shape[-1]),
            generators={} if self.generator is None else {0: self.generator},
        )
        return sample_from_logits(step_logits, batch, self.sampler).token_ids[0]
