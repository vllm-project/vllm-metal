# SPDX-License-Identifier: Apache-2.0
"""Tests for MLX-native temperature/top-k/top-p sampling."""

import mlx.core as mx
import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p

from vllm_metal.v1.sampling_batch import SamplingBatch

VOCAB_SIZE = 512
BATCH_SIZE = 4


def _random_params(**overrides) -> SamplingParams:
    defaults = {"temperature": 0.7, "top_k": 20, "top_p": 0.95}
    defaults.update(overrides)
    return SamplingParams(**defaults)


class TestNativeRandomEligibility:
    def test_plain_temperature_top_k_top_p_batch_is_eligible(self) -> None:
        params = [
            _random_params(temperature=0.7),
            _random_params(temperature=1.0),
        ]

        assert SamplingBatch.params_allow_native_random(params)

    @pytest.mark.parametrize(
        ("label", "params_list"),
        [
            ("empty batch", []),
            ("greedy request", [_random_params(temperature=0.0)]),
            ("seeded request", [_random_params(seed=7)]),
            ("frequency penalty", [_random_params(frequency_penalty=0.5)]),
            ("presence penalty", [_random_params(presence_penalty=0.5)]),
            ("repetition penalty", [_random_params(repetition_penalty=1.2)]),
            ("sample logprobs", [_random_params(logprobs=1)]),
            (
                "logprob token ids",
                [_random_params(logprob_token_ids=[1, 2])],
            ),
            (
                "mixed greedy and random",
                [_random_params(), _random_params(temperature=0.0)],
            ),
            (
                "allowed token ids",
                [_random_params(allowed_token_ids=[1, 2])],
            ),
            (
                "mixed top_k",
                [_random_params(top_k=20), _random_params(top_k=40)],
            ),
            (
                "mixed top_p",
                [_random_params(top_p=0.95), _random_params(top_p=0.8)],
            ),
        ],
    )
    def test_ineligible_batches_fall_back(self, label, params_list) -> None:
        assert not SamplingBatch.params_allow_native_random(params_list)

    def test_bad_words_fall_back(self) -> None:
        params = _random_params()
        params._bad_words_token_ids = [[99]]

        assert not SamplingBatch.params_allow_native_random([params])


class TestTopKTopPMaskParity:
    """The MLX mask must keep exactly the candidates vLLM's mask keeps."""

    @pytest.mark.parametrize(
        ("top_k", "top_p"),
        [
            (20, 1.0),
            (0, 0.95),
            (20, 0.95),
            (1, 1.0),
            (0, 1.0),
            (VOCAB_SIZE, 0.5),
            (0, 0.999999),
        ],
    )
    def test_mask_matches_vllm_reference(self, top_k: int, top_p: float) -> None:
        logits = mx.random.normal((BATCH_SIZE, VOCAB_SIZE), key=mx.random.key(42))
        mx.eval(logits)
        logits_torch = torch.tensor(logits.tolist())
        k_arg = (
            None
            if top_k <= 0 or top_k >= VOCAB_SIZE
            else torch.full((BATCH_SIZE,), top_k)
        )
        p_arg = None if top_p == 1.0 else torch.full((BATCH_SIZE,), top_p)

        masked_mlx = SamplingBatch._top_k_top_p_masked_logits(logits, top_k, top_p)
        mx.eval(masked_mlx)
        masked_ref = apply_top_k_top_p(logits_torch.clone(), k_arg, p_arg)

        kept_mlx = [[v != float("-inf") for v in row] for row in masked_mlx.tolist()]
        kept_ref = (masked_ref != float("-inf")).tolist()
        assert kept_mlx == kept_ref

    def test_top_p_boundary_ties_are_masked_positionally(self) -> None:
        """Boundary ties must not all survive (vLLM masks sorted positions).

        Four equal logits at top_p=0.25: each holds 0.25 mass, so exactly
        one sorted position has leading mass < 0.25 and only one survives.
        """
        logits = mx.array([[2.0, 2.0, 2.0, 2.0]])
        logits_torch = torch.full((1, 4), 2.0)

        masked = SamplingBatch._top_k_top_p_masked_logits(logits, 0, 0.25)
        mx.eval(masked)
        masked_ref = apply_top_k_top_p(logits_torch, None, torch.full((1,), 0.25))

        kept_count = sum(v != float("-inf") for v in masked.tolist()[0])
        kept_count_ref = int((masked_ref != float("-inf")).sum().item())
        assert kept_count == kept_count_ref == 1


class TestMlxRandomTokens:
    def test_samples_stay_inside_candidate_set(self) -> None:
        candidate_count = 8
        logits = mx.full((BATCH_SIZE, VOCAB_SIZE), -100.0)
        logits[:, :candidate_count] = mx.linspace(5.0, 1.0, candidate_count)[None, :]
        mx.eval(logits)
        params = [_random_params(top_k=candidate_count)] * BATCH_SIZE

        sampled = []
        key = mx.random.key(0)
        for _ in range(200):
            key, subkey = mx.random.split(key)
            tokens = SamplingBatch._native_random_tokens(logits, params, subkey)
            mx.eval(tokens)
            sampled.extend(tokens.tolist())

        assert all(0 <= token < candidate_count for token in sampled)
        assert len(set(sampled)) > 1

    def test_same_key_is_deterministic(self) -> None:
        logits = mx.random.normal((BATCH_SIZE, VOCAB_SIZE), key=mx.random.key(3))
        mx.eval(logits)
        params = [_random_params()] * BATCH_SIZE
        key = mx.random.key(11)

        first = SamplingBatch._native_random_tokens(logits, params, key)
        second = SamplingBatch._native_random_tokens(logits, params, key)
        mx.eval(first, second)

        assert first.tolist() == second.tolist()

    def test_per_row_temperatures_shape_distinct_distributions(self) -> None:
        """Row temperatures are per-row facts: a near-zero-temperature row
        must stick to the argmax while a hot row keeps exploring."""
        logits = mx.array([[5.0, 4.5, 4.0, 3.5]] * 2)
        mx.eval(logits)
        params = [
            _random_params(temperature=0.01, top_k=0, top_p=1.0),
            _random_params(temperature=2.0, top_k=0, top_p=1.0),
        ]

        cold_row, hot_row = set(), set()
        key = mx.random.key(7)
        for _ in range(100):
            key, subkey = mx.random.split(key)
            tokens = SamplingBatch._native_random_tokens(logits, params, subkey)
            mx.eval(tokens)
            cold_row.add(tokens.tolist()[0])
            hot_row.add(tokens.tolist()[1])

        assert cold_row == {0}
        assert len(hot_row) > 1

    def test_distribution_tracks_probabilities(self) -> None:
        probs = [0.5, 0.3, 0.2]
        logits = mx.log(mx.array([probs] * 64))
        mx.eval(logits)
        params = [_random_params(temperature=1.0, top_k=0, top_p=1.0)] * 64

        counts = [0, 0, 0]
        key = mx.random.key(5)
        draws = 0
        for _ in range(400):
            key, subkey = mx.random.split(key)
            tokens = SamplingBatch._native_random_tokens(logits, params, subkey)
            mx.eval(tokens)
            for token in tokens.tolist():
                counts[token] += 1
                draws += 1

        for index, expected in enumerate(probs):
            assert abs(counts[index] / draws - expected) < 0.02
