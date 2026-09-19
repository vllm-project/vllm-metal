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

    def test_empty_specific_token_list_is_eligible(self) -> None:
        assert SamplingBatch.params_allow_native_random(
            [_random_params(logprob_token_ids=[])]
        )

    @pytest.mark.parametrize(
        ("label", "params_list"),
        [
            ("greedy request", [_random_params(temperature=0.0)]),
            (
                "mixed greedy and random",
                [_random_params(), _random_params(temperature=0.0)],
            ),
            (
                "mixed top_k",
                [_random_params(top_k=20), _random_params(top_k=40)],
            ),
            (
                "mixed top_p",
                [_random_params(top_p=0.95), _random_params(top_p=0.8)],
            ),
            (
                "mixed min_p",
                [_random_params(min_p=0.1), _random_params(min_p=0.0)],
            ),
        ],
    )
    def test_per_row_masks_keep_mixed_batches_eligible(
        self, label, params_list
    ) -> None:
        """Masks are per row: one request must never push the whole batch
        onto the torch sampler (that serialises every decode step)."""
        assert SamplingBatch.params_allow_native_random(params_list)

    @pytest.mark.parametrize(
        ("label", "params_list"),
        [
            ("empty batch", []),
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
                "allowed token ids",
                [_random_params(allowed_token_ids=[1, 2])],
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

    def test_per_row_masks_match_vllm_reference(self) -> None:
        """Each row keeps exactly the candidates vLLM keeps for its own k/p."""
        logits = mx.random.normal((4, VOCAB_SIZE), key=mx.random.key(9))
        mx.eval(logits)
        top_k = [20, 0, 7, VOCAB_SIZE]
        top_p = [1.0, 0.9, 0.5, 0.75]
        sorted_desc, sorted_idx = SamplingBatch._sorted_candidate_logits(
            logits, top_k, top_p, [0.0] * 4
        )
        masked_mlx = mx.put_along_axis(
            mx.full(logits.shape, -mx.inf), sorted_idx, sorted_desc, axis=-1
        )
        mx.eval(masked_mlx)
        logits_torch = torch.tensor(logits.tolist())
        k_arg = torch.tensor([k if 0 < k < VOCAB_SIZE else VOCAB_SIZE for k in top_k])
        masked_ref = apply_top_k_top_p(logits_torch.clone(), k_arg, torch.tensor(top_p))

        kept_mlx = [[v != float("-inf") for v in row] for row in masked_mlx.tolist()]
        kept_ref = (masked_ref != float("-inf")).tolist()
        assert kept_mlx == kept_ref

    def test_partitioned_top_k_keeps_ties_at_the_kth_value(self) -> None:
        """Top-k masks by value, so ties at the k-th largest survive when the
        partition (sized by the batch's widest top-k) contains them."""
        logits = mx.array(
            [[3.0, 2.0, 2.0, 2.0, 1.0, 0.0], [5.0, 4.0, 3.0, 2.0, 1.0, 0.0]]
        )
        sorted_desc, sorted_idx = SamplingBatch._sorted_candidate_logits(
            logits, [2, 4], [1.0, 1.0], [0.0, 0.0]
        )
        mx.eval(sorted_desc, sorted_idx)
        kept_rows = [
            {
                int(t)
                for t, v in zip(idx_row, val_row, strict=True)
                if v != float("-inf")
            }
            for idx_row, val_row in zip(
                sorted_idx.tolist(), sorted_desc.tolist(), strict=True
            )
        ]
        assert kept_rows == [{0, 1, 2, 3}, {0, 1, 2, 3}]


class TestMlxRandomTokens:
    def test_mixed_batch_rows_follow_their_own_params(self) -> None:
        """A greedy row sticks to its argmax and a narrow top-k row stays inside
        its candidates while a wide row keeps exploring — all in one draw."""
        logits = mx.random.normal((3, VOCAB_SIZE), key=mx.random.key(21))
        mx.eval(logits)
        argmax = int(mx.argmax(logits[0]).item())
        narrow_candidates = {int(i) for i in mx.argsort(logits[1])[::-1][:4].tolist()}
        params = [
            _random_params(temperature=0.0),
            _random_params(temperature=1.5, top_k=4, top_p=1.0),
            _random_params(temperature=2.0, top_k=0, top_p=1.0),
        ]
        assert SamplingBatch.params_allow_native_random(params)

        greedy_row, narrow_row, wide_row = set(), set(), set()
        key = mx.random.key(1)
        for _ in range(200):
            key, subkey = mx.random.split(key)
            tokens = SamplingBatch._native_random_tokens(logits, params, subkey)
            mx.eval(tokens)
            g, n, w = tokens.tolist()
            greedy_row.add(g)
            narrow_row.add(n)
            wide_row.add(w)

        assert greedy_row == {argmax}
        assert narrow_row <= narrow_candidates and len(narrow_row) > 1
        assert len(wide_row) > len(narrow_candidates)

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
