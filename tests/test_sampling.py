# SPDX-License-Identifier: Apache-2.0
"""Tests for Metal sampling with vLLM Sampler integration."""

import mlx.core as mx
import numpy as np
import pytest
import torch
from vllm.utils.torch_utils import make_tensor_with_pad
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler

from vllm_metal.pytorch_backend.tensor_bridge import mlx_to_torch

VOCAB_SIZE = 1024
MAX_NUM_PROMPT_TOKENS = 64
NUM_OUTPUT_TOKENS = 20


def create_prompt_tokens_tensor(
    prompt_token_ids: list[list[int]],
    vocab_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Create padded tensor from variable-length prompt token lists."""
    return make_tensor_with_pad(
        prompt_token_ids,
        pad=vocab_size,
        device=device,
        dtype=torch.int64,
        pin_memory=False,
    )


def create_fake_logits_mlx(batch_size: int, vocab_size: int) -> mx.array:
    """Create uniform logits as MLX array."""
    logits = mx.full((batch_size, vocab_size), 1e-2)
    mx.eval(logits)
    return logits


def create_sampling_metadata(
    batch_size: int,
    vocab_size: int,
    device: torch.device,
    temperature: float = 0.0,
    top_p: float | None = None,
    top_k: int | None = None,
) -> SamplingMetadata:
    """Create SamplingMetadata matching vLLM's structure."""
    output_token_ids = [
        np.random.randint(0, vocab_size, size=NUM_OUTPUT_TOKENS).tolist()
        for _ in range(batch_size)
    ]
    prompt_token_ids = [
        np.random.randint(
            0, vocab_size, size=np.random.randint(1, MAX_NUM_PROMPT_TOKENS)
        ).tolist()
        for _ in range(batch_size)
    ]

    # Determine if all greedy or all random
    all_greedy = temperature < 1e-5
    all_random = not all_greedy

    return SamplingMetadata(
        temperature=None
        if all_greedy
        else torch.full((batch_size,), temperature, device=device),
        all_greedy=all_greedy,
        all_random=all_random,
        top_p=None
        if top_p is None
        else torch.full((batch_size,), top_p, device=device),
        top_k=None
        if top_k is None
        else torch.full((batch_size,), top_k, dtype=torch.int, device=device),
        generators={},
        max_num_logprobs=None,
        prompt_token_ids=create_prompt_tokens_tensor(
            prompt_token_ids, vocab_size, device
        ),
        output_token_ids=output_token_ids,
        frequency_penalties=torch.zeros(batch_size, device=device),
        presence_penalties=torch.zeros(batch_size, device=device),
        repetition_penalties=torch.ones(batch_size, device=device),
        no_penalties=True,
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
    )


class TestMLXToTorchSampling:
    """Test MLX logits -> torch conversion -> vLLM Sampler."""

    @pytest.mark.parametrize("batch_size", [1, 2, 32])
    def test_greedy_sampling_with_vllm_sampler(self, batch_size: int) -> None:
        """Test greedy sampling using vLLM's Sampler with MLX logits."""
        # Create MLX logits with known maximum
        logits_mlx = create_fake_logits_mlx(batch_size, VOCAB_SIZE)
        # Set a clear maximum at different positions per batch
        logits_list = []
        for i in range(batch_size):
            target_idx = (i * 7) % VOCAB_SIZE  # Different max per request
            row = mx.full((VOCAB_SIZE,), 1e-2)
            row = row.at[target_idx].add(10.0)
            logits_list.append(row)
        logits_mlx = mx.stack(logits_list)
        mx.eval(logits_mlx)

        # Convert to torch
        logits_torch = mlx_to_torch(logits_mlx, device="cpu")

        # Create sampling metadata (greedy)
        metadata = create_sampling_metadata(
            batch_size, VOCAB_SIZE, torch.device("cpu"), temperature=0.0
        )

        # Use vLLM's Sampler
        sampler = Sampler()
        output = sampler.forward(logits_torch, metadata)

        # Verify correct tokens sampled
        for i in range(batch_size):
            expected_idx = (i * 7) % VOCAB_SIZE
            assert output.sampled_token_ids[i, 0].item() == expected_idx

    @pytest.mark.parametrize("batch_size", [1, 8])
    @pytest.mark.parametrize("temperature", [0.5, 1.0, 2.0])
    def test_temperature_sampling(self, batch_size: int, temperature: float) -> None:
        """Test temperature-based sampling with vLLM Sampler."""
        logits_mlx = create_fake_logits_mlx(batch_size, VOCAB_SIZE)
        mx.eval(logits_mlx)

        logits_torch = mlx_to_torch(logits_mlx, device="cpu")
        metadata = create_sampling_metadata(
            batch_size, VOCAB_SIZE, torch.device("cpu"), temperature=temperature
        )

        sampler = Sampler()
        output = sampler.forward(logits_torch, metadata)

        # Should produce valid token IDs
        assert output.sampled_token_ids.shape == (batch_size, 1)
        assert (output.sampled_token_ids >= 0).all()
        assert (output.sampled_token_ids < VOCAB_SIZE).all()

    @pytest.mark.parametrize("top_k", [1, 10, 50])
    def test_top_k_sampling(self, top_k: int) -> None:
        """Test top-k sampling limits candidates."""
        batch_size = 4
        logits_mlx = mx.random.normal((batch_size, VOCAB_SIZE))
        mx.eval(logits_mlx)

        logits_torch = mlx_to_torch(logits_mlx, device="cpu")
        metadata = create_sampling_metadata(
            batch_size,
            VOCAB_SIZE,
            torch.device("cpu"),
            temperature=1.0,
            top_k=top_k,
        )

        sampler = Sampler()
        output = sampler.forward(logits_torch, metadata)

        # With top_k=1, should always pick the maximum
        if top_k == 1:
            expected = torch.argmax(logits_torch, dim=-1)
            assert (output.sampled_token_ids.squeeze() == expected).all()

    @pytest.mark.parametrize("top_p", [0.1, 0.5, 0.9])
    def test_top_p_sampling(self, top_p: float) -> None:
        """Test top-p (nucleus) sampling."""
        batch_size = 4
        # Create skewed logits
        logits_mlx = mx.zeros((batch_size, VOCAB_SIZE))
        # Set first token to have high probability
        logits_list = []
        for _ in range(batch_size):
            row = mx.zeros((VOCAB_SIZE,))
            row = row.at[0].add(10.0)
            logits_list.append(row)
        logits_mlx = mx.stack(logits_list)
        mx.eval(logits_mlx)

        logits_torch = mlx_to_torch(logits_mlx, device="cpu")
        metadata = create_sampling_metadata(
            batch_size,
            VOCAB_SIZE,
            torch.device("cpu"),
            temperature=1.0,
            top_p=top_p,
        )

        sampler = Sampler()
        output = sampler.forward(logits_torch, metadata)

        # With very low top_p and skewed distribution, should mostly pick token 0
        if top_p <= 0.1:
            assert (output.sampled_token_ids == 0).sum() >= batch_size // 2


class TestBatchedPerRequestSampling:
    """Test batched sampling with different params per request."""

    def test_mixed_greedy_and_random(self) -> None:
        """Test batch with some greedy, some random requests."""
        batch_size = 4
        # Create logits with different maxima per request
        logits_list = []
        for i in range(batch_size):
            row = mx.full((VOCAB_SIZE,), 1e-2)
            row = row.at[i * 10].add(10.0)
            logits_list.append(row)
        logits_mlx = mx.stack(logits_list)
        mx.eval(logits_mlx)

        logits_torch = mlx_to_torch(logits_mlx, device="cpu")

        # Create per-request temperatures (half greedy, half random)
        temperatures = torch.tensor([0.0, 1.0, 0.0, 1.0])

        metadata = SamplingMetadata(
            temperature=temperatures,
            all_greedy=False,
            all_random=False,
            top_p=None,
            top_k=None,
            generators={i: torch.Generator().manual_seed(42 + i) for i in [1, 3]},
            max_num_logprobs=None,
            prompt_token_ids=None,
            output_token_ids=[[] for _ in range(batch_size)],
            frequency_penalties=torch.zeros(batch_size),
            presence_penalties=torch.zeros(batch_size),
            repetition_penalties=torch.ones(batch_size),
            no_penalties=True,
            allowed_token_ids_mask=None,
            bad_words_token_ids={},
            logitsprocs=LogitsProcessors(),
        )

        sampler = Sampler()
        output = sampler.forward(logits_torch, metadata)

        # Greedy requests (0, 2) should pick the max
        assert output.sampled_token_ids[0, 0].item() == 0
        assert output.sampled_token_ids[2, 0].item() == 20
