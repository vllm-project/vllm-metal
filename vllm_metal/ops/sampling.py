# SPDX-License-Identifier: Apache-2.0
"""Metal sampling operations."""

from typing import Optional

import torch

from vllm_metal.mlx import to_mlx, to_torch


def sampling_from_probs(
    probs: torch.Tensor,
    random_numbers: torch.Tensor,
    output_indices: torch.Tensor,
) -> None:
    """Sample token indices from probability distribution.

    Uses the inverse CDF method for sampling.

    Args:
        probs: Probability distribution [batch, vocab_size].
        random_numbers: Uniform random numbers [batch].
        output_indices: Output tensor for sampled indices [batch].
    """
    import mlx.core as mx

    # Convert to MLX
    probs_mlx = to_mlx(probs)
    random_mlx = to_mlx(random_numbers)

    # Compute cumulative probabilities
    cum_probs = mx.cumsum(probs_mlx, axis=-1)

    # Sample using inverse CDF
    # Find first index where cum_prob > random
    batch_size = probs.shape[0]
    indices = []

    for i in range(batch_size):
        r = random_mlx[i]
        # Find index where cumsum exceeds random number
        mask = cum_probs[i] > r
        # Get first True index
        idx = mx.argmax(mask.astype(mx.int32))
        indices.append(idx)

    result = mx.stack(indices)

    # Convert back and copy to output
    result_torch = to_torch(result, device=output_indices.device, dtype=output_indices.dtype)
    output_indices.copy_(result_torch)


def multinomial_sample(
    probs: torch.Tensor,
    num_samples: int = 1,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample from multinomial distribution.

    Args:
        probs: Probability distribution [batch, vocab_size].
        num_samples: Number of samples per batch element.
        generator: Optional random number generator.

    Returns:
        Sampled indices [batch, num_samples].
    """
    # Use PyTorch's multinomial since it handles edge cases well
    return torch.multinomial(probs, num_samples, generator=generator)


def top_k_sampling(
    logits: torch.Tensor,
    top_k: int,
    temperature: float = 1.0,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Top-k sampling from logits.

    Args:
        logits: Unnormalized logits [batch, vocab_size].
        top_k: Number of top tokens to sample from.
        temperature: Sampling temperature.
        generator: Optional random number generator.

    Returns:
        Sampled token indices [batch].
    """
    import mlx.core as mx

    # Apply temperature
    if temperature != 1.0:
        logits = logits / temperature

    # Convert to MLX for top-k
    logits_mlx = to_mlx(logits)

    batch_size, vocab_size = logits.shape
    sampled_indices = []

    for i in range(batch_size):
        # Get top-k values and indices
        if top_k < vocab_size:
            top_k_values, top_k_indices = torch.topk(logits[i], top_k)

            # Softmax over top-k
            probs = torch.softmax(top_k_values, dim=-1)

            # Sample from top-k
            sample_idx = torch.multinomial(probs.unsqueeze(0), 1, generator=generator)
            sampled_token = top_k_indices[sample_idx.squeeze()]
        else:
            # Sample from full distribution
            probs = torch.softmax(logits[i], dim=-1)
            sampled_token = torch.multinomial(probs.unsqueeze(0), 1, generator=generator).squeeze()

        sampled_indices.append(sampled_token)

    return torch.stack(sampled_indices)


def top_p_sampling(
    logits: torch.Tensor,
    top_p: float,
    temperature: float = 1.0,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Top-p (nucleus) sampling from logits.

    Args:
        logits: Unnormalized logits [batch, vocab_size].
        top_p: Cumulative probability threshold.
        temperature: Sampling temperature.
        generator: Optional random number generator.

    Returns:
        Sampled token indices [batch].
    """
    # Apply temperature
    if temperature != 1.0:
        logits = logits / temperature

    batch_size = logits.shape[0]
    sampled_indices = []

    for i in range(batch_size):
        # Sort by probability
        sorted_logits, sorted_indices = torch.sort(logits[i], descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(probs, dim=-1)

        # Find cutoff
        cutoff_mask = cumulative_probs > top_p
        cutoff_mask[1:] = cutoff_mask[:-1].clone()
        cutoff_mask[0] = False

        # Zero out tokens beyond cutoff
        sorted_logits[cutoff_mask] = float("-inf")

        # Sample from filtered distribution
        filtered_probs = torch.softmax(sorted_logits, dim=-1)
        sample_idx = torch.multinomial(filtered_probs.unsqueeze(0), 1, generator=generator)
        sampled_token = sorted_indices[sample_idx.squeeze()]
        sampled_indices.append(sampled_token)

    return torch.stack(sampled_indices)
