# SPDX-License-Identifier: Apache-2.0
"""Metal-compatible Gumbel sampling.

This module provides PyTorch/MLX implementations of Gumbel-max sampling
that replace Triton kernels on the Metal backend.
"""

import torch


def gumbel_sample(
    logits: torch.Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample from a categorical distribution using Gumbel-max trick.

    The Gumbel-max trick provides an efficient way to sample from
    categorical distributions. It's equivalent to:
        1. Add Gumbel noise to logits
        2. Take argmax

    This is more numerically stable than softmax + multinomial for
    large vocabularies.

    Args:
        logits: Unnormalized log probabilities [batch_size, vocab_size].
        generator: Optional random number generator.

    Returns:
        Sampled token indices [batch_size].
    """
    # Generate Gumbel noise: -log(-log(uniform))
    # Using a small epsilon to avoid log(0)
    eps = 1e-10

    # Generate uniform random numbers
    uniform = torch.rand_like(logits, generator=generator)

    # Clip to avoid numerical issues
    uniform = uniform.clamp(min=eps, max=1.0 - eps)

    # Compute Gumbel noise
    gumbel_noise = -torch.log(-torch.log(uniform))

    # Add noise to logits and take argmax
    noisy_logits = logits + gumbel_noise
    samples = torch.argmax(noisy_logits, dim=-1)

    return samples


def gumbel_sample_batch(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: torch.Tensor | None = None,
    top_p: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Batch Gumbel sampling with temperature and filtering.

    Args:
        logits: Logits tensor [batch_size, vocab_size].
        temperatures: Temperature values [batch_size].
        top_k: Optional top-k values [batch_size].
        top_p: Optional top-p values [batch_size].
        generator: Optional random generator.

    Returns:
        Sampled token indices [batch_size].
    """
    batch_size = logits.shape[0]
    device = logits.device

    # Apply temperature
    temp_mask = temperatures > 0
    logits_scaled = logits.clone()
    if temp_mask.any():
        logits_scaled[temp_mask] = (
            logits[temp_mask] / temperatures[temp_mask].unsqueeze(-1)
        )

    # Apply top-k filtering if provided
    if top_k is not None:
        vocab_size = logits.shape[1]
        for i in range(batch_size):
            k = int(top_k[i])
            if k > 0 and k < vocab_size:
                top_k_values, _ = torch.topk(logits_scaled[i], k)
                threshold = top_k_values[-1]
                logits_scaled[i] = torch.where(
                    logits_scaled[i] >= threshold,
                    logits_scaled[i],
                    torch.full_like(logits_scaled[i], float("-inf")),
                )

    # Apply top-p filtering if provided
    if top_p is not None:
        for i in range(batch_size):
            p = float(top_p[i])
            if p < 1.0:
                sorted_logits, sorted_indices = torch.sort(
                    logits_scaled[i], descending=True
                )
                cumulative_probs = torch.cumsum(
                    torch.softmax(sorted_logits, dim=-1), dim=-1
                )

                sorted_mask = cumulative_probs > p
                sorted_mask[1:] = sorted_mask[:-1].clone()
                sorted_mask[0] = False

                sorted_logits[sorted_mask] = float("-inf")
                logits_scaled[i] = sorted_logits.scatter(
                    0, sorted_indices, sorted_logits
                )

    # Handle greedy sampling (temperature = 0)
    samples = torch.zeros(batch_size, dtype=torch.long, device=device)

    greedy_mask = temperatures == 0
    if greedy_mask.any():
        samples[greedy_mask] = torch.argmax(logits[greedy_mask], dim=-1)

    # Gumbel sampling for non-greedy
    non_greedy_mask = ~greedy_mask
    if non_greedy_mask.any():
        samples[non_greedy_mask] = gumbel_sample(
            logits_scaled[non_greedy_mask],
            generator=generator,
        )

    return samples


def sample_with_multinomial(
    probs: torch.Tensor,
    num_samples: int = 1,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample from probability distribution using multinomial.

    This is an alternative to Gumbel sampling that uses PyTorch's
    multinomial function directly.

    Args:
        probs: Probability distribution [batch_size, vocab_size].
        num_samples: Number of samples per batch.
        generator: Optional random generator.

    Returns:
        Sampled indices [batch_size, num_samples].
    """
    # Ensure probs are normalized
    probs = probs / probs.sum(dim=-1, keepdim=True)

    # Handle potential NaN values
    probs = torch.nan_to_num(probs, nan=0.0)

    # Sample
    samples = torch.multinomial(
        probs,
        num_samples=num_samples,
        generator=generator,
    )

    return samples
