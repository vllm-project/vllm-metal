# SPDX-License-Identifier: Apache-2.0
"""Metal-compatible penalty and temperature application.

This module provides PyTorch/MLX implementations of penalty functions
that replace Triton kernels on the Metal backend.
"""

import torch


def apply_penalties_and_temperature(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    presence_penalties: torch.Tensor,
    frequency_penalties: torch.Tensor,
    repetition_penalties: torch.Tensor,
    output_token_ids: torch.Tensor,
    bin_counts: torch.Tensor,
    vocab_size: int,
) -> torch.Tensor:
    """Apply penalties and temperature to logits.

    This function applies various sampling penalties and temperature
    scaling to logits for controlled text generation.

    Args:
        logits: Raw model logits [batch_size, vocab_size].
        temperatures: Temperature values [batch_size].
        presence_penalties: Presence penalty values [batch_size].
        frequency_penalties: Frequency penalty values [batch_size].
        repetition_penalties: Repetition penalty values [batch_size].
        output_token_ids: Previously generated tokens [batch_size, max_len].
        bin_counts: Token frequency counts [batch_size, vocab_size].
        vocab_size: Size of the vocabulary.

    Returns:
        Modified logits with penalties and temperature applied.
    """
    batch_size = logits.shape[0]
    device = logits.device

    # Apply temperature scaling
    # Temperature of 0 means greedy (handled separately)
    temp_mask = temperatures > 0
    if temp_mask.any():
        logits[temp_mask] = logits[temp_mask] / temperatures[temp_mask].unsqueeze(-1)

    # Apply repetition penalty
    # Repetition penalty multiplies logits of previously seen tokens
    rep_mask = repetition_penalties != 1.0
    if rep_mask.any():
        for i in range(batch_size):
            if repetition_penalties[i] != 1.0:
                # Get unique tokens that have been generated
                seen_tokens = output_token_ids[i][output_token_ids[i] >= 0].unique()

                if len(seen_tokens) > 0:
                    penalty = repetition_penalties[i]

                    # Apply penalty: divide positive logits, multiply negative
                    for token in seen_tokens:
                        if token < vocab_size:
                            if logits[i, token] > 0:
                                logits[i, token] = logits[i, token] / penalty
                            else:
                                logits[i, token] = logits[i, token] * penalty

    # Apply presence penalty
    # Presence penalty subtracts from logits of tokens that appear
    pres_mask = presence_penalties != 0.0
    if pres_mask.any():
        for i in range(batch_size):
            if presence_penalties[i] != 0.0:
                # Create presence mask from bin_counts
                present = bin_counts[i] > 0
                logits[i] = logits[i] - presence_penalties[i] * present.float()

    # Apply frequency penalty
    # Frequency penalty subtracts proportionally to occurrence count
    freq_mask = frequency_penalties != 0.0
    if freq_mask.any():
        for i in range(batch_size):
            if frequency_penalties[i] != 0.0:
                logits[i] = logits[i] - frequency_penalties[i] * bin_counts[i].float()

    return logits


def apply_temperature(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
) -> torch.Tensor:
    """Apply temperature scaling to logits.

    Args:
        logits: Raw logits [batch_size, vocab_size].
        temperatures: Temperature values [batch_size].

    Returns:
        Temperature-scaled logits.
    """
    # Handle temperature = 0 (greedy) separately
    temp_mask = temperatures > 0
    if temp_mask.any():
        logits[temp_mask] = logits[temp_mask] / temperatures[temp_mask].unsqueeze(-1)

    return logits


def apply_top_k(
    logits: torch.Tensor,
    top_k: torch.Tensor,
) -> torch.Tensor:
    """Apply top-k filtering to logits.

    Args:
        logits: Logits tensor [batch_size, vocab_size].
        top_k: Top-k values [batch_size].

    Returns:
        Filtered logits with non-top-k values set to -inf.
    """
    batch_size = logits.shape[0]
    vocab_size = logits.shape[1]

    for i in range(batch_size):
        k = int(top_k[i])
        if k > 0 and k < vocab_size:
            # Get threshold value
            top_k_values, _ = torch.topk(logits[i], k)
            threshold = top_k_values[-1]

            # Mask out values below threshold
            logits[i] = torch.where(
                logits[i] >= threshold,
                logits[i],
                torch.full_like(logits[i], float("-inf")),
            )

    return logits


def apply_top_p(
    logits: torch.Tensor,
    top_p: torch.Tensor,
) -> torch.Tensor:
    """Apply top-p (nucleus) filtering to logits.

    Args:
        logits: Logits tensor [batch_size, vocab_size].
        top_p: Top-p values [batch_size].

    Returns:
        Filtered logits with low-probability values set to -inf.
    """
    batch_size = logits.shape[0]

    for i in range(batch_size):
        p = float(top_p[i])
        if p < 1.0:
            # Sort logits and get cumulative probabilities
            sorted_logits, sorted_indices = torch.sort(logits[i], descending=True)
            cumulative_probs = torch.cumsum(
                torch.softmax(sorted_logits, dim=-1), dim=-1
            )

            # Find cutoff index
            sorted_mask = cumulative_probs > p
            # Shift mask to keep at least one token
            sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
            sorted_mask[..., 0] = False

            # Apply mask
            sorted_logits[sorted_mask] = float("-inf")

            # Restore original order
            logits[i] = sorted_logits.scatter(0, sorted_indices, sorted_logits)

    return logits
