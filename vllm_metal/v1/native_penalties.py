# SPDX-License-Identifier: Apache-2.0
"""Lazy MLX penalties with vLLM's prompt/output token semantics."""

from collections.abc import Sequence

import mlx.core as mx
from vllm.sampling_params import SamplingParams


def uses_penalties(params: SamplingParams) -> bool:
    return (
        params.repetition_penalty != 1.0
        or params.frequency_penalty != 0.0
        or params.presence_penalty != 0.0
    )


def _token_counts(token_ids: Sequence[list[int]], vocab_size: int) -> mx.array:
    """Count repeated IDs per row on device; the extra bin absorbs padding."""
    counts = mx.zeros((len(token_ids), vocab_size + 1), dtype=mx.int32)
    width = max(map(len, token_ids), default=0)
    if width:
        tokens = mx.array(
            [ids + [vocab_size] * (width - len(ids)) for ids in token_ids],
            dtype=mx.int32,
        )
        # Match vLLM's treatment of placeholder output IDs in unpenalized rows.
        tokens = mx.where(tokens == -1, vocab_size, tokens)
        rows = mx.arange(len(token_ids), dtype=mx.int32)[:, None]
        counts = counts.at[rows, tokens].add(1)
    return counts[:, :vocab_size]


def apply_native_penalties(
    logits: mx.array,
    params: Sequence[SamplingParams],
    prompt_token_ids: Sequence[list[int]],
    output_token_ids: Sequence[list[int]],
) -> mx.array:
    """Apply repetition, frequency, then presence without evaluating logits.

    As in MRv2's ``gpu/sample/penalties.py``, repetition uses prompt or output
    presence; frequency and presence use only committed output counts.
    """
    if not any(uses_penalties(sp) for sp in params):
        return logits
    logits = logits.astype(mx.float32)
    output_counts = _token_counts(output_token_ids, logits.shape[-1])
    output_mask = output_counts > 0
    if any(sp.repetition_penalty != 1.0 for sp in params):
        prompt_mask = _token_counts(prompt_token_ids, logits.shape[-1]) > 0
        repetition = mx.array([sp.repetition_penalty for sp in params])[:, None]
        scale = mx.where(prompt_mask | output_mask, repetition, 1.0)
        logits = mx.where(logits > 0, logits / scale, logits * scale)
    frequency = mx.array([sp.frequency_penalty for sp in params])[:, None]
    presence = mx.array([sp.presence_penalty for sp in params])[:, None]
    return logits - frequency * output_counts - presence * output_mask
