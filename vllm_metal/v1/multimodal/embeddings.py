# SPDX-License-Identifier: Apache-2.0
"""Embedding splice helpers for multimodal placeholders."""

from __future__ import annotations

import mlx.core as mx
import numpy as np

# === Flattening ===


def _flatten_embeddings(multimodal_embeddings: mx.array | list[mx.array]) -> mx.array:
    if isinstance(multimodal_embeddings, list):
        return mx.concatenate(multimodal_embeddings, axis=0)
    return multimodal_embeddings


def _normalize_mask(inputs_embeds: mx.array, is_multimodal: mx.array) -> mx.array:
    # Packed prefill builds one embeddings batch at a time; promote its 1D
    # placeholder mask to match the rank-3 embeddings layout.
    if (
        is_multimodal.ndim == 1
        and inputs_embeds.ndim == 3
        and inputs_embeds.shape[0] == 1
    ):
        return is_multimodal[None, :]
    return is_multimodal


# === Public Helpers ===


def merge_multimodal_embeddings(
    inputs_embeds: mx.array,
    multimodal_embeddings: mx.array | list[mx.array],
    is_multimodal: mx.array,
) -> mx.array:
    """Splice multimodal embeddings into placeholder positions.

    Port of ``vllm/model_executor/models/utils.py:458-498``
    ``_merge_multimodal_embeddings``.  This helper builds and returns the
    merged tensor; callers should not rely on ``inputs_embeds`` being mutated.
    """
    if len(multimodal_embeddings) == 0:
        return inputs_embeds

    mm_embeds_flat = _flatten_embeddings(multimodal_embeddings)
    mask = _normalize_mask(inputs_embeds, is_multimodal)
    mask_flat = mx.flatten(mask)
    num_actual_tokens = int(mm_embeds_flat.shape[0])
    num_expected_tokens = int(mask_flat.sum().item())

    if num_actual_tokens != num_expected_tokens:
        raise ValueError(
            f"Attempted to assign {num_actual_tokens} multimodal tokens to "
            f"{num_expected_tokens} placeholders"
        )

    input_dtype = inputs_embeds.dtype
    hidden_size = inputs_embeds.shape[-1]
    flat = inputs_embeds.reshape(-1, hidden_size)
    positions = mx.array(np.where(mask_flat)[0], dtype=mx.uint32)
    flat[positions] = mm_embeds_flat.astype(input_dtype)
    return flat.reshape(inputs_embeds.shape)
