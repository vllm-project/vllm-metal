# SPDX-License-Identifier: Apache-2.0
"""Embedding splice helpers for multimodal placeholders."""

from __future__ import annotations

from collections.abc import Sequence

import mlx.core as mx
import numpy as np


def merge_multimodal_embeddings(
    inputs_embeds: mx.array,
    multimodal_embeddings: mx.array | Sequence[mx.array],
    is_multimodal: mx.array,
) -> mx.array:
    """Splice multimodal embeddings into placeholder positions.

    Mirrors ``vllm/model_executor/models/utils.py``
    ``_merge_multimodal_embeddings`` for MLX arrays.  This helper builds and
    returns the merged tensor; callers should not rely on ``inputs_embeds``
    being mutated.
    """
    if len(multimodal_embeddings) == 0:
        return inputs_embeds

    mm_embeds_flat = _flatten_embeddings(multimodal_embeddings)
    mask = _normalize_mask(inputs_embeds, is_multimodal)
    if mask.shape != inputs_embeds.shape[:-1]:
        raise ValueError(
            f"Multimodal mask shape {mask.shape} must match inputs_embeds "
            f"leading shape {inputs_embeds.shape[:-1]}."
        )
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
    mask_np = np.asarray(mask_flat)
    positions = mx.array(np.where(mask_np)[0], dtype=mx.uint32)
    flat[positions] = mm_embeds_flat.astype(input_dtype)
    return flat.reshape(inputs_embeds.shape)


def _flatten_embeddings(
    multimodal_embeddings: mx.array | Sequence[mx.array],
) -> mx.array:
    if isinstance(multimodal_embeddings, Sequence):
        return mx.concatenate(multimodal_embeddings, axis=0)
    return multimodal_embeddings


def _normalize_mask(inputs_embeds: mx.array, is_multimodal: mx.array) -> mx.array:
    # Packed prefill builds one embeddings batch at a time; promote its 1D
    # placeholder mask to match the rank-3 embeddings layout.
    if is_multimodal.ndim == 1 and inputs_embeds.ndim == 3:
        if inputs_embeds.shape[0] == 1:
            return is_multimodal[None, :]
        raise ValueError(
            "1D multimodal mask is only supported for packed batch size 1, "
            f"got inputs_embeds batch size {inputs_embeds.shape[0]}."
        )
    return is_multimodal
