# SPDX-License-Identifier: Apache-2.0
"""Multimodal helpers for vLLM Metal v1."""

from __future__ import annotations

from vllm_metal.v1.multimodal.embeddings import merge_multimodal_embeddings
from vllm_metal.v1.multimodal.encoder_cache import EncoderCache
from vllm_metal.v1.multimodal.feature_spec import (
    MultiModalFeatureSpec,
    PlaceholderRange,
    iter_image_grid_thw,
)
from vllm_metal.v1.multimodal.mrope import get_mrope_input_positions

__all__ = [
    "MultiModalFeatureSpec",
    "PlaceholderRange",
    "EncoderCache",
    "get_mrope_input_positions",
    "iter_image_grid_thw",
    "merge_multimodal_embeddings",
]
