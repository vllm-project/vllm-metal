# SPDX-License-Identifier: Apache-2.0
"""Generic multimodal helpers for vLLM Metal."""

from __future__ import annotations

from vllm_metal.multimodal.embeddings import merge_multimodal_embeddings
from vllm_metal.multimodal.feature_spec import (
    MultiModalFeatureSpec,
    PlaceholderRange,
)

__all__ = [
    "MultiModalFeatureSpec",
    "PlaceholderRange",
    "merge_multimodal_embeddings",
]
