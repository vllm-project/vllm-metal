# SPDX-License-Identifier: Apache-2.0
"""MLX encoder-output cache for multimodal requests."""

from __future__ import annotations

import mlx.core as mx

from vllm_metal.v1.multimodal.feature_spec import MultiModalFeatureSpec


class EncoderCache:
    """Store multimodal features and MLX encoder outputs by request/hash.

    Lifted from ``vllm/v1/worker/gpu/mm/encoder_cache.py:8-40`` with the
    tensor type changed from ``torch.Tensor`` to ``mlx.core.array``.
    """

    def __init__(self) -> None:
        self.mm_features: dict[str, list[MultiModalFeatureSpec]] = {}
        self.encoder_outputs: dict[str, mx.array] = {}

    def add_request(
        self, req_id: str, mm_features: list[MultiModalFeatureSpec]
    ) -> None:
        self.mm_features[req_id] = mm_features

    def remove_request(self, req_id: str) -> None:
        self.mm_features.pop(req_id, None)

    def reset_mm_cache(self) -> None:
        """Clear profiling-only multimodal cache state when needed."""

    def reset_encoder_cache(self) -> None:
        """Clear cached encoder outputs."""
        self.encoder_outputs.clear()

    def free_encoder_cache(self, mm_hash: str) -> None:
        self.encoder_outputs.pop(mm_hash, None)
