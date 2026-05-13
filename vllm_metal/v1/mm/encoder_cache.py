# SPDX-License-Identifier: Apache-2.0
"""MLX encoder-output cache for multimodal requests."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm_metal.multimodal.feature_spec import MultiModalFeatureSpec

if TYPE_CHECKING:
    from vllm_metal.v1.model_adapter import MultimodalEncodeResult


class EncoderCache:
    """Store multimodal features and encoder-output records by request/hash.

    Mirrors upstream vLLM's v1 GPU ``EncoderCache``.  Stores the adapter's
    full ``MultimodalEncodeResult`` (hidden_states + deepstack channel) so
    downstream splice has both fields available; consumers read
    ``.hidden_states`` for first-token splice and ``.deepstack_visual_embeds``
    for layer-residual injection.
    """

    def __init__(self) -> None:
        self.mm_features: dict[str, list[MultiModalFeatureSpec]] = {}
        self.encoder_outputs: dict[str, MultimodalEncodeResult] = {}

    def add_request(
        self, req_id: str, mm_features: list[MultiModalFeatureSpec]
    ) -> None:
        self.mm_features[req_id] = mm_features

    def remove_request(self, req_id: str) -> None:
        self.mm_features.pop(req_id, None)

    def reset_mm_cache(self) -> None:
        """Mirror upstream's profiling-cache reset hook."""
        # TODO: Implement when vllm-metal adds profiling-time MM cache state.
        pass

    def reset_encoder_cache(self) -> None:
        """Clear cached encoder outputs."""
        self.encoder_outputs.clear()

    def free_encoder_cache(self, mm_hash: str) -> None:
        self.encoder_outputs.pop(mm_hash, None)
