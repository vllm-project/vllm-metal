# SPDX-License-Identifier: Apache-2.0
"""Qwen3-VL multimodal adapter for vLLM Metal."""

from __future__ import annotations

from functools import cache
from types import SimpleNamespace
from typing import Any

import mlx.core as mx

from vllm_metal.multimodal.feature_spec import MultiModalFeatureSpec


class Qwen3VLMultimodalAdapter:
    """Model-owned multimodal helpers for the Qwen3-VL execution path."""

    def __init__(self, *, spatial_merge_size: int) -> None:
        if spatial_merge_size <= 0:
            raise ValueError(
                f"spatial_merge_size must be positive, got {spatial_merge_size}"
            )
        self._spatial_merge_size = spatial_merge_size

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[MultiModalFeatureSpec],
    ) -> tuple[mx.array, int]:
        """Return ``((3, seq_len) int32 positions, mrope_position_delta)``.

        Calls upstream vLLM's mm_features-driven Qwen3-VL M-RoPE helper with a
        minimal image-only config shim, then converts the returned torch tensor
        to an MLX array.  This keeps the position-building policy upstream-owned
        while the vllm-metal runner can consume MLX arrays.
        """
        if not input_tokens:
            return mx.zeros((3, 0), dtype=mx.int32), 0

        _validate_qwen3_vl_image_features(
            mm_features,
            spatial_merge_size=self._spatial_merge_size,
        )

        torch_positions, mrope_position_delta = (
            _qwen3_vl_cls()._get_mrope_input_positions(
                input_tokens=input_tokens,
                mm_features=mm_features,
                config=_qwen3_vl_image_config(self._spatial_merge_size),
            )
        )
        llm_positions = torch_positions.cpu().numpy()
        return mx.array(llm_positions, dtype=mx.int32), int(mrope_position_delta)


# === Internal Validation ===


def _validate_qwen3_vl_image_features(
    features: list[MultiModalFeatureSpec],
    *,
    spatial_merge_size: int,
) -> None:
    """Validate the image-only feature shape accepted by this PR series."""
    for feature in sorted(features, key=lambda feature: feature.mm_position.offset):
        modality = feature.modality
        if modality == "video":
            raise NotImplementedError(
                "Video multimodal features are out of scope for the initial "
                "Qwen3.5-4B multimodal PR series."
            )
        if modality != "image":
            raise ValueError(f"Unsupported modality: {modality}")
        if feature.data is None:
            raise ValueError("Image feature data is required to read image_grid_thw.")

        t, h, w = _grid_thw(feature.data, "image_grid_thw")
        if t != 1:
            raise ValueError(
                "Multi-frame images are out of scope for the initial "
                f"Qwen3.5-4B multimodal PR series, got t={t}"
            )

        llm_grid_h = h // spatial_merge_size
        llm_grid_w = w // spatial_merge_size
        num_grid_tokens = llm_grid_h * llm_grid_w
        num_embeds = feature.mm_position.get_num_embeds()
        if num_embeds != num_grid_tokens:
            raise ValueError(
                "image_grid_thw implies "
                f"{num_grid_tokens} multimodal embeddings, got "
                f"mm_position.get_num_embeds()={num_embeds}"
            )


# === Upstream Shim ===


@cache
def _qwen3_vl_cls():
    from vllm.model_executor.models.qwen3_vl import Qwen3VLForConditionalGeneration

    return Qwen3VLForConditionalGeneration


def _qwen3_vl_image_config(spatial_merge_size: int) -> SimpleNamespace:
    """Build the minimal config read by Qwen3-VL's image-only M-RoPE path."""
    # The token-id fields are only read for video; keep sentinels here so the
    # static upstream helper can be reused without constructing a full model.
    return SimpleNamespace(
        video_token_id=-1,
        vision_start_token_id=-1,
        vision_end_token_id=-1,
        vision_config=SimpleNamespace(spatial_merge_size=spatial_merge_size),
    )


# === Feature Data Access ===


def _feature_value(data: Any, key: str) -> Any:
    try:
        value = data[key]
    except KeyError as exc:
        raise ValueError(f"Multimodal feature data is missing {key}.") from exc
    return getattr(value, "data", value)


def _grid_thw(data: Any, key: str) -> tuple[int, int, int]:
    raw_values = _feature_value(data, key)
    values = raw_values.tolist() if hasattr(raw_values, "tolist") else raw_values
    if len(values) == 1 and isinstance(values[0], list):
        values = values[0]
    if len(values) != 3:
        raise ValueError(f"{key} must contain exactly 3 values, got {values}.")
    t, h, w = values
    return int(t), int(h), int(w)
