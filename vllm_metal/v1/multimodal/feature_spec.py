# SPDX-License-Identifier: Apache-2.0
"""Feature-spec wrappers for multimodal inputs."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange

__all__ = ["MultiModalFeatureSpec", "PlaceholderRange", "iter_image_grid_thw"]


# === Feature Data Access ===


def _feature_value(data: Any, key: str) -> Any:
    value = data[key]
    return getattr(value, "data", value)


def _grid_thw(data: Any, key: str) -> tuple[int, int, int]:
    values = _feature_value(data, key).tolist()
    if len(values) == 1 and isinstance(values[0], list):
        values = values[0]
    t, h, w = values
    return int(t), int(h), int(w)


# === Public Helpers ===


def iter_image_grid_thw(
    features: list[MultiModalFeatureSpec],
    *,
    spatial_merge_size: int,
) -> Iterator[tuple[int, int, int, int, float]]:
    """Yield ``(offset, llm_grid_t, llm_grid_h, llm_grid_w, t_factor)``.

    Port of
    ``vllm/model_executor/models/qwen2_5_vl.py:1035-1059``
    ``Qwen2_5_VLForConditionalGeneration.iter_mm_grid_thw``, restricted to
    image modality for the initial Qwen3.5-4B multimodal PR series.

    Video features raise ``NotImplementedError`` because video support is out
    of scope for this series.  Features must arrive in placeholder order,
    matching the upstream vLLM iterator contract.
    """
    if spatial_merge_size <= 0:
        raise ValueError(
            f"spatial_merge_size must be positive, got {spatial_merge_size}"
        )

    previous_offset = -1
    for feature in features:
        offset = feature.mm_position.offset
        if offset < previous_offset:
            raise ValueError(
                "Image multimodal features must be ordered by mm_position.offset."
            )
        previous_offset = offset

        modality = feature.modality
        if modality.startswith("video"):
            raise NotImplementedError(
                "Video multimodal features are out of scope for the initial "
                "Qwen3.5-4B multimodal PR series."
            )
        if not modality.startswith("image"):
            raise ValueError(f"Unsupported modality: {modality}")
        if feature.data is None:
            raise ValueError("Image feature data is required to read image_grid_thw.")

        t, h, w = _grid_thw(feature.data, "image_grid_thw")
        if t != 1:
            raise ValueError(f"Image must have 1 frame, got {t}")

        yield (
            offset,
            1,
            h // spatial_merge_size,
            w // spatial_merge_size,
            1.0,
        )
