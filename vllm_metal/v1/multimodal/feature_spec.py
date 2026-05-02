# SPDX-License-Identifier: Apache-2.0
"""Feature-spec wrappers for multimodal inputs."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange

__all__ = ["MultiModalFeatureSpec", "PlaceholderRange", "iter_image_grid_thw"]


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


# === Public Helpers ===


def iter_image_grid_thw(
    features: list[MultiModalFeatureSpec],
    *,
    spatial_merge_size: int,
) -> Iterator[tuple[int, int, int, int, float]]:
    """Yield ``(offset, llm_grid_t, llm_grid_h, llm_grid_w, t_factor)``.

    Matches the image branch of
    ``Qwen2_5_VLForConditionalGeneration.iter_mm_grid_thw`` for the initial
    Qwen3.5-4B multimodal PR series.

    Video features raise ``NotImplementedError`` because video support is out
    of scope for this series.  Features are sorted by placeholder offset to
    match upstream Qwen2.5-VL behavior.

    The image branch validates ``mm_position.get_num_embeds()`` against the
    grid-derived token count.  Future placeholder formats that include
    non-embed tokens must preserve that embed-count invariant.
    """
    if spatial_merge_size <= 0:
        raise ValueError(
            f"spatial_merge_size must be positive, got {spatial_merge_size}"
        )

    def _iter() -> Iterator[tuple[int, int, int, int, float]]:
        for feature in sorted(features, key=lambda feature: feature.mm_position.offset):
            offset = feature.mm_position.offset

            modality = feature.modality
            if modality.startswith("video"):
                raise NotImplementedError(
                    "Video multimodal features are out of scope for the initial "
                    "Qwen3.5-4B multimodal PR series."
                )
            if not modality.startswith("image"):
                raise ValueError(f"Unsupported modality: {modality}")
            if feature.data is None:
                raise ValueError(
                    "Image feature data is required to read image_grid_thw."
                )

            t, h, w = _grid_thw(feature.data, "image_grid_thw")
            if t != 1:
                raise ValueError(
                    "Multi-frame images are out of scope for the initial "
                    f"Qwen3.5-4B multimodal PR series, got t={t}"
                )

            llm_grid_t = 1
            llm_grid_h = h // spatial_merge_size
            llm_grid_w = w // spatial_merge_size
            num_grid_tokens = llm_grid_t * llm_grid_h * llm_grid_w
            num_embeds = feature.mm_position.get_num_embeds()
            if num_embeds != num_grid_tokens:
                raise ValueError(
                    "image_grid_thw implies "
                    f"{num_grid_tokens} multimodal embeddings, got "
                    f"mm_position.get_num_embeds()={num_embeds}"
                )

            yield (
                offset,
                llm_grid_t,
                llm_grid_h,
                llm_grid_w,
                1.0,
            )

    return _iter()
