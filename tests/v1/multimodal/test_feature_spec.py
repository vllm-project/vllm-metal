# SPDX-License-Identifier: Apache-2.0
"""Tests for multimodal feature-spec helpers."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from vllm.multimodal.inputs import MultiModalFeatureSpec as UpstreamFeatureSpec

from vllm_metal.v1.multimodal import (
    MultiModalFeatureSpec,
    PlaceholderRange,
    iter_image_grid_thw,
)

# === Test Helpers ===


def _feature(
    *,
    offset: int,
    length: int,
    grid_thw: tuple[int, int, int] = (1, 10, 10),
    modality: str = "image",
) -> MultiModalFeatureSpec:
    key = "video_grid_thw" if modality.startswith("video") else "image_grid_thw"
    return MultiModalFeatureSpec(
        data={key: SimpleNamespace(data=np.array(grid_thw))},
        modality=modality,
        identifier=f"{modality}-{offset}",
        mm_position=PlaceholderRange(offset=offset, length=length),
    )


# === Tests ===


def test_iter_image_grid_thw_single_image() -> None:
    [result] = list(
        iter_image_grid_thw([_feature(offset=10, length=25)], spatial_merge_size=2)
    )
    assert result == (10, 1, 5, 5, 1.0)


def test_iter_image_grid_thw_multiple_images_preserves_order() -> None:
    features = [
        _feature(offset=10, length=25, grid_thw=(1, 10, 10)),
        _feature(offset=30, length=4, grid_thw=(1, 4, 4)),
    ]
    results = list(iter_image_grid_thw(features, spatial_merge_size=2))
    assert results == [(10, 1, 5, 5, 1.0), (30, 1, 2, 2, 1.0)]


def test_iter_image_grid_thw_rejects_unordered_features() -> None:
    features = [
        _feature(offset=30, length=4, grid_thw=(1, 4, 4)),
        _feature(offset=10, length=25, grid_thw=(1, 10, 10)),
    ]

    with pytest.raises(ValueError, match="ordered by mm_position.offset"):
        list(iter_image_grid_thw(features, spatial_merge_size=2))


def test_iter_image_grid_thw_uses_spatial_merge_size() -> None:
    [result] = list(
        iter_image_grid_thw(
            [_feature(offset=0, length=4, grid_thw=(1, 12, 12))],
            spatial_merge_size=3,
        )
    )
    assert result == (0, 1, 4, 4, 1.0)


def test_iter_image_grid_thw_video_raises() -> None:
    with pytest.raises(NotImplementedError, match="Video multimodal features"):
        list(
            iter_image_grid_thw(
                [_feature(offset=0, length=4, modality="video")],
                spatial_merge_size=2,
            )
        )


def test_re_export_matches_upstream() -> None:
    assert MultiModalFeatureSpec is UpstreamFeatureSpec
