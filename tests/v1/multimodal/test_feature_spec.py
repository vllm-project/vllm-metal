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

_SPATIAL_MERGE_SIZE = 2
_SPATIAL_MERGE_SIZE_3 = 3
_FIRST_IMAGE_OFFSET = 10
_SECOND_IMAGE_OFFSET = 30
_GRID_THW_10X10 = (1, 10, 10)
_GRID_THW_4X4 = (1, 4, 4)
_GRID_THW_12X12 = (1, 12, 12)
_TOKENS_10X10_MERGE_2 = 25
_TOKENS_4X4_MERGE_2 = 4
_TOKENS_12X12_MERGE_3 = 16

# === Test Helpers ===


def _feature(
    *,
    offset: int,
    length: int,
    grid_thw: tuple[int, ...] = _GRID_THW_10X10,
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
        iter_image_grid_thw(
            [
                _feature(
                    offset=_FIRST_IMAGE_OFFSET,
                    length=_TOKENS_10X10_MERGE_2,
                )
            ],
            spatial_merge_size=_SPATIAL_MERGE_SIZE,
        )
    )
    assert result == (_FIRST_IMAGE_OFFSET, 1, 5, 5, 1.0)


def test_iter_image_grid_thw_multiple_images_preserves_order() -> None:
    features = [
        _feature(
            offset=_FIRST_IMAGE_OFFSET,
            length=_TOKENS_10X10_MERGE_2,
            grid_thw=_GRID_THW_10X10,
        ),
        _feature(
            offset=_SECOND_IMAGE_OFFSET,
            length=_TOKENS_4X4_MERGE_2,
            grid_thw=_GRID_THW_4X4,
        ),
    ]
    results = list(
        iter_image_grid_thw(features, spatial_merge_size=_SPATIAL_MERGE_SIZE)
    )
    assert results == [
        (_FIRST_IMAGE_OFFSET, 1, 5, 5, 1.0),
        (_SECOND_IMAGE_OFFSET, 1, 2, 2, 1.0),
    ]


def test_iter_image_grid_thw_sorts_unordered_features() -> None:
    features = [
        _feature(
            offset=_SECOND_IMAGE_OFFSET,
            length=_TOKENS_4X4_MERGE_2,
            grid_thw=_GRID_THW_4X4,
        ),
        _feature(
            offset=_FIRST_IMAGE_OFFSET,
            length=_TOKENS_10X10_MERGE_2,
            grid_thw=_GRID_THW_10X10,
        ),
    ]

    results = list(
        iter_image_grid_thw(features, spatial_merge_size=_SPATIAL_MERGE_SIZE)
    )

    assert results == [
        (_FIRST_IMAGE_OFFSET, 1, 5, 5, 1.0),
        (_SECOND_IMAGE_OFFSET, 1, 2, 2, 1.0),
    ]


def test_iter_image_grid_thw_uses_spatial_merge_size() -> None:
    [result] = list(
        iter_image_grid_thw(
            [
                _feature(
                    offset=0,
                    length=_TOKENS_12X12_MERGE_3,
                    grid_thw=_GRID_THW_12X12,
                )
            ],
            spatial_merge_size=_SPATIAL_MERGE_SIZE_3,
        )
    )
    assert result == (0, 1, 4, 4, 1.0)


def test_iter_image_grid_thw_rejects_invalid_spatial_merge_size() -> None:
    with pytest.raises(ValueError, match="spatial_merge_size must be positive"):
        iter_image_grid_thw(
            [
                _feature(
                    offset=0,
                    length=_TOKENS_4X4_MERGE_2,
                    grid_thw=_GRID_THW_4X4,
                )
            ],
            spatial_merge_size=0,
        )


def test_iter_image_grid_thw_rejects_bad_grid_shape() -> None:
    with pytest.raises(ValueError, match="image_grid_thw must contain exactly 3"):
        list(
            iter_image_grid_thw(
                [
                    _feature(
                        offset=0,
                        length=_TOKENS_4X4_MERGE_2,
                        grid_thw=(1, 4),
                    )
                ],
                spatial_merge_size=_SPATIAL_MERGE_SIZE,
            )
        )


def test_iter_image_grid_thw_rejects_embed_count_mismatch() -> None:
    with pytest.raises(ValueError, match="image_grid_thw implies 4"):
        list(
            iter_image_grid_thw(
                [
                    _feature(
                        offset=0,
                        length=_TOKENS_4X4_MERGE_2 - 1,
                        grid_thw=_GRID_THW_4X4,
                    )
                ],
                spatial_merge_size=_SPATIAL_MERGE_SIZE,
            )
        )


def test_iter_image_grid_thw_video_raises() -> None:
    with pytest.raises(NotImplementedError, match="Video multimodal features"):
        list(
            iter_image_grid_thw(
                [
                    _feature(
                        offset=0,
                        length=_TOKENS_4X4_MERGE_2,
                        grid_thw=_GRID_THW_4X4,
                        modality="video",
                    )
                ],
                spatial_merge_size=_SPATIAL_MERGE_SIZE,
            )
        )


def test_re_export_matches_upstream() -> None:
    assert MultiModalFeatureSpec is UpstreamFeatureSpec
