# SPDX-License-Identifier: Apache-2.0
"""Tests for Qwen3.5-4B multimodal M-RoPE helpers."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vllm_metal.multimodal import (
    MultiModalFeatureSpec,
    PlaceholderRange,
)
from vllm_metal.multimodal.qwen3_vl import Qwen3VLMultimodalAdapter

# Qwen3-VL's mm_features-driven helper reads image sequence length and media
# offsets from mm_features; the concrete token ids are irrelevant in these
# focused image-only tests.
_TEXT_TOKEN_ID = 1
_IMAGE_PLACEHOLDER_TOKEN_ID = 2
_SPATIAL_MERGE_SIZE = 2
_IMAGE_GRID_THW_2X2 = (1, 4, 4)
_IMAGE_TOKEN_COUNT_2X2 = 4


# === Test Helpers ===


def _adapter(spatial_merge_size: int = _SPATIAL_MERGE_SIZE) -> Qwen3VLMultimodalAdapter:
    return Qwen3VLMultimodalAdapter(spatial_merge_size=spatial_merge_size)


def _feature(
    *,
    offset: int,
    length: int,
    grid_thw: tuple[int, ...],
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


def test_text_only_positions() -> None:
    positions, delta = _adapter().get_mrope_input_positions(
        [_TEXT_TOKEN_ID] * 4,
        [],
    )
    assert positions.tolist() == [[0, 1, 2, 3]] * 3
    assert delta == 0


def test_invalid_spatial_merge_size_raises() -> None:
    with pytest.raises(ValueError, match="spatial_merge_size must be positive"):
        _adapter(spatial_merge_size=0)


def test_bad_grid_shape_raises() -> None:
    tokens = [_IMAGE_PLACEHOLDER_TOKEN_ID] * _IMAGE_TOKEN_COUNT_2X2
    feature = _feature(
        offset=0,
        length=_IMAGE_TOKEN_COUNT_2X2,
        grid_thw=(1, 4),
    )

    with pytest.raises(ValueError, match="image_grid_thw must contain exactly 3"):
        _adapter().get_mrope_input_positions(
            tokens,
            [feature],
        )


def test_embed_count_mismatch_raises() -> None:
    tokens = [_IMAGE_PLACEHOLDER_TOKEN_ID] * _IMAGE_TOKEN_COUNT_2X2
    feature = _feature(
        offset=0,
        length=_IMAGE_TOKEN_COUNT_2X2 - 1,
        grid_thw=_IMAGE_GRID_THW_2X2,
    )

    with pytest.raises(ValueError, match="image_grid_thw implies 4"):
        _adapter().get_mrope_input_positions(
            tokens,
            [feature],
        )


def test_video_feature_raises() -> None:
    tokens = [_IMAGE_PLACEHOLDER_TOKEN_ID] * _IMAGE_TOKEN_COUNT_2X2
    feature = _feature(
        offset=0,
        length=_IMAGE_TOKEN_COUNT_2X2,
        grid_thw=_IMAGE_GRID_THW_2X2,
        modality="video",
    )

    with pytest.raises(NotImplementedError, match="Video multimodal features"):
        _adapter().get_mrope_input_positions(
            tokens,
            [feature],
        )


def test_unsupported_modality_raises() -> None:
    tokens = [_IMAGE_PLACEHOLDER_TOKEN_ID] * _IMAGE_TOKEN_COUNT_2X2
    feature = _feature(
        offset=0,
        length=_IMAGE_TOKEN_COUNT_2X2,
        grid_thw=_IMAGE_GRID_THW_2X2,
        modality="image_embeds",
    )

    with pytest.raises(ValueError, match="Unsupported modality: image_embeds"):
        _adapter().get_mrope_input_positions(
            tokens,
            [feature],
        )


def test_single_image_positions_in_middle_of_sequence() -> None:
    tokens = (
        [_TEXT_TOKEN_ID] * 2
        + [_IMAGE_PLACEHOLDER_TOKEN_ID] * _IMAGE_TOKEN_COUNT_2X2
        + [_TEXT_TOKEN_ID] * 2
    )
    feature = _feature(
        offset=2,
        length=_IMAGE_TOKEN_COUNT_2X2,
        grid_thw=_IMAGE_GRID_THW_2X2,
    )
    positions, delta = _adapter().get_mrope_input_positions(
        tokens,
        [feature],
    )

    assert positions.tolist() == [
        [0, 1, 2, 2, 2, 2, 4, 5],
        [0, 1, 2, 2, 3, 3, 4, 5],
        [0, 1, 2, 3, 2, 3, 4, 5],
    ]
    assert delta == -2


def test_multi_image_positions_preserve_feature_offsets() -> None:
    tokens = (
        [_TEXT_TOKEN_ID]
        + [_IMAGE_PLACEHOLDER_TOKEN_ID] * _IMAGE_TOKEN_COUNT_2X2
        + [_TEXT_TOKEN_ID]
        + [_IMAGE_PLACEHOLDER_TOKEN_ID] * _IMAGE_TOKEN_COUNT_2X2
        + [_TEXT_TOKEN_ID]
    )
    # Intentionally reversed: the helper should mirror upstream Qwen3-VL and
    # sort features by placeholder offset before computing M-RoPE positions.
    features = [
        _feature(
            offset=6,
            length=_IMAGE_TOKEN_COUNT_2X2,
            grid_thw=_IMAGE_GRID_THW_2X2,
        ),
        _feature(
            offset=1,
            length=_IMAGE_TOKEN_COUNT_2X2,
            grid_thw=_IMAGE_GRID_THW_2X2,
        ),
    ]

    positions, delta = _adapter().get_mrope_input_positions(
        tokens,
        features,
    )

    assert positions.tolist() == [
        [0, 1, 1, 1, 1, 3, 4, 4, 4, 4, 6],
        [0, 1, 1, 2, 2, 3, 4, 4, 5, 5, 6],
        [0, 1, 2, 1, 2, 3, 4, 5, 4, 5, 6],
    ]
    assert delta == -4


def test_image_at_start_of_sequence() -> None:
    tokens = [_IMAGE_PLACEHOLDER_TOKEN_ID] * _IMAGE_TOKEN_COUNT_2X2
    tokens += [_TEXT_TOKEN_ID] * 6
    feature = _feature(
        offset=0,
        length=_IMAGE_TOKEN_COUNT_2X2,
        grid_thw=_IMAGE_GRID_THW_2X2,
    )

    positions, delta = _adapter().get_mrope_input_positions(
        tokens,
        [feature],
    )

    assert positions.tolist() == [
        [0, 0, 0, 0, 2, 3, 4, 5, 6, 7],
        [0, 0, 1, 1, 2, 3, 4, 5, 6, 7],
        [0, 1, 0, 1, 2, 3, 4, 5, 6, 7],
    ]
    assert delta == -2


def test_image_at_end_of_sequence() -> None:
    tokens = [_TEXT_TOKEN_ID] * 4
    tokens += [_IMAGE_PLACEHOLDER_TOKEN_ID] * _IMAGE_TOKEN_COUNT_2X2
    feature = _feature(
        offset=4,
        length=_IMAGE_TOKEN_COUNT_2X2,
        grid_thw=_IMAGE_GRID_THW_2X2,
    )

    positions, delta = _adapter().get_mrope_input_positions(
        tokens,
        [feature],
    )

    assert positions.tolist() == [
        [0, 1, 2, 3, 4, 4, 4, 4],
        [0, 1, 2, 3, 4, 4, 5, 5],
        [0, 1, 2, 3, 4, 5, 4, 5],
    ]
    assert delta == -2
