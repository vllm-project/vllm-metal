# SPDX-License-Identifier: Apache-2.0
"""Tests for Qwen3.5-4B multimodal M-RoPE helpers."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from vllm_metal.v1.multimodal import (
    MultiModalFeatureSpec,
    PlaceholderRange,
    get_mrope_input_positions,
)

_FIXTURES = Path(__file__).parent / "fixtures"

# Qwen3.5-4B config/tokenizer map <|image_pad|> to this token id.
_QWEN35_VL_IMAGE_PAD_TOKEN_ID = 248056
_QWEN35_VL_SPATIAL_MERGE_SIZE = 2


# === Test Helpers ===


def _feature(
    *,
    offset: int,
    length: int,
    grid_thw: tuple[int, int, int],
) -> MultiModalFeatureSpec:
    return MultiModalFeatureSpec(
        data={"image_grid_thw": SimpleNamespace(data=np.array(grid_thw))},
        modality="image",
        identifier=f"image-{offset}",
        mm_position=PlaceholderRange(offset=offset, length=length),
    )


# === Tests ===


def test_text_only_positions() -> None:
    positions, delta = get_mrope_input_positions(
        [1, 2, 3, 4],
        [],
        spatial_merge_size=_QWEN35_VL_SPATIAL_MERGE_SIZE,
    )
    assert positions.tolist() == [[0, 1, 2, 3]] * 3
    assert delta == 0


def test_single_image_positions_match_fixture() -> None:
    with (_FIXTURES / "mrope_qwen35_4b_single_image.json").open() as f:
        fixture = json.load(f)

    feature = _feature(
        offset=fixture["image_offset"],
        length=fixture["image_length"],
        grid_thw=tuple(fixture["image_grid_thw"]),
    )
    positions, delta = get_mrope_input_positions(
        fixture["input_tokens"],
        [feature],
        spatial_merge_size=_QWEN35_VL_SPATIAL_MERGE_SIZE,
    )

    assert positions.tolist() == fixture["llm_positions"]
    assert delta == fixture["mrope_position_delta"]


def test_image_at_start_of_sequence() -> None:
    tokens = [_QWEN35_VL_IMAGE_PAD_TOKEN_ID] * 4 + [1, 2, 3, 4, 5, 6]
    feature = _feature(offset=0, length=4, grid_thw=(1, 4, 4))

    positions, delta = get_mrope_input_positions(
        tokens,
        [feature],
        spatial_merge_size=_QWEN35_VL_SPATIAL_MERGE_SIZE,
    )

    assert positions.tolist() == [
        [0, 0, 0, 0, 2, 3, 4, 5, 6, 7],
        [0, 0, 1, 1, 2, 3, 4, 5, 6, 7],
        [0, 1, 0, 1, 2, 3, 4, 5, 6, 7],
    ]
    assert delta == -2


def test_image_at_end_of_sequence() -> None:
    tokens = [1, 2, 3, 4] + [_QWEN35_VL_IMAGE_PAD_TOKEN_ID] * 4
    feature = _feature(offset=4, length=4, grid_thw=(1, 4, 4))

    positions, delta = get_mrope_input_positions(
        tokens,
        [feature],
        spatial_merge_size=_QWEN35_VL_SPATIAL_MERGE_SIZE,
    )

    assert positions.tolist() == [
        [0, 1, 2, 3, 4, 4, 4, 4],
        [0, 1, 2, 3, 4, 4, 5, 5],
        [0, 1, 2, 3, 4, 5, 4, 5],
    ]
    assert delta == -2


def test_mrope_delta_negative_for_image() -> None:
    tokens = [1] * 10 + [_QWEN35_VL_IMAGE_PAD_TOKEN_ID] * 25 + [2] * 15
    feature = _feature(offset=10, length=25, grid_thw=(1, 10, 10))

    _, delta = get_mrope_input_positions(
        tokens,
        [feature],
        spatial_merge_size=_QWEN35_VL_SPATIAL_MERGE_SIZE,
    )

    assert delta < 0
