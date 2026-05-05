# SPDX-License-Identifier: Apache-2.0
"""Tests for Qwen3.5-4B multimodal M-RoPE helpers."""

from __future__ import annotations

import pytest
import torch
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItem

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


def _adapter(spatial_merge_size: int = _SPATIAL_MERGE_SIZE) -> Qwen3VLMultimodalAdapter:
    return Qwen3VLMultimodalAdapter(spatial_merge_size=spatial_merge_size)


def _grid_item(
    *,
    key: str,
    modality: str,
    grid_thw: tuple[int, ...],
) -> MultiModalKwargsItem:
    field_config = MultiModalFieldConfig.batched(modality, keep_on_cpu=True)
    field_elem = field_config.build_elems(key, torch.tensor([grid_thw]))[0]
    return MultiModalKwargsItem({key: field_elem})


def _feature(
    *,
    offset: int,
    length: int,
    grid_thw: tuple[int, ...] = _IMAGE_GRID_THW_2X2,
    modality: str = "image",
) -> MultiModalFeatureSpec:
    field_modality = "video" if modality == "video" else "image"
    key = "video_grid_thw" if modality == "video" else "image_grid_thw"
    return MultiModalFeatureSpec(
        data=_grid_item(key=key, modality=field_modality, grid_thw=grid_thw),
        modality=modality,
        identifier=f"{modality}-{offset}",
        mm_position=PlaceholderRange(offset=offset, length=length),
    )


def _single_image_tokens(*, text_before: int, text_after: int) -> list[int]:
    return (
        [_TEXT_TOKEN_ID] * text_before
        + [_IMAGE_PLACEHOLDER_TOKEN_ID] * _IMAGE_TOKEN_COUNT_2X2
        + [_TEXT_TOKEN_ID] * text_after
    )


class TestQwen3VLMultimodalAdapterValidation:
    def test_invalid_spatial_merge_size_raises(self) -> None:
        with pytest.raises(ValueError, match="spatial_merge_size must be positive"):
            _adapter(spatial_merge_size=0)

    @pytest.mark.parametrize(
        ("feature", "error_type", "match"),
        [
            pytest.param(
                _feature(
                    offset=0,
                    length=_IMAGE_TOKEN_COUNT_2X2,
                    grid_thw=(1, 4),
                ),
                ValueError,
                "image_grid_thw must contain exactly 3",
                id="bad-grid-shape",
            ),
            pytest.param(
                _feature(
                    offset=0,
                    length=_IMAGE_TOKEN_COUNT_2X2 - 1,
                ),
                ValueError,
                "image_grid_thw implies 4",
                id="embed-count-mismatch",
            ),
            pytest.param(
                _feature(
                    offset=0,
                    length=_IMAGE_TOKEN_COUNT_2X2,
                    modality="video",
                ),
                NotImplementedError,
                "Video multimodal features",
                id="video-out-of-scope",
            ),
            pytest.param(
                _feature(
                    offset=0,
                    length=_IMAGE_TOKEN_COUNT_2X2,
                    modality="image_embeds",
                ),
                ValueError,
                "Unsupported modality: image_embeds",
                id="unsupported-modality",
            ),
        ],
    )
    def test_invalid_image_features_raise(
        self,
        feature: MultiModalFeatureSpec,
        error_type: type[Exception],
        match: str,
    ) -> None:
        tokens = [_IMAGE_PLACEHOLDER_TOKEN_ID] * _IMAGE_TOKEN_COUNT_2X2

        with pytest.raises(error_type, match=match):
            _adapter().get_mrope_input_positions(tokens, [feature])


class TestQwen3VLMultimodalAdapterPositions:
    def test_text_only_positions(self) -> None:
        positions, delta = _adapter().get_mrope_input_positions(
            [_TEXT_TOKEN_ID] * 4,
            [],
        )
        assert positions.tolist() == [[0, 1, 2, 3]] * 3
        assert delta == 0

    @pytest.mark.parametrize(
        ("text_before", "text_after", "expected_positions", "expected_delta"),
        [
            pytest.param(
                0,
                6,
                [
                    [0, 0, 0, 0, 2, 3, 4, 5, 6, 7],
                    [0, 0, 1, 1, 2, 3, 4, 5, 6, 7],
                    [0, 1, 0, 1, 2, 3, 4, 5, 6, 7],
                ],
                -2,
                id="image-at-start",
            ),
            pytest.param(
                2,
                2,
                [
                    [0, 1, 2, 2, 2, 2, 4, 5],
                    [0, 1, 2, 2, 3, 3, 4, 5],
                    [0, 1, 2, 3, 2, 3, 4, 5],
                ],
                -2,
                id="image-in-middle",
            ),
            pytest.param(
                4,
                0,
                [
                    [0, 1, 2, 3, 4, 4, 4, 4],
                    [0, 1, 2, 3, 4, 4, 5, 5],
                    [0, 1, 2, 3, 4, 5, 4, 5],
                ],
                -2,
                id="image-at-end",
            ),
        ],
    )
    def test_single_image_positions(
        self,
        text_before: int,
        text_after: int,
        expected_positions: list[list[int]],
        expected_delta: int,
    ) -> None:
        tokens = _single_image_tokens(
            text_before=text_before,
            text_after=text_after,
        )
        feature = _feature(
            offset=text_before,
            length=_IMAGE_TOKEN_COUNT_2X2,
        )

        positions, delta = _adapter().get_mrope_input_positions(tokens, [feature])

        assert positions.tolist() == expected_positions
        assert delta == expected_delta

    def test_multi_image_positions_preserve_feature_offsets(self) -> None:
        tokens = (
            [_TEXT_TOKEN_ID]
            + [_IMAGE_PLACEHOLDER_TOKEN_ID] * _IMAGE_TOKEN_COUNT_2X2
            + [_TEXT_TOKEN_ID]
            + [_IMAGE_PLACEHOLDER_TOKEN_ID] * _IMAGE_TOKEN_COUNT_2X2
            + [_TEXT_TOKEN_ID]
        )
        features = [
            _feature(offset=6, length=_IMAGE_TOKEN_COUNT_2X2),
            _feature(offset=1, length=_IMAGE_TOKEN_COUNT_2X2),
        ]

        positions, delta = _adapter().get_mrope_input_positions(tokens, features)

        assert positions.tolist() == [
            [0, 1, 1, 1, 1, 3, 4, 4, 4, 4, 6],
            [0, 1, 1, 2, 2, 3, 4, 4, 5, 5, 6],
            [0, 1, 2, 1, 2, 3, 4, 5, 4, 5, 6],
        ]
        assert delta == -4
