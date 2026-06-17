# SPDX-License-Identifier: Apache-2.0
"""Tests for PaddleOCR-VL multimodal helpers and adapter capabilities."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import mlx.core as mx
import pytest
import torch
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItem

from vllm_metal.multimodal import MultiModalFeatureSpec, PlaceholderRange
from vllm_metal.multimodal.paddleocr_vl import PaddleOCRVLMultimodalAdapter

_SPATIAL_MERGE_SIZE = 2
_IMAGE_GRID_THW_2X2 = (1, 4, 4)
_IMAGE_TOKEN_COUNT_2X2 = 4
_RAW_PATCH_COUNT_2X2 = 16
_TEXT_TOKEN_ID = 10
_VISION_START_TOKEN_ID = 20
_IMAGE_PLACEHOLDER_TOKEN_ID = 30


class _RecordingLanguageModel:
    def __init__(self, *, scalar_delta: bool = False) -> None:
        self._scalar_delta = scalar_delta
        self.model = SimpleNamespace(embed_tokens=lambda input_ids: input_ids + 1)
        self.rope_calls: list[dict[str, Any]] = []
        self.call_lm_calls: list[dict[str, Any]] = []

    def get_rope_index(
        self,
        input_ids: mx.array,
        image_grid_thw: mx.array | None = None,
        video_grid_thw: mx.array | None = None,
        attention_mask: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        self.rope_calls.append(
            {
                "input_ids": input_ids,
                "image_grid_thw": image_grid_thw,
                "video_grid_thw": video_grid_thw,
                "attention_mask": attention_mask,
            }
        )
        seq_len = input_ids.shape[1]
        positions = mx.arange(seq_len, dtype=mx.int32)
        positions = mx.broadcast_to(positions[None, None, :], (3, 1, seq_len))
        if self._scalar_delta:
            return positions, mx.array(-2, dtype=mx.int32)
        return positions, mx.array([[-2]], dtype=mx.int32)

    def __call__(
        self,
        input_ids: mx.array,
        *,
        inputs_embeds: mx.array,
        cache: list[Any],
        position_ids: mx.array,
    ) -> Any:
        self.call_lm_calls.append(
            {
                "input_ids": input_ids,
                "inputs_embeds": inputs_embeds,
                "cache": cache,
                "position_ids": position_ids,
            }
        )
        return SimpleNamespace(logits=mx.zeros((1, input_ids.shape[1], 8)))


class _RecordingVisual:
    def __init__(self, *, weight_dtype: mx.Dtype = mx.float32) -> None:
        self.calls: list[dict[str, Any]] = []
        self.embeddings = SimpleNamespace(
            patch_embedding=SimpleNamespace(weight=mx.zeros((1,), dtype=weight_dtype))
        )

    def __call__(
        self,
        pixel_values: mx.array,
        image_grid_thw: mx.array,
        *,
        output_hidden_states: bool | None = None,
    ) -> mx.array:
        self.calls.append(
            {
                "pixel_values": pixel_values,
                "image_grid_thw": image_grid_thw,
                "output_hidden_states": output_hidden_states,
            }
        )
        parts = []
        for idx, grid in enumerate(image_grid_thw.tolist()):
            t, h, w = (int(item) for item in grid)
            tokens = t * (h // _SPATIAL_MERGE_SIZE) * (w // _SPATIAL_MERGE_SIZE)
            parts.append(mx.full((tokens, 8), idx + 1, dtype=mx.float32))
        return mx.concatenate(parts, axis=0)


class _RealPaddleOCRVLGetRopeIndexLanguageModel:
    def __init__(self) -> None:
        try:
            from mlx_vlm.models.paddleocr_vl.config import (
                ModelConfig,
                TextConfig,
                VisionConfig,
            )
            from mlx_vlm.models.paddleocr_vl.language import LanguageModel
        except ModuleNotFoundError as exc:
            if exc.name and exc.name.startswith("mlx_vlm"):
                pytest.skip("mlx-vlm is only installed on Darwin/arm64")
            raise
        except RuntimeError as exc:
            if "No Metal device available" in str(exc):
                pytest.skip("mlx-vlm import requires a Metal device")
            raise

        self._language_model_cls = LanguageModel
        self.config = ModelConfig(
            text_config=TextConfig(), vision_config=VisionConfig()
        )
        self.model = SimpleNamespace(embed_tokens=lambda input_ids: input_ids + 1)

    @property
    def image_token_id(self) -> int:
        return int(self.config.image_token_id)

    @property
    def vision_start_token_id(self) -> int:
        return int(self.config.vision_start_token_id)

    def get_rope_index(
        self,
        input_ids: mx.array,
        image_grid_thw: mx.array | None = None,
        video_grid_thw: mx.array | None = None,
        attention_mask: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        return self._language_model_cls.get_rope_index(
            self,
            input_ids,
            image_grid_thw,
            video_grid_thw,
            attention_mask,
        )


def _adapter(
    *,
    visual: Any | None = None,
    language_model: Any | None = None,
    spatial_merge_size: int = _SPATIAL_MERGE_SIZE,
) -> PaddleOCRVLMultimodalAdapter:
    if language_model is None:
        language_model = _RecordingLanguageModel()
    return PaddleOCRVLMultimodalAdapter(
        spatial_merge_size=spatial_merge_size,
        visual=visual,
        language_model=language_model,
        embed_tokens_fn=language_model.model.embed_tokens,
    )


def _feature(
    *,
    offset: int = 0,
    length: int = _IMAGE_TOKEN_COUNT_2X2,
    grid_thw: tuple[int, ...] = _IMAGE_GRID_THW_2X2,
    raw_patches: int = _RAW_PATCH_COUNT_2X2,
    modality: str = "image",
) -> MultiModalFeatureSpec:
    field_config = MultiModalFieldConfig.batched("image", keep_on_cpu=True)
    grid_elem = field_config.build_elems("image_grid_thw", torch.tensor([grid_thw]))[0]
    pixel_elem = field_config.build_elems(
        "pixel_values",
        torch.zeros((1, raw_patches, 3, 14, 14), dtype=torch.float32),
    )[0]
    data = MultiModalKwargsItem(
        {"image_grid_thw": grid_elem, "pixel_values": pixel_elem}
    )
    return MultiModalFeatureSpec(
        data=data,
        modality=modality,
        identifier=f"{modality}-{offset}",
        mm_position=PlaceholderRange(offset=offset, length=length),
    )


class TestPaddleOCRVLMultimodalAdapterValidation:
    def test_invalid_spatial_merge_size_raises(self) -> None:
        with pytest.raises(ValueError, match="spatial_merge_size must be positive"):
            PaddleOCRVLMultimodalAdapter(spatial_merge_size=0)

    @pytest.mark.parametrize(
        ("feature", "error_type", "match"),
        [
            pytest.param(
                _feature(grid_thw=(1, 4), raw_patches=4),
                ValueError,
                "image_grid_thw must contain exactly 3",
                id="bad-grid-shape",
            ),
            pytest.param(
                _feature(length=_IMAGE_TOKEN_COUNT_2X2 - 1),
                ValueError,
                "image_grid_thw implies 4",
                id="embed-count-mismatch",
            ),
            pytest.param(
                _feature(modality="video"),
                NotImplementedError,
                "Video multimodal features",
                id="video-out-of-scope",
            ),
            pytest.param(
                _feature(modality="image_embeds"),
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
        with pytest.raises(error_type, match=match) as exc_info:
            _adapter().get_mrope_input_positions([1, 2, 3], [feature])
        assert feature.identifier in str(exc_info.value)

    def test_multiple_images_per_request_delegated_in_offset_order(self) -> None:
        language_model = _RecordingLanguageModel()
        adapter = _adapter(language_model=language_model)
        features = [
            _feature(offset=8, grid_thw=(1, 6, 4), length=6, raw_patches=24),
            _feature(offset=2, grid_thw=(1, 4, 4), length=4, raw_patches=16),
        ]
        input_tokens = (
            [_TEXT_TOKEN_ID, _VISION_START_TOKEN_ID]
            + [_IMAGE_PLACEHOLDER_TOKEN_ID] * _IMAGE_TOKEN_COUNT_2X2
            + [_TEXT_TOKEN_ID, _VISION_START_TOKEN_ID]
            + [_IMAGE_PLACEHOLDER_TOKEN_ID] * 6
            + [_TEXT_TOKEN_ID, _TEXT_TOKEN_ID]
        )

        adapter.get_mrope_input_positions(input_tokens, features)

        call = language_model.rope_calls[0]
        assert call["input_ids"].tolist() == [input_tokens]
        assert call["image_grid_thw"].tolist() == [[1, 4, 4], [1, 6, 4]]

    def test_multiple_images_exercise_real_mlx_vlm_rope_index(self) -> None:
        language_model = _RealPaddleOCRVLGetRopeIndexLanguageModel()
        adapter = _adapter(language_model=language_model)
        features = [
            _feature(offset=8, grid_thw=(1, 6, 4), length=6, raw_patches=24),
            _feature(offset=2, grid_thw=(1, 4, 4), length=4, raw_patches=16),
        ]
        input_tokens = (
            [_TEXT_TOKEN_ID, language_model.vision_start_token_id]
            + [language_model.image_token_id] * _IMAGE_TOKEN_COUNT_2X2
            + [_TEXT_TOKEN_ID, language_model.vision_start_token_id]
            + [language_model.image_token_id] * 6
            + [_TEXT_TOKEN_ID, _TEXT_TOKEN_ID]
        )

        positions, delta = adapter.get_mrope_input_positions(input_tokens, features)

        assert positions.shape == (3, 1, 16)
        assert delta == -5
        assert positions[:, 0, :].tolist() == [
            [0, 1, 2, 2, 2, 2, 4, 5, 6, 6, 6, 6, 6, 6, 9, 10],
            [0, 1, 2, 2, 3, 3, 4, 5, 6, 6, 7, 7, 8, 8, 9, 10],
            [0, 1, 2, 3, 2, 3, 4, 5, 6, 7, 6, 7, 6, 7, 9, 10],
        ]


class TestPaddleOCRVLMultimodalAdapterPositions:
    def test_empty_positions_carry_batch_axis(self) -> None:
        positions, delta = _adapter().get_mrope_input_positions([], [])

        assert positions.shape == (3, 1, 0)
        assert delta == 0

    def test_delegates_rope_index_to_language_model(self) -> None:
        language_model = _RecordingLanguageModel()
        adapter = _adapter(language_model=language_model)
        features = [_feature(offset=1, grid_thw=(1, 4, 4), length=4, raw_patches=16)]

        positions, delta = adapter.get_mrope_input_positions([1] * 12, features)

        assert positions.shape == (3, 1, 12)
        assert delta == -2
        call = language_model.rope_calls[0]
        assert call["input_ids"].tolist() == [[1] * 12]
        assert call["image_grid_thw"].tolist() == [[1, 4, 4]]
        assert call["video_grid_thw"] is None
        assert call["attention_mask"] is None

    def test_accepts_scalar_rope_delta_from_real_mlx_vlm(self) -> None:
        language_model = _RecordingLanguageModel(scalar_delta=True)
        adapter = _adapter(language_model=language_model)

        _positions, delta = adapter.get_mrope_input_positions(
            [1] * 8,
            [_feature(offset=1)],
        )

        assert delta == -2


class TestPaddleOCRVLMultimodalAdapterEncodeMultimodal:
    def test_calls_visual_tower_with_batched_pixel_values_and_grid(self) -> None:
        visual = _RecordingVisual()
        adapter = _adapter(visual=visual)
        feature = _feature()

        outputs = adapter.encode_multimodal([feature])

        assert len(outputs) == 1
        assert outputs[0].hidden_states.shape == (_IMAGE_TOKEN_COUNT_2X2, 8)
        assert outputs[0].deepstack_visual_embeds is None
        call = visual.calls[0]
        assert call["pixel_values"].shape == (1, _RAW_PATCH_COUNT_2X2, 3, 14, 14)
        assert call["image_grid_thw"].tolist() == [[1, 4, 4]]
        assert call["output_hidden_states"] is False

    def test_encodes_multiple_image_features(self) -> None:
        visual = _RecordingVisual()
        adapter = _adapter(visual=visual)
        features = [
            _feature(offset=1, grid_thw=(1, 4, 4), raw_patches=16),
            _feature(offset=8, grid_thw=(1, 6, 4), raw_patches=24),
        ]

        outputs = adapter.encode_multimodal(features)

        assert len(outputs) == 2
        assert [call["image_grid_thw"].tolist() for call in visual.calls] == [
            [[1, 4, 4]],
            [[1, 6, 4]],
        ]
        assert visual.calls[1]["pixel_values"].shape == (1, 24, 3, 14, 14)

    @pytest.mark.parametrize("weight_dtype", [mx.float16, mx.bfloat16, mx.float32])
    def test_casts_pixel_values_to_visual_weight_dtype(
        self,
        weight_dtype: mx.Dtype,
    ) -> None:
        visual = _RecordingVisual(weight_dtype=weight_dtype)
        adapter = _adapter(visual=visual)

        adapter.encode_multimodal([_feature()])

        assert visual.calls[0]["pixel_values"].dtype == weight_dtype

    def test_rejects_patch_count_mismatch(self) -> None:
        adapter = _adapter(visual=_RecordingVisual())

        feature = _feature(raw_patches=15)

        with pytest.raises(ValueError, match="patch count does not match") as exc_info:
            adapter.encode_multimodal([feature])
        assert feature.identifier in str(exc_info.value)

    def test_requires_visual_for_encode(self) -> None:
        with pytest.raises(RuntimeError, match="visual tower not loaded"):
            _adapter(visual=None).encode_multimodal([_feature()])


class TestPaddleOCRVLMultimodalAdapterCallLm:
    def test_call_lm_forwards_inputs_embeds_and_positions(self) -> None:
        language_model = _RecordingLanguageModel()
        adapter = _adapter(language_model=language_model)
        input_ids = mx.array([[1, 2]], dtype=mx.int32)
        inputs_embeds = mx.ones((1, 2, 8), dtype=mx.float32)
        position_ids = mx.zeros((3, 1, 2), dtype=mx.int32)
        cache = [object()]

        output = adapter.call_lm(input_ids, inputs_embeds, cache, position_ids)

        assert output.logits.shape == (1, 2, 8)
        call = language_model.call_lm_calls[0]
        assert call["input_ids"] is input_ids
        assert call["inputs_embeds"] is inputs_embeds
        assert call["cache"] is cache
        assert call["position_ids"] is position_ids

    def test_rejects_unexpected_deepstack(self) -> None:
        adapter = _adapter()

        with pytest.raises(RuntimeError, match="does not expose deepstack"):
            adapter.call_lm(
                mx.array([[1]], dtype=mx.int32),
                mx.ones((1, 1, 8), dtype=mx.float32),
                [None],
                mx.zeros((3, 1, 1), dtype=mx.int32),
                deepstack_visual_embeds=[mx.ones((1, 8))],
            )


class TestPaddleOCRVLMultimodalAdapterFromLoadedModel:
    def test_from_loaded_model_resolves_components(self) -> None:
        language_model = _RecordingLanguageModel()
        visual = _RecordingVisual()
        model = SimpleNamespace(
            config=SimpleNamespace(
                vision_config=SimpleNamespace(spatial_merge_size=2),
            ),
            visual=visual,
            language_model=language_model,
        )

        adapter = PaddleOCRVLMultimodalAdapter.from_loaded_model(model)

        assert adapter.text_model() is language_model
        input_ids = mx.array([[1]], dtype=mx.int32)
        assert adapter.embed_tokens(input_ids).tolist() == [[2]]

    def test_requires_explicit_positions(self) -> None:
        # PaddleOCR-VL's LM derives positions from model-level state, so
        # the runner must route text-only batches through the mm forward.
        assert PaddleOCRVLMultimodalAdapter.requires_explicit_positions is True
