# SPDX-License-Identifier: Apache-2.0
"""Tests for Qwen3.5-4B multimodal helpers and adapter capabilities."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import mlx.core as mx
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
    def test_empty_positions_carry_batch_axis(self) -> None:
        positions, delta = _adapter().get_mrope_input_positions([], [])

        assert positions.shape == (3, 1, 0)
        assert delta == 0

    def test_text_only_positions(self) -> None:
        positions, delta = _adapter().get_mrope_input_positions(
            [_TEXT_TOKEN_ID] * 4,
            [],
        )
        assert positions.shape == (3, 1, 4)
        assert positions.tolist() == [[[0, 1, 2, 3]]] * 3
        assert delta == 0

    @pytest.mark.parametrize(
        ("text_before", "text_after", "expected_rows", "expected_delta"),
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
        expected_rows: list[list[int]],
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

        assert positions.shape == (3, 1, len(tokens))
        assert positions.tolist() == [[row] for row in expected_rows]
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

        assert positions.shape == (3, 1, len(tokens))
        assert positions.tolist() == [
            [[0, 1, 1, 1, 1, 3, 4, 4, 4, 4, 6]],
            [[0, 1, 1, 2, 2, 3, 4, 4, 5, 5, 6]],
            [[0, 1, 2, 1, 2, 3, 4, 5, 4, 5, 6]],
        ]
        assert delta == -4


class _RecordingVisionTower:
    """Records its calls and returns deterministic ``(hidden, deepstack)`` pairs.

    Exposes ``patch_embed.proj.weight.dtype`` so the adapter's input-dtype
    cast — mirroring ``mlx_vlm.Model.get_input_embeddings`` — has a real
    target to read.
    """

    def __init__(
        self,
        hidden_factory: Any | None = None,
        deepstack_factory: Any | None = None,
        *,
        weight_dtype: mx.Dtype = mx.float32,
    ) -> None:
        self.calls: list[tuple[Any, Any]] = []
        self._hidden_factory = hidden_factory or (
            lambda call_idx: mx.full((4, 8), float(call_idx), dtype=mx.float32)
        )
        self._deepstack_factory = deepstack_factory or (lambda call_idx: None)
        # ``mlx_vlm.Model.get_input_embeddings`` reads
        # ``vision_tower.patch_embed.proj.weight.dtype``; mirror that path.
        self.patch_embed = SimpleNamespace(
            proj=SimpleNamespace(weight=mx.zeros((1,), dtype=weight_dtype))
        )

    def __call__(self, pixel_values: Any, grid_thw: Any) -> tuple[mx.array, Any]:
        self.calls.append((pixel_values, grid_thw))
        call_idx = len(self.calls) - 1
        return self._hidden_factory(call_idx), self._deepstack_factory(call_idx)


def _vision_feature(
    *,
    offset: int = 0,
    length: int = _IMAGE_TOKEN_COUNT_2X2,
    grid_thw: tuple[int, ...] = _IMAGE_GRID_THW_2X2,
    pixel_rows: int | None = None,
    modality: str = "image",
) -> MultiModalFeatureSpec:
    """Build a feature with both ``pixel_values`` and ``image_grid_thw`` fields."""
    if pixel_rows is None:
        pixel_rows = length
    field_config = MultiModalFieldConfig.batched("image", keep_on_cpu=True)
    grid_elem = field_config.build_elems("image_grid_thw", torch.tensor([grid_thw]))[0]
    pixel_elem = field_config.build_elems(
        "pixel_values", torch.zeros((1, pixel_rows, 8), dtype=torch.float32)
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


class TestQwen3VLMultimodalAdapterEncodeMultimodal:
    def test_calls_vision_tower_and_preserves_deepstack(self) -> None:
        deepstack = [mx.full((4, 8), 9.0, dtype=mx.float32)]
        tower = _RecordingVisionTower(deepstack_factory=lambda _: deepstack)
        adapter = Qwen3VLMultimodalAdapter(
            spatial_merge_size=_SPATIAL_MERGE_SIZE,
            vision_tower=tower,
        )
        feature = _vision_feature()

        outputs = adapter.encode_multimodal([feature])

        assert len(outputs) == 1
        assert outputs[0].hidden_states.shape == (4, 8)
        assert outputs[0].hidden_states.tolist() == [[0.0] * 8] * 4
        assert outputs[0].deepstack_visual_embeds is deepstack
        assert len(tower.calls) == 1

    def test_returns_one_output_per_feature(self) -> None:
        tower = _RecordingVisionTower()
        adapter = Qwen3VLMultimodalAdapter(
            spatial_merge_size=_SPATIAL_MERGE_SIZE,
            vision_tower=tower,
        )
        features = [
            _vision_feature(offset=0),
            _vision_feature(offset=10),
        ]

        outputs = adapter.encode_multimodal(features)

        assert len(outputs) == 2
        assert outputs[0].hidden_states.tolist()[0][0] == 0.0
        assert outputs[1].hidden_states.tolist()[0][0] == 1.0
        assert len(tower.calls) == 2

    def test_passes_mlx_arrays_to_vision_tower(self) -> None:
        tower = _RecordingVisionTower()
        adapter = Qwen3VLMultimodalAdapter(
            spatial_merge_size=_SPATIAL_MERGE_SIZE,
            vision_tower=tower,
        )

        adapter.encode_multimodal([_vision_feature()])

        pixel_values, grid_thw = tower.calls[0]
        assert isinstance(pixel_values, mx.array)
        assert isinstance(grid_thw, mx.array)

    def test_reshapes_grid_thw_to_two_dim_for_vision_tower(self) -> None:
        """vLLM batched fields deliver ``(3,)``; tower indexes ``grid_thw[:, 1:]``."""
        tower = _RecordingVisionTower()
        adapter = Qwen3VLMultimodalAdapter(
            spatial_merge_size=_SPATIAL_MERGE_SIZE,
            vision_tower=tower,
        )

        adapter.encode_multimodal([_vision_feature(grid_thw=_IMAGE_GRID_THW_2X2)])

        _, grid_thw = tower.calls[0]
        assert grid_thw.shape == (1, 3)
        assert grid_thw.tolist() == [list(_IMAGE_GRID_THW_2X2)]

    @pytest.mark.parametrize(
        "weight_dtype",
        [mx.float16, mx.bfloat16, mx.float32],
    )
    def test_casts_pixel_values_to_vision_tower_weight_dtype(
        self,
        weight_dtype: mx.Dtype,
    ) -> None:
        tower = _RecordingVisionTower(weight_dtype=weight_dtype)
        adapter = Qwen3VLMultimodalAdapter(
            spatial_merge_size=_SPATIAL_MERGE_SIZE,
            vision_tower=tower,
        )

        adapter.encode_multimodal([_vision_feature()])

        pixel_values, _ = tower.calls[0]
        assert pixel_values.dtype == weight_dtype

    def test_raises_on_video_feature(self) -> None:
        tower = _RecordingVisionTower()
        adapter = Qwen3VLMultimodalAdapter(
            spatial_merge_size=_SPATIAL_MERGE_SIZE,
            vision_tower=tower,
        )
        feature = _vision_feature()
        feature = MultiModalFeatureSpec(
            data=feature.data,
            modality="video",
            identifier=feature.identifier,
            mm_position=feature.mm_position,
        )

        with pytest.raises(ValueError, match="image features"):
            adapter.encode_multimodal([feature])

    def test_raises_when_vision_tower_is_none(self) -> None:
        adapter = Qwen3VLMultimodalAdapter(spatial_merge_size=_SPATIAL_MERGE_SIZE)

        with pytest.raises(RuntimeError, match="vision_tower not loaded"):
            adapter.encode_multimodal([_vision_feature()])

    def test_raises_on_missing_feature_data(self) -> None:
        tower = _RecordingVisionTower()
        adapter = Qwen3VLMultimodalAdapter(
            spatial_merge_size=_SPATIAL_MERGE_SIZE,
            vision_tower=tower,
        )
        feature = MultiModalFeatureSpec(
            data=None,
            modality="image",
            identifier="image-0",
            mm_position=PlaceholderRange(offset=0, length=4),
        )

        with pytest.raises(ValueError, match="feature.data is required"):
            adapter.encode_multimodal([feature])


class _RecordingLanguageModel:
    """Captures call kwargs for ``call_lm`` assertions.

    Mirrors mlx_vlm 0.4.x's ``LanguageModel.__call__`` shape: ``inputs_embeds``
    is a named parameter so signature sniffing finds it.  ``self.model.embed_tokens``
    mirrors the bottom-level embedding callable the real LM exposes.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.embed_calls: list[mx.array] = []

        def _embed(input_ids: mx.array) -> mx.array:
            self.embed_calls.append(input_ids)
            return input_ids

        self.model = SimpleNamespace(embed_tokens=_embed)

    def __call__(
        self,
        inputs: mx.array,
        inputs_embeds: mx.array | None = None,
        cache: list[Any] | None = None,
        position_ids: mx.array | None = None,
    ) -> str:
        self.calls.append(
            {
                "inputs": inputs,
                "inputs_embeds": inputs_embeds,
                "cache": cache,
                "position_ids": position_ids,
            }
        )
        return "lm-output"


class _LegacyLanguageModel:
    """LM that uses the older ``input_embeddings`` keyword."""

    def __init__(self) -> None:
        self.model = SimpleNamespace(embed_tokens=lambda input_ids: input_ids)

    def __call__(
        self,
        inputs: mx.array,
        input_embeddings: mx.array | None = None,
        cache: list[Any] | None = None,
        position_ids: mx.array | None = None,
    ) -> str:
        return "legacy-output"


class _UnknownLanguageModel:
    """LM that exposes neither supported embeds keyword."""

    def __call__(self, inputs: mx.array, cache: list[Any] | None = None) -> str:
        return "no-embeds"


class _DeepstackLanguageModel:
    """Mirrors mlx_vlm 0.5.x Qwen3-VL ``LanguageModel.__call__``.

    Declares ``visual_pos_masks`` and ``deepstack_visual_embeds`` as named
    parameters so the deepstack detector flips to True.  Records the kwargs
    each call receives.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.model = SimpleNamespace(embed_tokens=lambda input_ids: input_ids)

    def __call__(
        self,
        inputs: mx.array,
        inputs_embeds: mx.array | None = None,
        cache: list[Any] | None = None,
        position_ids: mx.array | None = None,
        visual_pos_masks: Any | None = None,
        deepstack_visual_embeds: Any | None = None,
    ) -> str:
        self.calls.append(
            {
                "inputs": inputs,
                "inputs_embeds": inputs_embeds,
                "cache": cache,
                "position_ids": position_ids,
                "visual_pos_masks": visual_pos_masks,
                "deepstack_visual_embeds": deepstack_visual_embeds,
            }
        )
        return "deepstack-output"


class _KwargsOnlyLanguageModel:
    """LM with ``inputs_embeds`` plus ``**kwargs`` — qwen3_5 0.4.x shape.

    Catches deepstack-named kwargs into the catch-all rather than via
    explicit parameters, so ``_detect_deepstack_kwargs`` should return
    False and the adapter must omit deepstack kwargs entirely.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.model = SimpleNamespace(embed_tokens=lambda input_ids: input_ids)

    def __call__(
        self,
        inputs: mx.array,
        inputs_embeds: mx.array | None = None,
        cache: list[Any] | None = None,
        position_ids: mx.array | None = None,
        **kwargs: Any,
    ) -> str:
        self.calls.append(
            {
                "inputs": inputs,
                "inputs_embeds": inputs_embeds,
                "cache": cache,
                "position_ids": position_ids,
                "extra": dict(kwargs),
            }
        )
        return "kwargs-only-output"


class TestQwen3VLMultimodalAdapterDetectEmbedsKwarg:
    def test_detects_inputs_embeds(self) -> None:
        kwarg = Qwen3VLMultimodalAdapter._detect_embeds_kwarg(_RecordingLanguageModel())
        assert kwarg == "inputs_embeds"

    def test_detects_input_embeddings(self) -> None:
        kwarg = Qwen3VLMultimodalAdapter._detect_embeds_kwarg(_LegacyLanguageModel())
        assert kwarg == "input_embeddings"

    def test_raises_on_drift(self) -> None:
        with pytest.raises(RuntimeError, match="mlx_vlm version drift"):
            Qwen3VLMultimodalAdapter._detect_embeds_kwarg(_UnknownLanguageModel())


class TestQwen3VLMultimodalAdapterDetectDeepstackKwargs:
    def test_detects_explicit_deepstack_params(self) -> None:
        assert (
            Qwen3VLMultimodalAdapter._detect_deepstack_kwargs(_DeepstackLanguageModel())
            is True
        )

    def test_rejects_kwargs_catch_all(self) -> None:
        assert (
            Qwen3VLMultimodalAdapter._detect_deepstack_kwargs(
                _KwargsOnlyLanguageModel()
            )
            is False
        )

    def test_rejects_lm_missing_both_params(self) -> None:
        assert (
            Qwen3VLMultimodalAdapter._detect_deepstack_kwargs(_RecordingLanguageModel())
            is False
        )

    def test_rejects_when_only_one_param_present(self) -> None:
        class _PartialDeepstackLM:
            def __call__(
                self,
                inputs: mx.array,
                inputs_embeds: mx.array | None = None,
                visual_pos_masks: Any | None = None,
            ) -> None:
                return None

        assert (
            Qwen3VLMultimodalAdapter._detect_deepstack_kwargs(_PartialDeepstackLM())
            is False
        )


class TestQwen3VLMultimodalAdapterCallLm:
    def test_passes_sniffed_kwarg_to_language_model(self) -> None:
        lm = _RecordingLanguageModel()
        adapter = Qwen3VLMultimodalAdapter(
            spatial_merge_size=_SPATIAL_MERGE_SIZE,
            language_model=lm,
            embeds_kwarg="inputs_embeds",
        )
        input_ids = mx.array([[1, 2, 3]], dtype=mx.int32)
        inputs_embeds = mx.zeros((1, 3, 8), dtype=mx.float32)
        position_ids = mx.zeros((3, 1, 3), dtype=mx.int32)

        output = adapter.call_lm(input_ids, inputs_embeds, [None], position_ids)

        assert output == "lm-output"
        assert len(lm.calls) == 1
        call = lm.calls[0]
        assert "inputs_embeds" in call
        assert call["inputs_embeds"] is inputs_embeds
        assert call["position_ids"] is position_ids
        assert call["cache"] == [None]

    def test_routes_through_legacy_input_embeddings_kwarg(self) -> None:
        captured: dict[str, Any] = {}

        class _Capture:
            def __call__(
                self,
                inputs: mx.array,
                input_embeddings: mx.array | None = None,
                cache: list[Any] | None = None,
                position_ids: mx.array | None = None,
            ) -> None:
                captured["input_embeddings"] = input_embeddings

        adapter = Qwen3VLMultimodalAdapter(
            spatial_merge_size=_SPATIAL_MERGE_SIZE,
            language_model=_Capture(),
            embeds_kwarg="input_embeddings",
        )
        inputs_embeds = mx.zeros((1, 1, 8), dtype=mx.float32)

        adapter.call_lm(
            mx.array([[1]], dtype=mx.int32),
            inputs_embeds,
            [None],
            mx.zeros((3, 1, 1), dtype=mx.int32),
        )

        assert captured["input_embeddings"] is inputs_embeds

    def test_raises_when_language_model_missing(self) -> None:
        adapter = Qwen3VLMultimodalAdapter(spatial_merge_size=_SPATIAL_MERGE_SIZE)

        with pytest.raises(RuntimeError, match="language_model not loaded"):
            adapter.call_lm(
                mx.array([[1]], dtype=mx.int32),
                mx.zeros((1, 1, 8), dtype=mx.float32),
                [None],
                mx.zeros((3, 1, 1), dtype=mx.int32),
            )

    def test_raises_when_embeds_kwarg_not_detected(self) -> None:
        adapter = Qwen3VLMultimodalAdapter(
            spatial_merge_size=_SPATIAL_MERGE_SIZE,
            language_model=_RecordingLanguageModel(),
        )

        with pytest.raises(RuntimeError, match="embeds_kwarg not detected"):
            adapter.call_lm(
                mx.array([[1]], dtype=mx.int32),
                mx.zeros((1, 1, 8), dtype=mx.float32),
                [None],
                mx.zeros((3, 1, 1), dtype=mx.int32),
            )

    def test_omits_deepstack_kwargs_when_unsupported_and_none(self) -> None:
        lm = _KwargsOnlyLanguageModel()
        adapter = Qwen3VLMultimodalAdapter(
            spatial_merge_size=_SPATIAL_MERGE_SIZE,
            language_model=lm,
            embeds_kwarg="inputs_embeds",
            supports_deepstack=False,
        )

        adapter.call_lm(
            mx.array([[1, 2, 3]], dtype=mx.int32),
            mx.zeros((1, 3, 8), dtype=mx.float32),
            [None],
            mx.zeros((3, 1, 3), dtype=mx.int32),
            visual_pos_masks=mx.array([[True, False, True]]),
            deepstack_visual_embeds=None,
        )

        assert len(lm.calls) == 1
        assert lm.calls[0]["extra"] == {}

    def test_raises_when_unsupported_lm_receives_deepstack_residuals(self) -> None:
        lm = _KwargsOnlyLanguageModel()
        adapter = Qwen3VLMultimodalAdapter(
            spatial_merge_size=_SPATIAL_MERGE_SIZE,
            language_model=lm,
            embeds_kwarg="inputs_embeds",
            supports_deepstack=False,
        )

        with pytest.raises(RuntimeError, match="deepstack_visual_embeds were produced"):
            adapter.call_lm(
                mx.array([[1, 2, 3]], dtype=mx.int32),
                mx.zeros((1, 3, 8), dtype=mx.float32),
                [None],
                mx.zeros((3, 1, 3), dtype=mx.int32),
                visual_pos_masks=mx.array([[True, False, True]]),
                deepstack_visual_embeds=[mx.zeros((1, 2, 8))],
            )
        assert lm.calls == []

    def test_forwards_deepstack_kwargs_when_supported(self) -> None:
        lm = _DeepstackLanguageModel()
        adapter = Qwen3VLMultimodalAdapter(
            spatial_merge_size=_SPATIAL_MERGE_SIZE,
            language_model=lm,
            embeds_kwarg="inputs_embeds",
            supports_deepstack=True,
        )
        visual_pos_masks = mx.array([[True, False, True]])
        deepstack_visual_embeds = [mx.zeros((1, 2, 8))]

        adapter.call_lm(
            mx.array([[1, 2, 3]], dtype=mx.int32),
            mx.zeros((1, 3, 8), dtype=mx.float32),
            [None],
            mx.zeros((3, 1, 3), dtype=mx.int32),
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )

        assert len(lm.calls) == 1
        call = lm.calls[0]
        assert call["visual_pos_masks"] is visual_pos_masks
        assert call["deepstack_visual_embeds"] is deepstack_visual_embeds

    def test_passes_none_deepstack_kwargs_when_supported_and_unset(self) -> None:
        lm = _DeepstackLanguageModel()
        adapter = Qwen3VLMultimodalAdapter(
            spatial_merge_size=_SPATIAL_MERGE_SIZE,
            language_model=lm,
            embeds_kwarg="inputs_embeds",
            supports_deepstack=True,
        )

        adapter.call_lm(
            mx.array([[1]], dtype=mx.int32),
            mx.zeros((1, 1, 8), dtype=mx.float32),
            [None],
            mx.zeros((3, 1, 1), dtype=mx.int32),
        )

        assert len(lm.calls) == 1
        assert lm.calls[0]["visual_pos_masks"] is None
        assert lm.calls[0]["deepstack_visual_embeds"] is None


class TestQwen3VLMultimodalAdapterFromLoadedModel:
    def test_sniffs_embeds_kwarg_from_language_model(self) -> None:
        class _LoadedModel:
            class _Config:
                class _VisionConfig:
                    spatial_merge_size = 2

                vision_config = _VisionConfig()

            config = _Config()
            vision_tower = _RecordingVisionTower()
            language_model = _RecordingLanguageModel()

        adapter = Qwen3VLMultimodalAdapter.from_loaded_model(_LoadedModel())

        assert adapter._embeds_kwarg == "inputs_embeds"
        assert adapter._spatial_merge_size == 2

    def test_resolves_embed_tokens_at_load_time(self) -> None:
        language_model = _RecordingLanguageModel()

        class _LoadedModel:
            class _Config:
                class _VisionConfig:
                    spatial_merge_size = 2

                vision_config = _VisionConfig()

            config = _Config()
            vision_tower = _RecordingVisionTower()

        loaded = _LoadedModel()
        loaded.language_model = language_model

        adapter = Qwen3VLMultimodalAdapter.from_loaded_model(loaded)

        assert adapter._embed_tokens_fn is language_model.model.embed_tokens

    def test_detects_no_deepstack_on_qwen3_5_style_lm(self) -> None:
        class _LoadedModel:
            class _Config:
                class _VisionConfig:
                    spatial_merge_size = 2

                vision_config = _VisionConfig()

            config = _Config()
            vision_tower = _RecordingVisionTower()
            language_model = _RecordingLanguageModel()

        adapter = Qwen3VLMultimodalAdapter.from_loaded_model(_LoadedModel())

        assert adapter._supports_deepstack is False

    def test_detects_deepstack_on_qwen3_vl_style_lm(self) -> None:
        class _LoadedModel:
            class _Config:
                class _VisionConfig:
                    spatial_merge_size = 2

                vision_config = _VisionConfig()

            config = _Config()
            vision_tower = _RecordingVisionTower()
            language_model = _DeepstackLanguageModel()

        adapter = Qwen3VLMultimodalAdapter.from_loaded_model(_LoadedModel())

        assert adapter._supports_deepstack is True


class TestQwen3VLMultimodalAdapterResolveEmbedTokens:
    def test_returns_inner_embed_tokens_callable(self) -> None:
        language_model = _RecordingLanguageModel()

        resolved = Qwen3VLMultimodalAdapter._resolve_embed_tokens(language_model)

        assert resolved is language_model.model.embed_tokens

    def test_raises_when_inner_model_missing(self) -> None:
        class _NoInner:
            def __call__(self, *args: object, **kwargs: object) -> None:
                return None

        with pytest.raises(RuntimeError, match="language_model.model attribute"):
            Qwen3VLMultimodalAdapter._resolve_embed_tokens(_NoInner())

    def test_raises_when_embed_tokens_missing(self) -> None:
        bad = SimpleNamespace(model=SimpleNamespace())

        with pytest.raises(RuntimeError, match="embed_tokens missing"):
            Qwen3VLMultimodalAdapter._resolve_embed_tokens(bad)

    def test_raises_when_embed_tokens_not_callable(self) -> None:
        bad = SimpleNamespace(model=SimpleNamespace(embed_tokens="not-a-callable"))

        with pytest.raises(RuntimeError, match="not callable"):
            Qwen3VLMultimodalAdapter._resolve_embed_tokens(bad)


class TestQwen3VLMultimodalAdapterEmbedTokens:
    def test_forwards_to_resolved_callable(self) -> None:
        recorded: list[mx.array] = []

        def _embed(input_ids: mx.array) -> mx.array:
            recorded.append(input_ids)
            return input_ids + 1

        adapter = Qwen3VLMultimodalAdapter(
            spatial_merge_size=_SPATIAL_MERGE_SIZE,
            embed_tokens_fn=_embed,
        )
        input_ids = mx.array([[1, 2, 3]], dtype=mx.int32)

        out = adapter.embed_tokens(input_ids)

        assert mx.allclose(out, input_ids + 1).item()
        assert len(recorded) == 1
        assert recorded[0] is input_ids

    def test_raises_when_not_resolved(self) -> None:
        adapter = Qwen3VLMultimodalAdapter(spatial_merge_size=_SPATIAL_MERGE_SIZE)

        with pytest.raises(RuntimeError, match="embed_tokens_fn not resolved"):
            adapter.embed_tokens(mx.array([[1]], dtype=mx.int32))
