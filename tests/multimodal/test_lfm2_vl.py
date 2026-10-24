# SPDX-License-Identifier: Apache-2.0
"""Tests for the LFM2-VL multimodal adapter."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import mlx.core as mx
import pytest
import torch
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItem

from vllm_metal.multimodal import MultiModalFeatureSpec, PlaceholderRange
from vllm_metal.multimodal.lfm2_vl import Lfm2VlMultimodalAdapter

_IMAGE_TOKEN_INDEX = 396
_FEATURE_DIM = 8
_HIDDEN = 4
# Two tiles plus a thumbnail, padded to the largest tile's patch count.
_TILE_SHAPES = ((4, 4), (4, 4), (2, 2))
_PADDED_PATCHES = 16
_PROJECTED_ROWS = 3


class _RecordingModel:
    """mlx-vlm ``Model`` stub recording what the adapter passes to the tower."""

    def __init__(self, *, rows_per_view: int = 1, flat: bool = False) -> None:
        self.config = SimpleNamespace(image_token_index=_IMAGE_TOKEN_INDEX)
        self.language_model = _RecordingLanguageModel()
        self.encode_calls: list[dict[str, Any]] = []
        self._rows_per_view = rows_per_view
        self._flat = flat

    def encode_images(self, pixel_values: mx.array, **kwargs: Any) -> Any:
        self.encode_calls.append({"pixel_values": pixel_values, **kwargs})
        views = [
            mx.full((self._rows_per_view, _HIDDEN), float(i))
            for i in range(int(pixel_values.shape[0]))
        ]
        if self._flat:
            return mx.concatenate(views, axis=0)
        return views


class _RecordingBackbone:
    """Headless ``language_model.model`` stub for ``call_lm_hidden_states``."""

    def __init__(self) -> None:
        self.embed_tokens = lambda input_ids: input_ids + 1
        self.calls: list[dict[str, Any]] = []

    def __call__(self, inputs: mx.array, **kwargs: Any) -> mx.array:
        self.calls.append({"inputs": inputs, **kwargs})
        return mx.zeros((1, inputs.shape[1], _HIDDEN))


class _RecordingLanguageModel:
    def __init__(self) -> None:
        self.model = _RecordingBackbone()
        self.calls: list[dict[str, Any]] = []

    def __call__(self, inputs: mx.array, **kwargs: Any) -> Any:
        self.calls.append({"inputs": inputs, **kwargs})
        return SimpleNamespace(logits=mx.zeros((1, inputs.shape[1], 8)))


def _feature(
    *,
    offset: int = 0,
    length: int = _PROJECTED_ROWS,
    tile_shapes: tuple[tuple[int, int], ...] = _TILE_SHAPES,
    padded_patches: int = _PADDED_PATCHES,
    modality: str = "image",
    drop_key: str | None = None,
) -> MultiModalFeatureSpec:
    field = MultiModalFieldConfig.batched("image")
    cpu_field = MultiModalFieldConfig.batched("image", keep_on_cpu=True)
    pixel_elem = field.build_elems(
        "pixel_values",
        torch.zeros(
            (1, len(tile_shapes), padded_patches, _FEATURE_DIM), dtype=torch.float32
        ),
    )[0]
    shapes_elem = cpu_field.build_elems(
        "spatial_shapes", torch.tensor([list(tile_shapes)], dtype=torch.int32)
    )[0]
    entries = {"pixel_values": pixel_elem, "spatial_shapes": shapes_elem}
    if drop_key is not None:
        entries.pop(drop_key)
    return MultiModalFeatureSpec(
        data=MultiModalKwargsItem(entries),
        modality=modality,
        identifier=f"{modality}-{offset}",
        mm_position=PlaceholderRange(offset=offset, length=length),
    )


def _adapter(model: _RecordingModel | None = None) -> Lfm2VlMultimodalAdapter:
    model = model or _RecordingModel()
    return Lfm2VlMultimodalAdapter.from_loaded_model(model)


class TestCapabilityFlags:
    def test_plain_rope_flags_match_the_paged_text_path(self) -> None:
        # LFM2 derives RoPE from ctx.offsets, so the runner must not hand it
        # caller-supplied positions and text-only batches stay on the text path.
        assert Lfm2VlMultimodalAdapter.supplies_segment_positions is False
        assert Lfm2VlMultimodalAdapter.requires_explicit_positions is False

    def test_forward_ready_and_selective_logits_enabled(self) -> None:
        assert Lfm2VlMultimodalAdapter.forward_ready is True
        assert Lfm2VlMultimodalAdapter.text_path_selective_logits_ok is True


class TestConstruction:
    def test_from_loaded_model_resolves_the_text_tower(self) -> None:
        model = _RecordingModel()
        adapter = _adapter(model)
        assert adapter.text_model() is model.language_model

    def test_embed_tokens_uses_the_backbone_embedding(self) -> None:
        adapter = _adapter()
        out = adapter.embed_tokens(mx.array([[1, 2, 3]], dtype=mx.int32))
        assert out.tolist() == [[2, 3, 4]]

    def test_missing_encode_images_is_rejected(self) -> None:
        broken = SimpleNamespace(
            language_model=_RecordingLanguageModel(),
            config=SimpleNamespace(image_token_index=_IMAGE_TOKEN_INDEX),
        )
        with pytest.raises(RuntimeError, match="encode_images missing"):
            Lfm2VlMultimodalAdapter.from_loaded_model(broken)

    def test_missing_image_token_index_is_rejected(self) -> None:
        model = _RecordingModel()
        model.config = SimpleNamespace(image_token_index=None)
        with pytest.raises(RuntimeError, match="image_token_index missing"):
            Lfm2VlMultimodalAdapter.from_loaded_model(model)

    def test_encode_without_construction_helper_is_rejected(self) -> None:
        with pytest.raises(RuntimeError, match="model not loaded"):
            Lfm2VlMultimodalAdapter().encode_multimodal([_feature()])

    def test_embed_tokens_without_construction_helper_is_rejected(self) -> None:
        with pytest.raises(RuntimeError, match="embed_tokens_fn not resolved"):
            Lfm2VlMultimodalAdapter().embed_tokens(mx.array([[1]], dtype=mx.int32))


class TestEncodeMultimodal:
    def test_tile_mask_marks_each_tile_real_patch_count(self) -> None:
        # The tower recovers per-tile lengths from the mask, so it must be
        # derived from spatial_shapes rather than assumed rectangular.
        model = _RecordingModel()
        _adapter(model).encode_multimodal([_feature()])
        mask = model.encode_calls[0]["pixel_attention_mask"]
        assert tuple(mask.shape) == (len(_TILE_SHAPES), _PADDED_PATCHES)
        assert mask.sum(axis=1).tolist() == [h * w for h, w in _TILE_SHAPES]

    def test_spatial_shapes_are_forwarded_unchanged(self) -> None:
        model = _RecordingModel()
        _adapter(model).encode_multimodal([_feature()])
        assert model.encode_calls[0]["spatial_shapes"].tolist() == [
            list(shape) for shape in _TILE_SHAPES
        ]

    def test_views_are_concatenated_in_prompt_order(self) -> None:
        model = _RecordingModel()
        result = _adapter(model).encode_multimodal([_feature()])
        # One row per view, each filled with its view index.
        assert result[0].hidden_states.shape[0] == _PROJECTED_ROWS
        assert [row[0] for row in result[0].hidden_states.tolist()] == [0.0, 1.0, 2.0]

    def test_a_flat_encoder_return_is_accepted(self) -> None:
        model = _RecordingModel(flat=True)
        result = _adapter(model).encode_multimodal([_feature()])
        assert result[0].hidden_states.shape[0] == _PROJECTED_ROWS

    def test_row_count_mismatch_is_rejected(self) -> None:
        model = _RecordingModel(rows_per_view=2)
        with pytest.raises(ValueError, match="placeholder embeddings"):
            _adapter(model).encode_multimodal([_feature()])

    def test_video_is_rejected_as_unsupported(self) -> None:
        with pytest.raises(NotImplementedError, match="video input"):
            _adapter().encode_multimodal([_feature(modality="video")])

    def test_other_modalities_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="only"):
            _adapter().encode_multimodal([_feature(modality="audio")])

    def test_missing_feature_data_is_rejected(self) -> None:
        feature = _feature()
        bare = MultiModalFeatureSpec(
            data=None,
            modality=feature.modality,
            identifier=feature.identifier,
            mm_position=feature.mm_position,
        )
        with pytest.raises(ValueError, match="feature.data is required"):
            _adapter().encode_multimodal([bare])

    def test_missing_spatial_shapes_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="spatial_shapes"):
            _adapter().encode_multimodal([_feature(drop_key="spatial_shapes")])

    def test_tile_count_disagreement_is_rejected(self) -> None:
        # pixel_values carries 3 tiles; spatial_shapes describes 2.
        feature = _feature()
        field = MultiModalFieldConfig.batched("image", keep_on_cpu=True)
        feature.data["spatial_shapes"] = field.build_elems(
            "spatial_shapes", torch.tensor([[[4, 4], [4, 4]]], dtype=torch.int32)
        )[0]
        with pytest.raises(ValueError, match="describes 2 tiles"):
            _adapter().encode_multimodal([feature])

    def test_tile_larger_than_the_padding_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="only carries"):
            _adapter().encode_multimodal(
                [_feature(tile_shapes=((8, 8),), padded_patches=16, length=1)]
            )

    def test_non_positive_spatial_shape_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-positive"):
            _adapter().encode_multimodal(
                [_feature(tile_shapes=((0, 4),), padded_patches=16, length=1)]
            )


class TestPositions:
    def test_positions_are_sequential_and_broadcast_to_three_rows(self) -> None:
        positions, delta = _adapter().get_mrope_input_positions([5, 6, 7, 8], [])
        assert tuple(positions.shape) == (3, 1, 4)
        assert positions.dtype == mx.int32
        assert positions.tolist() == [[[0, 1, 2, 3]]] * 3

    def test_delta_is_zero_because_lfm2_has_no_mrope(self) -> None:
        _, delta = _adapter().get_mrope_input_positions([1, 2, 3], [])
        assert delta == 0

    def test_empty_prompt_returns_an_empty_position_array(self) -> None:
        positions, delta = _adapter().get_mrope_input_positions([], [])
        assert tuple(positions.shape) == (3, 1, 0)
        assert delta == 0

    def test_video_features_are_rejected(self) -> None:
        with pytest.raises(NotImplementedError, match="video input"):
            _adapter().get_mrope_input_positions([1], [_feature(modality="video")])


class TestCallLm:
    def test_call_lm_passes_input_embeddings_not_inputs_embeds(self) -> None:
        # The LFM2 backbone only declares ``input_embeddings``; passing
        # ``inputs_embeds`` raises TypeError inside mlx-vlm.
        model = _RecordingModel()
        adapter = _adapter(model)
        ids = mx.array([[1, 2]], dtype=mx.int32)
        embeds = mx.zeros((1, 2, _HIDDEN))
        adapter.call_lm(ids, embeds, [], mx.zeros((3, 1, 2), dtype=mx.int32))
        call = model.language_model.calls[0]
        assert "input_embeddings" in call
        assert "inputs_embeds" not in call

    def test_call_lm_hidden_states_uses_the_headless_backbone(self) -> None:
        model = _RecordingModel()
        adapter = _adapter(model)
        ids = mx.array([[1, 2]], dtype=mx.int32)
        out = adapter.call_lm_hidden_states(
            ids, mx.zeros((1, 2, _HIDDEN)), [], mx.zeros((3, 1, 2), dtype=mx.int32)
        )
        assert tuple(out.shape) == (1, 2, _HIDDEN)
        assert "input_embeddings" in model.language_model.model.calls[0]

    def test_deepstack_residuals_are_rejected(self) -> None:
        adapter = _adapter()
        with pytest.raises(RuntimeError, match="deepstack"):
            adapter.call_lm(
                mx.array([[1]], dtype=mx.int32),
                mx.zeros((1, 1, _HIDDEN)),
                [],
                mx.zeros((3, 1, 1), dtype=mx.int32),
                deepstack_visual_embeds=object(),
            )

    def test_call_lm_without_construction_helper_is_rejected(self) -> None:
        with pytest.raises(RuntimeError, match="language_model not loaded"):
            Lfm2VlMultimodalAdapter().call_lm(
                mx.array([[1]], dtype=mx.int32),
                mx.zeros((1, 1, _HIDDEN)),
                [],
                mx.zeros((3, 1, 1), dtype=mx.int32),
            )
