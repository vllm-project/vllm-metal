# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the LFM2-VL adapter's integration contracts."""

from types import SimpleNamespace
from unittest.mock import Mock

import mlx.core as mx
import numpy as np
import pytest
import torch
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItem

from vllm_metal.compat import ensure_lfm2_vl_torch_outputs
from vllm_metal.multimodal import MultiModalFeatureSpec, PlaceholderRange
from vllm_metal.multimodal.lfm2_vl import Lfm2VlMultimodalAdapter
from vllm_metal.v1 import model_lifecycle
from vllm_metal.v1.model_adapter import DefaultModelAdapter


@pytest.fixture
def model():
    def forward(inputs, *, cache, input_embeddings):
        return input_embeddings

    backbone = Mock(side_effect=forward)
    backbone.embed_tokens = lambda ids: ids
    language_model = Mock(side_effect=forward)
    language_model.model = backbone
    # Two 4x4 patch grids and a 2x2 thumbnail, downsampled by two.
    views = [mx.full((4, 4), 0.0), mx.full((4, 4), 1.0), mx.full((1, 4), 2.0)]
    return SimpleNamespace(
        language_model=language_model, encode_images=Mock(return_value=views)
    )


def _feature(length=9):
    field = MultiModalFieldConfig.batched("image")
    cpu_field = MultiModalFieldConfig.batched("image", keep_on_cpu=True)
    data = MultiModalKwargsItem(
        {
            "pixel_values": field.build_elems(
                "pixel_values", torch.zeros((1, 3, 16, 8))
            )[0],
            "spatial_shapes": cpu_field.build_elems(
                "spatial_shapes", torch.tensor([[[4, 4], [4, 4], [2, 2]]])
            )[0],
        }
    )
    return MultiModalFeatureSpec(
        data=data,
        modality="image",
        identifier="image",
        mm_position=PlaceholderRange(offset=0, length=length),
    )


def test_tiled_image_encoding_preserves_view_layout(model):
    adapter = Lfm2VlMultimodalAdapter.from_loaded_model(model)
    result = adapter.encode_multimodal([_feature()])[0]
    call = model.encode_images.call_args
    assert call.args[0].shape == (3, 16, 8)
    assert call.kwargs["spatial_shapes"].tolist() == [[4, 4], [4, 4], [2, 2]]
    assert call.kwargs["pixel_attention_mask"].tolist() == [
        [True] * 16,
        [True] * 16,
        [True] * 4 + [False] * 12,
    ]
    assert result.hidden_states.shape == (9, 4)
    assert result.hidden_states[:, 0].tolist() == [0.0] * 4 + [1.0] * 4 + [2.0]


def test_encoder_output_must_match_placeholder_rows(model):
    adapter = Lfm2VlMultimodalAdapter.from_loaded_model(model)
    with pytest.raises(ValueError, match="placeholder embeddings"):
        adapter.encode_multimodal([_feature(length=8)])


@pytest.mark.parametrize("tokens", [[], [5, 6, 7, 8]])
def test_positions_are_sequential_with_zero_delta(model, tokens):
    adapter = Lfm2VlMultimodalAdapter.from_loaded_model(model)
    positions, delta = adapter.get_mrope_input_positions(tokens, [])
    assert positions.shape == (3, 1, len(tokens))
    assert positions.dtype == mx.int32
    assert positions.tolist() == [[list(range(len(tokens)))]] * 3
    assert delta == 0


@pytest.mark.parametrize("method", ["call_lm", "call_lm_hidden_states"])
def test_forward_passes_embeddings_to_the_selected_text_entrypoint(model, method):
    adapter = Lfm2VlMultimodalAdapter.from_loaded_model(model)
    ids = mx.array([[1, 2]], dtype=mx.int32)
    embeds = mx.ones((1, 2, 4))
    cache = []
    result = getattr(adapter, method)(ids, embeds, cache, mx.zeros((3, 1, 2)))
    target = model.language_model
    if method == "call_lm_hidden_states":
        target = target.model
    target.assert_called_once_with(ids, cache=cache, input_embeddings=embeds)
    assert result is embeds


@pytest.fixture
def processor(monkeypatch):
    from transformers.feature_extraction_utils import BatchFeature

    module = pytest.importorskip("transformers.models.lfm2_vl.processing_lfm2_vl")
    cls = module.Lfm2VlProcessor

    def numpy_call(self, **kwargs):
        return BatchFeature({"pixel_values": np.ones((2, 4, 3), dtype=np.float32)})

    monkeypatch.setattr(cls, "__call__", numpy_call)
    return object.__new__(cls)


@pytest.mark.parametrize("model_type", ["lfm2-vl", "lfm2_vl"])
@pytest.mark.parametrize("tensor_type", ["pt", "mlx"])
def test_model_load_preserves_processor_pixels_in_requested_format(
    processor, tensor_type, model_type, tmp_path, monkeypatch
):
    model = object()
    monkeypatch.setattr(
        model_lifecycle, "mlx_vlm_model_path", lambda *_, **__: tmp_path
    )
    monkeypatch.setattr(model_lifecycle, "mlx_vlm_load_model", lambda *_, **__: model)
    runner = SimpleNamespace(
        model_config=SimpleNamespace(
            revision=None, hf_config=SimpleNamespace(model_type=model_type)
        )
    )
    lifecycle = model_lifecycle.ModelLifecycle(runner, DefaultModelAdapter())
    assert lifecycle._load_generation_model("lfm2-checkpoint", is_vlm=True) is model
    pixels = processor(return_tensors=tensor_type)["pixel_values"]
    if tensor_type == "pt":
        assert isinstance(pixels, torch.Tensor)
        pixels = pixels.numpy()
    else:
        assert isinstance(pixels, np.ndarray)
    np.testing.assert_array_equal(pixels, np.ones((2, 4, 3), dtype=np.float32))


def test_processor_wrapper_is_installed_once_and_only_for_lfm2(processor):
    cls = type(processor)
    original = cls.__call__
    ensure_lfm2_vl_torch_outputs("qwen3_vl")
    assert cls.__call__ is original

    ensure_lfm2_vl_torch_outputs("lfm2_vl")
    wrapped = cls.__call__
    assert wrapped is not original
    ensure_lfm2_vl_torch_outputs("lfm2_vl")
    assert cls.__call__ is wrapped
