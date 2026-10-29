# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the LFM3-VL adapter's image encoding contract."""

from types import SimpleNamespace
from unittest.mock import Mock

import mlx.core as mx
import pytest
import torch
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItem

from vllm_metal.multimodal import MultiModalFeatureSpec, PlaceholderRange
from vllm_metal.multimodal.lfm3_vl import Lfm3VlMultimodalAdapter


@pytest.fixture
def model():
    backbone = Mock()
    backbone.embed_tokens = lambda ids: ids
    language_model = Mock()
    language_model.model = backbone
    # A 4x4 patch grid merges 2x2 into four rows.
    return SimpleNamespace(
        language_model=language_model,
        encode_images=Mock(return_value=mx.zeros((4, 8))),
    )


def _feature(modality="image", length=4):
    # Mirror the field layout of vLLM's Qwen2-VL processor.
    pixels = MultiModalFieldConfig.flat_from_sizes(modality, torch.tensor([16]))
    grid = MultiModalFieldConfig.batched(modality, keep_on_cpu=True)
    data = MultiModalKwargsItem(
        {
            "pixel_values": pixels.build_elems("pixel_values", torch.zeros((16, 12)))[
                0
            ],
            "image_grid_thw": grid.build_elems(
                "image_grid_thw", torch.tensor([[1, 4, 4]])
            )[0],
        }
    )
    return MultiModalFeatureSpec(
        data=data,
        modality=modality,
        identifier="image",
        mm_position=PlaceholderRange(offset=0, length=length),
    )


def test_image_encoding_passes_patches_with_their_grid(model):
    adapter = Lfm3VlMultimodalAdapter.from_loaded_model(model)
    result = adapter.encode_multimodal([_feature()])[0]
    call = model.encode_images.call_args
    assert call.args[0].shape == (16, 12)
    assert call.kwargs["image_grid_thw"].tolist() == [[1, 4, 4]]
    assert result.hidden_states.shape == (4, 8)


def test_encoder_output_must_match_placeholder_rows(model):
    adapter = Lfm3VlMultimodalAdapter.from_loaded_model(model)
    with pytest.raises(ValueError, match="placeholder embeddings"):
        adapter.encode_multimodal([_feature(length=5)])


def test_video_is_rejected_by_name(model):
    adapter = Lfm3VlMultimodalAdapter.from_loaded_model(model)
    with pytest.raises(NotImplementedError, match="LFM3-VL video"):
        adapter.encode_multimodal([_feature(modality="video")])


def test_processor_is_the_mlx_vlm_one():
    # Official MLX checkpoints carry no AutoProcessor mapping, so the
    # processing info must name mlx-vlm's processor instead of resolving it.
    from mlx_vlm.models.lfm3_vl.processing_lfm3_vl import Lfm3VlProcessor

    from vllm_metal.patches.lfm3_registry import Lfm3VlProcessingInfo

    ctx = Mock()
    info = Lfm3VlProcessingInfo(ctx)
    info.get_hf_processor(min_pixels=64)
    ctx.get_hf_processor.assert_called_once_with(Lfm3VlProcessor, min_pixels=64)
