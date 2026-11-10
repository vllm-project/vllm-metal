# SPDX-License-Identifier: Apache-2.0
"""Mixed LFM3-Omni media contracts without checkpoint weights."""

from types import SimpleNamespace
from unittest.mock import Mock

import mlx.core as mx
import numpy as np
import pytest
import torch
from PIL import Image
from vllm.config import MultiModalConfig
from vllm.exceptions import VLLMValidationError

from vllm_metal.multimodal.lfm3_omni import Lfm3OmniMultimodalAdapter
from vllm_metal.patches.lfm3_omni_registry import Lfm3OmniProcessingInfo


@pytest.fixture
def info():
    ctx = Mock()
    ctx.get_mm_config.return_value = MultiModalConfig(enable_mm_embeds=True)
    return Lfm3OmniProcessingInfo(ctx)


def test_image_and_audio_are_preserved_in_one_request(info):
    items = info.parse_mm_data(
        {"image": Image.new("RGB", (32, 32)), "audio": (np.zeros(16000), 16000)}
    )
    assert items["image"].get_count() == 1
    assert items["audio"].get_audio_length(0) == 16000


@pytest.mark.parametrize("samples", [0, 480001])
def test_audio_duration_is_checked_in_mixed_request(info, samples):
    with pytest.raises(VLLMValidationError, match="nonempty"):
        info.parse_mm_data(
            {"image": Image.new("RGB", (32, 32)), "audio": (np.zeros(samples), 16000)}
        )


@pytest.mark.parametrize("modality", ["image", "audio"])
def test_embedding_inputs_are_rejected(info, modality):
    with pytest.raises(VLLMValidationError, match="[Ee]mbeddings"):
        info.parse_mm_data({modality: torch.zeros((1, 4, 8))})


def test_encoders_preserve_interleaved_feature_order():
    from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItem

    from tests.multimodal.test_lfm3_vl import _feature
    from vllm_metal.multimodal import MultiModalFeatureSpec, PlaceholderRange

    field = MultiModalFieldConfig.batched("audio")
    data = MultiModalKwargsItem(
        {
            "input_features": field.build_elems(
                "input_features", torch.zeros((1, 128, 16))
            )[0],
            "feature_attention_mask": field.build_elems(
                "feature_attention_mask", torch.ones((1, 16))
            )[0],
        }
    )
    audio = MultiModalFeatureSpec(
        data=data,
        modality="audio",
        identifier="clip",
        mm_position=PlaceholderRange(offset=4, length=2),
    )
    model = SimpleNamespace(
        language_model=Mock(_placeholder_ids=[4, 5]),
        config=SimpleNamespace(audio_start_token_id=2, mixed_start_token_id=3),
        audio_tower=Mock(return_value=mx.full((2, 8), 2)),
        audio_projector=lambda x: x,
        encode_images=Mock(return_value=mx.full((4, 8), 1)),
    )
    adapter = Lfm3OmniMultimodalAdapter.from_loaded_model(model)
    results = adapter.encode_multimodal([_feature(), audio, _feature()])
    assert [r.hidden_states.shape[0] for r in results] == [4, 2, 4]
    assert [r.hidden_states[0, 0].item() for r in results] == [1, 2, 1]
    assert model.language_model._placeholder_ids == [4, 5, 2, 3]
    assert not adapter.text_path_selective_logits_ok


@pytest.mark.parametrize("audio_first", [False, True])
def test_mixed_processor_expands_both_modalities_without_losing_spans(
    monkeypatch, audio_first
):
    from mlx_vlm.models.lfm3_omni.processing_lfm3_omni import Lfm3OmniProcessor
    from mlx_vlm.models.lfm3_vl.processing_lfm3_vl import (
        Lfm3VlVideoProcessor,
        Qwen3VLImageProcessor,
    )
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PretrainedConfig, PreTrainedTokenizerFast
    from vllm.multimodal.processing import InputProcessingContext
    from vllm.multimodal.processing.context import TimingContext
    from vllm.multimodal.processing.inputs import ProcessorInputs

    from vllm_metal.patches.lfm3_omni_registry import (
        Lfm3OmniDummyInputsBuilder,
        Lfm3OmniMultiModalProcessor,
    )

    tokens = [
        "[UNK]",
        "<|image|>",
        "<|audio|>",
        "<|video|>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|audio_start|>",
        "<|mixed_start|>",
    ]
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel(dict(zip(tokens, range(len(tokens)), strict=True)))
        ),
        unk_token="[UNK]",
        pad_token="[UNK]",
        additional_special_tokens=tokens[1:],
    )
    native = Lfm3OmniProcessor(
        Qwen3VLImageProcessor(size={"shortest_edge": 1024, "longest_edge": 1024}),
        tokenizer,
        Lfm3VlVideoProcessor(),
    )
    mm_config = MultiModalConfig()
    config = SimpleNamespace(
        model="test-omni",
        dtype=torch.float32,
        multimodal_config=mm_config,
        get_multimodal_config=lambda: mm_config,
        hf_config=PretrainedConfig(vision_config=SimpleNamespace(spatial_merge_size=2)),
    )
    ctx = InputProcessingContext(config, tokenizer)
    monkeypatch.setattr(
        InputProcessingContext, "get_hf_processor", lambda *a, **kw: native
    )
    info = Lfm3OmniProcessingInfo(ctx)
    processor = Lfm3OmniMultiModalProcessor(info, Lfm3OmniDummyInputsBuilder(info))
    prompt = (
        "<|vision_start|><|image|><|vision_end|><|vision_start|><|audio|><|vision_end|>"
    )
    if audio_first:
        prompt = "<|vision_start|><|audio|><|vision_end|><|vision_start|><|image|><|vision_end|>"
    items = info.parse_mm_data(
        {"image": Image.new("RGB", (32, 32)), "audio": np.zeros(16000)}
    )
    result = processor.apply(
        ProcessorInputs(prompt=tokenizer.encode(prompt), mm_data_items=items),
        TimingContext(enabled=False),
    )
    assert result["mm_placeholders"]["audio"][0].length == 13
    assert result["mm_placeholders"]["image"][0].length > 0
    ids = result["prompt_token_ids"]
    assert ids.count(4) == ids.count(5) == 2
    assert set(result["mm_kwargs"]) == {"image", "audio"}

    reference = native(
        text=prompt,
        images=[Image.new("RGB", (32, 32))],
        audio=[np.zeros(16000)],
        return_tensors="pt",
    )
    assert ids == reference["input_ids"][0].tolist()
