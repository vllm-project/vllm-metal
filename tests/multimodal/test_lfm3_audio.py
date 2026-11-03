# SPDX-License-Identifier: Apache-2.0
"""Audio input contracts, exercised without checkpoint weights."""

from types import SimpleNamespace
from unittest.mock import Mock

import mlx.core as mx
import numpy as np
import pytest
import torch
from vllm.config import MultiModalConfig
from vllm.exceptions import VLLMValidationError
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItem

from vllm_metal.multimodal import MultiModalFeatureSpec, PlaceholderRange
from vllm_metal.multimodal.lfm3_audio import Lfm3AudioMultimodalAdapter
from vllm_metal.patches.lfm3_audio_registry import Lfm3AudioProcessingInfo


@pytest.fixture
def info():
    ctx = Mock()
    ctx.get_mm_config.return_value = MultiModalConfig(enable_mm_embeds=True)
    return Lfm3AudioProcessingInfo(ctx)


@pytest.mark.parametrize("samples", [0, 480001])
def test_audio_duration_is_validated_before_encoding(info, samples):
    with pytest.raises(VLLMValidationError, match="nonempty clips"):
        info.parse_mm_data({"audio": (np.zeros(samples), 16000)})


def test_precomputed_audio_is_rejected(info):
    with pytest.raises(VLLMValidationError, match="requires waveforms"):
        info.parse_mm_data({"audio": torch.zeros((1, 4, 8))})


def test_cached_audio_reference_does_not_require_waveform_length(info):
    items = info.parse_mm_data({"audio": [None, np.zeros(16000)]})
    assert items["audio"].get(0) is None
    assert items["audio"].get_audio_length(1) == 16000


def test_encoder_receives_unpadded_lengths_and_preserves_rows():
    field = MultiModalFieldConfig.batched("audio")
    data = MultiModalKwargsItem(
        {
            "input_features": field.build_elems(
                "input_features", torch.zeros((1, 128, 16))
            )[0],
            "feature_attention_mask": field.build_elems(
                "feature_attention_mask", torch.tensor([[1] * 9 + [0] * 7])
            )[0],
        }
    )
    model = SimpleNamespace(
        language_model=Mock(_placeholder_ids=[4, 5]),
        config=SimpleNamespace(audio_start_token_id=2, mixed_start_token_id=3),
        audio_tower=Mock(return_value=mx.ones((2, 8))),
        audio_projector=lambda x: x,
    )
    adapter = Lfm3AudioMultimodalAdapter.from_loaded_model(model)
    feature = MultiModalFeatureSpec(
        data=data,
        modality="audio",
        identifier="clip",
        mm_position=PlaceholderRange(offset=0, length=2),
    )
    result = adapter.encode_multimodal([feature])[0]
    assert model.audio_tower.call_args.args[1].tolist() == [9]
    assert result.hidden_states.shape == (2, 8)
    assert not adapter.text_path_selective_logits_ok


def test_text_generation_masks_speech_tokens():
    from mlx_vlm.models.lfm3.config import ModelConfig
    from mlx_vlm.models.lfm3_audio.lfm3_audio import LanguageModel

    language_model = LanguageModel(
        ModelConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            vocab_size=32,
            layer_types=["full_attention"],
        ),
        placeholder_ids=[4, 5],
    )
    model = SimpleNamespace(
        language_model=language_model,
        config=SimpleNamespace(audio_start_token_id=2, mixed_start_token_id=3),
    )
    adapter = Lfm3AudioMultimodalAdapter.from_loaded_model(model)
    ids = mx.array([[1]])
    embeds = adapter.embed_tokens(ids)
    output = adapter.call_lm(ids, embeds, [None], mx.array([0]))
    assert mx.isneginf(output.logits[..., [2, 3, 4, 5]]).all().item()
    assert mx.isfinite(output.logits[..., [0, 1, 6]]).all().item()
    text_output = adapter.text_model()(ids)
    assert mx.isneginf(text_output.logits[..., [2, 3, 4, 5]]).all().item()


def test_waveforms_expand_to_separate_unpadded_placeholder_ranges():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast
    from vllm.multimodal.processing import InputProcessingContext
    from vllm.multimodal.processing.context import TimingContext
    from vllm.multimodal.processing.inputs import ProcessorInputs

    from vllm_metal.patches.lfm3_audio_registry import (
        Lfm3AudioDummyInputsBuilder,
        Lfm3AudioMultiModalProcessor,
    )

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0, "<|audio|>": 1})),
        unk_token="[UNK]",
        pad_token="[UNK]",
        additional_special_tokens=["<|audio|>"],
    )
    mm_config = MultiModalConfig()
    config = SimpleNamespace(
        model="test-audio",
        dtype=torch.float32,
        multimodal_config=mm_config,
        get_multimodal_config=lambda: mm_config,
    )
    info = Lfm3AudioProcessingInfo(InputProcessingContext(config, tokenizer))
    processor = Lfm3AudioMultiModalProcessor(info, Lfm3AudioDummyInputsBuilder(info))
    items = info.parse_mm_data({"audio": [np.zeros(16000), np.zeros(8000)]})
    result = processor.apply(
        ProcessorInputs(prompt=[1, 0, 1], mm_data_items=items),
        TimingContext(enabled=False),
    )
    assert [p.length for p in result["mm_placeholders"]["audio"]] == [13, 7]
    assert result["prompt_token_ids"] == [1] * 13 + [0] + [1] * 7
    features = result["mm_kwargs"]["audio"]
    assert [int(item["feature_attention_mask"].data.sum()) for item in features] == [
        99,
        49,
    ]
