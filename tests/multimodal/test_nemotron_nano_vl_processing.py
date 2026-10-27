# SPDX-License-Identifier: Apache-2.0
"""VL V2 processor regression coverage, with no checkpoint download."""

import hashlib

import numpy as np
import pytest
import torch
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast
from vllm.config import ModelConfig
from vllm.exceptions import VLLMValidationError
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.processing import ProcessorInputs, TimingContext

from vllm_metal.multimodal.nemotron_nano_vl.config import (
    NemotronNanoVLConfig,
    register_config,
    register_processing,
)


@pytest.fixture
def processor(tmp_path):
    tokenizer_impl = Tokenizer(
        WordLevel(
            {
                "[UNK]": 0,
                "<image>": 1,
                "<img>": 2,
                "</img>": 3,
                "before": 4,
                "between": 5,
                "after": 6,
            },
            unk_token="[UNK]",
        )
    )
    tokenizer_impl.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_impl,
        unk_token="[UNK]",
        additional_special_tokens=["<image>", "<img>", "</img>"],
    )
    tokenizer.save_pretrained(tmp_path)
    config = NemotronNanoVLConfig(
        architectures=["NemotronH_Nano_VL_V2"],
        llm_config={
            "model_type": "nemotron_h",
            "num_hidden_layers": 3,
            "hybrid_override_pattern": "M*-",
            "hidden_size": 32,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "vocab_size": 32,
            "max_position_embeddings": 4096,
        },
        vision_config={"patch_size": 16, "args": {}},
        force_image_size=512,
        patch_size=16,
        downsample_ratio=0.5,
        use_thumbnail=True,
        image_tag_type="internvl",
        img_context_token_id=1,
        norm_mean=[0.48145466, 0.4578275, 0.40821073],
        norm_std=[0.26862954, 0.26130258, 0.27577711],
        dtype="bfloat16",
    )
    config.save_pretrained(tmp_path)
    register_config()
    register_processing()
    model_config = ModelConfig(
        model=str(tmp_path),
        max_model_len=2048,
        mm_processor_kwargs={"max_num_tiles": 2},
    )
    return MULTIMODAL_REGISTRY.create_processor(model_config, tokenizer=tokenizer)


@pytest.mark.parametrize(
    "size,tiles,digest",
    [
        (
            (512, 512),
            1,
            "392ed61fe22d26856cacff2d6ec99f3adf0fd16003d21a105d651b97361a7aa9",
        ),
        (
            (379, 291),
            1,
            "da83c7a4a4a6d0725f93b41ba9b589dbb032e1d3a2284ea64dc719590ac5d948",
        ),
        (
            (1024, 512),
            3,
            "a60d104152905a283bf7086b3c0af30b315ffb0ee8f1098c6f47aee404e28ff5",
        ),
        (
            (512, 1024),
            3,
            "8153c8493e632678d00126fc384fc0c8fb335ba9ae05888c7255cee58f773e53",
        ),
    ],
)
def test_pixels_match_pinned_nvidia_bf16_fixture(processor, size, tiles, digest):
    # Independent NVIDIA image_processing.py at ca9543b126e8bf3176916d3d305ccc415f89fd4d.
    # numpy seed 11, torchvision tensor bicubic, max_num_tiles=2, thumbnail on;
    # BF16 at encoder entry, as in NVIDIA modeling.py. Digests include tile order.
    width, height = size
    image = Image.fromarray(
        np.random.default_rng(11).integers(0, 256, (height, width, 3), dtype=np.uint8)
    )
    pixels = processor.info.get_hf_processor()._images_to_pixel_values_lst([image], 2)[
        0
    ]
    assert pixels.shape == (tiles, 3, 512, 512)
    assert pixels.dtype == torch.bfloat16
    assert (
        hashlib.sha256(pixels.view(torch.uint16).numpy().tobytes()).hexdigest()
        == digest
    )


def test_full_processor_preserves_text_image_order_and_marker_masks(processor):
    tokenizer = processor.info.get_tokenizer()
    prompt = "before <image> between <image> after"
    images = [Image.new("RGB", (379, 291)), Image.new("RGB", (1024, 512))]
    output = processor.apply(
        ProcessorInputs(
            prompt=tokenizer.encode(prompt, add_special_tokens=False),
            mm_data_items=processor.info.parse_mm_data({"image": images}),
        ),
        TimingContext(enabled=False),
    )
    expected = (
        "before <img>"
        + "<image>" * 256
        + "</img> between <img>"
        + "<image>" * 768
        + "</img> after"
    )
    assert output["prompt_token_ids"] == tokenizer.encode(
        expected, add_special_tokens=False
    )
    ranges = output["mm_placeholders"]["image"]
    assert [item.get_num_embeds() for item in ranges] == [256, 768]
    assert [item.length for item in ranges] == [258, 770]
    assert all(not item.is_embed[0] and not item.is_embed[-1] for item in ranges)
    assert processor.info.get_supported_mm_limits() == {"image": None}


@pytest.mark.parametrize("requested", [0, -1, True, 1.5, 3, 13])
def test_request_cannot_exceed_profiled_tile_limit(processor, requested):
    with pytest.raises(VLLMValidationError, match="configured limit 2"):
        processor.apply(
            ProcessorInputs(
                prompt=[1],
                mm_data_items=processor.info.parse_mm_data(
                    {"image": Image.new("RGB", (512, 512))}
                ),
                hf_processor_mm_kwargs={"max_num_tiles": requested},
            ),
            TimingContext(enabled=False),
        )


def test_request_can_reduce_tile_limit(processor):
    output = processor.apply(
        ProcessorInputs(
            prompt=[1],
            mm_data_items=processor.info.parse_mm_data(
                {"image": Image.new("RGB", (1024, 512))}
            ),
            hf_processor_mm_kwargs={"max_num_tiles": 1},
        ),
        TimingContext(enabled=False),
    )
    assert output["mm_placeholders"]["image"][0].get_num_embeds() == 256


@pytest.mark.parametrize("modality", ["audio", "video"])
def test_unsupported_modalities_rejected_during_input_processing(processor, modality):
    data = (
        {"audio": (np.zeros(1600, dtype=np.float32), 16000)}
        if modality == "audio"
        else {"video": np.zeros((1, 32, 32, 3), dtype=np.uint8)}
    )
    with pytest.raises(VLLMValidationError, match=modality) as error:
        processor.info.parse_mm_data(data)
    assert error.value.parameter == "multi_modal_data"


def test_precomputed_image_embeddings_rejected_before_scheduling(processor):
    processor.info.ctx.model_config.get_multimodal_config().enable_mm_embeds = True
    with pytest.raises(VLLMValidationError, match="precomputed image embeddings"):
        processor.apply(
            ProcessorInputs(
                prompt=[1],
                mm_data_items=processor.info.parse_mm_data(
                    {"image": torch.zeros(1, 256, 32)}
                ),
            ),
            TimingContext(enabled=False),
        )
