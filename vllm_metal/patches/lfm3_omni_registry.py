# SPDX-License-Identifier: Apache-2.0
"""LFM3-Omni image/audio processing and scheduler metadata for MLX."""

from mlx_vlm.models.lfm3_omni.processing_lfm3_omni import Lfm3OmniProcessor
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.qwen2_vl import (
    Qwen2VLDummyInputsBuilder,
    Qwen2VLMultiModalProcessor,
)
from vllm.multimodal import MULTIMODAL_REGISTRY

from vllm_metal.patches.lfm3_audio_registry import (
    Lfm3AudioDummyInputsBuilder,
    Lfm3AudioMultiModalProcessor,
    Lfm3AudioProcessingInfo,
)
from vllm_metal.patches.lfm3_registry import Lfm3ForCausalLM, Lfm3VlProcessingInfo


class Lfm3OmniProcessingInfo(Lfm3AudioProcessingInfo, Lfm3VlProcessingInfo):
    def get_hf_processor(self, **kwargs):
        return self.ctx.get_hf_processor(Lfm3OmniProcessor, **kwargs)

    def get_supported_mm_limits(self):
        return {"image": None, "audio": None}

    def get_mm_max_tokens_per_item(self, seq_len, mm_counts):
        return {
            **Lfm3VlProcessingInfo.get_mm_max_tokens_per_item(self, seq_len, mm_counts),
            **Lfm3AudioProcessingInfo.get_mm_max_tokens_per_item(
                self, seq_len, mm_counts
            ),
        }


class Lfm3OmniDummyInputsBuilder(Qwen2VLDummyInputsBuilder):
    def get_dummy_text(self, mm_counts):
        return "<|vision_start|><|image|><|vision_end|>" * mm_counts.get(
            "image", 0
        ) + "<|vision_start|><|audio|><|vision_end|>" * mm_counts.get("audio", 0)

    def get_dummy_mm_data(self, seq_len, mm_counts, mm_options):
        images = super().get_dummy_mm_data(seq_len, mm_counts, mm_options)
        images.pop("video", None)
        return {
            **images,
            **Lfm3AudioDummyInputsBuilder.get_dummy_mm_data(
                self, seq_len, mm_counts, mm_options
            ),
        }


class Lfm3OmniMultiModalProcessor(Qwen2VLMultiModalProcessor):
    def _get_hf_mm_text(self, mm_counts):
        return self.dummy_inputs.get_dummy_text(mm_counts)

    def _get_mm_fields_config(self, hf_inputs, hf_processor_mm_kwargs):
        return {
            **super()._get_mm_fields_config(hf_inputs, hf_processor_mm_kwargs),
            **Lfm3AudioMultiModalProcessor._get_mm_fields_config(
                self, hf_inputs, hf_processor_mm_kwargs
            ),
        }

    def _get_prompt_updates(self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs):
        return [
            *super()._get_prompt_updates(
                mm_items, hf_processor_mm_kwargs, out_mm_kwargs
            ),
            *Lfm3AudioMultiModalProcessor._get_prompt_updates(
                self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs
            ),
        ]


@MULTIMODAL_REGISTRY.register_processor(
    Lfm3OmniMultiModalProcessor,
    info=Lfm3OmniProcessingInfo,
    dummy_inputs=Lfm3OmniDummyInputsBuilder,
)
class Lfm3OmniForConditionalGeneration(Lfm3ForCausalLM, SupportsMultiModal):
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str:
        if modality in ("image", "audio"):
            return f"<|vision_start|><|{modality}|><|vision_end|>"
        raise ValueError("LFM3-Omni supports image and audio input only")
