# SPDX-License-Identifier: Apache-2.0
"""LFM3-Audio input processing and scheduler metadata for Metal MLX."""

from collections.abc import Mapping

from mlx_vlm.models.lfm3_audio.audio import SAMPLE_RATE, num_encoder_tokens, num_frames
from mlx_vlm.models.lfm3_audio.processing_lfm3_audio import Lfm3AudioProcessor
from vllm.exceptions import VLLMValidationError
from vllm.inputs import MultiModalDataDict
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig
from vllm.multimodal.parse import (
    AudioProcessorItems,
    DictEmbeddingItems,
    EmbeddingItems,
    MultiModalDataItems,
    MultiModalDataParser,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
)

from vllm_metal.patches.lfm3_registry import Lfm3ForCausalLM

MAX_AUDIO_SAMPLES = 30 * SAMPLE_RATE


class Lfm3AudioProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config()

    def get_hf_processor(self, **kwargs: object) -> Lfm3AudioProcessor:
        # The standalone mlx-vlm processor only needs the shared tokenizer.
        return Lfm3AudioProcessor(self.get_tokenizer())

    def get_data_parser(self) -> MultiModalDataParser:
        return MultiModalDataParser(target_sr=SAMPLE_RATE, target_channels=1)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": None}

    def get_mm_max_tokens_per_item(
        self, seq_len: int, mm_counts: Mapping[str, int]
    ) -> Mapping[str, int]:
        return {"audio": num_encoder_tokens(num_frames(MAX_AUDIO_SAMPLES))}

    def parse_mm_data(
        self, mm_data: MultiModalDataDict, *, validate: bool = True
    ) -> MultiModalDataItems:
        items = super().parse_mm_data(mm_data, validate=validate)
        audio = items.get("audio")
        if validate and isinstance(audio, (EmbeddingItems, DictEmbeddingItems)):
            raise VLLMValidationError(
                "LFM3-Audio requires waveforms; precomputed embeddings are unsupported.",
                parameter="audio",
            )
        if validate and isinstance(audio, AudioProcessorItems):
            for i in range(len(audio)):
                length = audio.get_audio_length(i)
                if not 0 < length <= MAX_AUDIO_SAMPLES:
                    raise VLLMValidationError(
                        "LFM3-Audio requires nonempty clips of at most 30 seconds.",
                        parameter="audio",
                    )
        return items


class Lfm3AudioDummyInputsBuilder(BaseDummyInputsBuilder[Lfm3AudioProcessingInfo]):
    def get_dummy_text(self, mm_counts):
        return "<|audio|>" * mm_counts.get("audio", 0)

    def get_dummy_mm_data(self, seq_len, mm_counts, mm_options):
        return {
            "audio": self._get_dummy_audios(
                length=MAX_AUDIO_SAMPLES,
                num_audios=mm_counts.get("audio", 0),
                overrides=mm_options.get("audio"),
            )
        }


class Lfm3AudioMultiModalProcessor(BaseMultiModalProcessor[Lfm3AudioProcessingInfo]):
    def _get_hf_mm_text(self, mm_counts):
        return self.dummy_inputs.get_dummy_text(mm_counts)

    def _get_mm_fields_config(self, hf_inputs, hf_processor_mm_kwargs):
        return {
            "input_features": MultiModalFieldConfig.batched("audio"),
            "feature_attention_mask": MultiModalFieldConfig.batched("audio"),
        }

    def _get_prompt_updates(self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs):
        token_id = self.info.get_tokenizer().convert_tokens_to_ids("<|audio|>")

        def replacement(index):
            mask = out_mm_kwargs["audio"][index]["feature_attention_mask"].data
            return [token_id] * num_encoder_tokens(int(mask.sum()))

        return [
            PromptReplacement(
                modality="audio", target=[token_id], replacement=replacement
            )
        ]


@MULTIMODAL_REGISTRY.register_processor(
    Lfm3AudioMultiModalProcessor,
    info=Lfm3AudioProcessingInfo,
    dummy_inputs=Lfm3AudioDummyInputsBuilder,
)
class Lfm3AudioForConditionalGeneration(Lfm3ForCausalLM, SupportsMultiModal):
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str:
        if modality == "audio":
            return "<|audio|>"
        raise ValueError("LFM3-Audio supports audio input only")
