# SPDX-License-Identifier: Apache-2.0
"""HuggingFace-compatible PretrainedConfig and model stub for Qwen3-ASR.

These config classes allow ``transformers.AutoConfig.from_pretrained()`` and
vLLM's ``_CONFIG_REGISTRY`` to recognise the ``qwen3_asr`` model type.

The stub model class (``Qwen3ASRStub``) implements vLLM's
``SupportsTranscription`` interface so the API server can initialise
without errors.  vllm-metal's STT executor never instantiates this class —
the real inference runs through the MLX ``Qwen3ASRModel``.
"""

from __future__ import annotations

import logging

from transformers.configuration_utils import PretrainedConfig

from vllm_metal.stt.config import get_whisper_languages
from vllm_metal.stt.qwen3_asr.config import Qwen3ASRAudioConfig

logger = logging.getLogger(__name__)

# Map language names (from Qwen3-ASR config) to ISO 639-1 codes.
# Built from the Whisper language table which covers all 30 Qwen3-ASR languages.
_NAME_TO_CODE: dict[str, str] = {
    v.title(): k for k, v in get_whisper_languages().items()
}
# Qwen3-ASR uses "Cantonese" which Whisper maps as "yue"
_NAME_TO_CODE["Cantonese"] = "yue"
# Qwen3-ASR uses "Filipino" which Whisper maps under "tl" (Tagalog)
_NAME_TO_CODE["Filipino"] = "tl"
# Qwen3-ASR uses "Macedonian" which Whisper maps as "mk"
_NAME_TO_CODE["Macedonian"] = "mk"

# ---------------------------------------------------------------------------
# HuggingFace PretrainedConfig subclasses
# ---------------------------------------------------------------------------


class Qwen3ASRAudioEncoderConfig(PretrainedConfig):
    """Audio encoder configuration for Qwen3-ASR."""

    model_type = "qwen3_asr_audio_encoder"

    def __init__(
        self,
        num_mel_bins: int = 128,
        encoder_layers: int = 32,
        encoder_attention_heads: int = 20,
        encoder_ffn_dim: int = 5120,
        d_model: int = 1280,
        max_source_positions: int = 1500,
        n_window: int = 100,
        output_dim: int = 3584,
        downsample_hidden_size: int = 480,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_mel_bins = num_mel_bins
        self.d_model = d_model
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_ffn_dim = encoder_ffn_dim
        self.num_hidden_layers = encoder_layers
        self.max_source_positions = max_source_positions
        self.n_window = n_window
        self.output_dim = output_dim
        self.downsample_hidden_size = downsample_hidden_size


class Qwen3ASRTextConfig(PretrainedConfig):
    """Text decoder configuration for Qwen3-ASR."""

    model_type = "qwen3_asr_text"

    def __init__(
        self,
        vocab_size: int = 151936,
        hidden_size: int = 4096,
        intermediate_size: int = 22016,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 32,
        head_dim: int = 128,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 5000000.0,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = (
            num_key_value_heads
            if num_key_value_heads is not None
            else num_attention_heads
        )
        self.head_dim = head_dim
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


class Qwen3ASRThinkerConfig(PretrainedConfig):
    """Thinker (audio + text) configuration for Qwen3-ASR."""

    model_type = "qwen3_asr_thinker"

    def __init__(
        self,
        audio_config=None,
        text_config=None,
        audio_token_id: int = 151676,
        audio_start_token_id: int = 151669,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if isinstance(audio_config, dict):
            audio_config = Qwen3ASRAudioEncoderConfig(**audio_config)
        elif audio_config is None:
            audio_config = Qwen3ASRAudioEncoderConfig()
        self.audio_config = audio_config

        if isinstance(text_config, dict):
            text_config = Qwen3ASRTextConfig(**text_config)
        elif text_config is None:
            text_config = Qwen3ASRTextConfig()
        self.text_config = text_config
        self.audio_token_id = audio_token_id
        self.audio_start_token_id = audio_start_token_id


class Qwen3ASRHFConfig(PretrainedConfig):
    """Top-level HuggingFace config for Qwen3-ASR models.

    Registered as ``model_type = "qwen3_asr"`` with both
    ``transformers.AutoConfig`` and vLLM's internal config registry.
    """

    model_type = "qwen3_asr"

    def __init__(
        self,
        thinker_config=None,
        support_languages=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if thinker_config is None:
            thinker_config = {}
        if isinstance(thinker_config, dict):
            self.thinker_config = Qwen3ASRThinkerConfig(**thinker_config)
        else:
            self.thinker_config = thinker_config
        self.support_languages = support_languages

    def get_text_config(self, decoder=False):
        """Return the text sub-config (needed by vLLM internals)."""
        return self.thinker_config.text_config


# ---------------------------------------------------------------------------
# Stub model class implementing SupportsTranscription
# ---------------------------------------------------------------------------


def _make_stub_class():
    """Build the stub inside a function to defer heavy imports."""
    import hashlib
    import math
    from collections.abc import Mapping, Sequence
    from typing import Any, Literal

    import numpy as np
    import torch
    import torch.nn as nn
    from transformers import BatchFeature, WhisperFeatureExtractor
    from vllm.config import ModelConfig, VllmConfig
    from vllm.inputs.data import PromptType, TokensPrompt
    from vllm.model_executor.models.interfaces import (
        SpeechToTextConfig,
        SupportsMultiModal,
        SupportsTranscription,
    )
    from vllm.multimodal.inputs import (
        MultiModalBatchedField,
        MultiModalFieldConfig,
        MultiModalFieldElem,
        MultiModalInputs,
        MultiModalKwargsItem,
        MultiModalKwargsItems,
        PlaceholderRange,
    )
    from vllm.multimodal.parse import MultiModalDataItems
    from vllm.multimodal.processing import (
        BaseDummyInputsBuilder,
        BaseMultiModalProcessor,
        BaseProcessingInfo,
        PromptUpdate,
    )
    from vllm.multimodal.registry import _ProcessorFactories
    from vllm.tokenizers import get_tokenizer

    # ------------------------------------------------------------------
    # Minimal multimodal processor so audio reaches the model_runner
    # ------------------------------------------------------------------

    class _Info(BaseProcessingInfo):
        """Minimal processing info for Qwen3-ASR audio."""

        _fe: WhisperFeatureExtractor | None = None

        def get_supported_mm_limits(self) -> Mapping[str, int | None]:
            return {"audio": 1}

        def get_feature_extractor(self) -> WhisperFeatureExtractor:
            if self._fe is None:
                self._fe = WhisperFeatureExtractor.from_pretrained(self.model_id)
            return self._fe

        def get_hf_processor(self, **kwargs: object):  # type: ignore[override]
            return self.get_feature_extractor()

        def get_tokenizer(self):
            return self.ctx.get_tokenizer()

    class _DummyInputs(BaseDummyInputsBuilder[_Info]):
        """Minimal dummy inputs builder for Qwen3-ASR audio."""

        def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
            return "<|audio_pad|>"

        def get_dummy_mm_data(
            self,
            seq_len: int,
            mm_counts: Mapping[str, int],
            mm_options: Mapping[str, Any] | None = None,
        ) -> dict[str, list[np.ndarray]]:
            n = mm_counts.get("audio", 0)
            fe = self.info.get_feature_extractor()
            audio_len = fe.chunk_length * fe.sampling_rate
            return {"audio": [np.zeros(audio_len, dtype=np.float32)] * n}

    class _Processor(BaseMultiModalProcessor[_Info]):
        """Minimal processor: run WhisperFeatureExtractor, emit input_features."""

        # Required abstract methods (unused since we override apply)
        def _get_mm_fields_config(
            self,
            hf_inputs: BatchFeature,
            hf_processor_mm_kwargs: Mapping[str, object],
        ) -> Mapping[str, MultiModalFieldConfig]:
            return {"input_features": MultiModalFieldConfig.batched("audio")}

        def _get_prompt_updates(
            self,
            mm_items: MultiModalDataItems,
            hf_processor_mm_kwargs: Mapping[str, object],
            out_mm_kwargs: MultiModalKwargsItems,
        ) -> Sequence[PromptUpdate]:
            return []

        # Override apply to bypass the complex HF processor pipeline.
        def apply(
            self,
            prompt: str | list[int],
            mm_data: dict[str, Any],
            hf_processor_mm_kwargs: Mapping[str, object],
            tokenization_kwargs: Mapping[str, object] | None = None,
            *,
            mm_uuids: dict[str, list[str]] | None = None,
        ) -> MultiModalInputs:
            # 1. Resolve token IDs
            if isinstance(prompt, list):
                prompt_ids = list(prompt)
            else:
                tokenizer = self.info.get_tokenizer()
                prompt_ids = tokenizer.encode(prompt)  # type: ignore[union-attr]

            # 2. Extract audio and compute mel spectrogram
            audio = mm_data.get("audio")
            if audio is None:
                return MultiModalInputs(
                    type="multimodal",
                    prompt_token_ids=prompt_ids,
                    mm_kwargs=MultiModalKwargsItems({}),
                    mm_hashes={},
                    mm_placeholders={},
                )

            if isinstance(audio, (list, tuple)):
                audio = audio[0]
            # Handle (audio, sr) tuple from some codepaths
            if isinstance(audio, (list, tuple)):
                audio = audio[0]

            fe = self.info.get_feature_extractor()
            features = fe(
                audio,
                sampling_rate=fe.sampling_rate,
                return_tensors="pt",
                padding=False,
            )
            mel = features["input_features"]  # (1, n_mels, time)

            # 3. Build MultiModalKwargsItem with input_features
            elem = MultiModalFieldElem(
                modality="audio",
                key="input_features",
                data=mel,
                field=MultiModalBatchedField(),
            )
            item = MultiModalKwargsItem.from_elems([elem])
            mm_kwargs = MultiModalKwargsItems({"audio": [item]})

            # 4. Placeholders & hashes
            audio_hash = hashlib.md5(np.asarray(audio).tobytes()).hexdigest()
            mm_hashes: dict[str, list[str]] = {"audio": [audio_hash]}
            mm_placeholders: dict[str, list[PlaceholderRange]] = {
                "audio": [PlaceholderRange(offset=0, length=1)]
            }

            return MultiModalInputs(
                type="multimodal",
                prompt_token_ids=prompt_ids,
                mm_kwargs=mm_kwargs,
                mm_hashes=mm_hashes,
                mm_placeholders=mm_placeholders,
            )

    # ------------------------------------------------------------------
    # Stub model class
    # ------------------------------------------------------------------

    class Qwen3ASRStub(nn.Module, SupportsTranscription, SupportsMultiModal):
        """Minimal stub so vLLM's API server can initialise for Qwen3-ASR.

        Never instantiated for inference — vllm-metal's MLX path handles that.

        Implements ``SupportsTranscription`` (for STT API) and
        ``SupportsMultiModal`` (so audio data flows through the pipeline).
        Transcription-only — text generation is not supported.
        """

        # Populated from HF config.support_languages in get_speech_to_text_config.
        supported_languages: Mapping[str, str] = {}
        supports_transcription = True
        supports_transcription_only = True
        supports_segment_timestamp = False
        supports_multimodal = True

        # Set by register_qwen3_asr_config via _ProcessorFactories
        _processor_factory: Any = None

        def __init__(
            self,
            *,
            vllm_config: VllmConfig | None = None,
            prefix: str = "",
        ) -> None:
            super().__init__()

        def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
            raise NotImplementedError("Stub — real inference uses MLX path")

        def forward(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
        ) -> torch.Tensor:
            raise NotImplementedError("Stub — real inference uses MLX path")

        def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
            raise NotImplementedError("Stub — real inference uses MLX path")

        def embed_multimodal(self, **kwargs: object):
            raise NotImplementedError("Stub — real inference uses MLX path")

        @classmethod
        def get_placeholder_str(cls, modality: str, i: int) -> str | None:
            if modality.startswith("audio"):
                return "<|audio_start|><|audio_pad|><|audio_end|>"
            return None

        @classmethod
        def get_speech_to_text_config(
            cls,
            model_config: ModelConfig,
            task_type: Literal["transcribe", "translate"],
        ) -> SpeechToTextConfig:
            # Populate supported_languages from HF config on first call.
            # This runs during OpenAISpeechToText.__init__ (before
            # validate_language), and model_cls is the class not an instance,
            # so __init__ never fires — we must set it here.
            if not cls.supported_languages:
                lang_names = getattr(model_config.hf_config, "support_languages", None)
                if lang_names:
                    cls.supported_languages = _build_supported_languages(lang_names)
            return SpeechToTextConfig(
                sample_rate=16000,
                max_audio_clip_s=30,
                overlap_chunk_second=1,
                min_energy_split_window_size=1600,
            )

        @classmethod
        def get_generation_prompt(
            cls,
            audio: np.ndarray,
            stt_config: SpeechToTextConfig,
            model_config: ModelConfig,
            language: str | None,
            task_type: Literal["transcribe", "translate"],
            request_prompt: str,
            to_language: str | None,
        ) -> PromptType:
            if task_type == "translate":
                raise ValueError(
                    "Qwen3-ASR does not support translation, only transcription"
                )
            if language:
                logger.warning(
                    "Qwen3-ASR ignores language parameter %r; "
                    "model auto-detects language",
                    language,
                )
            if request_prompt:
                logger.warning(
                    "Qwen3-ASR ignores prompt parameter; "
                    "prompt-guided transcription is not supported"
                )
            tokenizer = get_tokenizer(model_config.tokenizer)
            prompt_text = (
                "<|im_start|>user\n"
                "<|audio_start|><|audio_pad|><|audio_end|>\n"
                "<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            prompt_token_ids = tokenizer.encode(prompt_text)
            return TokensPrompt(
                prompt_token_ids=prompt_token_ids,
                multi_modal_data={"audio": audio},
            )

        @classmethod
        def get_num_audio_tokens(
            cls,
            audio_duration_s: float,
            stt_config: SpeechToTextConfig,
            model_config: ModelConfig,
        ) -> int | None:
            # Derive hop_length from WhisperFeatureExtractor defaults
            hop_length = WhisperFeatureExtractor().hop_length

            # Derive n_window from audio encoder config
            n_window = model_config.hf_config.thinker_config.audio_config.n_window

            mel_frames = math.ceil(
                audio_duration_s * stt_config.sample_rate / hop_length
            )
            return Qwen3ASRAudioConfig(n_window=n_window).feat_extract_output_length(
                mel_frames
            )

    # Attach multimodal processor factory to the stub class
    Qwen3ASRStub._processor_factory = _ProcessorFactories(
        info=_Info,
        dummy_inputs=_DummyInputs,
        processor=_Processor,
    )

    return Qwen3ASRStub


def _build_supported_languages(lang_names: list[str] | None) -> dict[str, str]:
    """Convert Qwen3-ASR ``support_languages`` (name list) to ``{code: name}``."""
    if not lang_names:
        return {}
    result: dict[str, str] = {}
    for name in lang_names:
        code = _NAME_TO_CODE.get(name)
        if code:
            result[code] = name.lower()
        else:
            logger.debug("Unknown language in Qwen3-ASR config: %s", name)
    return result


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register_qwen3_asr_config() -> None:
    """Register Qwen3-ASR config with transformers AutoConfig and vLLM.

    Registers three things so that ``vllm serve`` recognises ``qwen3_asr``:
    1. ``transformers.AutoConfig`` — so ``from_pretrained()`` works.
    2. vLLM's ``_CONFIG_REGISTRY`` — so the config is loaded before the
       ``AutoConfig`` fallback.
    3. vLLM's ``ModelRegistry`` — so the architecture
       ``Qwen3ASRForConditionalGeneration`` passes validation and
       the API server can access ``SupportsTranscription`` methods.

    Safe to call multiple times — registration is idempotent.
    """
    from transformers import AutoConfig

    # 1. Register with transformers AutoConfig
    try:
        AutoConfig.register("qwen3_asr", Qwen3ASRHFConfig)
    except ValueError:
        logger.debug("AutoConfig qwen3_asr already registered")

    # 2. Register with vLLM's internal _CONFIG_REGISTRY if available
    try:
        from vllm.transformers_utils.config import _CONFIG_REGISTRY

        if "qwen3_asr" not in _CONFIG_REGISTRY:
            _CONFIG_REGISTRY["qwen3_asr"] = Qwen3ASRHFConfig
    except (ImportError, AttributeError) as exc:
        logger.debug("_CONFIG_REGISTRY registration skipped: %s", exc)

    # 3. Register architecture with vLLM's ModelRegistry
    # Use our stub class that implements SupportsTranscription.
    try:
        from vllm.model_executor.models.registry import ModelRegistry

        arch = "Qwen3ASRForConditionalGeneration"
        if arch not in ModelRegistry.get_supported_archs():
            stub_cls = _make_stub_class()
            ModelRegistry.register_model(arch, stub_cls)
    except (ImportError, AttributeError) as exc:
        logger.debug("ModelRegistry registration skipped: %s", exc)
