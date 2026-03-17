# SPDX-License-Identifier: Apache-2.0
"""Speech-to-Text model constructor registry."""

from __future__ import annotations

from collections.abc import Callable

import mlx.core as mx

from vllm_metal.stt.qwen3_asr.config import Qwen3ASRConfig
from vllm_metal.stt.qwen3_asr.model import Qwen3ASRModel
from vllm_metal.stt.whisper.config import WhisperConfig
from vllm_metal.stt.whisper.model import WhisperModel

STTModel = WhisperModel | Qwen3ASRModel
STTModelConstructor = Callable[[dict, mx.Dtype], STTModel]


def get_stt_model_constructor(model_type: str) -> STTModelConstructor:
    """Return the model constructor for an STT ``model_type``."""
    model_type = model_type.lower()
    try:
        return _STT_MODEL_CONSTRUCTORS[model_type]
    except KeyError:
        raise ValueError(
            f"Unsupported STT model_type: {model_type!r}. "
            "Expected 'whisper' or 'qwen3_asr'."
        ) from None


def _construct_whisper_model(config_dict: dict, dtype: mx.Dtype) -> WhisperModel:
    config = WhisperConfig.from_dict(config_dict)
    return WhisperModel(config, dtype)


def _construct_qwen3_asr_model(config_dict: dict, dtype: mx.Dtype) -> Qwen3ASRModel:
    config = Qwen3ASRConfig.from_dict(config_dict)
    return Qwen3ASRModel(config, dtype)


_STT_MODEL_CONSTRUCTORS: dict[str, STTModelConstructor] = {
    # Default to Whisper for backward compatibility.
    "": _construct_whisper_model,
    "whisper": _construct_whisper_model,
    "qwen3_asr": _construct_qwen3_asr_model,
}
