# SPDX-License-Identifier: Apache-2.0
"""Speech-to-Text support for vLLM Metal."""

from vllm_metal.stt.base import TranscriptionResult
from vllm_metal.stt.config import (
    SpeechToTextConfig,
    get_supported_languages,
    get_whisper_languages,
    is_stt_model,
    validate_language,
)
from vllm_metal.stt.formatting import format_as_srt, format_as_vtt
from vllm_metal.stt.protocol import TranscriptionSegment
from vllm_metal.stt.transcribe import (
    Qwen3ASRTranscriber,
    WhisperTranscriber,
    load_model,
    transcribe,
)

__all__ = [
    "Qwen3ASRTranscriber",
    "SpeechToTextConfig",
    "TranscriptionResult",
    "TranscriptionSegment",
    "WhisperTranscriber",
    "format_as_srt",
    "format_as_vtt",
    "get_supported_languages",
    "get_whisper_languages",
    "is_stt_model",
    "load_model",
    "transcribe",
    "validate_language",
]
