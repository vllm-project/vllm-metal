# SPDX-License-Identifier: Apache-2.0
"""Speech-to-Text support for vLLM Metal."""

from vllm_metal.stt.config import (
    SpeechToTextConfig,
    get_supported_languages,
    get_whisper_languages,
    is_stt_model,
    validate_language,
)
from vllm_metal.stt.transcribe import (
    TranscriptionResult,
    WhisperTranscriber,
    load_model,
    transcribe,
)

__all__ = [
    "SpeechToTextConfig",
    "TranscriptionResult",
    "WhisperTranscriber",
    "get_supported_languages",
    "get_whisper_languages",
    "is_stt_model",
    "load_model",
    "transcribe",
    "validate_language",
]
