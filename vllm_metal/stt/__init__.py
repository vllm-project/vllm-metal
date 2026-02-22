# SPDX-License-Identifier: Apache-2.0
"""Speech-to-Text support for vLLM Metal."""

from vllm_metal.stt.config import (
    SpeechToTextConfig,
    get_supported_languages,
    get_whisper_languages,
    is_stt_model,
    validate_language,
)
from vllm_metal.stt.formatting import format_as_srt, format_as_vtt
from vllm_metal.stt.protocol import TranscriptionSegment

__all__ = [
    "SpeechToTextConfig",
    "TranscriptionSegment",
    "format_as_srt",
    "format_as_vtt",
    "get_supported_languages",
    "get_whisper_languages",
    "is_stt_model",
    "validate_language",
]
