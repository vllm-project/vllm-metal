# SPDX-License-Identifier: Apache-2.0
"""Speech-to-Text support for vLLM Metal."""

from vllm_metal.stt.config import (
    ISO639_1_SUPPORTED_LANGS,
    SpeechToTextConfig,
    is_stt_model,
    validate_language,
)
from vllm_metal.stt.transcribe import TranscriptionResult, load_model, transcribe

__all__ = [
    "ISO639_1_SUPPORTED_LANGS",
    "SpeechToTextConfig",
    "TranscriptionResult",
    "is_stt_model",
    "load_model",
    "transcribe",
    "validate_language",
]
