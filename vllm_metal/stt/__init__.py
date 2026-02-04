# SPDX-License-Identifier: Apache-2.0
"""Speech-to-Text support for vLLM Metal."""

from vllm_metal.stt.api import init_stt_model
from vllm_metal.stt.api import router as stt_router
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
    "init_stt_model",
    "is_stt_model",
    "load_model",
    "stt_router",
    "transcribe",
    "validate_language",
]
