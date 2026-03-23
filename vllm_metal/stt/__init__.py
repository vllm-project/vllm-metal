# SPDX-License-Identifier: Apache-2.0
"""Speech-to-Text support for vLLM Metal."""

from vllm_metal.stt.config import (
    SpeechToTextConfig,
)
from vllm_metal.stt.formatting import format_as_srt, format_as_vtt
from vllm_metal.stt.loader import load_model
from vllm_metal.stt.protocol import TranscriptionResult, TranscriptionSegment
from vllm_metal.stt.qwen3_asr.transcriber import Qwen3ASRTranscriber
from vllm_metal.stt.whisper import WhisperTranscriber

__all__ = [
    "Qwen3ASRTranscriber",
    "SpeechToTextConfig",
    "TranscriptionResult",
    "TranscriptionSegment",
    "WhisperTranscriber",
    "format_as_srt",
    "format_as_vtt",
    "load_model",
]
