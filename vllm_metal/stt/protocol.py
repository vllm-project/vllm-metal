# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible response types for Speech-to-Text."""

from typing import Literal

from pydantic import BaseModel

# Accepted values for ``response_format`` query parameter.
ResponseFormat = Literal["json", "text", "verbose_json", "srt", "vtt"]


class TranscriptionSegment(BaseModel):
    """Single segment of a transcription with timing information."""

    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: list[int]
    avg_logprob: float = 0.0
    compression_ratio: float = 0.0
    no_speech_prob: float = 0.0


class TranscriptionWord(BaseModel):
    """Word-level timestamp."""

    word: str
    start: float
    end: float


class TranscriptionResponse(BaseModel):
    """Simple JSON transcription response (``response_format="json"``)."""

    text: str


class TranscriptionResponseVerbose(BaseModel):
    """Verbose JSON response with segments, duration, and language."""

    task: str = "transcribe"
    language: str | None = None
    duration: float | None = None
    text: str
    segments: list[TranscriptionSegment] | None = None


class TranscriptionUsage(BaseModel):
    """Duration-based usage statistics for STT requests."""

    prompt_duration_seconds: float
    total_duration_seconds: float
