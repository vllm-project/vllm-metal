# SPDX-License-Identifier: Apache-2.0
"""Response types for Speech-to-Text."""

from dataclasses import dataclass, field

from pydantic import BaseModel


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


@dataclass
class TranscriptionResult:
    """Result of a transcription operation."""

    text: str
    language: str | None = None
    segments: list[TranscriptionSegment] = field(default_factory=list)
    duration: float = 0.0
