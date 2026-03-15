# SPDX-License-Identifier: Apache-2.0
"""Shared Speech-to-Text result types."""

from __future__ import annotations

from dataclasses import dataclass, field

from vllm_metal.stt.protocol import TranscriptionSegment


@dataclass
class TranscriptionResult:
    """Result of a transcription operation.

    Attributes:
        text: Full transcribed text.
        language: Language code used for transcription.
        segments: Timestamped segments (populated only with ``with_timestamps``).
        duration: Total audio duration in seconds.
    """

    text: str
    language: str | None = None
    segments: list[TranscriptionSegment] = field(default_factory=list)
    duration: float = 0.0
