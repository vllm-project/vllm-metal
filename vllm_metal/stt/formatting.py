# SPDX-License-Identifier: Apache-2.0
"""Subtitle formatting utilities (SRT / WebVTT)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm_metal.stt.protocol import TranscriptionSegment


def _format_timestamp_srt(seconds: float) -> str:
    """Format *seconds* as ``HH:MM:SS,mmm`` for SRT."""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{millis:03d}"


def _format_timestamp_vtt(seconds: float) -> str:
    """Format *seconds* as ``HH:MM:SS.mmm`` for WebVTT."""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    return f"{hrs:02d}:{mins:02d}:{secs:02d}.{millis:03d}"


def format_as_srt(segments: list[TranscriptionSegment]) -> str:
    """Render segments as an SRT subtitle string."""
    lines: list[str] = []
    for seg in segments:
        lines.append(str(seg.id + 1))
        lines.append(
            f"{_format_timestamp_srt(seg.start)} --> {_format_timestamp_srt(seg.end)}"
        )
        lines.append(seg.text.strip())
        lines.append("")
    return "\n".join(lines)


def format_as_vtt(segments: list[TranscriptionSegment]) -> str:
    """Render segments as a WebVTT subtitle string."""
    lines: list[str] = ["WEBVTT", ""]
    for seg in segments:
        lines.append(
            f"{_format_timestamp_vtt(seg.start)} --> {_format_timestamp_vtt(seg.end)}"
        )
        lines.append(seg.text.strip())
        lines.append("")
    return "\n".join(lines)
