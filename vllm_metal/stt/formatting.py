# SPDX-License-Identifier: Apache-2.0
"""Subtitle formatting utilities (SRT / WebVTT)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm_metal.stt.protocol import TranscriptionSegment


def _format_timestamp(seconds: float, ms_sep: str = ".") -> str:
    """Format *seconds* as ``HH:MM:SS<sep>mmm``.

    Args:
        seconds: Time in seconds.
        ms_sep: Millisecond separator — ``,`` for SRT, ``.`` for WebVTT.

    Returns:
        Formatted timestamp string, e.g. ``"01:02:03.456"``.
    """
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    return f"{hrs:02d}:{mins:02d}:{secs:02d}{ms_sep}{millis:03d}"


def format_as_srt(segments: list[TranscriptionSegment]) -> str:
    """Render segments as an SRT subtitle string."""
    lines: list[str] = []
    for seg in segments:
        lines.append(str(seg.id + 1))
        start = _format_timestamp(seg.start, ms_sep=",")
        end = _format_timestamp(seg.end, ms_sep=",")
        lines.append(f"{start} --> {end}")
        lines.append(seg.text.strip())
        lines.append("")
    return "\n".join(lines)


def format_as_vtt(segments: list[TranscriptionSegment]) -> str:
    """Render segments as a WebVTT subtitle string."""
    lines: list[str] = ["WEBVTT", ""]
    for seg in segments:
        start = _format_timestamp(seg.start, ms_sep=".")
        end = _format_timestamp(seg.end, ms_sep=".")
        lines.append(f"{start} --> {end}")
        lines.append(seg.text.strip())
        lines.append("")
    return "\n".join(lines)
