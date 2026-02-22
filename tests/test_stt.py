# SPDX-License-Identifier: Apache-2.0
"""Tests for STT data types, config, and formatting."""

from __future__ import annotations

import pytest

from vllm_metal.stt.config import (
    SpeechToTextConfig,
    get_supported_languages,
    get_whisper_languages,
    validate_language,
)
from vllm_metal.stt.formatting import format_as_srt, format_as_vtt
from vllm_metal.stt.protocol import TranscriptionSegment

# ===========================================================================
# SpeechToTextConfig
# ===========================================================================


class TestSpeechToTextConfig:
    """Tests for SpeechToTextConfig dataclass."""

    def test_default_values(self) -> None:
        cfg = SpeechToTextConfig()
        assert cfg.max_audio_clip_s == 30.0
        assert cfg.overlap_chunk_second == 1.0
        assert cfg.min_energy_split_window_size == 1600
        assert cfg.sample_rate == 16000  # deprecated but still accepted

    def test_custom_values(self) -> None:
        cfg = SpeechToTextConfig(
            max_audio_clip_s=15.0,
            overlap_chunk_second=0.5,
            min_energy_split_window_size=800,
        )
        assert cfg.max_audio_clip_s == 15.0
        assert cfg.overlap_chunk_second == 0.5
        assert cfg.min_energy_split_window_size == 800


# ===========================================================================
# validate_language
# ===========================================================================


class TestValidateLanguage:
    """Tests for validate_language() — three-tier validation."""

    def test_none_defaults_to_en(self) -> None:
        assert validate_language(None) == "en"

    def test_none_with_no_default(self) -> None:
        assert validate_language(None, default=None) is None

    def test_officially_supported(self) -> None:
        assert validate_language("en") == "en"
        assert validate_language("zh") == "zh"
        assert validate_language("ja") == "ja"

    def test_known_but_unsupported(self) -> None:
        code = "yue"  # cantonese — in Whisper but not officially supported
        whisper_langs = get_whisper_languages()
        supported = get_supported_languages()
        assert code in whisper_langs
        assert code not in supported
        assert validate_language(code) == "yue"

    def test_unknown_code_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported language"):
            validate_language("zz")

    def test_case_insensitive(self) -> None:
        assert validate_language("EN") == "en"
        assert validate_language("Zh") == "zh"

    def test_strips_whitespace(self) -> None:
        assert validate_language("  fr  ") == "fr"

    def test_whisper_languages_count(self) -> None:
        assert len(get_whisper_languages()) == 100

    def test_supported_is_subset_of_whisper(self) -> None:
        assert get_supported_languages().issubset(set(get_whisper_languages()))


# ===========================================================================
# TranscriptionSegment
# ===========================================================================


class TestTranscriptionSegment:
    """Tests for TranscriptionSegment pydantic model."""

    def test_create_segment(self) -> None:
        seg = TranscriptionSegment(
            id=0, seek=0, start=0.0, end=2.5, text=" Hello.", tokens=[1, 2]
        )
        assert seg.id == 0
        assert seg.start == 0.0
        assert seg.end == 2.5
        assert seg.text == " Hello."
        assert seg.tokens == [1, 2]

    def test_default_values(self) -> None:
        seg = TranscriptionSegment(
            id=0, seek=0, start=0.0, end=1.0, text="hi", tokens=[]
        )
        assert seg.avg_logprob == 0.0
        assert seg.compression_ratio == 0.0
        assert seg.no_speech_prob == 0.0


# ===========================================================================
# Formatting (SRT / VTT)
# ===========================================================================


class TestFormatting:
    """Tests for SRT and VTT subtitle formatting."""

    @pytest.fixture()
    def sample_segments(self) -> list[TranscriptionSegment]:
        return [
            TranscriptionSegment(
                id=0,
                seek=0,
                start=0.0,
                end=2.5,
                text=" Hello world.",
                tokens=[1, 2, 3],
            ),
            TranscriptionSegment(
                id=1,
                seek=250,
                start=2.5,
                end=5.0,
                text=" How are you?",
                tokens=[4, 5, 6],
            ),
        ]

    def test_srt_format(self, sample_segments: list[TranscriptionSegment]) -> None:
        srt = format_as_srt(sample_segments)
        lines = srt.split("\n")
        assert lines[0] == "1"
        assert "00:00:00,000 --> 00:00:02,500" in lines[1]
        assert "Hello world." in lines[2]
        assert lines[4] == "2"
        assert "00:00:02,500 --> 00:00:05,000" in lines[5]

    def test_vtt_format(self, sample_segments: list[TranscriptionSegment]) -> None:
        vtt = format_as_vtt(sample_segments)
        lines = vtt.split("\n")
        assert lines[0] == "WEBVTT"
        assert lines[1] == ""
        assert "00:00:00.000 --> 00:00:02.500" in lines[2]
        assert "Hello world." in lines[3]

    def test_srt_empty_segments(self) -> None:
        assert format_as_srt([]) == ""

    def test_vtt_empty_segments(self) -> None:
        vtt = format_as_vtt([])
        assert vtt.startswith("WEBVTT")

    def test_srt_long_timestamps(self) -> None:
        seg = TranscriptionSegment(
            id=0,
            seek=0,
            start=3661.123,
            end=3665.456,
            text=" One hour in.",
            tokens=[1],
        )
        srt = format_as_srt([seg])
        assert "01:01:01,123" in srt
        assert "01:01:05,456" in srt
