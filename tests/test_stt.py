# SPDX-License-Identifier: Apache-2.0
"""Tests for STT data types, config, formatting, and audio pipeline."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from vllm_metal.stt.audio import SAMPLE_RATE, audio_duration, split_audio
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


# ===========================================================================
# Audio pipeline (log_mel_spectrogram, _stft)
# ===========================================================================


class TestAudioPipeline:
    """Tests for core audio processing functions."""

    def test_log_mel_spectrogram_shape(self) -> None:
        """Log-mel spectrogram should have expected shape."""
        from vllm_metal.stt.audio import N_MELS_DEFAULT, log_mel_spectrogram

        audio = mx.zeros(SAMPLE_RATE)  # 1 second
        mel = log_mel_spectrogram(audio)
        assert mel.ndim == 2
        assert mel.shape[0] == N_MELS_DEFAULT

    def test_log_mel_spectrogram_values_bounded(self) -> None:
        """Output should be normalised (roughly in [-1, 1] range)."""
        from vllm_metal.stt.audio import log_mel_spectrogram

        audio = mx.array(np.random.randn(SAMPLE_RATE).astype(np.float32))
        mel = log_mel_spectrogram(audio)
        assert mel.min().item() >= -2.0
        assert mel.max().item() <= 2.0

    def test_log_mel_spectrogram_accepts_numpy(self) -> None:
        """Should accept numpy arrays."""
        from vllm_metal.stt.audio import log_mel_spectrogram

        audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
        mel = log_mel_spectrogram(audio)
        assert mel.ndim == 2

    def test_stft_output_shape(self) -> None:
        """STFT should produce expected frequency bins."""
        from vllm_metal.stt.audio import HOP_LENGTH, N_FFT, _hanning, _stft

        audio = mx.zeros(SAMPLE_RATE)
        window = _hanning(N_FFT)
        freqs = _stft(audio, window, N_FFT, HOP_LENGTH)
        assert freqs.shape[0] == N_FFT // 2 + 1

    def test_hanning_window_properties(self) -> None:
        """Hanning window should be symmetric and peak in centre."""
        from vllm_metal.stt.audio import _hanning

        window = _hanning(400)
        assert window.shape[0] == 400
        assert abs(window[0].item()) < 1e-6
        assert window[200].item() > window[0].item()


# ===========================================================================
# Audio chunking
# ===========================================================================


class TestAudioChunking:
    """Tests for audio_duration() and split_audio()."""

    def test_audio_duration(self) -> None:
        audio = mx.zeros(SAMPLE_RATE)
        assert audio_duration(audio) == pytest.approx(1.0)

    def test_audio_duration_half_second(self) -> None:
        audio = mx.zeros(SAMPLE_RATE // 2)
        assert audio_duration(audio) == pytest.approx(0.5)

    def test_split_short_audio_is_noop(self) -> None:
        """Audio shorter than max_clip_s is returned as-is."""
        audio = mx.zeros(5 * SAMPLE_RATE)
        chunks = split_audio(audio)
        assert len(chunks) == 1
        assert chunks[0][1] == 0.0
        assert chunks[0][0].shape[0] == 5 * SAMPLE_RATE

    def test_split_long_audio(self) -> None:
        """90 seconds should produce multiple chunks."""
        audio = mx.zeros(90 * SAMPLE_RATE)
        chunks = split_audio(audio, max_clip_s=30.0)
        assert len(chunks) >= 3
        for _, start in chunks:
            assert start >= 0.0

    def test_split_covers_all_audio(self) -> None:
        """All samples should be covered (no gaps at the end)."""
        duration_s = 65
        audio = mx.zeros(duration_s * SAMPLE_RATE)
        chunks = split_audio(audio, max_clip_s=30.0, overlap_s=0.0)
        last_chunk, last_start = chunks[-1]
        last_end_sample = int(last_start * SAMPLE_RATE) + last_chunk.shape[0]
        assert last_end_sample == duration_s * SAMPLE_RATE

    def test_split_with_energy_based_split_point(self) -> None:
        """Splitter should find quiet regions near chunk boundaries."""
        sr = SAMPLE_RATE
        loud = mx.array(np.random.randn(28 * sr).astype(np.float32))
        silent = mx.zeros(2 * sr)
        loud2 = mx.array(np.random.randn(20 * sr).astype(np.float32))
        audio = mx.concatenate([loud, silent, loud2])

        chunks = split_audio(audio, max_clip_s=30.0, overlap_s=0.5)
        assert len(chunks) >= 2
        _, second_start = chunks[1]
        assert 26.0 <= second_start <= 31.0
