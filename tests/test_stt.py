# SPDX-License-Identifier: Apache-2.0
"""Tests for STT data types, config, and audio pipeline."""

from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from transformers.models.whisper.tokenization_whisper import LANGUAGES
from vllm.model_executor.models.whisper_utils import ISO639_1_SUPPORTED_LANGS

from vllm_metal.stt import audio as audio_mod
from vllm_metal.stt.audio import (
    HOP_LENGTH,
    N_FFT,
    N_MELS_DEFAULT,
    SAMPLE_RATE,
    _hanning,
    _rms_energy,
    _stft,
    audio_duration,
    log_mel_spectrogram,
    split_audio,
)
from vllm_metal.stt.whisper.transcriber import WhisperTranscriber


class TestValidateLanguage:
    """Tests for Whisper language validation."""

    def test_none_defaults_to_en(self) -> None:
        assert WhisperTranscriber.validate_language(None) == "en"

    def test_none_with_no_default(self) -> None:
        assert WhisperTranscriber.validate_language(None, default=None) is None

    def test_officially_supported(self) -> None:
        assert WhisperTranscriber.validate_language("en") == "en"
        assert WhisperTranscriber.validate_language("zh") == "zh"
        assert WhisperTranscriber.validate_language("ja") == "ja"

    def test_known_but_unsupported(self) -> None:
        code = "yue"  # cantonese — in Whisper but not officially supported
        assert code in LANGUAGES
        assert code not in ISO639_1_SUPPORTED_LANGS
        assert WhisperTranscriber.validate_language(code) == "yue"

    def test_unknown_code_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported language"):
            WhisperTranscriber.validate_language("zz")

    def test_case_insensitive(self) -> None:
        assert WhisperTranscriber.validate_language("EN") == "en"
        assert WhisperTranscriber.validate_language("Zh") == "zh"

    def test_language_name_maps_to_iso_code(self) -> None:
        assert WhisperTranscriber.validate_language("French") == "fr"

    def test_strips_whitespace(self) -> None:
        assert WhisperTranscriber.validate_language("  fr  ") == "fr"

    def test_supported_languages_match_upstream_vllm_subset(self) -> None:
        assert set(ISO639_1_SUPPORTED_LANGS) <= set(LANGUAGES)


class TestAudioPipeline:
    """Tests for core audio processing functions."""

    def test_log_mel_spectrogram_shape(self) -> None:
        """Log-mel spectrogram should have expected shape."""
        audio = mx.zeros(SAMPLE_RATE)  # 1 second
        mel = log_mel_spectrogram(audio)
        assert mel.ndim == 2
        assert mel.shape[0] == N_MELS_DEFAULT

    def test_log_mel_spectrogram_values_bounded(self) -> None:
        """Output should be normalised (roughly in [-1, 1] range)."""
        audio = mx.array(np.random.randn(SAMPLE_RATE).astype(np.float32))
        mel = log_mel_spectrogram(audio)
        assert mel.min().item() >= -2.0
        assert mel.max().item() <= 2.0

    def test_log_mel_spectrogram_accepts_numpy(self) -> None:
        """Should accept numpy arrays."""
        audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
        mel = log_mel_spectrogram(audio)
        assert mel.ndim == 2

    def test_stft_output_shape(self) -> None:
        """STFT should produce expected frequency bins."""
        audio = mx.zeros(SAMPLE_RATE)
        window = _hanning(N_FFT)
        freqs = _stft(audio, window, N_FFT, HOP_LENGTH)
        assert freqs.shape[0] == N_FFT // 2 + 1

    def test_hanning_window_properties(self) -> None:
        """Hanning window should be symmetric and peak in centre."""
        window = _hanning(400)
        assert window.shape[0] == 400
        assert abs(window[0].item()) < 1e-6
        assert window[200].item() > window[0].item()

    def test_rms_energy_partial_last_window_matches_unpadded_mean(self) -> None:
        """Partial trailing windows should use their real sample count."""
        audio = mx.array([1.0, 1.0, 2.0], mx.float32)
        energies = _rms_energy(audio, window_size=2)

        # Window 1: sqrt((1^2 + 1^2) / 2) = 1.0
        # Window 2: sqrt((2^2) / 1) = 2.0 (real sample count, not padded count)
        assert energies.shape[0] == 2
        assert energies[0].item() == pytest.approx(1.0)
        assert energies[1].item() == pytest.approx(2.0)


class TestAudioLoading:
    """Tests for audio loading fallback and ffmpeg timeout behavior."""

    def test_load_audio_falls_back_to_ffmpeg(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        called: dict[str, object] = {}

        def fake_librosa_load(*_args: object, **_kwargs: object) -> None:
            raise ValueError("force ffmpeg fallback")

        def fake_load_audio_ffmpeg(
            file_path: str,
            sample_rate: int,
            timeout_s: float = audio_mod._FFMPEG_TIMEOUT_S,
        ) -> mx.array:
            called["file_path"] = file_path
            called["sample_rate"] = sample_rate
            called["timeout_s"] = timeout_s
            return mx.array(np.zeros(4, dtype=np.float32))

        monkeypatch.setitem(
            sys.modules, "librosa", SimpleNamespace(load=fake_librosa_load)
        )
        monkeypatch.setattr(audio_mod, "_load_audio_ffmpeg", fake_load_audio_ffmpeg)

        audio = audio_mod.load_audio("dummy.wav")

        assert called["file_path"] == "dummy.wav"
        assert called["sample_rate"] == SAMPLE_RATE
        assert called["timeout_s"] == pytest.approx(audio_mod._FFMPEG_TIMEOUT_S)
        assert audio.shape[0] == 4

    def test_load_audio_ffmpeg_timeout_uses_configured_value(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        timeout_s = 12.0
        monkeypatch.setattr("shutil.which", lambda _binary: "/usr/bin/ffmpeg")

        def fake_run(
            cmd: list[str], capture_output: bool, timeout: float
        ) -> subprocess.CompletedProcess[bytes]:
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout)

        monkeypatch.setattr(audio_mod.subprocess, "run", fake_run)

        with pytest.raises(RuntimeError, match="ffmpeg timed out after 12.0s"):
            audio_mod._load_audio_ffmpeg("dummy.wav", SAMPLE_RATE, timeout_s=timeout_s)

    def test_load_audio_ffmpeg_rejects_non_positive_timeout(self) -> None:
        timeout_s = 0.0

        with pytest.raises(ValueError, match="ffmpeg timeout must be > 0"):
            audio_mod._load_audio_ffmpeg("dummy.wav", SAMPLE_RATE, timeout_s=timeout_s)


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

    def test_split_chunks_never_exceed_max_clip(self) -> None:
        """No chunk should be longer than max_clip_s samples."""
        max_clip_s = 10.0
        max_samples = int(max_clip_s * SAMPLE_RATE)
        audio = mx.array(np.random.randn(50 * SAMPLE_RATE).astype(np.float32))
        chunks = split_audio(audio, max_clip_s=max_clip_s, overlap_s=0.0)

        for chunk, _ in chunks:
            assert chunk.shape[0] <= max_samples

    def test_split_small_clip_silent_audio_avoids_tiny_chunks(self) -> None:
        """Small max_clip_s should not degrade into 1-sample chunks."""
        max_clip_s = 0.4
        max_samples = int(max_clip_s * SAMPLE_RATE)
        audio = mx.zeros(2 * SAMPLE_RATE)

        chunks = split_audio(audio, max_clip_s=max_clip_s, overlap_s=0.0)

        assert len(chunks) == 5
        for chunk, _ in chunks:
            assert chunk.shape[0] == max_samples
