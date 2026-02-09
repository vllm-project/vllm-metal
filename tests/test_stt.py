# SPDX-License-Identifier: Apache-2.0
"""Tests for Speech-to-Text functionality."""

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

# ---------------------------------------------------------------------------
# Audio pipeline (log_mel_spectrogram, _stft)
# ---------------------------------------------------------------------------


class TestAudioPipeline:
    """Tests for core audio processing functions."""

    def test_log_mel_spectrogram_shape(self) -> None:
        """Log-mel spectrogram should have expected shape."""
        from vllm_metal.stt.audio import N_MELS_DEFAULT, log_mel_spectrogram

        # 1 second of audio
        audio = mx.zeros(SAMPLE_RATE)
        mel = log_mel_spectrogram(audio)
        assert mel.ndim == 2
        assert mel.shape[0] == N_MELS_DEFAULT  # 80 mel bins

    def test_log_mel_spectrogram_values_bounded(self) -> None:
        """Output should be normalized (roughly in [-1, 1] range)."""
        from vllm_metal.stt.audio import log_mel_spectrogram

        audio = mx.array(np.random.randn(SAMPLE_RATE).astype(np.float32))
        mel = log_mel_spectrogram(audio)
        # Whisper's normalization: (log_spec + 4) / 4
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
        # rfft produces N_FFT // 2 + 1 bins
        assert freqs.shape[0] == N_FFT // 2 + 1

    def test_hanning_window_properties(self) -> None:
        """Hanning window should be symmetric and peak in center."""
        from vllm_metal.stt.audio import _hanning

        window = _hanning(400)
        assert window.shape[0] == 400
        # First element is 0 (or very close)
        assert abs(window[0].item()) < 1e-6
        # Peak near center
        assert window[200].item() > window[0].item()


# ---------------------------------------------------------------------------
# SpeechToTextConfig
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# validate_language
# ---------------------------------------------------------------------------


class TestValidateLanguage:
    """Tests for validate_language() — three-tier validation."""

    def test_none_defaults_to_en(self) -> None:
        # Matches upstream Whisper: default to "en" when language is None
        assert validate_language(None) == "en"

    def test_none_with_no_default(self) -> None:
        assert validate_language(None, default=None) is None

    def test_officially_supported(self) -> None:
        # Tier 1: officially supported languages pass silently
        assert validate_language("en") == "en"
        assert validate_language("zh") == "zh"
        assert validate_language("ja") == "ja"

    def test_known_but_unsupported(self) -> None:
        # Tier 2: in Whisper but not officially supported
        code = "yue"  # cantonese — in Whisper but not officially supported
        whisper_langs = get_whisper_languages()
        supported = get_supported_languages()
        assert code in whisper_langs
        assert code not in supported
        assert validate_language(code) == "yue"

    def test_unknown_code_raises(self) -> None:
        # Tier 3: completely unknown code
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
        # All officially supported langs must exist in the Whisper map
        assert get_supported_languages().issubset(set(get_whisper_languages()))


# ---------------------------------------------------------------------------
# Audio chunking
# ---------------------------------------------------------------------------


class TestAudioChunking:
    """Tests for audio_duration() and split_audio()."""

    def test_audio_duration(self) -> None:
        # 1 second of audio at 16 kHz
        audio = mx.zeros(SAMPLE_RATE)
        assert audio_duration(audio) == pytest.approx(1.0)

    def test_audio_duration_half_second(self) -> None:
        audio = mx.zeros(SAMPLE_RATE // 2)
        assert audio_duration(audio) == pytest.approx(0.5)

    def test_split_short_audio_is_noop(self) -> None:
        # 5 seconds — shorter than max_clip_s=30
        audio = mx.zeros(5 * SAMPLE_RATE)
        chunks = split_audio(audio)
        assert len(chunks) == 1
        assert chunks[0][1] == 0.0
        assert chunks[0][0].shape[0] == 5 * SAMPLE_RATE

    def test_split_long_audio(self) -> None:
        # 90 seconds — should produce multiple chunks
        audio = mx.zeros(90 * SAMPLE_RATE)
        chunks = split_audio(audio, max_clip_s=30.0)
        assert len(chunks) >= 3
        # Every chunk starts at a non-negative offset
        for _, start in chunks:
            assert start >= 0.0

    def test_split_covers_all_audio(self) -> None:
        # Verify that all samples are covered (no gaps at the end)
        duration_s = 65
        audio = mx.zeros(duration_s * SAMPLE_RATE)
        chunks = split_audio(audio, max_clip_s=30.0, overlap_s=0.0)
        last_chunk, last_start = chunks[-1]
        last_end_sample = int(last_start * SAMPLE_RATE) + last_chunk.shape[0]
        assert last_end_sample == duration_s * SAMPLE_RATE

    def test_split_with_energy_based_split_point(self) -> None:
        # Create audio with a silent gap near the boundary where the
        # splitter searches so it can find a natural split point.
        sr = SAMPLE_RATE
        loud = mx.array(np.random.randn(28 * sr).astype(np.float32))
        silent = mx.zeros(2 * sr)
        loud2 = mx.array(np.random.randn(20 * sr).astype(np.float32))
        audio = mx.concatenate([loud, silent, loud2])

        chunks = split_audio(audio, max_clip_s=30.0, overlap_s=0.5)
        assert len(chunks) >= 2
        # The split should happen near the silent region (around 28-30s)
        _, second_start = chunks[1]
        assert 26.0 <= second_start <= 31.0


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


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
        # First segment index is 1-based
        assert lines[0] == "1"
        assert "00:00:00,000 --> 00:00:02,500" in lines[1]
        assert "Hello world." in lines[2]
        # Second segment
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


# ---------------------------------------------------------------------------
# Segment extraction
# ---------------------------------------------------------------------------


class TestExtractSegments:
    """Tests for _extract_segments (timestamp token parsing).

    These tests use the actual Whisper tokenizer, so they require
    ``transformers`` but do **not** need a model checkpoint.
    """

    @pytest.fixture(autouse=True)
    def _import_extract(self):
        """Import _extract_segments; skip if tokenizer unavailable."""
        try:
            from vllm_metal.stt.transcribe import _extract_segments, _get_token_id

            self._extract = _extract_segments
            self._tid = _get_token_id
        except Exception:
            pytest.skip("Whisper tokenizer not available")

    def _make_tokens(self, start_ts: str, end_ts: str) -> list[int]:
        """Build a token list: <|start_ts|> text_token <|end_ts|>."""
        start_tok = self._tid(f"<|{start_ts}|>")
        end_tok = self._tid(f"<|{end_ts}|>")
        # Token 2425 = " Hello" in Whisper vocab
        text_tok = 2425
        return [start_tok, text_tok, end_tok]

    def test_paired_timestamps(self) -> None:
        """Two timestamp tokens surrounding text produce one segment."""
        tokens = self._make_tokens("0.00", "2.00")
        segments = self._extract(tokens)
        assert len(segments) == 1
        assert segments[0].start == 0.0
        assert segments[0].end == 2.0

    def test_time_offset(self) -> None:
        """Time offset is added to segment boundaries."""
        tokens = self._make_tokens("0.00", "3.00")
        segments = self._extract(tokens, time_offset=10.0)
        assert len(segments) == 1
        assert segments[0].start == pytest.approx(10.0)
        assert segments[0].end == pytest.approx(13.0)

    def test_empty_tokens(self) -> None:
        """No tokens yields no segments."""
        assert self._extract([]) == []

    def test_segment_id_offset(self) -> None:
        """Segments start numbering from the given offset."""
        tokens = self._make_tokens("0.00", "1.00")
        segments = self._extract(tokens, segment_id_offset=5)
        assert len(segments) == 1
        assert segments[0].id == 5


# ---------------------------------------------------------------------------
# Prompt encoding
# ---------------------------------------------------------------------------


class TestEncodePrompt:
    """Tests for _encode_prompt (initial context injection)."""

    @pytest.fixture(autouse=True)
    def _import_encode(self):
        try:
            from vllm_metal.stt.transcribe import _encode_prompt, _get_token_id

            self._encode = _encode_prompt
            self._tid = _get_token_id
        except Exception:
            pytest.skip("Whisper tokenizer not available")

    def test_none_returns_empty(self) -> None:
        assert self._encode(None) == []

    def test_empty_string_returns_empty(self) -> None:
        assert self._encode("") == []

    def test_prompt_starts_with_startofprev(self) -> None:
        result = self._encode("Kubernetes")
        startofprev = self._tid("<|startofprev|>")
        assert result[0] == startofprev
        assert len(result) > 1

    def test_prompt_contains_text_tokens(self) -> None:
        result = self._encode("hello world")
        # Should have startofprev + at least one text token
        assert len(result) >= 2

    def test_long_prompt_truncated(self) -> None:
        # Very long prompt should be truncated to 224 text tokens + 1 special
        long_text = "word " * 500
        result = self._encode(long_text)
        # startofprev (1) + at most 224 text tokens
        assert len(result) <= 225


# ---------------------------------------------------------------------------
# Integration tests (require model download)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Model runner STT path (deterministic, no model required)
# ---------------------------------------------------------------------------


class TestExecuteSTT:
    """Deterministic tests for _execute_stt and helpers in model_runner."""

    @pytest.fixture(autouse=True)
    def _import_model_runner(self):
        """Import model_runner; skip if vllm not available."""
        try:
            from vllm_metal.v1 import model_runner

            self.mr = model_runner
        except ImportError:
            pytest.skip("vllm not available")

    def test_greedy_decode_empty_prompt_returns_eot(self) -> None:
        """Empty prompt should return EOT immediately without decoding."""
        from unittest.mock import MagicMock

        runner = MagicMock()
        runner.model = MagicMock()

        eot_token = 50257
        audio_features = mx.zeros((1, 1500, 512))

        result = self.mr.MetalModelRunner._greedy_decode_stt(
            runner, audio_features, [], eot_token
        )

        assert result == [eot_token]
        runner.model.decode.assert_not_called()

    def test_extract_audio_features_missing_mm_features(self) -> None:
        """Request without mm_features should return None."""
        from unittest.mock import MagicMock

        runner = MagicMock()
        runner.model = MagicMock()

        new_req = MagicMock()
        del new_req.mm_features

        result = self.mr.MetalModelRunner._extract_audio_features(runner, new_req)
        assert result is None

    def test_extract_audio_features_empty_list(self) -> None:
        """Request with empty mm_features list should return None."""
        from unittest.mock import MagicMock

        runner = MagicMock()
        new_req = MagicMock()
        new_req.mm_features = []

        result = self.mr.MetalModelRunner._extract_audio_features(runner, new_req)
        assert result is None


# ---------------------------------------------------------------------------
# Weight sanitization
# ---------------------------------------------------------------------------


class TestWeightSanitize:
    """Tests for WhisperModel.sanitize() weight mapping."""

    @pytest.fixture
    def model(self):
        """Create a minimal WhisperModel for testing sanitize()."""
        from vllm_metal.stt.whisper import WhisperConfig, WhisperModel

        config = WhisperConfig(
            n_mels=80,
            n_vocab=51865,
            n_audio_ctx=1500,
            n_audio_state=512,
            n_audio_head=8,
            n_audio_layer=6,
            n_text_ctx=448,
            n_text_state=512,
            n_text_head=8,
            n_text_layer=6,
        )
        return WhisperModel(config, dtype=mx.float16)

    def test_sanitize_hf_key_rename(self, model) -> None:
        """HuggingFace keys should be renamed to MLX format."""
        weights = {
            "model.encoder.layers.0.self_attn.q_proj.weight": mx.zeros((512, 512)),
        }
        sanitized = model.sanitize(weights)
        assert "encoder.blocks.0.attn.query.weight" in sanitized
        assert "model.encoder.layers.0.self_attn.q_proj.weight" not in sanitized

    def test_sanitize_skips_encoder_positions(self, model) -> None:
        """encoder.embed_positions should be skipped (None mapping)."""
        weights = {
            "model.encoder.embed_positions.weight": mx.zeros((1500, 512)),
            "model.decoder.embed_tokens.weight": mx.zeros((51865, 512)),
        }
        sanitized = model.sanitize(weights)
        assert "encoder.embed_positions.weight" not in sanitized
        assert "decoder.token_embedding.weight" in sanitized

    def test_sanitize_transposes_conv_weights(self, model) -> None:
        """Conv1d weights should be transposed from HF format."""
        # HF format: (out_channels, in_channels, kernel_size)
        hf_conv = mx.zeros((512, 80, 3))
        weights = {"model.encoder.conv1.weight": hf_conv}
        sanitized = model.sanitize(weights)
        # MLX expects (out_channels, kernel_size, in_channels)
        assert sanitized["encoder.conv1.weight"].shape == (512, 3, 80)

    def test_sanitize_preserves_mlx_format(self, model) -> None:
        """Already-MLX-format weights pass through unchanged."""
        weights = {
            "encoder.blocks.0.attn.query.weight": mx.zeros((512, 512)),
        }
        sanitized = model.sanitize(weights)
        assert "encoder.blocks.0.attn.query.weight" in sanitized

    def test_sanitize_casts_dtype(self, model) -> None:
        """Weights should be cast to model dtype."""
        weights = {"encoder.ln_post.weight": mx.ones((512,), dtype=mx.float32)}
        sanitized = model.sanitize(weights)
        assert sanitized["encoder.ln_post.weight"].dtype == mx.float16


# ---------------------------------------------------------------------------
# Error paths for mm_features
# ---------------------------------------------------------------------------


class TestMalformedMMFeatures:
    """Error-path tests for malformed multimodal features."""

    @pytest.fixture(autouse=True)
    def _import_model_runner(self):
        """Import model_runner; skip if vllm not available."""
        try:
            from vllm_metal.v1 import model_runner

            self.mr = model_runner
        except ImportError:
            pytest.skip("vllm not available")

    def test_mm_features_wrong_type(self) -> None:
        """mm_features with wrong type should be handled gracefully."""
        from unittest.mock import MagicMock

        runner = MagicMock()
        new_req = MagicMock()
        new_req.mm_features = "not_a_list"  # Wrong type

        result = self.mr.MetalModelRunner._extract_audio_features(runner, new_req)
        # Should not crash, returns None for invalid input
        assert result is None

    def test_mm_features_none_value(self) -> None:
        """mm_features=None should return None."""
        from unittest.mock import MagicMock

        runner = MagicMock()
        new_req = MagicMock()
        new_req.mm_features = None

        result = self.mr.MetalModelRunner._extract_audio_features(runner, new_req)
        assert result is None

    def test_mm_features_missing_audio_features_key(self) -> None:
        """mm_features item without 'audio_features' key."""
        from unittest.mock import MagicMock

        runner = MagicMock()
        new_req = MagicMock()
        new_req.mm_features = [{"wrong_key": mx.zeros((1, 10, 512))}]

        result = self.mr.MetalModelRunner._extract_audio_features(runner, new_req)
        assert result is None


@pytest.mark.slow
class TestTranscribeIntegration:
    """End-to-end transcription tests that need a model checkpoint.

    Run with ``pytest -m slow`` to include these.
    """

    def test_transcribe_silence(self) -> None:
        """Transcribing silence should return an empty or near-empty string."""
        from vllm_metal.stt.transcribe import load_model, transcribe

        model = load_model("openai/whisper-tiny")
        audio = mx.zeros(SAMPLE_RATE * 3)  # 3s silence
        result = transcribe(model, audio)
        # Whisper may hallucinate on silence; just check it doesn't crash
        assert isinstance(result.text, str)

    def test_transcribe_with_timestamps(self) -> None:
        """Timestamp mode should populate segments and duration."""
        from vllm_metal.stt.transcribe import load_model, transcribe

        model = load_model("openai/whisper-tiny")
        audio = mx.zeros(SAMPLE_RATE * 3)
        result = transcribe(model, audio, with_timestamps=True)
        assert result.duration == pytest.approx(3.0)
