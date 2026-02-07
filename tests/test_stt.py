# SPDX-License-Identifier: Apache-2.0
"""Tests for Speech-to-Text functionality."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from vllm_metal.stt.audio import SAMPLE_RATE, audio_duration, split_audio
from vllm_metal.stt.config import (
    ISO639_1_SUPPORTED_LANGS,
    WHISPER_LANGUAGES,
    SpeechToTextConfig,
    validate_language,
)
from vllm_metal.stt.formatting import format_as_srt, format_as_vtt
from vllm_metal.stt.protocol import TranscriptionSegment

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
        # Tier 1: languages in ISO639_1_SUPPORTED_LANGS pass silently
        assert validate_language("en") == "en"
        assert validate_language("zh") == "zh"
        assert validate_language("ja") == "ja"

    def test_known_but_unsupported_warns(self) -> None:
        # Tier 2: in WHISPER_LANGUAGES but not in ISO639_1_SUPPORTED_LANGS
        # Should pass but emit a warning
        code = "yue"  # cantonese — in Whisper but not officially supported
        assert code in WHISPER_LANGUAGES
        assert code not in ISO639_1_SUPPORTED_LANGS
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
        assert len(WHISPER_LANGUAGES) == 100

    def test_supported_is_subset_of_whisper(self) -> None:
        # All officially supported langs must exist in the Whisper map
        assert set(ISO639_1_SUPPORTED_LANGS).issubset(set(WHISPER_LANGUAGES))


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

    def test_greedy_decode_empty_prompt_returns_eot(self) -> None:
        """Empty prompt should return EOT immediately without decoding."""
        from unittest.mock import MagicMock

        from vllm_metal.v1 import model_runner as mr

        runner = MagicMock()
        runner.model = MagicMock()

        eot_token = 50257
        audio_features = mx.zeros((1, 1500, 512))

        # Call the helper directly
        result = mr.MetalModelRunner._greedy_decode_stt(
            runner, audio_features, [], eot_token
        )

        assert result == [eot_token]
        # Model.decode should NOT be called for empty prompt
        runner.model.decode.assert_not_called()

    def test_extract_audio_features_missing_mm_features(self) -> None:
        """Request without mm_features should return None."""
        from unittest.mock import MagicMock

        from vllm_metal.v1 import model_runner as mr

        runner = MagicMock()
        runner.model = MagicMock()

        new_req = MagicMock()
        del new_req.mm_features  # Simulate missing attribute

        result = mr.MetalModelRunner._extract_audio_features(runner, new_req)
        assert result is None

    def test_extract_audio_features_empty_list(self) -> None:
        """Request with empty mm_features list should return None."""
        from unittest.mock import MagicMock

        from vllm_metal.v1 import model_runner as mr

        runner = MagicMock()
        new_req = MagicMock()
        new_req.mm_features = []

        result = mr.MetalModelRunner._extract_audio_features(runner, new_req)
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
