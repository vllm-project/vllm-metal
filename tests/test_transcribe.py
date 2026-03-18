# SPDX-License-Identifier: Apache-2.0
"""Tests for transcription orchestration: segment extraction, prompt encoding."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest

from vllm_metal.stt.loader import load_model
from vllm_metal.stt.protocol import TranscriptionResult
from vllm_metal.stt.whisper import WhisperTranscriber
from vllm_metal.stt.whisper.transcriber import MAX_PROMPT_TOKENS

# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture()
def transcriber():
    """Create a WhisperTranscriber with tokenizer only (no model needed).

    Skips if transformers is not available.
    """
    try:
        from transformers import WhisperTokenizer
    except ImportError:
        pytest.skip("transformers not available")

    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small")
    t = WhisperTranscriber(model=None, tokenizer=tokenizer)  # type: ignore[arg-type]
    return t


# ===========================================================================
# TranscriptionResult
# ===========================================================================


class TestTranscriptionResult:
    """Tests for TranscriptionResult dataclass."""

    def test_defaults(self) -> None:
        """Default fields should be None / empty / zero."""
        result = TranscriptionResult(text="hello")
        assert result.text == "hello"
        assert result.language is None
        assert result.segments == []
        assert result.duration == 0.0

    def test_with_all_fields(self) -> None:
        """All fields should be stored correctly."""
        result = TranscriptionResult(
            text="hello", language="en", segments=[], duration=5.0
        )
        assert result.language == "en"
        assert result.duration == 5.0


# ===========================================================================
# Segment extraction
# ===========================================================================


class TestExtractSegments:
    """Tests for WhisperTranscriber._extract_segments.

    These tests use the actual Whisper tokenizer, so they require
    ``transformers`` but do **not** need a model checkpoint.
    """

    def _make_tokens(
        self, transcriber: WhisperTranscriber, start_ts: str, end_ts: str
    ) -> list[int]:
        """Build a token list: ``<|start_ts|> text_token <|end_ts|>``.

        Args:
            transcriber: Transcriber with a loaded tokenizer.
            start_ts: Start timestamp string (e.g. ``"0.00"``).
            end_ts: End timestamp string (e.g. ``"2.00"``).

        Returns:
            List of three token IDs.
        """
        start_tok = transcriber._get_token_id(f"<|{start_ts}|>")
        end_tok = transcriber._get_token_id(f"<|{end_ts}|>")
        # Token 2425 = " Hello" in Whisper vocab
        text_tok = 2425
        return [start_tok, text_tok, end_tok]

    def test_paired_timestamps(self, transcriber: WhisperTranscriber) -> None:
        """Two timestamp tokens surrounding text produce one segment."""
        tokens = self._make_tokens(transcriber, "0.00", "2.00")
        segments = transcriber._extract_segments(tokens)
        assert len(segments) == 1
        assert segments[0].start == 0.0
        assert segments[0].end == 2.0

    def test_time_offset(self, transcriber: WhisperTranscriber) -> None:
        """Time offset is added to segment boundaries."""
        tokens = self._make_tokens(transcriber, "0.00", "3.00")
        segments = transcriber._extract_segments(tokens, time_offset=10.0)
        assert len(segments) == 1
        assert segments[0].start == pytest.approx(10.0)
        assert segments[0].end == pytest.approx(13.0)

    def test_empty_tokens(self, transcriber: WhisperTranscriber) -> None:
        """No tokens yields no segments."""
        assert transcriber._extract_segments([]) == []

    def test_segment_id_offset(self, transcriber: WhisperTranscriber) -> None:
        """Segments start numbering from the given offset."""
        tokens = self._make_tokens(transcriber, "0.00", "1.00")
        segments = transcriber._extract_segments(tokens, segment_id_offset=5)
        assert len(segments) == 1
        assert segments[0].id == 5


# ===========================================================================
# Prompt encoding
# ===========================================================================


class TestEncodePrompt:
    """Tests for WhisperTranscriber._encode_prompt."""

    def test_none_returns_empty(self, transcriber: WhisperTranscriber) -> None:
        """None prompt should return an empty list."""
        assert transcriber._encode_prompt(None) == []

    def test_empty_string_returns_empty(self, transcriber: WhisperTranscriber) -> None:
        """Empty string should return an empty list."""
        assert transcriber._encode_prompt("") == []

    def test_prompt_starts_with_startofprev(
        self, transcriber: WhisperTranscriber
    ) -> None:
        """Encoded prompt should begin with ``<|startofprev|>``."""
        result = transcriber._encode_prompt("Kubernetes")
        startofprev = transcriber._get_token_id("<|startofprev|>")
        assert result[0] == startofprev
        assert len(result) > 1

    def test_prompt_contains_text_tokens(self, transcriber: WhisperTranscriber) -> None:
        """Prompt should have at least startofprev + one text token."""
        result = transcriber._encode_prompt("hello world")
        assert len(result) >= 2

    def test_long_prompt_truncated(self, transcriber: WhisperTranscriber) -> None:
        """Very long prompt should be truncated to MAX_PROMPT_TOKENS + 1."""
        long_text = "word " * 500
        result = transcriber._encode_prompt(long_text)
        # startofprev (1) + at most MAX_PROMPT_TOKENS text tokens
        assert len(result) <= MAX_PROMPT_TOKENS + 1


# ===========================================================================
# Model loading (error paths)
# ===========================================================================


class TestLoadModel:
    """Tests for load_model() error paths using temporary directories."""

    def test_missing_config_json(self, tmp_path: Path) -> None:
        """Should raise FileNotFoundError when config.json is absent."""
        with pytest.raises(FileNotFoundError, match="config.json not found"):
            load_model(tmp_path)

    def test_missing_weight_files(self, tmp_path: Path) -> None:
        """Should raise FileNotFoundError when no weight files exist."""
        config = {
            "n_mels": 80,
            "n_audio_ctx": 10,
            "n_audio_state": 64,
            "n_audio_head": 2,
            "n_audio_layer": 1,
            "n_vocab": 100,
            "n_text_ctx": 32,
            "n_text_state": 64,
            "n_text_head": 2,
            "n_text_layer": 1,
        }
        (tmp_path / "config.json").write_text(json.dumps(config))

        with pytest.raises(FileNotFoundError, match="No weight files"):
            load_model(tmp_path)

    def test_empty_model_path_raises(self) -> None:
        """Whitespace-only model paths should fail fast."""
        with pytest.raises(ValueError, match="model_path"):
            load_model("   ")

    def test_invalid_dtype_raises(self, tmp_path: Path) -> None:
        """Non-floating dtypes are rejected before any file I/O."""
        with pytest.raises(TypeError, match="Unsupported STT model dtype"):
            load_model(tmp_path, dtype=mx.int32)

    def test_unknown_model_type_raises(self, tmp_path: Path) -> None:
        """Unknown model_type values should not fall through to Whisper."""
        (tmp_path / "config.json").write_text(json.dumps({"model_type": "mystery_stt"}))

        with pytest.raises(ValueError, match="Unsupported STT model_type"):
            load_model(tmp_path)


class TestResolveDecodeOptions:
    """Tests for WhisperTranscriber task/language validation."""

    def test_multilingual_model_normalizes_inputs(self) -> None:
        transcriber = WhisperTranscriber(
            model=SimpleNamespace(is_multilingual=True),
            model_path=None,
        )

        language, task = transcriber._resolve_decode_options(" EN ", "Transcribe")

        assert language == "en"
        assert task == "transcribe"

    def test_invalid_task_raises(self) -> None:
        transcriber = WhisperTranscriber(
            model=SimpleNamespace(is_multilingual=True),
            model_path=None,
        )

        with pytest.raises(ValueError, match="Unsupported STT task"):
            transcriber._resolve_decode_options("en", "summarize")

    def test_english_only_model_rejects_translation(self) -> None:
        transcriber = WhisperTranscriber(
            model=SimpleNamespace(is_multilingual=False),
            model_path=None,
        )

        with pytest.raises(ValueError, match="do not support translation"):
            transcriber._resolve_decode_options(None, "translate")

    def test_english_only_model_rejects_non_english_language(self) -> None:
        transcriber = WhisperTranscriber(
            model=SimpleNamespace(is_multilingual=False),
            model_path=None,
        )

        with pytest.raises(ValueError, match="only support English transcription"):
            transcriber._resolve_decode_options("fr", "transcribe")


# ===========================================================================
# Greedy decode and encode chunk (require tiny model)
# ===========================================================================


class TestGreedyDecode:
    """Tests for WhisperTranscriber._greedy_decode with a tiny model."""

    @pytest.fixture()
    def tiny_transcriber(self):
        """Create a WhisperTranscriber with a tiny model and tokenizer."""
        try:
            from transformers import WhisperTokenizer
        except ImportError:
            pytest.skip("transformers not available")

        import mlx.core as mx

        from vllm_metal.stt.whisper import WhisperConfig, WhisperModel

        config = WhisperConfig(
            n_mels=80,
            n_audio_ctx=10,
            n_audio_state=64,
            n_audio_head=2,
            n_audio_layer=1,
            n_vocab=100,
            n_text_ctx=32,
            n_text_state=64,
            n_text_head=2,
            n_text_layer=1,
        )
        model = WhisperModel(config, dtype=mx.float32)
        t = WhisperTranscriber(model=model, model_path=None)
        t._tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small")
        return t

    def test_greedy_decode_returns_list(self, tiny_transcriber) -> None:
        """Greedy decode should return a list of integer token IDs."""
        import mlx.core as mx

        mel = mx.random.normal((1, 20, 80))
        audio_features = tiny_transcriber.model.encode(mel)
        mx.eval(audio_features)

        tokens = tiny_transcriber._greedy_decode(
            audio_features, language="en", max_tokens=5
        )
        assert isinstance(tokens, list)
        assert all(isinstance(t, int) for t in tokens)

    def test_greedy_decode_with_timestamps(self, tiny_transcriber) -> None:
        """Timestamp mode should not include <|notimestamps|> in prefix."""
        import mlx.core as mx

        mel = mx.random.normal((1, 20, 80))
        audio_features = tiny_transcriber.model.encode(mel)
        mx.eval(audio_features)

        tokens = tiny_transcriber._greedy_decode(
            audio_features, language="en", with_timestamps=True, max_tokens=5
        )
        assert isinstance(tokens, list)


class TestEncodeChunk:
    """Tests for WhisperTranscriber._encode_chunk."""

    def test_encode_chunk_output_shape(self) -> None:
        """Encoded features should have shape (1, n_audio_ctx, n_audio_state).

        ``_encode_chunk`` calls ``pad_or_trim(mel, N_FRAMES=3000)`` then
        conv2 (stride=2), so the output is always 1500 frames regardless
        of input length.  ``n_audio_ctx`` must match (1500).
        """
        import mlx.core as mx

        from vllm_metal.stt.whisper import WhisperConfig, WhisperModel

        config = WhisperConfig(
            n_mels=80,
            n_audio_ctx=1500,
            n_audio_state=64,
            n_audio_head=2,
            n_audio_layer=1,
            n_vocab=100,
            n_text_ctx=32,
            n_text_state=64,
            n_text_head=2,
            n_text_layer=1,
        )
        model = WhisperModel(config, dtype=mx.float32)
        t = WhisperTranscriber(model=model, model_path=None)

        audio_chunk = mx.zeros(16000)  # 1 second
        features = t._encode_chunk(audio_chunk)

        assert features.ndim == 3
        assert features.shape == (1, 1500, 64)


# ===========================================================================
# Integration tests (require model download)
# ===========================================================================


@pytest.mark.slow
class TestTranscribeIntegration:
    """End-to-end transcription tests that need a model checkpoint.

    Run with ``pytest -m slow`` to include these.
    """

    def test_transcribe_silence(self) -> None:
        """Transcribing silence should return an empty or near-empty string."""
        import mlx.core as mx

        from vllm_metal.stt.audio import SAMPLE_RATE
        from vllm_metal.stt.loader import load_model
        from vllm_metal.stt.whisper import WhisperTranscriber

        model = load_model("openai/whisper-tiny")
        audio = mx.zeros(SAMPLE_RATE * 3)  # 3s silence
        result = WhisperTranscriber(model).transcribe(audio)
        # Whisper may hallucinate on silence; just check it doesn't crash
        assert isinstance(result.text, str)

    def test_transcribe_with_timestamps(self) -> None:
        """Timestamp mode should populate segments and duration."""
        import mlx.core as mx

        from vllm_metal.stt.audio import SAMPLE_RATE
        from vllm_metal.stt.loader import load_model
        from vllm_metal.stt.whisper import WhisperTranscriber

        model = load_model("openai/whisper-tiny")
        audio = mx.zeros(SAMPLE_RATE * 3)
        result = WhisperTranscriber(model).transcribe(audio, with_timestamps=True)
        assert result.duration == pytest.approx(3.0)
