# SPDX-License-Identifier: Apache-2.0
"""Tests for Whisper transcription orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import mlx.core as mx
import pytest
from transformers import WhisperTokenizer
from vllm.config import SpeechToTextConfig

from vllm_metal.stt.audio import SAMPLE_RATE
from vllm_metal.stt.loader import load_model
from vllm_metal.stt.whisper import WhisperConfig, WhisperModel, WhisperTranscriber
from vllm_metal.stt.whisper.transcriber import (
    DEFAULT_SEGMENT_DURATION,
    MAX_PROMPT_TOKENS,
)


class _ChunkingTestTranscriber(WhisperTranscriber):
    def __init__(self, config: SpeechToTextConfig) -> None:
        tokenizer = cast(
            WhisperTokenizer, SimpleNamespace(decode=lambda *_a, **_k: "ok")
        )
        super().__init__(
            model=cast(WhisperModel, SimpleNamespace(is_multilingual=True)),
            model_path=None,
            config=config,
            tokenizer=tokenizer,
        )

    def _encode_chunk(self, audio: mx.array) -> mx.array:
        del audio
        return mx.zeros((1, 1, 1), dtype=mx.float32)

    def _greedy_decode(
        self,
        audio_features: mx.array,
        language: str | None = None,
        task: str = "transcribe",
        prompt: str | None = None,
        with_timestamps: bool = False,
        max_tokens: int | None = None,
    ) -> list[int]:
        del audio_features, language, task, prompt, with_timestamps, max_tokens
        return [1]


def _make_tiny_whisper_model(*, n_audio_ctx: int) -> WhisperModel:
    config = WhisperConfig(
        n_mels=80,
        n_audio_ctx=n_audio_ctx,
        n_audio_state=64,
        n_audio_head=2,
        n_audio_layer=1,
        n_vocab=100,
        n_text_ctx=32,
        n_text_state=64,
        n_text_head=2,
        n_text_layer=1,
    )
    return WhisperModel(config, dtype=mx.float32)


@pytest.fixture(scope="session")
def whisper_tokenizer() -> WhisperTokenizer:
    return WhisperTokenizer.from_pretrained("openai/whisper-small")


@pytest.fixture()
def transcriber(whisper_tokenizer: WhisperTokenizer) -> WhisperTranscriber:
    return WhisperTranscriber(
        model=cast(WhisperModel, SimpleNamespace(is_multilingual=True)),
        tokenizer=whisper_tokenizer,
    )


class TestExtractSegments:
    def _make_tokens(
        self, transcriber: WhisperTranscriber, start_ts: str, end_ts: str
    ) -> list[int]:
        start_tok = transcriber._get_token_id(f"<|{start_ts}|>")
        end_tok = transcriber._get_token_id(f"<|{end_ts}|>")
        return [start_tok, 2425, end_tok]

    def test_paired_timestamps(self, transcriber: WhisperTranscriber) -> None:
        tokens = self._make_tokens(transcriber, "0.00", "2.00")
        segments = transcriber._extract_segments(tokens)

        assert len(segments) == 1
        assert segments[0].start == 0.0
        assert segments[0].end == 2.0

    def test_time_offset(self, transcriber: WhisperTranscriber) -> None:
        tokens = self._make_tokens(transcriber, "0.00", "3.00")
        segments = transcriber._extract_segments(tokens, time_offset=10.0)

        assert len(segments) == 1
        assert segments[0].start == pytest.approx(10.0)
        assert segments[0].end == pytest.approx(13.0)

    def test_empty_tokens(self, transcriber: WhisperTranscriber) -> None:
        assert transcriber._extract_segments([]) == []

    def test_segment_id_offset(self, transcriber: WhisperTranscriber) -> None:
        tokens = self._make_tokens(transcriber, "0.00", "1.00")
        segments = transcriber._extract_segments(tokens, segment_id_offset=5)

        assert len(segments) == 1
        assert segments[0].id == 5


class TestEncodePrompt:
    def test_none_returns_empty(self, transcriber: WhisperTranscriber) -> None:
        assert transcriber._encode_prompt(None) == []

    def test_empty_string_returns_empty(self, transcriber: WhisperTranscriber) -> None:
        assert transcriber._encode_prompt("") == []

    def test_prompt_starts_with_startofprev(
        self, transcriber: WhisperTranscriber
    ) -> None:
        result = transcriber._encode_prompt("Kubernetes")

        assert result[0] == transcriber._get_token_id("<|startofprev|>")
        assert len(result) > 1

    def test_long_prompt_truncated(self, transcriber: WhisperTranscriber) -> None:
        result = transcriber._encode_prompt("word " * 500)

        assert result[0] == transcriber._get_token_id("<|startofprev|>")
        assert len(result) <= MAX_PROMPT_TOKENS + 1


class TestLoadModel:
    def test_missing_config_json(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="config.json not found"):
            load_model(tmp_path)

    def test_missing_weight_files(self, tmp_path: Path) -> None:
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
        with pytest.raises(ValueError, match="model_path"):
            load_model("   ")

    def test_invalid_dtype_raises(self, tmp_path: Path) -> None:
        with pytest.raises(TypeError, match="Unsupported STT model dtype"):
            load_model(tmp_path, dtype=mx.int32)

    def test_unknown_model_type_raises(self, tmp_path: Path) -> None:
        (tmp_path / "config.json").write_text(json.dumps({"model_type": "mystery_stt"}))

        with pytest.raises(ValueError, match="Unsupported STT model_type"):
            load_model(tmp_path)


class TestResolveDecodeOptions:
    def test_multilingual_model_normalizes_inputs(self) -> None:
        transcriber = WhisperTranscriber(
            model=cast(WhisperModel, SimpleNamespace(is_multilingual=True)),
            model_path=None,
        )

        language, task = transcriber._resolve_decode_options(" EN ", "Transcribe")

        assert language == "en"
        assert task == "transcribe"

    def test_invalid_task_raises(self) -> None:
        transcriber = WhisperTranscriber(
            model=cast(WhisperModel, SimpleNamespace(is_multilingual=True)),
            model_path=None,
        )

        with pytest.raises(ValueError, match="Unsupported STT task"):
            transcriber._resolve_decode_options("en", "summarize")

    def test_english_only_model_rejects_translation(self) -> None:
        transcriber = WhisperTranscriber(
            model=cast(WhisperModel, SimpleNamespace(is_multilingual=False)),
            model_path=None,
        )

        with pytest.raises(ValueError, match="do not support translation"):
            transcriber._resolve_decode_options(None, "translate")

    def test_english_only_model_rejects_non_english_language(self) -> None:
        transcriber = WhisperTranscriber(
            model=cast(WhisperModel, SimpleNamespace(is_multilingual=False)),
            model_path=None,
        )

        with pytest.raises(ValueError, match="only support English transcription"):
            transcriber._resolve_decode_options("fr", "transcribe")


class TestChunkingPolicy:
    @pytest.mark.parametrize(
        "config",
        [
            SpeechToTextConfig(max_audio_clip_s=None),
            SpeechToTextConfig(min_energy_split_window_size=None),
        ],
    )
    def test_transcribe_allows_short_audio_when_chunking_disabled(
        self,
        config: SpeechToTextConfig,
    ) -> None:
        transcriber = _ChunkingTestTranscriber(config)

        result = transcriber.transcribe(mx.zeros(1600, dtype=mx.float32))

        assert result.text == "ok"
        assert result.duration > 0

    @pytest.mark.parametrize(
        "config",
        [
            SpeechToTextConfig(max_audio_clip_s=None),
            SpeechToTextConfig(min_energy_split_window_size=None),
        ],
    )
    def test_transcribe_raises_when_chunking_disabled_for_long_audio(
        self,
        config: SpeechToTextConfig,
    ) -> None:
        transcriber = _ChunkingTestTranscriber(config)
        long_audio = mx.zeros(
            int((DEFAULT_SEGMENT_DURATION + 1) * SAMPLE_RATE),
            dtype=mx.float32,
        )

        with pytest.raises(ValueError, match="Audio chunking is disabled"):
            transcriber.transcribe(long_audio)

    def test_transcribe_raises_when_max_clip_exceeds_whisper_window_for_long_audio(
        self,
    ) -> None:
        config = SpeechToTextConfig(
            max_audio_clip_s=int(DEFAULT_SEGMENT_DURATION) + 1,
            min_energy_split_window_size=1600,
        )
        transcriber = _ChunkingTestTranscriber(config)
        long_audio = mx.zeros(
            int((DEFAULT_SEGMENT_DURATION + 1) * SAMPLE_RATE),
            dtype=mx.float32,
        )

        with pytest.raises(ValueError, match="max_audio_clip_s="):
            transcriber.transcribe(long_audio)

    def test_transcribe_raises_when_max_clip_exceeds_whisper_window_for_short_audio(
        self,
    ) -> None:
        config = SpeechToTextConfig(
            max_audio_clip_s=int(DEFAULT_SEGMENT_DURATION) + 1,
            min_energy_split_window_size=1600,
        )
        transcriber = _ChunkingTestTranscriber(config)
        short_audio = mx.zeros(
            int((DEFAULT_SEGMENT_DURATION - 1) * SAMPLE_RATE),
            dtype=mx.float32,
        )

        with pytest.raises(ValueError, match="max_audio_clip_s="):
            transcriber.transcribe(short_audio)


class TestGreedyDecode:
    @pytest.fixture()
    def tiny_transcriber(
        self, whisper_tokenizer: WhisperTokenizer
    ) -> WhisperTranscriber:
        transcriber = WhisperTranscriber(model=_make_tiny_whisper_model(n_audio_ctx=10))
        transcriber.tokenizer = whisper_tokenizer
        return transcriber

    def test_greedy_decode_returns_list(
        self, tiny_transcriber: WhisperTranscriber
    ) -> None:
        mel = mx.random.normal((1, 20, 80))
        audio_features = tiny_transcriber.model.encode(mel)
        mx.eval(audio_features)

        tokens = tiny_transcriber._greedy_decode(
            audio_features, language="en", max_tokens=5
        )

        assert isinstance(tokens, list)
        assert all(isinstance(token, int) for token in tokens)

    def test_greedy_decode_with_timestamps(
        self, tiny_transcriber: WhisperTranscriber
    ) -> None:
        mel = mx.random.normal((1, 20, 80))
        audio_features = tiny_transcriber.model.encode(mel)
        mx.eval(audio_features)

        tokens = tiny_transcriber._greedy_decode(
            audio_features, language="en", with_timestamps=True, max_tokens=5
        )

        assert isinstance(tokens, list)


class TestEncodeChunk:
    def test_encode_chunk_output_shape(self) -> None:
        transcriber = WhisperTranscriber(
            model=_make_tiny_whisper_model(n_audio_ctx=1500)
        )

        features = transcriber._encode_chunk(mx.zeros(SAMPLE_RATE))

        assert features.ndim == 3
        assert features.shape == (1, 1500, 64)


@pytest.mark.slow
class TestTranscribeIntegration:
    def test_transcribe_silence(self) -> None:
        model = load_model("openai/whisper-tiny")
        audio = mx.zeros(SAMPLE_RATE * 3)
        result = WhisperTranscriber(model).transcribe(audio)

        assert isinstance(result.text, str)

    def test_transcribe_with_timestamps(self) -> None:
        model = load_model("openai/whisper-tiny")
        audio = mx.zeros(SAMPLE_RATE * 3)
        result = WhisperTranscriber(model).transcribe(audio, with_timestamps=True)

        assert result.duration == pytest.approx(3.0)
