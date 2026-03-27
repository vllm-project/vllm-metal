# SPDX-License-Identifier: Apache-2.0
"""Whisper transcription policy and decode loop."""

from __future__ import annotations

import logging
import re
from typing import cast

import mlx.core as mx
import numpy as np
from transformers import WhisperTokenizer
from vllm.config import SpeechToTextConfig

from vllm_metal.stt.audio import (
    N_FRAMES,
    SAMPLE_RATE,
    audio_duration,
    load_audio,
    log_mel_spectrogram,
    pad_or_trim,
    split_audio,
)
from vllm_metal.stt.config import validate_language
from vllm_metal.stt.protocol import TranscriptionResult, TranscriptionSegment

from .config import WHISPER_MAX_DECODE_TOKENS
from .model import WhisperModel

logger = logging.getLogger(__name__)

SEEK_MULTIPLIER = 100
DEFAULT_SEGMENT_DURATION = 30.0
MAX_PROMPT_TOKENS = 224

TIMESTAMP_RE = re.compile(r"<\|(\d+\.\d+)\|>")
WHISPER_TASKS = frozenset({"transcribe", "translate"})


class WhisperTranscriber:
    def __init__(
        self,
        model: WhisperModel,
        model_path: str | None = None,
        config: SpeechToTextConfig | None = None,
        tokenizer: WhisperTokenizer | None = None,
    ) -> None:
        self.model = model
        self.config = config or SpeechToTextConfig()
        self._model_path = model_path
        self._tokenizer = tokenizer

    @staticmethod
    def load_tokenizer(model_path: str | None) -> WhisperTokenizer:
        if model_path:
            try:
                return WhisperTokenizer.from_pretrained(model_path)
            except (OSError, ValueError) as e:
                logger.debug("Local tokenizer load failed for %s: %s", model_path, e)

        try:
            return WhisperTokenizer.from_pretrained("openai/whisper-small")
        except OSError:
            return WhisperTokenizer.from_pretrained(
                "openai/whisper-small", local_files_only=True
            )

    @property
    def tokenizer(self) -> WhisperTokenizer:
        if self._tokenizer is None:
            self._tokenizer = self.load_tokenizer(self._model_path)
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, tokenizer: WhisperTokenizer | None) -> None:
        self._tokenizer = tokenizer

    def transcribe(
        self,
        audio: str | np.ndarray | mx.array,
        language: str | None = None,
        task: str = "transcribe",
        prompt: str | None = None,
        with_timestamps: bool = False,
    ) -> TranscriptionResult:
        language, task = self._resolve_decode_options(language, task)

        if isinstance(audio, str):
            audio = load_audio(audio, sample_rate=SAMPLE_RATE)
        elif isinstance(audio, np.ndarray):
            audio = mx.array(audio, mx.float32)

        total_duration = audio_duration(audio, SAMPLE_RATE)

        chunks = self._prepare_audio_chunks(audio)

        all_segments: list[TranscriptionSegment] = []
        all_text_parts: list[str] = []
        seg_id_offset = 0

        for chunk_audio, chunk_start in chunks:
            features = self._encode_chunk(chunk_audio)
            output_tokens = self._greedy_decode(
                features, language, task, prompt, with_timestamps=with_timestamps
            )

            if with_timestamps:
                segments = self._extract_segments(
                    output_tokens, chunk_start, seg_id_offset
                )
                for seg in segments:
                    all_segments.append(seg)
                    all_text_parts.append(seg.text)
                seg_id_offset += len(segments)
                if not segments:
                    text = self.tokenizer.decode(
                        output_tokens, skip_special_tokens=True
                    )
                    if text.strip():
                        all_text_parts.append(text.strip())
            else:
                text = self.tokenizer.decode(output_tokens, skip_special_tokens=True)
                if text.strip():
                    all_text_parts.append(text.strip())

        return TranscriptionResult(
            text=" ".join(all_text_parts).strip(),
            language=language,
            segments=all_segments if with_timestamps else [],
            duration=total_duration,
        )

    def _prepare_audio_chunks(self, audio: mx.array) -> list[tuple[mx.array, float]]:
        """Return one chunk or split chunks based on STT chunking policy."""
        max_clip_s = self.config.max_audio_clip_s
        window_size = self.config.min_energy_split_window_size
        if max_clip_s is None or window_size is None:
            return [(audio, 0.0)]

        return split_audio(
            audio,
            max_clip_s=max_clip_s,
            overlap_s=self.config.overlap_chunk_second,
            window_size=window_size,
            sample_rate=SAMPLE_RATE,
        )

    def greedy_decode_tokens(
        self,
        audio_features: mx.array,
        prompt_token_ids: list[int],
        max_tokens: int | None = None,
    ) -> list[int]:
        if max_tokens is None:
            max_tokens = WHISPER_MAX_DECODE_TOKENS

        if not prompt_token_ids:
            logger.warning("Empty prompt_token_ids; returning no tokens")
            return []

        remaining = self.model.config.n_text_ctx - len(prompt_token_ids)
        if remaining <= 0:
            logger.warning(
                "Prompt (%d tokens) already fills context window (%d)",
                len(prompt_token_ids),
                self.model.config.n_text_ctx,
            )
            return []
        max_tokens = min(max_tokens, remaining)

        eot_token = self._get_token_id("<|endoftext|>")
        tokens = mx.array([prompt_token_ids], dtype=mx.int32)
        kv_cache = None
        output_tokens: list[int] = []

        for _ in range(max_tokens):
            logits, kv_cache = self.model.decode(tokens, audio_features, kv_cache)
            next_token = int(mx.argmax(logits[:, -1, :], axis=-1).item())
            if next_token == eot_token:
                break
            output_tokens.append(next_token)
            tokens = mx.array([[next_token]], dtype=mx.int32)

        return output_tokens

    def _get_token_id(self, token: str) -> int:
        return cast(int, self.tokenizer.convert_tokens_to_ids(token))

    def _resolve_decode_options(
        self,
        language: str | None,
        task: str,
    ) -> tuple[str | None, str]:
        task = task.strip().lower()
        if task not in WHISPER_TASKS:
            supported = ", ".join(sorted(WHISPER_TASKS))
            raise ValueError(
                f"Unsupported STT task: {task!r}. Must be one of {supported}."
            )

        if self.model.is_multilingual:
            return validate_language(language, default=None), task

        resolved_language = validate_language(language, default=None)
        if task == "translate":
            raise ValueError("English-only Whisper models do not support translation.")
        if resolved_language not in (None, "en"):
            raise ValueError(
                "English-only Whisper models only support English transcription."
            )
        return resolved_language, task

    def _encode_prompt(self, prompt: str | None) -> list[int]:
        if not prompt:
            return []
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        prompt_ids = prompt_ids[-MAX_PROMPT_TOKENS:]
        return [self._get_token_id("<|startofprev|>"), *prompt_ids]

    def _greedy_decode(
        self,
        audio_features: mx.array,
        language: str | None = None,
        task: str = "transcribe",
        prompt: str | None = None,
        with_timestamps: bool = False,
        max_tokens: int | None = None,
    ) -> list[int]:
        if max_tokens is None:
            max_tokens = (
                WHISPER_MAX_DECODE_TOKENS if with_timestamps else MAX_PROMPT_TOKENS
            )

        prefix = self._encode_prompt(prompt)
        prefix.append(self._get_token_id("<|startoftranscript|>"))
        if self.model.is_multilingual:
            prefix.append(self._get_token_id(f"<|{language or 'en'}|>"))
            prefix.append(self._get_token_id(f"<|{task}|>"))
        if not with_timestamps:
            prefix.append(self._get_token_id("<|notimestamps|>"))

        return self.greedy_decode_tokens(audio_features, prefix, max_tokens)

    def _extract_segments(
        self,
        token_ids: list[int],
        time_offset: float = 0.0,
        segment_id_offset: int = 0,
    ) -> list[TranscriptionSegment]:
        raw_tokens = [self.tokenizer.convert_ids_to_tokens(tid) for tid in token_ids]

        segments: list[TranscriptionSegment] = []
        seg_start: float | None = None
        seg_tokens: list[int] = []
        seg_id = segment_id_offset

        for tid, text in zip(token_ids, raw_tokens, strict=True):
            m = TIMESTAMP_RE.match(text)
            if m:
                ts = float(m.group(1))
                if seg_start is None:
                    seg_start = ts
                    seg_tokens = []
                else:
                    seg_text = self.tokenizer.decode(
                        seg_tokens, skip_special_tokens=True
                    )
                    if seg_text.strip():
                        segments.append(
                            TranscriptionSegment(
                                id=seg_id,
                                seek=int(seg_start * SEEK_MULTIPLIER),
                                start=round(seg_start + time_offset, 2),
                                end=round(ts + time_offset, 2),
                                text=seg_text,
                                tokens=list(seg_tokens),
                            )
                        )
                        seg_id += 1
                    seg_start = None
                    seg_tokens = []
            else:
                seg_tokens.append(tid)

        if seg_start is not None and seg_tokens:
            seg_text = self.tokenizer.decode(seg_tokens, skip_special_tokens=True)
            if seg_text.strip():
                segments.append(
                    TranscriptionSegment(
                        id=seg_id,
                        seek=int(seg_start * SEEK_MULTIPLIER),
                        start=round(seg_start + time_offset, 2),
                        end=round(
                            seg_start + time_offset + DEFAULT_SEGMENT_DURATION, 2
                        ),
                        text=seg_text,
                        tokens=list(seg_tokens),
                    )
                )

        return segments

    def _encode_chunk(self, audio_chunk: mx.array) -> mx.array:
        mel = log_mel_spectrogram(audio_chunk, n_mels=self.model.config.n_mels)
        mel = pad_or_trim(mel, N_FRAMES, axis=-1)
        if mel.ndim == 2:
            mel = mel[None, ...]
        mel = mel.transpose(0, 2, 1)
        features = self.model.encode(mel)
        mx.eval(features)
        return features
