# SPDX-License-Identifier: Apache-2.0
"""Qwen3-ASR runtime adapter for vLLM STT execution."""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from vllm_metal.stt.runtime import STTRuntimeAdapter

from .model import Qwen3ASRModel
from .transcriber import Qwen3ASRTranscriber


class Qwen3ASRRuntimeAdapter(STTRuntimeAdapter):
    def __init__(self, model: Qwen3ASRModel, model_path: str) -> None:
        super().__init__(model, model_path)
        self._transcriber: Qwen3ASRTranscriber | None = None
        self._asr_text_token_id: int | None = None
        self._im_end_token_id: int | None = None

    @property
    def transcriber(self) -> Qwen3ASRTranscriber:
        if self._transcriber is None:
            self._transcriber = Qwen3ASRTranscriber(
                self.model, model_path=self._model_path
            )
        return self._transcriber

    @property
    def eot_token(self) -> int:
        return self.model.config.eos_token_id

    def extract_audio_features(self, input_features: Any) -> mx.array:
        mel = self._to_mx_float16(input_features)

        # Qwen3-ASR encoder expects: (n_mels, time) or (batch, n_mels, time).
        # HF WhisperFeatureExtractor output shape is already (n_mels, time).
        if mel.ndim == 3:
            mel = mel[0]  # drop batch dim -> (n_mels, time)
        elif mel.ndim != 2:
            raise ValueError(f"Qwen3-ASR expects 2D or 3D mel, got rank {mel.ndim}")

        features = self.model.encode(mel)
        mx.eval(features)
        return features

    def warm_up(self) -> None:
        n_mels = self.model.config.audio_config.num_mel_bins
        dummy_mel = mx.zeros((n_mels, 100), dtype=mx.float16)
        features = self.extract_audio_features(dummy_mel)
        mx.eval(features)

    def decode_tokens(
        self,
        audio_features: mx.array,
        prompt_token_ids: list[int],
    ) -> list[int]:
        del prompt_token_ids

        n_audio_frames = audio_features.shape[0]
        prompt_token_ids = self.transcriber.build_prompt_tokens(n_audio_frames)
        tokens = self.transcriber.greedy_decode_tokens(audio_features, prompt_token_ids)
        tokens = self._extract_asr_text_tokens(tokens)
        tokens.append(self.eot_token)
        return tokens

    def _extract_asr_text_tokens(self, tokens: list[int]) -> list[int]:
        """Extract tokens between <asr_text> and <|im_end|>."""
        if self._asr_text_token_id is None:
            tokenizer = self.transcriber.tokenizer
            self._asr_text_token_id = tokenizer.encode(
                "<asr_text>", add_special_tokens=False
            )[0]
            self._im_end_token_id = tokenizer.encode(
                "<|im_end|>", add_special_tokens=False
            )[0]
        asr_text_token = self._asr_text_token_id
        im_end_token = self._im_end_token_id

        start = -1
        for i, t in enumerate(tokens):
            if t == asr_text_token:
                start = i + 1

        if start < 0 or start >= len(tokens):
            return tokens

        end = len(tokens)
        for i in range(start, len(tokens)):
            if tokens[i] == im_end_token:
                end = i
                break

        return tokens[start:end]
