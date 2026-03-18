# SPDX-License-Identifier: Apache-2.0
"""Whisper runtime adapter for vLLM STT execution."""

from __future__ import annotations

import logging
from typing import Any

import mlx.core as mx

from vllm_metal.stt.runtime import STTRuntimeAdapter

from .model import WhisperModel
from .transcriber import WhisperTranscriber

logger = logging.getLogger(__name__)


class WhisperRuntimeAdapter(STTRuntimeAdapter):
    def __init__(self, model: WhisperModel, model_path: str) -> None:
        super().__init__(model, model_path)
        self._transcriber: WhisperTranscriber | None = None

    @property
    def transcriber(self) -> WhisperTranscriber:
        if self._transcriber is None:
            self._transcriber = WhisperTranscriber(
                self.model, model_path=self._model_path
            )
        return self._transcriber

    @property
    def eot_token(self) -> int:
        return int(self.transcriber.tokenizer.convert_tokens_to_ids("<|endoftext|>"))

    def extract_audio_features(self, input_features: Any) -> mx.array:
        mel = self._to_mx_float16(input_features)

        # Whisper encoder expects: (batch, time, n_mels).
        # HF WhisperFeatureExtractor output shape: (n_mels, time).
        if mel.ndim == 2:
            mel = mel[None, ...].transpose(0, 2, 1)  # (1, time, n_mels)
        elif mel.ndim == 3:
            mel = mel.transpose(0, 2, 1)  # (batch, time, n_mels)
        else:
            raise ValueError(
                f"Unexpected mel spectrogram rank {mel.ndim}; expected 2D or 3D"
            )

        features = self.model.encode(mel)
        mx.eval(features)
        return features

    def warm_up(self) -> None:
        n_mels = self.model.config.n_mels
        n_audio_ctx = self.model.config.n_audio_ctx
        # Warm up with feature-extractor shaped input (n_mels, time) to reuse the
        # same normalization path as real STT requests.
        dummy_mel = mx.zeros((n_mels, n_audio_ctx * 2), dtype=mx.float16)
        features = self.extract_audio_features(dummy_mel)
        mx.eval(features)

    def decode_tokens(
        self,
        audio_features: mx.array,
        prompt_token_ids: list[int],
    ) -> list[int]:
        if not prompt_token_ids:
            logger.warning("STT: empty prompt_token_ids, returning EOT only")
            return [self.eot_token]
        tokens = self.transcriber.greedy_decode_tokens(audio_features, prompt_token_ids)
        tokens.append(self.eot_token)
        return tokens
