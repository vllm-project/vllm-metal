# SPDX-License-Identifier: Apache-2.0
"""STT runtime glue used by the vLLM runner.

This module owns the STT execution helper that `vllm_metal/v1/model_runner.py`
delegates to, so the runner stays thin and STT logic stays under `stt/`.
"""

from __future__ import annotations

import logging
from typing import Any

import mlx.core as mx
import numpy as np
import torch

logger = logging.getLogger(__name__)


class STTExecutor:
    """Encapsulates STT-specific audio extraction and decoding.

    Holds a lazily-created :class:`WhisperTranscriber` and provides
    :meth:`extract_audio_features` and :meth:`decode` so that the model runner
    can delegate without embedding STT-specific logic.
    """

    def __init__(self, model: Any, model_path: str) -> None:
        self.model = model
        self._model_path = model_path
        self._transcriber: Any = None
        self._model_type: str = getattr(model, "model_type", "whisper")
        # Cached Qwen3-ASR special token IDs (resolved once on first use)
        self._asr_text_token_id: int | None = None
        self._im_end_token_id: int | None = None

    @property
    def transcriber(self):
        """Lazily-created transcriber (Whisper or Qwen3-ASR)."""
        if self._transcriber is None:
            if self._model_type == "qwen3_asr":
                from vllm_metal.stt.transcribe import Qwen3ASRTranscriber

                self._transcriber = Qwen3ASRTranscriber(
                    self.model, model_path=self._model_path
                )
            else:
                from vllm_metal.stt.transcribe import WhisperTranscriber

                self._transcriber = WhisperTranscriber(
                    self.model, model_path=self._model_path
                )
        return self._transcriber

    @property
    def eot_token(self) -> int:
        """End-of-text token ID resolved from the tokenizer or config."""
        if self._model_type == "qwen3_asr":
            return self.model.config.eos_token_id
        return self.transcriber.tokenizer.convert_tokens_to_ids("<|endoftext|>")

    def extract_audio_features(self, input_features: Any) -> mx.array:
        """Extract and encode STT input features."""
        # Convert to MLX array — handle numpy, torch, and lists
        if isinstance(input_features, np.ndarray):
            mel = mx.array(input_features, dtype=mx.float16)
        elif isinstance(input_features, torch.Tensor):
            # .cpu() for device safety, .float() because bfloat16 has
            # no numpy dtype support.
            mel = mx.array(input_features.cpu().float().numpy(), dtype=mx.float16)
        else:
            mel = mx.array(np.array(input_features), dtype=mx.float16)

        if self._model_type == "qwen3_asr":
            # Qwen3-ASR encoder expects: (n_mels, time) or (batch, n_mels, time)
            # HF WhisperFeatureExtractor output shape is already (n_mels, time)
            if mel.ndim == 3:
                mel = mel[0]  # drop batch dim -> (n_mels, time)
            elif mel.ndim != 2:
                raise ValueError(f"Qwen3-ASR expects 2D or 3D mel, got rank {mel.ndim}")
            features = self.model.encode(mel)
            mx.eval(features)
            return features

        # Whisper encoder expects: (batch, time, n_mels)
        # HF WhisperFeatureExtractor output shape: (n_mels, time)
        if mel.ndim == 2:
            mel = mel[None, ...]  # add batch dim -> (1, n_mels, time)
            mel = mel.transpose(0, 2, 1)  # -> (1, time, n_mels)
        elif mel.ndim == 3:
            mel = mel.transpose(
                0, 2, 1
            )  # (batch, n_mels, time) -> (batch, time, n_mels)
        else:
            raise ValueError(
                f"Unexpected mel spectrogram rank {mel.ndim}; expected 2D or 3D"
            )

        features = self.model.encode(mel)
        mx.eval(features)
        return features

    def decode(
        self,
        audio_features: mx.array,
        prompt_token_ids: list[int],
    ) -> list[int]:
        """Decode audio features into token IDs (ending with EOT).

        Delegates the core decode loop to the transcriber.
        """
        eot = self.eot_token

        if self._model_type == "qwen3_asr":
            # Qwen3-ASR uses a fixed prompt format. Rebuild prompt with the
            # correct number of audio_pad tokens matching the audio encoder
            # output length.
            n_audio_frames = audio_features.shape[0]
            prompt_token_ids = self.transcriber.build_prompt_tokens(n_audio_frames)
        elif not prompt_token_ids:
            logger.warning("STT: empty prompt_token_ids, returning EOT")
            return [eot]

        tokens = self.transcriber.greedy_decode_tokens(audio_features, prompt_token_ids)

        if self._model_type == "qwen3_asr":
            # Extract tokens between <asr_text> and <|im_end|>
            tokens = self._extract_asr_text_tokens(tokens)

        # Always end with EOT so vLLM marks the request as finished
        tokens.append(eot)
        return tokens

    def _extract_asr_text_tokens(self, tokens: list[int]) -> list[int]:
        """Extract content tokens between <asr_text> and <|im_end|>.

        Qwen3-ASR outputs: ``language {lang}<asr_text>{text}<|im_end|>``.
        We extract only the ``{text}`` portion.
        """
        if self._asr_text_token_id is None:
            tok = self.transcriber.tokenizer
            self._asr_text_token_id = tok.encode(
                "<asr_text>", add_special_tokens=False
            )[0]
            self._im_end_token_id = tok.encode("<|im_end|>", add_special_tokens=False)[
                0
            ]
        asr_text_token = self._asr_text_token_id
        im_end_token = self._im_end_token_id

        # Find last <asr_text> tag
        start = -1
        for i, t in enumerate(tokens):
            if t == asr_text_token:
                start = i + 1

        if start < 0 or start >= len(tokens):
            return tokens  # No <asr_text> found; return as-is

        # Find first <|im_end|> after <asr_text>
        end = len(tokens)
        for i in range(start, len(tokens)):
            if tokens[i] == im_end_token:
                end = i
                break

        return tokens[start:end]
