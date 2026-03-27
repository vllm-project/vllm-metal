# SPDX-License-Identifier: Apache-2.0
"""Qwen3-ASR transcription policy and decode loop."""

from __future__ import annotations

import logging
from typing import Any

import mlx.core as mx
from transformers import AutoTokenizer

from .config import QWEN3_ASR_MAX_DECODE_TOKENS
from .model import Qwen3ASRModel

logger = logging.getLogger(__name__)

ASR_TEXT_TAG = "<asr_text>"


class Qwen3ASRTranscriber:
    def __init__(
        self,
        model: Qwen3ASRModel,
        model_path: str | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = (
            tokenizer if tokenizer is not None else self.load_tokenizer(model_path)
        )

    @staticmethod
    def load_tokenizer(model_path: str | None) -> Any:
        if not model_path:
            raise ValueError("Qwen3-ASR requires a local tokenizer model_path.")
        return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    def greedy_decode_tokens(
        self,
        audio_features: mx.array,
        prompt_token_ids: list[int],
        max_tokens: int | None = None,
    ) -> list[int]:
        if max_tokens is None:
            max_tokens = QWEN3_ASR_MAX_DECODE_TOKENS

        if not prompt_token_ids:
            logger.warning("Empty prompt_token_ids; returning no tokens")
            return []

        eos_token = self.model.config.eos_token_id
        tokens = mx.array([prompt_token_ids], dtype=mx.int32)

        logits, cache = self.model.prefill(tokens, audio_features)
        mx.eval(logits)

        output_tokens: list[int] = []
        next_token = int(mx.argmax(logits[:, -1, :], axis=-1).item())
        if next_token == eos_token:
            return output_tokens
        output_tokens.append(next_token)

        for _ in range(max_tokens - 1):
            token_input = mx.array([[next_token]], dtype=mx.int32)
            logits, cache = self.model.decode_step(token_input, cache)
            mx.eval(logits)
            next_token = int(mx.argmax(logits[:, -1, :], axis=-1).item())
            if next_token == eos_token:
                break
            output_tokens.append(next_token)

        return output_tokens

    def build_prompt_tokens(self, n_audio_frames: int) -> list[int]:
        tok = self.tokenizer
        audio_pad_id = self.model.config.audio_token_id
        audio_start_id = self.model.config.audio_start_token_id
        audio_end_id = self.model.config.audio_end_token_id

        im_start = tok.encode("<|im_start|>", add_special_tokens=False)
        im_end = tok.encode("<|im_end|>", add_special_tokens=False)
        user = tok.encode("user\n", add_special_tokens=False)
        assistant = tok.encode("assistant\n", add_special_tokens=False)
        newline = tok.encode("\n", add_special_tokens=False)

        prompt = (
            im_start
            + user
            + [audio_start_id]
            + [audio_pad_id] * n_audio_frames
            + [audio_end_id]
            + newline
            + im_end
            + newline
            + im_start
            + assistant
        )
        return prompt

    @staticmethod
    def post_process_output(text: str) -> str:
        if not text:
            return ""
        if ASR_TEXT_TAG not in text:
            return text
        _, text_part = text.rsplit(ASR_TEXT_TAG, 1)
        for marker in ("<|im_end|>", "<|im_start|>", "<|endoftext|>"):
            idx = text_part.find(marker)
            if idx >= 0:
                text_part = text_part[:idx]
        return text_part.strip()
