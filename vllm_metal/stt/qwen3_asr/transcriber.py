# SPDX-License-Identifier: Apache-2.0
"""Qwen3-ASR transcription policy and decode loop."""

from __future__ import annotations

from typing import cast

import mlx.core as mx
from transformers import AutoTokenizer
from vllm.tokenizers import TokenizerLike

from vllm_metal.stt.sampling import STTSampling

from .config import QWEN3_ASR_MAX_DECODE_TOKENS
from .model import Qwen3ASRModel

ASR_TEXT_TAG = "<asr_text>"


class Qwen3ASRTranscriber:
    def __init__(
        self,
        model: Qwen3ASRModel,
        model_path: str | None = None,
        tokenizer: TokenizerLike | None = None,
    ) -> None:
        self.model = model
        self.tokenizer: TokenizerLike = (
            tokenizer if tokenizer is not None else self.load_tokenizer(model_path)
        )
        self._asr_text_token_id: int | None = None

    @property
    def asr_text_token_id(self) -> int:
        """The tag that opens the transcript inside the decode stream."""
        if self._asr_text_token_id is None:
            self._asr_text_token_id = self.tokenizer.encode(
                ASR_TEXT_TAG, add_special_tokens=False
            )[0]
        return self._asr_text_token_id

    @staticmethod
    def load_tokenizer(model_path: str | None) -> TokenizerLike:
        if not model_path:
            raise ValueError("Qwen3-ASR requires a local tokenizer model_path.")
        return cast(
            TokenizerLike,
            AutoTokenizer.from_pretrained(model_path, trust_remote_code=True),
        )

    def decode_tokens(
        self,
        audio_features: mx.array,
        prompt_token_ids: list[int],
        sampling: STTSampling,
        max_tokens: int | None = None,
    ) -> list[int]:
        if max_tokens is None:
            max_tokens = QWEN3_ASR_MAX_DECODE_TOKENS

        if not prompt_token_ids:
            raise ValueError("Qwen3-ASR decode requires non-empty prompt_token_ids.")

        budget = sampling.decode_budget(max_tokens)
        asr_text_token = self.asr_text_token_id
        # Qwen3-ASR also ends assistant turns with the tokenizer's <|im_end|>.
        eos_tokens = (self.model.config.eos_token_id, self.tokenizer.eos_token_id)
        tokens = mx.array([prompt_token_ids], dtype=mx.int32)

        logits, cache = self.model.prefill(tokens, audio_features)
        mx.eval(logits)

        output_tokens: list[int] = []
        transcript_len: int | None = None
        for step in range(max_tokens):
            if step:
                token_input = mx.array([[output_tokens[-1]]], dtype=mx.int32)
                logits, cache = self.model.decode_step(token_input, cache)
                mx.eval(logits)
            next_token = sampling.next_token(
                prompt_token_ids, output_tokens, logits[:, -1, :]
            )
            if next_token in eos_tokens:
                break
            output_tokens.append(next_token)
            # The request's budget counts transcript, not the tags around it.
            if next_token == asr_text_token:
                transcript_len = 0
            elif transcript_len is not None:
                transcript_len += 1
                if transcript_len >= budget:
                    break

        return output_tokens

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
