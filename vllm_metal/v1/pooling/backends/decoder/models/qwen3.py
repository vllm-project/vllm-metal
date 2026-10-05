# SPDX-License-Identifier: Apache-2.0
"""Qwen3 reranker ``classify`` pooling behavior."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import torch
from vllm.pooling_params import PoolingParams
from vllm.tasks import PoolingTask

from vllm_metal.tensor_bridge import mlx_to_torch
from vllm_metal.v1.pooling.contract import CLASSIFY_TASK, DecoderPoolingSpan
from vllm_metal.v1.pooling.validation import PoolingConfigView

QWEN3_RERANKER_TOKENS = ("no", "yes")
QWEN3_RERANKER_ARCH = "Qwen3ForSequenceClassification"
QWEN3_RERANKER_TASKS: tuple[PoolingTask | None, ...] = (None, CLASSIFY_TASK)


@dataclass(frozen=True, slots=True)
class Qwen3ClassifierHead:
    token_ids: mx.array
    logits: Callable[[mx.array], mx.array]


@dataclass(frozen=True, slots=True)
class Qwen3RerankerConfigView:
    config: PoolingConfigView

    @property
    def classifier_tokens(self) -> tuple[str, str] | None:
        tokens = getattr(self.config.hf_config, "classifier_from_token", None)
        if not isinstance(tokens, (list, tuple)) or len(tokens) != 2:
            return None
        return (str(tokens[0]), str(tokens[1]))

    @property
    def is_original_reranker(self) -> bool:
        return (
            getattr(self.config.hf_config, "is_original_qwen3_reranker", False) is True
        )

    @property
    def is_supported(self) -> bool:
        return (
            self.config.is_text_only
            and self.config.task in QWEN3_RERANKER_TASKS
            and self.config.uses_last_pooling
            and not self.config.chunked_processing_enabled
            and QWEN3_RERANKER_ARCH in self.config.architectures
            and self.is_original_reranker
            and self.classifier_tokens == QWEN3_RERANKER_TOKENS
        )


class Qwen3RerankerPooler:
    """Pool Qwen3 reranker hidden states as yes-minus-no scores."""

    task: PoolingTask = CLASSIFY_TASK

    def __init__(
        self,
        model: Any,
        sequence_model: Any | None,
        model_config: Any,
        tokenizer: Any,
    ) -> None:
        self.model = model
        self.config = PoolingConfigView(model_config)
        self.qwen_config = Qwen3RerankerConfigView(self.config)
        self.tokenizer = tokenizer
        self.sequence_model = sequence_model
        self.classifier_head = self._classifier_head()

    def is_supported(self) -> bool:
        return self.qwen_config.is_supported and self.classifier_head is not None

    def pool_one(
        self,
        hidden_states: mx.array,
        span: DecoderPoolingSpan,
    ) -> torch.Tensor:
        token_index = span.start_row + span.num_tokens - 1
        return self._pool_token(hidden_states, token_index, span.pooling_params)

    def _pool_token(
        self,
        hidden_states: mx.array,
        token_index: int,
        pooling_params: PoolingParams,
    ) -> torch.Tensor:
        if hidden_states.ndim != 3 or hidden_states.shape[0] != 1:
            raise ValueError(
                "Metal classify pooling expected hidden states with shape "
                f"[1, tokens, hidden], got {hidden_states.shape} for model="
                f"{self.config.label}."
            )
        if token_index < 0 or token_index >= hidden_states.shape[1]:
            raise ValueError(
                f"Metal classify pooling token index {token_index} is outside hidden "
                f"state shape {hidden_states.shape} for model={self.config.label}."
            )

        assert self.classifier_head is not None

        vector = hidden_states[0, token_index, :].astype(mx.float32)
        vocab_logits = mx.squeeze(
            self.classifier_head.logits(vector).astype(mx.float32)
        )
        if vocab_logits.ndim != 1:
            raise ValueError(
                "Metal classify pooling expected classifier logits with shape "
                f"[vocab], got {vocab_logits.shape} for model={self.config.label}."
            )

        token_logits = vocab_logits[self.classifier_head.token_ids]
        score = token_logits[1] - token_logits[0]
        if self.config.pooler_config.logit_mean is not None:
            score = score - float(self.config.pooler_config.logit_mean)
        if self.config.pooler_config.logit_sigma is not None:
            score = score / float(self.config.pooler_config.logit_sigma)
        if self._classifier_use_activation(pooling_params):
            score = mx.sigmoid(score)

        tensor = mlx_to_torch(score.reshape((1,)), device="cpu")
        return tensor.detach().clone()

    def _word_embeddings_tied(self) -> bool | None:
        for source in (
            self.model,
            getattr(self.model, "args", None),
            self.config.hf_config,
        ):
            value = getattr(source, "tie_word_embeddings", None)
            if value is not None:
                return bool(value)
        return None

    def _tied_embedding_logits_fn(self) -> Any | None:
        if self.sequence_model is None:
            return None
        embed_tokens = getattr(self.sequence_model, "embed_tokens", None)
        as_linear = getattr(embed_tokens, "as_linear", None)
        return as_linear if callable(as_linear) else None

    def _classifier_logits_fn(self) -> Any | None:
        if self.sequence_model is None:
            return None

        lm_head = getattr(self.model, "lm_head", None)
        tied_embedding_logits = self._tied_embedding_logits_fn()
        tied = self._word_embeddings_tied()

        if tied is False:
            return lm_head if callable(lm_head) else None
        if tied is True:
            return tied_embedding_logits
        return None

    def _resolve_token_id(self, token: str) -> int | None:
        if self.tokenizer is None:
            return None

        convert = getattr(self.tokenizer, "convert_tokens_to_ids", None)
        if callable(convert):
            token_id = convert(token)
            if isinstance(token_id, int) and token_id >= 0:
                return token_id

        encode = getattr(self.tokenizer, "encode", None)
        if callable(encode):
            token_ids = encode(token, add_special_tokens=False)
            if isinstance(token_ids, list) and len(token_ids) == 1:
                return int(token_ids[0])

        return None

    def _classifier_token_ids(self) -> tuple[int, int] | None:
        tokens = self.qwen_config.classifier_tokens
        if tokens is None:
            return None
        token_ids = tuple(self._resolve_token_id(token) for token in tokens)
        if any(token_id is None for token_id in token_ids):
            return None
        no_id, yes_id = token_ids
        assert no_id is not None and yes_id is not None
        return (no_id, yes_id)

    def _classifier_head(self) -> Qwen3ClassifierHead | None:
        token_ids = self._classifier_token_ids()
        logits_fn = self._classifier_logits_fn()
        if token_ids is None or logits_fn is None:
            return None
        return Qwen3ClassifierHead(mx.array(token_ids, dtype=mx.int32), logits_fn)

    def _classifier_use_activation(self, pooling_params: PoolingParams) -> bool:
        if pooling_params.use_activation is not None:
            return pooling_params.use_activation
        return self.config.pooler_config.use_activation is not False
