# SPDX-License-Identifier: Apache-2.0
"""Generic decoder pooling backend.

This module owns packed decoder execution and LAST-token embedding pooling.
Model-family files provide only task-specific poolers, for example Qwen3
reranker ``classify`` scoring.
"""

from __future__ import annotations

from typing import Any, NoReturn

import mlx.core as mx
import torch
from vllm.pooling_params import PoolingParams
from vllm.tasks import PoolingTask

from vllm_metal.attention.context import OffsetCache
from vllm_metal.pytorch_backend.tensor_bridge import mlx_to_torch
from vllm_metal.v1.pooling.contract import (
    EMBED_TASK,
    DecoderPooler,
    DecoderPoolingBatch,
    DecoderPoolingSpan,
    PoolingCapabilities,
)
from vllm_metal.v1.pooling.validation import PoolingConfigView


class DecoderModelView:
    """Access the MLX decoder model shape used by pooling."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self.sequence_model = self._sequence_model()

    @property
    def has_sequence_model(self) -> bool:
        return self.sequence_model is not None

    def forward_packed(
        self,
        input_ids: mx.array,
        offset_caches: list[OffsetCache] | None,
    ) -> mx.array:
        assert self.sequence_model is not None
        if offset_caches is None:
            return self.sequence_model(input_ids)
        return self.sequence_model(input_ids, cache=offset_caches)

    def _sequence_model(self) -> Any | None:
        body = getattr(self.model, "model", None)
        return body if callable(body) else None


class LastTokenEmbeddingPooler:
    """Pool decoder hidden states into normalized LAST-token embeddings."""

    task: PoolingTask = EMBED_TASK

    def __init__(
        self,
        model_view: DecoderModelView,
        config: PoolingConfigView,
    ) -> None:
        self.model_view = model_view
        self.config = config

    def is_supported(self) -> bool:
        return (
            self.config.supports_decoder_embed_config
            and self.model_view.has_sequence_model
            and self._is_decoder_embedding()
        )

    def pool_one(
        self,
        hidden_states: mx.array,
        span: DecoderPoolingSpan,
    ) -> torch.Tensor:
        token_index = span.start_row + span.num_tokens - 1
        return self._pool_token(hidden_states, token_index)

    def _pool_token(self, hidden_states: mx.array, token_index: int) -> torch.Tensor:
        if hidden_states.ndim != 3 or hidden_states.shape[0] != 1:
            raise ValueError(
                "Metal embed pooling expected hidden states with shape "
                f"[1, tokens, hidden], got {hidden_states.shape} "
                f"for model={self.config.label}."
            )
        if token_index < 0 or token_index >= hidden_states.shape[1]:
            raise ValueError(
                f"Metal embed pooling token index {token_index} is outside hidden "
                f"state shape {hidden_states.shape} for model={self.config.label}."
            )

        vector = hidden_states[0, token_index, :].astype(mx.float32)
        vector = self._normalize_vector(vector)
        tensor = mlx_to_torch(vector, device="cpu", already_contiguous=True)
        return tensor.detach().clone()

    def _normalize_vector(self, vector: mx.array) -> mx.array:
        norm = mx.sqrt(mx.sum(vector * vector))
        norm = mx.maximum(norm, mx.array(1e-12, dtype=mx.float32))
        return mx.contiguous(vector / norm)

    def _is_decoder_embedding(self) -> bool:
        return any(
            architecture.endswith("ForCausalLM")
            or architecture.endswith("ForTextEncoding")
            or architecture.endswith("EmbeddingModel")
            for architecture in self.config.architectures
        )


class MetalDecoderPoolingBackend:
    """Decoder pooling backend for current Metal text pooling behavior."""

    capabilities = PoolingCapabilities(
        execution_kind="decoder",
        requires_paged_attention=True,
        uses_kv_cache=True,
        supports_chunked_requests=True,
    )

    def __init__(
        self,
        config: PoolingConfigView,
        model_view: DecoderModelView,
        poolers: tuple[DecoderPooler, ...],
    ) -> None:
        self.config = config
        self.model_view = model_view
        self.poolers_by_task = self._supported_poolers_by_task(poolers)

    def supported_tasks(self) -> tuple[PoolingTask, ...]:
        return tuple(self.poolers_by_task)

    def validate_params(self, pooling_params: PoolingParams) -> None:
        self.config.reject_unsupported_pooler_config()
        task = pooling_params.task or EMBED_TASK
        if task not in self.poolers_by_task:
            self._raise_unsupported_task(pooling_params.task)

    def profile_forward(self, input_ids: mx.array) -> mx.array:
        return self.forward_packed(input_ids, None)

    def forward_packed(
        self,
        input_ids: mx.array,
        offset_caches: list[OffsetCache] | None,
    ) -> mx.array:
        if not self.model_view.has_sequence_model:
            raise NotImplementedError(
                "Metal pooling requires an MLX model with a callable "
                f"'.model' transformer body; model={self.config.label}; "
                "runner='pooling'."
            )
        hidden_states = self.model_view.forward_packed(input_ids, offset_caches)
        if not hasattr(hidden_states, "shape") or not hasattr(hidden_states, "dtype"):
            raise ValueError(
                "Metal pooling expected MLX hidden states from model body; "
                f"got {type(hidden_states).__name__} for model={self.config.label}."
            )
        return hidden_states

    def pool_packed(
        self,
        hidden_states: mx.array,
        batch: DecoderPoolingBatch,
    ) -> tuple[torch.Tensor | None, ...]:
        outputs: list[torch.Tensor | None] = []
        for span in batch.spans:
            if not span.is_complete:
                outputs.append(None)
                continue
            outputs.append(self._pool_complete_span(hidden_states, span))
        return tuple(outputs)

    def _pool_complete_span(
        self,
        hidden_states: mx.array,
        span: DecoderPoolingSpan,
    ) -> torch.Tensor:
        task = span.pooling_params.task or EMBED_TASK
        pooler = self.poolers_by_task.get(task)
        if pooler is None:
            self._raise_unsupported_task(span.pooling_params.task)
        return pooler.pool_one(hidden_states, span)

    def _supported_poolers_by_task(
        self,
        poolers: tuple[DecoderPooler, ...],
    ) -> dict[PoolingTask, DecoderPooler]:
        poolers_by_task: dict[PoolingTask, DecoderPooler] = {}
        for pooler in poolers:
            if not pooler.is_supported():
                continue
            if pooler.task in poolers_by_task:
                raise RuntimeError(
                    "Metal pooling found multiple supported poolers for "
                    f"task={pooler.task!r} on model={self.config.label}."
                )
            poolers_by_task[pooler.task] = pooler
        return poolers_by_task

    def _raise_unsupported_task(self, task: PoolingTask | None) -> NoReturn:
        supported_tasks = ", ".join(repr(value) for value in self.supported_tasks())
        supported_hint = supported_tasks or "none"
        raise NotImplementedError(
            f"Metal pooling does not support task={task or EMBED_TASK!r} "
            f"for model={self.config.label}; supported tasks: {supported_hint}."
        )
