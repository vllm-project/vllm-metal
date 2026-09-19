# SPDX-License-Identifier: Apache-2.0
"""BGE-M3 dense and sparse encoder pooling."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import torch
from vllm.pooling_params import PoolingParams
from vllm.tasks import PoolingTask

from vllm_metal.pytorch_backend.tensor_bridge import (
    TORCH_TO_MLX_DTYPE,
    mlx_to_torch,
)
from vllm_metal.v1.pooling.backends.encoder.models.loading import (
    encoder_model_path,
    load_encoder_weight_file,
)
from vllm_metal.v1.pooling.backends.encoder.models.xlm_roberta import (
    load_xlm_roberta_backend,
)
from vllm_metal.v1.pooling.backends.encoder.runtime import (
    EncoderEmbeddingPooler,
    MetalEncoderPoolingBackend,
)
from vllm_metal.v1.pooling.contract import (
    EMBED_TASK,
    TOKEN_CLASSIFY_TASK,
    EncoderPoolingRequest,
    LoadedEncoderBackend,
)
from vllm_metal.v1.pooling.validation import PoolingConfigView

_ARCHITECTURES = frozenset({"BgeM3EmbeddingModel"})
_TOKEN_POOLING_TYPES = (None, "ALL")
_SPARSE_LINEAR_FILE = "sparse_linear.pt"


class BgeM3Pooler:
    """BGE-M3 dense embedding and sparse lexical-weight pooling."""

    def __init__(
        self,
        config: PoolingConfigView,
        sparse_linear: nn.Linear,
    ) -> None:
        self.config = config
        self.sparse_linear = sparse_linear
        self.embedding_pooler = EncoderEmbeddingPooler(config)
        self.bos_token_id = int(config.hf_config.bos_token_id)
        self.eos_token_id = int(config.hf_config.eos_token_id)

    def supported_tasks(self) -> tuple[PoolingTask, ...]:
        tasks: list[PoolingTask] = []
        if self.embedding_pooler.supported_tasks():
            tasks.append(EMBED_TASK)
        if self._supports_token_classify():
            tasks.append(TOKEN_CLASSIFY_TASK)
        return tuple(tasks)

    def validate_params(self, pooling_params: PoolingParams) -> None:
        if pooling_params.task in (None, EMBED_TASK):
            self.embedding_pooler.validate_params(pooling_params)
            return
        if (
            pooling_params.task == TOKEN_CLASSIFY_TASK
            and self._supports_token_classify()
        ):
            return
        raise NotImplementedError(
            "Metal BGE-M3 pooling supports only task='embed' or "
            f"task='token_classify' for model={self.config.label}."
        )

    def pool_one(
        self,
        hidden_states: mx.array,
        request: EncoderPoolingRequest,
    ) -> torch.Tensor:
        """Dispatch one BGE-M3 request to dense embedding or sparse pooling."""
        if request.pooling_params.task == TOKEN_CLASSIFY_TASK:
            return self._pool_sparse(hidden_states, request)
        return self.embedding_pooler.pool_one(hidden_states, request)

    def _supports_token_classify(self) -> bool:
        return (
            self.config.is_text_only
            and self.config.task in (None, TOKEN_CLASSIFY_TASK)
            and self.config.pooler_config.tok_pooling_type in _TOKEN_POOLING_TYPES
            and not self.config.chunked_processing_enabled
        )

    def _pool_sparse(
        self,
        hidden_states: mx.array,
        request: EncoderPoolingRequest,
    ) -> torch.Tensor:
        """Return sparse lexical scores after BOS/EOS filtering."""
        if not request.token_ids:
            raise ValueError("Metal BGE-M3 sparse pooling requires at least one token.")

        token_hidden_states = hidden_states[0, : len(request.token_ids), :]
        scores = self.sparse_linear(token_hidden_states).astype(mx.float32)
        if request.pooling_params.use_activation is not False:
            scores = mx.maximum(scores, mx.zeros_like(scores))

        tensor = mlx_to_torch(scores, device="cpu")
        tensor = tensor.detach().clone().squeeze(-1)
        start = 1 if request.token_ids[0] == self.bos_token_id else 0
        end = (
            -1 if request.token_ids[-1] == self.eos_token_id else len(request.token_ids)
        )
        return tensor[start:end]


def supports_bge_m3_encoder(model_config: Any) -> bool:
    architectures = tuple(
        str(value) for value in model_config.hf_config.architectures or ()
    )
    return any(architecture in _ARCHITECTURES for architecture in architectures)


def load_bge_m3_backend(
    model_config: Any,
) -> LoadedEncoderBackend:
    loaded_backbone = load_xlm_roberta_backend(model_config)
    model_path = encoder_model_path(model_config)
    config = PoolingConfigView(model_config)
    sparse_linear = load_sparse_linear(
        model_path,
        int(loaded_backbone.model_args["hidden_size"]),
        TORCH_TO_MLX_DTYPE[model_config.dtype],
    )
    pooling_backend = MetalEncoderPoolingBackend(
        config,
        loaded_backbone.model,
        BgeM3Pooler(config, sparse_linear),
    )
    return LoadedEncoderBackend(
        model=loaded_backbone.model,
        tokenizer=loaded_backbone.tokenizer,
        model_args=loaded_backbone.model_args,
        pooling_backend=pooling_backend,
    )


def load_sparse_linear(
    model_path: Path,
    hidden_size: int,
    target_dtype: mx.Dtype,
) -> nn.Linear:
    sparse_linear_path = model_path / _SPARSE_LINEAR_FILE
    if not sparse_linear_path.exists():
        raise FileNotFoundError(f"Missing BGE-M3 sparse head: {sparse_linear_path}.")

    sparse_linear = nn.Linear(hidden_size, 1)
    weights = {
        name: value.astype(target_dtype)
        if mx.issubdtype(value.dtype, mx.floating)
        else value
        for name, value in load_encoder_weight_file(sparse_linear_path).items()
    }
    sparse_linear.load_weights(list(weights.items()), strict=True)
    mx.eval(sparse_linear.parameters())
    return sparse_linear
