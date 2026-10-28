# SPDX-License-Identifier: Apache-2.0
"""mlx-vlm EmbeddingGemma 2 text integration for encoder pooling."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, cast

import mlx.core as mx
import mlx.nn as nn
from mlx_vlm.models.embedding_gemma2 import Model, ModelConfig

from vllm_metal.tensor_bridge import TORCH_TO_MLX_DTYPE
from vllm_metal.v1.pooling.backends.encoder.models.loading import (
    encoder_model_path,
    load_encoder_weights,
)
from vllm_metal.v1.pooling.backends.encoder.runtime import MetalEncoderPoolingBackend
from vllm_metal.v1.pooling.contract import LoadedEncoderBackend
from vllm_metal.v1.pooling.validation import PoolingConfigView


class EmbeddingGemma2EncoderAdapter(nn.Module):
    def __init__(self, model: Model) -> None:
        super().__init__()
        self.model = model
        self.config = model.config

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        # Upstream includes embedding_projection in last_hidden_state.
        return cast(
            mx.array,
            self.model(input_ids, attention_mask=attention_mask).last_hidden_state,
        )


def supports_embedding_gemma2_encoder(model_config: Any) -> bool:
    return model_config.hf_config.model_type == "embedding_gemma2"


def load_embedding_gemma2_backend(model_config: Any) -> LoadedEncoderBackend:
    config = model_config.hf_config.to_dict()
    if (
        model_config.quantization is not None
        or config.get("quantization") is not None
        or config.get("quantization_config") is not None
    ):
        raise NotImplementedError(
            "Metal EmbeddingGemma 2 encoder pooling does not support quantization yet."
        )

    # Text requests need neither tower; sanitize drops their checkpoint weights.
    config.update(vision_config=None, audio_config=None)
    args = ModelConfig.from_dict(config)
    model = Model(args)
    target_dtype = TORCH_TO_MLX_DTYPE[model_config.dtype]
    weights = model.sanitize(load_encoder_weights(encoder_model_path(model_config)))
    weights = {
        name: value.astype(target_dtype)
        if mx.issubdtype(value.dtype, mx.floating)
        else value
        for name, value in weights.items()
    }
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())

    encoder = EmbeddingGemma2EncoderAdapter(model)
    return LoadedEncoderBackend(
        model=encoder,
        model_args=asdict(args.text_config),
        pooling_backend=MetalEncoderPoolingBackend(
            PoolingConfigView(model_config), encoder
        ),
    )
