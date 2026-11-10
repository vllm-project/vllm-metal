# SPDX-License-Identifier: Apache-2.0
"""mlx-vlm EmbeddingGemma 2 text integration for encoder pooling."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from mlx_vlm.models.embedding_gemma2 import Model, ModelConfig

from vllm_metal.v1.pooling.backends.encoder.models.adapter import MLXVLMEncoderAdapter
from vllm_metal.v1.pooling.backends.encoder.models.loading import (
    load_encoder_checkpoint,
)
from vllm_metal.v1.pooling.backends.encoder.runtime import MetalEncoderPoolingBackend
from vllm_metal.v1.pooling.contract import LoadedEncoderBackend
from vllm_metal.v1.pooling.validation import PoolingConfigView


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
    load_encoder_checkpoint(model, model_config)

    encoder = MLXVLMEncoderAdapter(model)
    return LoadedEncoderBackend(
        model=encoder,
        model_args=asdict(args.text_config),
        pooling_backend=MetalEncoderPoolingBackend(
            PoolingConfigView(model_config), encoder
        ),
    )
