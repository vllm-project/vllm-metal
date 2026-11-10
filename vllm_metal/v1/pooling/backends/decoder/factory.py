# SPDX-License-Identifier: Apache-2.0
"""Factory for the decoder pooling backend."""

from __future__ import annotations

from typing import Any

from vllm_metal.v1.pooling.backends.decoder.models.qwen3 import (
    Qwen3RerankerPooler,
)
from vllm_metal.v1.pooling.backends.decoder.runtime import (
    DecoderModelView,
    LastTokenEmbeddingPooler,
    MetalDecoderPoolingBackend,
)
from vllm_metal.v1.pooling.contract import DecoderPoolingBackend
from vllm_metal.v1.pooling.validation import PoolingConfigView


def build_decoder_pooling_backend(
    model: Any,
    model_config: Any,
    tokenizer: Any = None,
) -> DecoderPoolingBackend:
    config = PoolingConfigView(model_config)
    model_view = DecoderModelView(model)
    poolers = (
        LastTokenEmbeddingPooler(model_view, config),
        Qwen3RerankerPooler(
            model,
            model_view.sequence_model,
            model_config,
            tokenizer,
        ),
    )
    return MetalDecoderPoolingBackend(config, model_view, poolers)
