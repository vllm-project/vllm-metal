# SPDX-License-Identifier: Apache-2.0
"""Factory for the decoder pooling backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vllm.logger import init_logger

from vllm_metal.v1.pooling.backends.decoder.models.d3 import D3_ARCH, D3ReadoutPooler
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

logger = init_logger(__name__)


def build_decoder_pooling_backend(
    model: Any,
    model_config: Any,
    tokenizer: Any = None,
    checkpoint_dir: Path | None = None,
) -> DecoderPoolingBackend:
    config = PoolingConfigView(model_config)
    model_view = DecoderModelView(model)
    d3_pooler = D3ReadoutPooler(model_config, checkpoint_dir=checkpoint_dir)
    poolers = (
        LastTokenEmbeddingPooler(model_view, config),
        Qwen3RerankerPooler(
            model,
            model_view.sequence_model,
            model_config,
            tokenizer,
        ),
        d3_pooler,
    )
    backend = MetalDecoderPoolingBackend(config, model_view, poolers)
    if D3_ARCH in config.architectures and not d3_pooler.is_supported():
        # Without this the only symptom is a bare 501 from /v1/systemone.
        logger.warning(
            "d3 checkpoint %s loaded without readout.safetensors in %s: no "
            "classify pooler is installed and /v1/systemone will answer 501.",
            config.label,
            checkpoint_dir if checkpoint_dir is not None else "<unresolved>",
        )
    return backend
