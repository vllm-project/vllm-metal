# SPDX-License-Identifier: Apache-2.0
"""Factory for the decoder pooling backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vllm_metal.v1.pooling.backends.decoder.models.d3 import (
    D3_READOUT_FILE,
    D3ReadoutPooler,
)
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
    if d3_pooler.d3_config.is_supported and d3_pooler.readout is None:
        # Without the readout the server would start and answer 501 on every
        # decision request; the checkpoint is unusable as served.
        raise ValueError(
            f"d3 checkpoint {config.label} is missing {D3_READOUT_FILE} in "
            f"{checkpoint_dir if checkpoint_dir is not None else '<unresolved>'}: "
            "the classify head cannot score prompts without it."
        )
    return MetalDecoderPoolingBackend(config, model_view, poolers)
