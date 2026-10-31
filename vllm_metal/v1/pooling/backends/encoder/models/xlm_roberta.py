# SPDX-License-Identifier: Apache-2.0
"""mlx-vlm XLM-RoBERTa / RoBERTa integration for encoder pooling."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from mlx_vlm.models.xlm_roberta import Model, ModelConfig

from vllm_metal.v1.pooling.backends.encoder.models.adapter import MLXVLMEncoderAdapter
from vllm_metal.v1.pooling.backends.encoder.models.loading import (
    load_encoder_checkpoint,
)
from vllm_metal.v1.pooling.backends.encoder.runtime import MetalEncoderPoolingBackend
from vllm_metal.v1.pooling.contract import LoadedEncoderBackend
from vllm_metal.v1.pooling.validation import PoolingConfigView

_MODEL_TYPES = frozenset({"xlm-roberta", "roberta"})
_ARCHITECTURES = frozenset({"XLMRobertaModel", "RobertaModel"})


def supports_xlm_roberta_encoder(model_config: Any) -> bool:
    hf_config = model_config.hf_config
    architectures = tuple(str(value) for value in hf_config.architectures or ())
    model_type = str(hf_config.model_type).replace("_", "-")
    if architectures:
        return any(architecture in _ARCHITECTURES for architecture in architectures)
    return model_type in _MODEL_TYPES


def load_xlm_roberta_backend(
    model_config: Any,
) -> LoadedEncoderBackend:
    hf_config = model_config.hf_config
    config = hf_config.to_dict()
    if model_config.quantization is not None or config.get("quantization") is not None:
        raise NotImplementedError(
            "Metal XLM-R encoder pooling does not support quantization yet."
        )
    if hf_config.position_embedding_type != "absolute":
        raise NotImplementedError(
            "Metal XLM-R encoder pooling supports only absolute position embeddings."
        )
    if hf_config.hidden_act != "gelu":
        raise NotImplementedError(
            "Metal XLM-R encoder pooling supports only GELU activation."
        )

    args = ModelConfig.from_dict(config)
    model = Model(args)
    load_encoder_checkpoint(model, model_config)

    encoder = MLXVLMEncoderAdapter(model)
    pooling_backend = MetalEncoderPoolingBackend(
        PoolingConfigView(model_config),
        encoder,
    )
    return LoadedEncoderBackend(
        model=encoder,
        model_args=asdict(args),
        pooling_backend=pooling_backend,
    )
