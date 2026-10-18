# SPDX-License-Identifier: Apache-2.0
"""mlx-vlm XLM-RoBERTa / RoBERTa integration for encoder pooling."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_vlm.models.xlm_roberta import Model, ModelConfig
from transformers import AutoTokenizer

from vllm_metal.tensor_bridge import TORCH_TO_MLX_DTYPE
from vllm_metal.v1.pooling.backends.encoder.models.loading import (
    encoder_model_path,
    load_encoder_weights,
)
from vllm_metal.v1.pooling.backends.encoder.runtime import MetalEncoderPoolingBackend
from vllm_metal.v1.pooling.contract import LoadedEncoderBackend
from vllm_metal.v1.pooling.validation import PoolingConfigView

_MODEL_TYPES = frozenset({"xlm-roberta", "roberta"})
_ARCHITECTURES = frozenset({"XLMRobertaModel", "RobertaModel"})


class XLMRobertaEncoderAdapter(nn.Module):
    """Expose upstream hidden states to vllm-metal's dense and sparse poolers."""

    def __init__(self, model: Model) -> None:
        super().__init__()
        self.model = model
        self.config = model.config

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        return self.model(input_ids, attention_mask=attention_mask).last_hidden_state


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

    target_dtype = TORCH_TO_MLX_DTYPE[model_config.dtype]
    args = ModelConfig.from_dict(config)
    model = Model(args)
    model_path = encoder_model_path(model_config)
    weights = load_encoder_weights(model_path)
    weights = model.sanitize(weights)
    weights = {
        name: value.astype(target_dtype)
        if mx.issubdtype(value.dtype, mx.floating)
        else value
        for name, value in weights.items()
    }
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())

    tokenizer = AutoTokenizer.from_pretrained(
        model_config.tokenizer,
        revision=model_config.tokenizer_revision,
        trust_remote_code=model_config.trust_remote_code,
    )
    encoder = XLMRobertaEncoderAdapter(model)
    pooling_backend = MetalEncoderPoolingBackend(
        PoolingConfigView(model_config),
        encoder,
    )
    return LoadedEncoderBackend(
        model=encoder,
        tokenizer=tokenizer,
        model_args=asdict(args),
        pooling_backend=pooling_backend,
    )
