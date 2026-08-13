# SPDX-License-Identifier: Apache-2.0
"""Factory for encoder pooling backends."""

from __future__ import annotations

from typing import Any

from vllm_metal.v1.pooling.backends.encoder.models.xlm_roberta import (
    load_xlm_roberta_backend,
    supports_xlm_roberta_encoder,
)
from vllm_metal.v1.pooling.contract import EncoderPoolingBackend
from vllm_metal.v1.pooling.validation import PoolingConfigView


def supports_encoder_pooling_backend(model_config: Any) -> bool:
    config = PoolingConfigView(model_config)
    return (
        config.runner_type == "pooling"
        and config.is_text_only
        and supports_xlm_roberta_encoder(model_config)
    )


def load_encoder_pooling_backend(
    model_config: Any,
) -> tuple[Any, Any, EncoderPoolingBackend]:
    return load_xlm_roberta_backend(model_config)
