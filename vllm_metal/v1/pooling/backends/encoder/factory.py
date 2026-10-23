# SPDX-License-Identifier: Apache-2.0
"""Factory for encoder pooling backends."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from vllm_metal.v1.pooling.backends.encoder.models.bge_m3 import (
    load_bge_m3_backend,
    supports_bge_m3_encoder,
)
from vllm_metal.v1.pooling.backends.encoder.models.laya import (
    load_laya_backend,
    supports_laya_encoder,
)
from vllm_metal.v1.pooling.backends.encoder.models.xlm_roberta import (
    load_xlm_roberta_backend,
    supports_xlm_roberta_encoder,
)
from vllm_metal.v1.pooling.contract import LoadedEncoderBackend
from vllm_metal.v1.pooling.validation import PoolingConfigView


@dataclass(frozen=True, slots=True)
class EncoderBackendLoader:
    supports: Callable[[Any], bool]
    load: Callable[[Any], LoadedEncoderBackend]


_ENCODER_BACKEND_LOADERS = (
    EncoderBackendLoader(supports=supports_laya_encoder, load=load_laya_backend),
    # Decoder pooling wraps the already-loaded generation model. Encoder pooling
    # has model-family-owned loaders because it does not use the generation
    # loader, paged attention, or KV cache.
    EncoderBackendLoader(
        supports=supports_bge_m3_encoder,
        load=load_bge_m3_backend,
    ),
    EncoderBackendLoader(
        supports=supports_xlm_roberta_encoder,
        load=load_xlm_roberta_backend,
    ),
)


def supports_encoder_pooling_backend(model_config: Any) -> bool:
    config = PoolingConfigView(model_config)
    return (
        config.runner_type == "pooling"
        and config.is_text_only
        and any(loader.supports(model_config) for loader in _ENCODER_BACKEND_LOADERS)
    )


def load_encoder_pooling_backend(
    model_config: Any,
) -> LoadedEncoderBackend:
    for loader in _ENCODER_BACKEND_LOADERS:
        if loader.supports(model_config):
            return loader.load(model_config)

    config = PoolingConfigView(model_config)
    raise NotImplementedError(
        f"Metal encoder pooling has no backend for model={config.label}."
    )
