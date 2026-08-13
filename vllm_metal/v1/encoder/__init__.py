# SPDX-License-Identifier: Apache-2.0
"""Typed encoder embedding backends for Metal pooling."""

from vllm_metal.v1.encoder.backend import (
    EncoderEmbeddingBackend,
    EncoderPoolingPolicy,
    EncoderSequenceModel,
)
from vllm_metal.v1.encoder.registry import (
    backend_from_loaded_model,
    encoder_family_for_config,
    encoder_pooling_policy,
    load_encoder_backend,
)

__all__ = [
    "EncoderEmbeddingBackend",
    "EncoderPoolingPolicy",
    "EncoderSequenceModel",
    "backend_from_loaded_model",
    "encoder_family_for_config",
    "encoder_pooling_policy",
    "load_encoder_backend",
]
