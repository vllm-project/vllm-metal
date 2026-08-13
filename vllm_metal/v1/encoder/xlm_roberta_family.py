# SPDX-License-Identifier: Apache-2.0
"""XLM-RoBERTa / RoBERTa dense encoder embedding family."""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from vllm_metal.v1.encoder.backend import (
    EncoderPoolingPolicy,
    EncoderSequenceModel,
    forward_packed_encoder_segments,
)
from vllm_metal.v1.encoder.loader import load_encoder_model

_ARCHITECTURES = frozenset(
    {
        "XLMRobertaModel",
        "RobertaModel",
        "RobertaEmbeddingModel",
        "BgeM3EmbeddingModel",
    }
)
_MODEL_TYPES = frozenset(
    {
        "xlm-roberta",
        "roberta",
        "xlm_roberta",
    }
)
_POOLING_POLICY = EncoderPoolingPolicy(
    default_sequence_pooling_type="CLS",
    allowed_sequence_pooling_types=(None, "CLS", "LAST"),
    skip_paged_attention_patch=True,
)


class XLMRobertaSequenceModel(EncoderSequenceModel):
    """Metal-compatible wrapper for a native XLM-RoBERTa / RoBERTa encoder."""


class XLMRobertaEmbeddingBackend:
    """Runtime backend for one loaded XLM-RoBERTa / RoBERTa encoder."""

    pooling_policy = _POOLING_POLICY

    def __init__(self, model: XLMRobertaSequenceModel) -> None:
        self.model = model

    def forward_sequence_hidden_states(
        self,
        input_ids: mx.array,
        *,
        segment_lengths: list[int] | None = None,
        model_label: str = "encoder-embedding",
    ) -> mx.array:
        return forward_packed_encoder_segments(
            self.model.model,
            input_ids,
            segment_lengths=segment_lengths,
            model_label=model_label,
        )


class XLMRobertaEmbeddingFamily:
    """Dense XLM-RoBERTa / RoBERTa / BGE-M3 encoder family."""

    @staticmethod
    def matches_config(model_config: Any) -> bool:
        hf_config = getattr(model_config, "hf_config", None)
        architectures: list[str] = []
        for source in (model_config, hf_config):
            values = getattr(source, "architectures", None)
            if isinstance(values, (list, tuple)):
                architectures.extend(str(arch) for arch in values)
        if any(arch in _ARCHITECTURES for arch in architectures):
            return True
        model_type = getattr(hf_config, "model_type", None)
        if model_type is None:
            return False
        normalized = str(model_type).replace("_", "-")
        return normalized in {value.replace("_", "-") for value in _MODEL_TYPES}

    @staticmethod
    def pooling_policy() -> EncoderPoolingPolicy:
        return _POOLING_POLICY

    @classmethod
    def load(
        cls,
        model_name: str,
        *,
        tokenizer_config: dict[str, Any] | None = None,
        lazy: bool = False,
    ) -> tuple[XLMRobertaSequenceModel, Any, XLMRobertaEmbeddingBackend]:
        raw_model, tokenizer = load_encoder_model(
            model_name,
            tokenizer_config=tokenizer_config,
            lazy=lazy,
        )
        model = XLMRobertaSequenceModel(raw_model)
        return model, tokenizer, XLMRobertaEmbeddingBackend(model)

    @classmethod
    def from_loaded_model(cls, model: Any) -> XLMRobertaEmbeddingBackend | None:
        if isinstance(model, XLMRobertaSequenceModel):
            return XLMRobertaEmbeddingBackend(model)
        return None
