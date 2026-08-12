# SPDX-License-Identifier: Apache-2.0
"""Encoder embedding adapter via native Apache-2.0 MLX XLM-RoBERTa.

Owns load, bidirectional segment forward, CLS pooling defaults, and paged
attention patch-skipping so encoder special-cases stay in one place.

Does **not** depend on GPLv3 ``mlx-embeddings``.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from vllm_metal.v1.encoder.loader import load_encoder_model
from vllm_metal.v1.encoder.xlm_roberta import XLMRobertaModel

_ENCODER_EMBEDDING_ARCHITECTURES = frozenset(
    {
        "XLMRobertaModel",
        "RobertaModel",
        "RobertaEmbeddingModel",
        "BgeM3EmbeddingModel",
    }
)
_ENCODER_EMBEDDING_MODEL_TYPES = frozenset(
    {
        "xlm-roberta",
        "roberta",
        "xlm_roberta",
    }
)
_ENCODER_SEQUENCE_POOLING = (None, "CLS", "LAST")


class _EncoderSequenceBody:
    """Callable transformer body returning ``[1, tokens, hidden]`` states."""

    def __init__(self, model: XLMRobertaModel) -> None:
        self._model = model

    def __call__(self, input_ids: mx.array, cache: Any = None) -> mx.array:
        del cache  # Encoder embeddings are bidirectional and cache-free.
        if input_ids.ndim == 1:
            input_ids = input_ids.reshape(1, -1)
        attention_mask = mx.ones(input_ids.shape, dtype=mx.int32)
        return self._model(input_ids, attention_mask=attention_mask)


class NativeEncoderModel:
    """Weight wrapper exposing a Metal-compatible ``.model`` body."""

    def __init__(self, model: XLMRobertaModel) -> None:
        self._model = model
        self.config = getattr(model, "config", None)
        self.args = self.config
        self.model = _EncoderSequenceBody(model)

    def modules(self) -> list[Any]:
        """Expose child modules for vLLM engine cleanup hooks."""
        inner_modules = getattr(self._model, "modules", None)
        if callable(inner_modules):
            return list(inner_modules())
        return [self._model]


# Back-compat alias used by older tests/docs wording.
MlxEmbeddingsEncoderModel = NativeEncoderModel


class EncoderEmbeddingAdapter:
    """Single owner for native encoder embedding behavior on Metal."""

    skip_paged_attention_patch = True
    default_sequence_pooling_type = "CLS"
    allowed_sequence_pooling_types = _ENCODER_SEQUENCE_POOLING

    def __init__(self, model: NativeEncoderModel) -> None:
        self.model = model

    @staticmethod
    def matches_architecture(architecture: str) -> bool:
        return architecture in _ENCODER_EMBEDDING_ARCHITECTURES

    @staticmethod
    def matches_model_type(model_type: str | None) -> bool:
        if model_type is None:
            return False
        return model_type.replace("_", "-") in {
            value.replace("_", "-") for value in _ENCODER_EMBEDDING_MODEL_TYPES
        }

    @classmethod
    def matches_config(cls, model_config: Any) -> bool:
        """Return whether vLLM resolved this as an encoder embedding checkpoint."""
        hf_config = getattr(model_config, "hf_config", None)
        architectures: list[str] = []
        for source in (model_config, hf_config):
            values = getattr(source, "architectures", None)
            if isinstance(values, (list, tuple)):
                architectures.extend(str(arch) for arch in values)
        if any(cls.matches_architecture(arch) for arch in architectures):
            return True
        model_type = getattr(hf_config, "model_type", None)
        return cls.matches_model_type(
            str(model_type) if model_type is not None else None
        )

    @classmethod
    def requires_load(cls, model_config: Any) -> bool:
        """True when Metal should load weights through the native encoder path."""
        return cls.matches_config(model_config)

    @classmethod
    def load(
        cls,
        model_name: str,
        *,
        tokenizer_config: dict[str, Any] | None = None,
        lazy: bool = False,
    ) -> tuple[NativeEncoderModel, Any, EncoderEmbeddingAdapter]:
        """Load a native encoder embedding checkpoint and return model + adapter."""
        raw_model, tokenizer = load_encoder_model(
            model_name,
            tokenizer_config=tokenizer_config,
            lazy=lazy,
        )
        model = NativeEncoderModel(raw_model)
        return model, tokenizer, cls(model)

    @classmethod
    def from_loaded_model(cls, model: Any) -> EncoderEmbeddingAdapter | None:
        """Return an adapter when ``model`` is the encoder weight wrapper."""
        if isinstance(model, NativeEncoderModel):
            return cls(model)
        return None

    def forward_sequence_hidden_states(
        self,
        input_ids: mx.array,
        *,
        segment_lengths: list[int] | None = None,
        model_label: str = "encoder-embedding",
    ) -> mx.array:
        """Forward each packed segment independently (bidirectional)."""
        body = self.model.model
        flat = input_ids.reshape(-1)
        total_tokens = int(flat.shape[0])
        if not segment_lengths:
            segment_lengths = [total_tokens]
        if sum(segment_lengths) != total_tokens:
            raise ValueError(
                "Encoder pooling segment_lengths "
                f"{segment_lengths!r} do not cover input tokens "
                f"{total_tokens} for model={model_label}."
            )

        parts: list[mx.array] = []
        offset = 0
        for length in segment_lengths:
            segment = flat[offset : offset + length].reshape(1, length)
            parts.append(body(segment))
            offset += length
        if len(parts) == 1:
            return parts[0]
        return mx.concatenate(parts, axis=1)


def is_encoder_embedding_architecture(architecture: str) -> bool:
    return EncoderEmbeddingAdapter.matches_architecture(architecture)


def is_encoder_embedding_model_type(model_type: str | None) -> bool:
    return EncoderEmbeddingAdapter.matches_model_type(model_type)


def is_encoder_embedding_config(model_config: Any) -> bool:
    return EncoderEmbeddingAdapter.matches_config(model_config)


def requires_mlx_embeddings_load(model_config: Any) -> bool:
    """Deprecated alias: native encoder load gate."""
    return EncoderEmbeddingAdapter.requires_load(model_config)


def load_mlx_embeddings_model(
    model_name: str,
    *,
    tokenizer_config: dict[str, Any] | None = None,
    lazy: bool = False,
) -> tuple[NativeEncoderModel, Any]:
    """Deprecated alias for native encoder load (no mlx-embeddings)."""
    model, tokenizer, _adapter = EncoderEmbeddingAdapter.load(
        model_name,
        tokenizer_config=tokenizer_config,
        lazy=lazy,
    )
    return model, tokenizer
