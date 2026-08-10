# SPDX-License-Identifier: Apache-2.0
"""Encoder embedding load path via optional mlx-embeddings (#589 PR1)."""

from __future__ import annotations

from typing import Any

import mlx.core as mx

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


def is_encoder_embedding_architecture(architecture: str) -> bool:
    return architecture in _ENCODER_EMBEDDING_ARCHITECTURES


def is_encoder_embedding_model_type(model_type: str | None) -> bool:
    if model_type is None:
        return False
    return model_type.replace("_", "-") in {
        value.replace("_", "-") for value in _ENCODER_EMBEDDING_MODEL_TYPES
    }


def is_encoder_embedding_config(model_config: Any) -> bool:
    """Return whether vLLM resolved this as an encoder embedding checkpoint."""
    hf_config = getattr(model_config, "hf_config", None)
    architectures: list[str] = []
    for source in (model_config, hf_config):
        values = getattr(source, "architectures", None)
        if isinstance(values, (list, tuple)):
            architectures.extend(str(arch) for arch in values)
    if any(is_encoder_embedding_architecture(arch) for arch in architectures):
        return True
    model_type = getattr(hf_config, "model_type", None)
    return is_encoder_embedding_model_type(
        str(model_type) if model_type is not None else None
    )


def requires_mlx_embeddings_load(model_config: Any) -> bool:
    """True when Metal should load weights through mlx-embeddings."""
    return is_encoder_embedding_config(model_config)


class _EncoderSequenceBody:
    """Callable transformer body returning ``[1, tokens, hidden]`` states."""

    def __init__(self, model: Any) -> None:
        self._model = model

    def __call__(self, input_ids: mx.array, cache: Any = None) -> mx.array:
        del cache  # Encoder embeddings are bidirectional and cache-free.
        if input_ids.ndim == 1:
            input_ids = input_ids.reshape(1, -1)
        attention_mask = mx.ones(input_ids.shape, dtype=mx.int32)
        output = self._model(input_ids, attention_mask=attention_mask)
        hidden_states = getattr(output, "last_hidden_state", None)
        if hidden_states is None:
            raise ValueError(
                "mlx-embeddings encoder forward did not return last_hidden_state; "
                f"got {type(output)!r}."
            )
        return hidden_states


class MlxEmbeddingsEncoderModel:
    """Thin adapter so Metal pooling can use mlx-embeddings models."""

    is_mlx_embeddings_encoder = True

    def __init__(self, model: Any) -> None:
        self._model = model
        self.config = getattr(model, "config", None)
        self.args = self.config
        self.model = _EncoderSequenceBody(model)


def _import_mlx_embeddings_load() -> Any:
    try:
        from mlx_embeddings import load as mlx_embeddings_load
    except ImportError as exc:
        raise ImportError(
            "Loading encoder embedding models such as XLM-RoBERTa / BGE-M3 "
            "requires the optional 'mlx-embeddings' package. Install it with: "
            'pip install "vllm-metal[embeddings]"'
        ) from exc
    return mlx_embeddings_load


def load_mlx_embeddings_model(
    model_name: str,
    *,
    tokenizer_config: dict[str, Any] | None = None,
    lazy: bool = False,
) -> tuple[MlxEmbeddingsEncoderModel, Any]:
    """Load an encoder embedding checkpoint through mlx-embeddings."""
    mlx_embeddings_load = _import_mlx_embeddings_load()
    model, tokenizer = mlx_embeddings_load(
        model_name,
        tokenizer_config=dict(tokenizer_config or {}),
        lazy=lazy,
    )
    return MlxEmbeddingsEncoderModel(model), tokenizer
