# SPDX-License-Identifier: Apache-2.0
"""Encoder embedding adapter via optional mlx-embeddings (#589 PR1).

Owns load, bidirectional segment forward, CLS pooling defaults, and paged
attention patch-skipping so encoder special-cases stay in one place.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import mlx.core as mx
import torch
from huggingface_hub import hf_hub_download

from vllm_metal.pytorch_backend.tensor_bridge import torch_to_mlx

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
_BGE_M3_SPARSE_MODEL_ID = "mlx-community/bge-m3-mlx-8bit"
_BGE_M3_SPARSE_HEAD_REPO_ID = "BAAI/bge-m3"
_BGE_M3_SPARSE_HEAD_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
_BGE_M3_SPARSE_HEAD_FILENAME = "sparse_linear.pt"


class BgeM3SparseHead:
    """Immutable MLX sparse lexical head owned by the encoder adapter."""

    def __init__(
        self,
        *,
        weight: mx.array,
        bias: mx.array,
        bos_token_id: int,
        eos_token_id: int,
    ) -> None:
        if weight.ndim != 2 or tuple(weight.shape[:1]) != (1,):
            raise ValueError(
                "BGE-M3 sparse head weight must have shape [1, hidden], "
                f"got {weight.shape}."
            )
        if bias.ndim != 1 or tuple(bias.shape) != (1,):
            raise ValueError(
                f"BGE-M3 sparse head bias must have shape [1], got {bias.shape}."
            )
        self._weight = weight.astype(mx.float32)
        self._bias = bias.astype(mx.float32)
        self._bos_token_id = bos_token_id
        self._eos_token_id = eos_token_id

    def supports_hidden_size(self, hidden_size: int) -> bool:
        return int(self._weight.shape[1]) == hidden_size

    def project_token_logits(self, hidden_states: mx.array) -> mx.array:
        """Apply the sparse linear projection to packed token states once."""
        if hidden_states.ndim != 2:
            raise ValueError(
                "BGE-M3 sparse pooling expected hidden states with shape "
                f"[tokens, hidden], got {hidden_states.shape}."
            )
        if not self.supports_hidden_size(int(hidden_states.shape[1])):
            raise ValueError(
                "BGE-M3 sparse head hidden size does not match encoder output; "
                f"head={self._weight.shape[1]}, hidden={hidden_states.shape[1]}."
            )
        return mx.squeeze(
            hidden_states.astype(mx.float32) @ self._weight.T + self._bias,
            axis=-1,
        )

    def filter_token_weights(
        self,
        token_logits: mx.array,
        *,
        token_ids: list[int],
        use_activation: bool,
    ) -> mx.array:
        """Apply activation and remove matching boundary BOS/EOS rows."""
        if token_logits.ndim != 1:
            raise ValueError(
                "BGE-M3 sparse pooling expected token logits with shape "
                f"[tokens], got {token_logits.shape}."
            )
        if int(token_logits.shape[0]) != len(token_ids):
            raise ValueError(
                "BGE-M3 sparse pooling token IDs must align with hidden states; "
                f"got {len(token_ids)} IDs for {token_logits.shape[0]} rows."
            )

        if use_activation:
            token_logits = mx.maximum(
                token_logits,
                mx.array(0.0, dtype=mx.float32),
            )

        retained_start = 1 if token_ids[0] == self._bos_token_id else 0
        retained_end = (
            len(token_ids) - 1
            if token_ids[-1] == self._eos_token_id
            else len(token_ids)
        )
        retained_indices = list(range(retained_start, retained_end))
        if not retained_indices:
            return mx.zeros((0,), dtype=mx.float32)
        return token_logits[mx.array(retained_indices, dtype=mx.int32)]


def _load_bge_m3_sparse_head(model_config: Any) -> BgeM3SparseHead:
    """Load the official BGE-M3 sparse head from a fixed upstream revision."""
    hidden_size = getattr(model_config, "hidden_size", None)
    bos_token_id = getattr(model_config, "bos_token_id", None)
    eos_token_id = getattr(model_config, "eos_token_id", None)
    if not all(
        isinstance(value, int) for value in (hidden_size, bos_token_id, eos_token_id)
    ):
        raise ValueError(
            "BGE-M3 sparse pooling requires integer hidden_size, bos_token_id, "
            "and eos_token_id in the encoder config."
        )

    head_path = hf_hub_download(
        repo_id=_BGE_M3_SPARSE_HEAD_REPO_ID,
        filename=_BGE_M3_SPARSE_HEAD_FILENAME,
        revision=_BGE_M3_SPARSE_HEAD_REVISION,
    )
    state = torch.load(head_path, map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping):
        raise ValueError("BGE-M3 sparse_linear.pt must contain a state mapping.")
    weight = state.get("weight")
    bias = state.get("bias")
    if not isinstance(weight, torch.Tensor) or not isinstance(bias, torch.Tensor):
        raise ValueError(
            "BGE-M3 sparse_linear.pt must contain tensor 'weight' and 'bias'."
        )
    if not weight.is_floating_point() or not bias.is_floating_point():
        raise ValueError(
            "BGE-M3 sparse_linear.pt weight and bias must be floating point."
        )
    if not torch.isfinite(weight).all() or not torch.isfinite(bias).all():
        raise ValueError("BGE-M3 sparse_linear.pt weight and bias must be finite.")
    if tuple(weight.shape) != (1, hidden_size) or tuple(bias.shape) != (1,):
        raise ValueError(
            "BGE-M3 sparse_linear.pt shape does not match the encoder; "
            f"weight={tuple(weight.shape)}, bias={tuple(bias.shape)}, "
            f"hidden_size={hidden_size}."
        )
    return BgeM3SparseHead(
        weight=torch_to_mlx(weight.float()),
        bias=torch_to_mlx(bias.float()),
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
    )


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
    """Weight wrapper exposing a Metal-compatible ``.model`` body."""

    def __init__(
        self,
        model: Any,
        *,
        sparse_head: BgeM3SparseHead | None = None,
    ) -> None:
        self._model = model
        self.config = getattr(model, "config", None)
        self.args = self.config
        self.model = _EncoderSequenceBody(model)
        self.sparse_head = sparse_head

    def modules(self) -> list[Any]:
        """Expose child modules for vLLM engine cleanup hooks."""
        inner_modules = getattr(self._model, "modules", None)
        if callable(inner_modules):
            return list(inner_modules())
        return [self._model]


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


class EncoderEmbeddingAdapter:
    """Single owner for mlx-embeddings encoder embedding behavior on Metal."""

    skip_paged_attention_patch = True
    default_sequence_pooling_type = "CLS"
    allowed_sequence_pooling_types = _ENCODER_SEQUENCE_POOLING

    def __init__(
        self,
        model: MlxEmbeddingsEncoderModel,
        *,
        sparse_head: BgeM3SparseHead | None = None,
    ) -> None:
        self.model = model
        self.sparse_head = sparse_head or model.sparse_head

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
        """True when Metal should load weights through mlx-embeddings."""
        return cls.matches_config(model_config)

    @staticmethod
    def requests_token_classify(model_config: Any | None) -> bool:
        pooler_config = getattr(model_config, "pooler_config", None)
        return getattr(pooler_config, "task", None) == "token_classify"

    @staticmethod
    def supports_sparse_model_config(model_config: Any | None) -> bool:
        return getattr(model_config, "model", None) == _BGE_M3_SPARSE_MODEL_ID

    @property
    def supports_token_classify(self) -> bool:
        return self.sparse_head is not None

    @classmethod
    def load(
        cls,
        model_name: str,
        *,
        model_config: Any | None = None,
        tokenizer_config: dict[str, Any] | None = None,
        lazy: bool = False,
    ) -> tuple[MlxEmbeddingsEncoderModel, Any, EncoderEmbeddingAdapter]:
        """Load an encoder embedding checkpoint and return model + adapter."""
        load_sparse_head = cls.requests_token_classify(model_config)
        if load_sparse_head and not cls.supports_sparse_model_config(model_config):
            raise NotImplementedError(
                "BGE-M3 sparse token classification currently supports only "
                f"{_BGE_M3_SPARSE_MODEL_ID!r}."
            )

        mlx_embeddings_load = _import_mlx_embeddings_load()
        raw_model, tokenizer = mlx_embeddings_load(
            model_name,
            tokenizer_config=dict(tokenizer_config or {}),
            lazy=lazy,
        )
        sparse_head = (
            _load_bge_m3_sparse_head(getattr(model_config, "hf_config", None))
            if load_sparse_head
            else None
        )
        model = MlxEmbeddingsEncoderModel(raw_model, sparse_head=sparse_head)
        return model, tokenizer, cls(model)

    @classmethod
    def from_loaded_model(cls, model: Any) -> EncoderEmbeddingAdapter | None:
        """Return an adapter when ``model`` is the encoder weight wrapper."""
        if isinstance(model, MlxEmbeddingsEncoderModel):
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

    def sparse_token_logits(self, hidden_states: mx.array) -> mx.array:
        return self._require_sparse_head().project_token_logits(hidden_states)

    def filter_sparse_token_weights(
        self,
        token_logits: mx.array,
        *,
        token_ids: list[int],
        use_activation: bool,
    ) -> mx.array:
        return self._require_sparse_head().filter_token_weights(
            token_logits,
            token_ids=token_ids,
            use_activation=use_activation,
        )

    def _require_sparse_head(self) -> BgeM3SparseHead:
        if self.sparse_head is None:
            raise NotImplementedError(
                "BGE-M3 token_classify requires a loaded sparse head."
            )
        return self.sparse_head


# Compatibility aliases used by older call sites / tests.
def is_encoder_embedding_architecture(architecture: str) -> bool:
    return EncoderEmbeddingAdapter.matches_architecture(architecture)


def is_encoder_embedding_model_type(model_type: str | None) -> bool:
    return EncoderEmbeddingAdapter.matches_model_type(model_type)


def is_encoder_embedding_config(model_config: Any) -> bool:
    return EncoderEmbeddingAdapter.matches_config(model_config)


def requires_mlx_embeddings_load(model_config: Any) -> bool:
    return EncoderEmbeddingAdapter.requires_load(model_config)


def load_mlx_embeddings_model(
    model_name: str,
    *,
    tokenizer_config: dict[str, Any] | None = None,
    lazy: bool = False,
) -> tuple[MlxEmbeddingsEncoderModel, Any]:
    model, tokenizer, _adapter = EncoderEmbeddingAdapter.load(
        model_name,
        tokenizer_config=tokenizer_config,
        lazy=lazy,
    )
    return model, tokenizer
