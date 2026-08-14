# SPDX-License-Identifier: Apache-2.0
"""Native MLX XLM-RoBERTa / RoBERTa encoder family."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from vllm_metal.pytorch_backend.tensor_bridge import TORCH_TO_MLX_DTYPE
from vllm_metal.v1.pooling.backends.encoder.runtime import MetalEncoderPoolingBackend
from vllm_metal.v1.pooling.validation import PoolingConfigView

_MODEL_TYPES = frozenset({"xlm-roberta", "roberta"})
_ARCHITECTURES = frozenset({"XLMRobertaModel", "RobertaModel"})


@dataclass(frozen=True, slots=True)
class XLMRobertaArgs:
    model_type: str = "xlm-roberta"
    hidden_size: int = 768
    num_hidden_layers: int = 12
    intermediate_size: int = 3072
    num_attention_heads: int = 12
    max_position_embeddings: int = 512
    vocab_size: int = 250002
    type_vocab_size: int = 1
    layer_norm_eps: float = 1e-5
    pad_token_id: int = 1

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> XLMRobertaArgs:
        names = {field.name for field in fields(cls)}
        return cls(**{name: config[name] for name in names if name in config})


class XLMRobertaEmbeddings(nn.Module):
    def __init__(self, args: XLMRobertaArgs) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        self.position_embeddings = nn.Embedding(
            args.max_position_embeddings,
            args.hidden_size,
        )
        self.token_type_embeddings = nn.Embedding(
            args.type_vocab_size,
            args.hidden_size,
        )
        self.LayerNorm = nn.LayerNorm(args.hidden_size, eps=args.layer_norm_eps)
        self.padding_idx = args.pad_token_id

    def __call__(
        self,
        input_ids: mx.array,
        token_type_ids: mx.array | None = None,
        position_ids: mx.array | None = None,
    ) -> mx.array:
        if token_type_ids is None:
            token_type_ids = mx.zeros(input_ids.shape, dtype=mx.int32)
        if position_ids is None:
            position_ids = self._position_ids(input_ids)
        return self.LayerNorm(
            self.word_embeddings(input_ids)
            + self.token_type_embeddings(token_type_ids)
            + self.position_embeddings(position_ids)
        )

    def _position_ids(self, input_ids: mx.array) -> mx.array:
        mask = (input_ids != self.padding_idx).astype(mx.int32)
        return mx.cumsum(mask, axis=1) * mask + self.padding_idx


class XLMRobertaSelfAttention(nn.Module):
    def __init__(self, args: XLMRobertaArgs) -> None:
        super().__init__()
        if args.hidden_size % args.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size={args.hidden_size} must divide by "
                f"num_attention_heads={args.num_attention_heads}."
            )
        self.num_heads = args.num_attention_heads
        self.head_dim = args.hidden_size // args.num_attention_heads
        self.hidden_size = args.hidden_size
        self.query = nn.Linear(args.hidden_size, args.hidden_size)
        self.key = nn.Linear(args.hidden_size, args.hidden_size)
        self.value = nn.Linear(args.hidden_size, args.hidden_size)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array,
    ) -> mx.array:
        query = self._split_heads(self.query(hidden_states))
        key = self._split_heads(self.key(hidden_states))
        value = self._split_heads(self.value(hidden_states))
        scores = (query @ key.transpose(0, 1, 3, 2)) / math.sqrt(self.head_dim)
        probs = mx.softmax(scores + attention_mask, axis=-1)
        context = probs @ value
        batch, _, seq_len, _ = context.shape
        return context.transpose(0, 2, 1, 3).reshape(
            batch,
            seq_len,
            self.hidden_size,
        )

    def _split_heads(self, tensor: mx.array) -> mx.array:
        batch, seq_len, _ = tensor.shape
        return tensor.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(
            0,
            2,
            1,
            3,
        )


class XLMRobertaSelfOutput(nn.Module):
    def __init__(self, args: XLMRobertaArgs) -> None:
        super().__init__()
        self.dense = nn.Linear(args.hidden_size, args.hidden_size)
        self.LayerNorm = nn.LayerNorm(args.hidden_size, eps=args.layer_norm_eps)

    def __call__(self, hidden_states: mx.array, residual: mx.array) -> mx.array:
        return self.LayerNorm(self.dense(hidden_states) + residual)


class XLMRobertaAttention(nn.Module):
    def __init__(self, args: XLMRobertaArgs) -> None:
        super().__init__()
        self.self = XLMRobertaSelfAttention(args)
        self.output = XLMRobertaSelfOutput(args)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array,
    ) -> mx.array:
        return self.output(
            self.self(hidden_states, attention_mask),
            hidden_states,
        )


class XLMRobertaIntermediate(nn.Module):
    def __init__(self, args: XLMRobertaArgs) -> None:
        super().__init__()
        self.dense = nn.Linear(args.hidden_size, args.intermediate_size)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return nn.gelu(self.dense(hidden_states))


class XLMRobertaOutput(nn.Module):
    def __init__(self, args: XLMRobertaArgs) -> None:
        super().__init__()
        self.dense = nn.Linear(args.intermediate_size, args.hidden_size)
        self.LayerNorm = nn.LayerNorm(args.hidden_size, eps=args.layer_norm_eps)

    def __call__(self, hidden_states: mx.array, residual: mx.array) -> mx.array:
        return self.LayerNorm(self.dense(hidden_states) + residual)


class XLMRobertaLayer(nn.Module):
    def __init__(self, args: XLMRobertaArgs) -> None:
        super().__init__()
        self.attention = XLMRobertaAttention(args)
        self.intermediate = XLMRobertaIntermediate(args)
        self.output = XLMRobertaOutput(args)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array,
    ) -> mx.array:
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        return self.output(intermediate_output, attention_output)


class XLMRobertaEncoder(nn.Module):
    def __init__(self, args: XLMRobertaArgs) -> None:
        super().__init__()
        self.layer = [XLMRobertaLayer(args) for _ in range(args.num_hidden_layers)]

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array,
    ) -> mx.array:
        for layer in self.layer:
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states


class XLMRobertaModel(nn.Module):
    """Encoder-only XLM-RoBERTa/RoBERTa backbone returning hidden states."""

    def __init__(self, args: XLMRobertaArgs) -> None:
        super().__init__()
        self.args = args
        self.config = args
        self.embeddings = XLMRobertaEmbeddings(args)
        self.encoder = XLMRobertaEncoder(args)

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        if attention_mask is None:
            attention_mask = mx.ones(input_ids.shape, dtype=mx.int32)
        hidden_states = self.embeddings(input_ids)
        return self.encoder(
            hidden_states, self._extended_attention_mask(attention_mask)
        )

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        return {
            key: value
            for key, value in weights.items()
            if not key.endswith(".position_ids") and not key.startswith("pooler.")
        }

    def _extended_attention_mask(self, attention_mask: mx.array) -> mx.array:
        mask = attention_mask.astype(mx.float32)
        mask = mx.expand_dims(mx.expand_dims(mask, 1), 1)
        return (1.0 - mask) * -1e4


def supports_xlm_roberta_encoder(model_config: Any) -> bool:
    hf_config = model_config.hf_config
    architectures = tuple(str(value) for value in hf_config.architectures or ())
    model_type = str(hf_config.model_type).replace("_", "-")
    if architectures:
        return any(architecture in _ARCHITECTURES for architecture in architectures)
    return model_type in _MODEL_TYPES


def load_xlm_roberta_backend(
    model_config: Any,
) -> tuple[Any, Any, dict[str, Any], MetalEncoderPoolingBackend]:
    hf_config = model_config.hf_config
    if model_config.quantization is not None:
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

    model_path = Path(model_config.model)
    if not model_path.exists():
        model_path = Path(
            snapshot_download(
                repo_id=model_config.model,
                revision=model_config.revision,
            )
        )

    weight_files = sorted(model_path.glob("model*.safetensors"))
    if not weight_files:
        weight_files = sorted(model_path.glob("*.safetensors"))
    if not weight_files:
        raise FileNotFoundError(f"No safetensors found in {model_path}.")

    config = hf_config.to_dict()
    target_dtype = TORCH_TO_MLX_DTYPE[model_config.dtype]
    args = XLMRobertaArgs.from_config(config)
    model = XLMRobertaModel(args)
    weights: dict[str, mx.array] = {}
    for weight_file in weight_files:
        weights.update(mx.load(str(weight_file)))
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
    pooling_backend = MetalEncoderPoolingBackend(
        PoolingConfigView(model_config),
        model,
    )
    return model, tokenizer, asdict(args), pooling_backend
