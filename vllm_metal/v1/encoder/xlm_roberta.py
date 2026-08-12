# SPDX-License-Identifier: Apache-2.0
"""Native MLX XLM-RoBERTa / RoBERTa encoder (Apache-2.0, no mlx-embeddings).

Architecture follows the public HuggingFace Bert/XLM-RoBERTa layout so MLX
quantized checkpoints (e.g. mlx-community BGE-M3) load by matching parameter
paths. Implemented in-tree to keep vllm-metal Apache-2.0 compatible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn


@dataclass
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
    hidden_dropout_prob: float = 0.0
    attention_probs_dropout_prob: float = 0.0
    pad_token_id: int = 1
    bos_token_id: int = 0
    eos_token_id: int = 2
    add_pooling_layer: bool = True
    position_embedding_type: str = "absolute"

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> XLMRobertaArgs:
        fields = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in params.items() if k in fields})


class XLMRobertaEmbeddings(nn.Module):
    def __init__(self, args: XLMRobertaArgs) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        self.position_embeddings = nn.Embedding(
            args.max_position_embeddings, args.hidden_size
        )
        self.token_type_embeddings = nn.Embedding(
            args.type_vocab_size, args.hidden_size
        )
        self.LayerNorm = nn.LayerNorm(args.hidden_size, eps=args.layer_norm_eps)
        self.padding_idx = args.pad_token_id

    def _position_ids(self, input_ids: mx.array) -> mx.array:
        # HuggingFace XLM-RoBERTa: cumulative positions skipping pads, then + pad_idx.
        mask = (input_ids != self.padding_idx).astype(mx.int32)
        incremental = mx.cumsum(mask, axis=1) * mask
        return incremental + self.padding_idx

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

        embeddings = (
            self.word_embeddings(input_ids)
            + self.token_type_embeddings(token_type_ids)
            + self.position_embeddings(position_ids)
        )
        return self.LayerNorm(embeddings)


class XLMRobertaSelfAttention(nn.Module):
    def __init__(self, args: XLMRobertaArgs) -> None:
        super().__init__()
        if args.hidden_size % args.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size {args.hidden_size} not divisible by "
                f"num_attention_heads {args.num_attention_heads}"
            )
        self.num_attention_heads = args.num_attention_heads
        self.attention_head_size = args.hidden_size // args.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(args.hidden_size, self.all_head_size)
        self.key = nn.Linear(args.hidden_size, self.all_head_size)
        self.value = nn.Linear(args.hidden_size, self.all_head_size)

    def _transpose_for_scores(self, x: mx.array) -> mx.array:
        batch, seq, _ = x.shape
        x = x.reshape(batch, seq, self.num_attention_heads, self.attention_head_size)
        return x.transpose(0, 2, 1, 3)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        query = self._transpose_for_scores(self.query(hidden_states))
        key = self._transpose_for_scores(self.key(hidden_states))
        value = self._transpose_for_scores(self.value(hidden_states))

        scale = 1.0 / math.sqrt(self.attention_head_size)
        scores = (query @ key.transpose(0, 1, 3, 2)) * scale
        if attention_mask is not None:
            scores = scores + attention_mask
        probs = mx.softmax(scores, axis=-1)
        context = probs @ value
        context = context.transpose(0, 2, 1, 3)
        batch, seq, _, _ = context.shape
        return context.reshape(batch, seq, self.all_head_size)


class XLMRobertaSelfOutput(nn.Module):
    def __init__(self, args: XLMRobertaArgs) -> None:
        super().__init__()
        self.dense = nn.Linear(args.hidden_size, args.hidden_size)
        self.LayerNorm = nn.LayerNorm(args.hidden_size, eps=args.layer_norm_eps)

    def __call__(self, hidden_states: mx.array, input_tensor: mx.array) -> mx.array:
        return self.LayerNorm(self.dense(hidden_states) + input_tensor)


class XLMRobertaAttention(nn.Module):
    def __init__(self, args: XLMRobertaArgs) -> None:
        super().__init__()
        self.self = XLMRobertaSelfAttention(args)
        self.output = XLMRobertaSelfOutput(args)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        self_out = self.self(hidden_states, attention_mask=attention_mask)
        return self.output(self_out, hidden_states)


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

    def __call__(self, hidden_states: mx.array, input_tensor: mx.array) -> mx.array:
        return self.LayerNorm(self.dense(hidden_states) + input_tensor)


class XLMRobertaLayer(nn.Module):
    def __init__(self, args: XLMRobertaArgs) -> None:
        super().__init__()
        self.attention = XLMRobertaAttention(args)
        self.intermediate = XLMRobertaIntermediate(args)
        self.output = XLMRobertaOutput(args)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        attention_out = self.attention(hidden_states, attention_mask=attention_mask)
        intermediate = self.intermediate(attention_out)
        return self.output(intermediate, attention_out)


class XLMRobertaEncoder(nn.Module):
    def __init__(self, args: XLMRobertaArgs) -> None:
        super().__init__()
        self.layer = [XLMRobertaLayer(args) for _ in range(args.num_hidden_layers)]

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        for layer in self.layer:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)
        return hidden_states


class XLMRobertaPooler(nn.Module):
    def __init__(self, args: XLMRobertaArgs) -> None:
        super().__init__()
        self.dense = nn.Linear(args.hidden_size, args.hidden_size)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return mx.tanh(self.dense(hidden_states[:, 0]))


class XLMRobertaModel(nn.Module):
    """Encoder-only XLM-RoBERTa / RoBERTa backbone returning last hidden states."""

    def __init__(self, args: XLMRobertaArgs) -> None:
        super().__init__()
        self.args = args
        self.config = args
        self.embeddings = XLMRobertaEmbeddings(args)
        self.encoder = XLMRobertaEncoder(args)
        self.pooler = XLMRobertaPooler(args) if args.add_pooling_layer else None

    def _extended_attention_mask(self, attention_mask: mx.array) -> mx.array:
        # HF convention: 1 = keep, 0 = mask. Broadcast to [B, 1, 1, S].
        mask = attention_mask.astype(mx.float32)
        mask = mx.expand_dims(mx.expand_dims(mask, 1), 1)
        return (1.0 - mask) * -1e4

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,
        token_type_ids: mx.array | None = None,
    ) -> mx.array:
        if attention_mask is None:
            attention_mask = mx.ones(input_ids.shape, dtype=mx.int32)
        extended = self._extended_attention_mask(attention_mask)
        hidden = self.embeddings(input_ids, token_type_ids=token_type_ids)
        return self.encoder(hidden, attention_mask=extended)

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Drop unused keys that some converters may include."""
        return {
            k: v
            for k, v in weights.items()
            if not k.endswith(".position_ids") and "rotary" not in k
        }
