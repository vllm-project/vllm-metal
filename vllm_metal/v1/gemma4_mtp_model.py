# SPDX-License-Identifier: Apache-2.0
"""MLX model classes for Gemma4 MTP assistant checkpoints."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models import gemma4_text
from mlx_lm.models.base import BaseModelArgs

from vllm_metal.attention.context import get_context
from vllm_metal.attention.impls.sdpa import sdpa_forward

GEMMA4_MTP_DEFAULT_VOCAB_SIZE = 262144
GEMMA4_MTP_DEFAULT_NUM_CENTROIDS = 2048
GEMMA4_MTP_DEFAULT_CENTROID_TOP_K = 32
GEMMA4_MTP_PREPROJECTION_INPUTS = 2


@dataclass
class Gemma4MTPAssistantModelArgs(BaseModelArgs):
    """MLX model args for raw Gemma4 assistant checkpoints."""

    model_type: str = "gemma4_assistant"
    text_config: dict[str, Any] | None = None
    vocab_size: int = GEMMA4_MTP_DEFAULT_VOCAB_SIZE
    backbone_hidden_size: int = 0
    tie_word_embeddings: bool = True
    use_ordered_embeddings: bool = False
    num_centroids: int = GEMMA4_MTP_DEFAULT_NUM_CENTROIDS
    centroid_intermediate_top_k: int = GEMMA4_MTP_DEFAULT_CENTROID_TOP_K

    def __post_init__(self) -> None:
        if self.text_config is None:
            self.text_config = {}
        else:
            self.text_config = dict(self.text_config)
            self.vocab_size = self.text_config.get("vocab_size", self.vocab_size)
        self.text_config.setdefault("vocab_size", self.vocab_size)


class Gemma4MTPMaskedEmbedding(nn.Module):
    """Centroid metadata for sparse Gemma4 assistant logits."""

    def __init__(
        self,
        *,
        hidden_size: int,
        vocab_size: int,
        num_centroids: int,
        centroid_intermediate_top_k: int,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_centroids = num_centroids
        self.centroid_intermediate_top_k = centroid_intermediate_top_k
        self.centroids = nn.Linear(hidden_size, num_centroids, bias=False)
        self.token_ordering = mx.zeros((vocab_size,), dtype=mx.int64)

    @property
    def vocab_size_per_centroid(self) -> int:
        return self.vocab_size // self.num_centroids

    @property
    def num_selected(self) -> int:
        return self.centroid_intermediate_top_k * self.vocab_size_per_centroid

    def _select_and_score(
        self,
        hidden_states: mx.array,
        lm_head_weight: mx.array,
    ) -> tuple[mx.array, mx.array]:
        top_k_indices = mx.argpartition(
            self.centroids(hidden_states),
            kth=-self.centroid_intermediate_top_k,
            axis=-1,
        )[..., -self.centroid_intermediate_top_k :]
        clusters = self.token_ordering.reshape(
            self.num_centroids,
            self.vocab_size_per_centroid,
        )
        selected = clusters[top_k_indices]
        embeddings = lm_head_weight[selected.reshape(-1)].reshape(
            hidden_states.shape[0],
            self.num_selected,
            self.hidden_size,
        )
        logits = mx.einsum("td,tsd->ts", hidden_states, embeddings)
        return logits, selected.reshape(hidden_states.shape[0], -1)

    def get_top_tokens(
        self,
        hidden_states: mx.array,
        lm_head_weight: mx.array,
    ) -> mx.array:
        """Return sparse argmax token ids without materializing full logits."""
        logits, indices = self._select_and_score(hidden_states, lm_head_weight)
        best = mx.argmax(logits, axis=-1, keepdims=True)
        return mx.take_along_axis(indices, best, axis=-1).squeeze(-1)


class Gemma4MTPAssistantBackbone(nn.Module):
    """Q-only Gemma4 assistant backbone."""

    def __init__(self, config: gemma4_text.ModelArgs) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            gemma4_text.DecoderLayer(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class Gemma4MTPAssistantModel(nn.Module):
    """MLX module matching Gemma4 assistant checkpoint keys."""

    def __init__(self, args: Gemma4MTPAssistantModelArgs) -> None:
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.tie_word_embeddings = args.tie_word_embeddings
        text_config = dict(args.text_config or {})
        num_hidden_layers = self._required_positive_int(
            text_config,
            "num_hidden_layers",
        )
        if args.backbone_hidden_size <= 0:
            raise ValueError("Gemma4 MTP assistant requires backbone_hidden_size > 0")

        hidden_size_per_layer_input = text_config.get("hidden_size_per_layer_input")
        if hidden_size_per_layer_input is None:
            text_config["hidden_size_per_layer_input"] = 0
        elif hidden_size_per_layer_input != 0:
            raise ValueError(
                "Gemma4 MTP assistant forward does not support per-layer inputs: "
                "hidden_size_per_layer_input must be 0"
            )
        vocab_size_per_layer_input = text_config.get("vocab_size_per_layer_input")
        if vocab_size_per_layer_input is None:
            text_config["vocab_size_per_layer_input"] = 0
        elif vocab_size_per_layer_input != 0:
            raise ValueError(
                "Gemma4 MTP assistant forward does not support per-layer inputs: "
                "vocab_size_per_layer_input must be 0"
            )

        # All assistant layers are Q-only and read target K/V in the later
        # KV-sharing PR. The assistant config may already declare all layers as
        # KV-shared; if present, it must match the Q-only layer count.
        num_kv_shared_layers = text_config.get("num_kv_shared_layers")
        if (
            num_kv_shared_layers is not None
            and num_kv_shared_layers != num_hidden_layers
        ):
            raise ValueError(
                "Gemma4 MTP assistant num_kv_shared_layers must equal "
                f"num_hidden_layers: num_kv_shared_layers={num_kv_shared_layers}, "
                f"num_hidden_layers={num_hidden_layers}"
            )
        text_config["num_kv_shared_layers"] = num_hidden_layers
        self.text_args = gemma4_text.ModelArgs.from_dict(text_config)
        self.model = Gemma4MTPAssistantBackbone(self.text_args)
        # The pre-projection consumes concatenated target-token embeddings and
        # the previous target/backbone hidden-state feedback, both in backbone
        # hidden size.
        pre_projection_input_size = (
            GEMMA4_MTP_PREPROJECTION_INPUTS * args.backbone_hidden_size
        )
        self.pre_projection = nn.Linear(
            pre_projection_input_size,
            self.text_args.hidden_size,
            bias=False,
        )
        self.post_projection = nn.Linear(
            self.text_args.hidden_size,
            args.backbone_hidden_size,
            bias=False,
        )
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(
                self.text_args.hidden_size,
                self.text_args.vocab_size,
                bias=False,
            )
        if args.use_ordered_embeddings:
            self.masked_embedding = Gemma4MTPMaskedEmbedding(
                hidden_size=self.text_args.hidden_size,
                vocab_size=self.text_args.vocab_size,
                num_centroids=args.num_centroids,
                centroid_intermediate_top_k=args.centroid_intermediate_top_k,
            )
        else:
            self.masked_embedding = None

    @property
    def layers(self) -> list[gemma4_text.DecoderLayer]:
        return self.model.layers

    def __call__(
        self,
        input_ids: mx.array | None,
        *,
        target_hidden_states: mx.array,
        target_input_embeddings: mx.array,
        target_kv_cache: Any,
        target_cache_indices: Sequence[int],
    ) -> tuple[mx.array, mx.array]:
        """Run one Gemma4 MTP assistant draft step over target KV cache.

        ``input_ids`` is accepted for parity with upstream's proposer API; this
        Metal path receives already-scaled target/backbone embeddings from the
        target model because the assistant's own embedding table is draft-dim
        and is kept for logits.
        """
        hidden_states, input_embeddings = self._normalize_forward_inputs(
            target_hidden_states,
            target_input_embeddings,
        )
        if len(target_cache_indices) != len(self.layers):
            raise ValueError(
                "Gemma4 MTP target_cache_indices must match assistant layers: "
                f"indices={len(target_cache_indices)}, layers={len(self.layers)}"
            )

        combined = mx.concatenate([input_embeddings, hidden_states], axis=-1)
        hidden_states = self.pre_projection(combined)
        for layer, cache_idx in zip(self.layers, target_cache_indices, strict=True):
            hidden_states = self._forward_q_only_layer(
                layer,
                hidden_states,
                target_kv_cache=target_kv_cache,
                target_cache_idx=cache_idx,
            )

        draft_hidden_states = self.model.norm(hidden_states)
        backbone_hidden_states = self.post_projection(draft_hidden_states)
        return draft_hidden_states, backbone_hidden_states

    def draft_token_ids(
        self,
        input_ids: mx.array | None,
        *,
        target_hidden_states: mx.array,
        target_input_embeddings: mx.array,
        target_kv_cache: Any,
        target_cache_indices: Sequence[int],
    ) -> mx.array:
        """Return one greedy draft token id per packed request row."""
        draft_hidden_states, _ = self(
            input_ids,
            target_hidden_states=target_hidden_states,
            target_input_embeddings=target_input_embeddings,
            target_kv_cache=target_kv_cache,
            target_cache_indices=target_cache_indices,
        )
        return self.get_top_tokens(draft_hidden_states[0])

    def compute_logits(self, hidden_states: mx.array) -> mx.array:
        """Compute assistant logits from draft-dim hidden states."""
        if self.masked_embedding is not None:
            raise NotImplementedError(
                "Gemma4 MTP masked embeddings expose sparse top-token "
                "selection on Metal; full masked logits are not materialized."
            )
        if self.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(hidden_states)
        else:
            logits = self.lm_head(hidden_states)
        final_softcap = getattr(self.text_args, "final_logit_softcapping", None)
        if final_softcap is not None:
            logits = mx.tanh(logits / final_softcap) * final_softcap
        return logits

    def get_top_tokens(self, hidden_states: mx.array) -> mx.array:
        """Greedy-sample assistant tokens."""
        if self.masked_embedding is not None:
            lm_head_weight = (
                self.model.embed_tokens.weight
                if self.tie_word_embeddings
                else self.lm_head.weight
            )
            return self.masked_embedding.get_top_tokens(
                hidden_states,
                lm_head_weight,
            )
        return mx.argmax(self.compute_logits(hidden_states), axis=-1)

    def _normalize_forward_inputs(
        self,
        target_hidden_states: mx.array,
        target_input_embeddings: mx.array,
    ) -> tuple[mx.array, mx.array]:
        hidden_states = self._ensure_single_batch_rows(
            target_hidden_states,
            "target_hidden_states",
        )
        input_embeddings = self._ensure_single_batch_rows(
            target_input_embeddings,
            "target_input_embeddings",
        )
        if hidden_states.shape != input_embeddings.shape:
            raise ValueError(
                "Gemma4 MTP target hidden states and input embeddings must "
                f"have matching shape, got {hidden_states.shape} and "
                f"{input_embeddings.shape}"
            )
        if hidden_states.shape[-1] != self.args.backbone_hidden_size:
            raise ValueError(
                "Gemma4 MTP target feedback must use backbone hidden size: "
                f"got {hidden_states.shape[-1]}, expected "
                f"{self.args.backbone_hidden_size}"
            )
        return hidden_states, input_embeddings

    def _ensure_single_batch_rows(self, value: mx.array, name: str) -> mx.array:
        ndim = len(value.shape)
        if ndim == 2:
            return value[None, :, :]
        if ndim == 3 and value.shape[0] == 1:
            return value
        raise ValueError(
            f"Gemma4 MTP {name} must be row-major [num_tokens, hidden] or "
            f"single-batch [1, num_tokens, hidden], got shape {value.shape}"
        )

    def _forward_q_only_layer(
        self,
        layer: gemma4_text.DecoderLayer,
        hidden_states: mx.array,
        *,
        target_kv_cache: Any,
        target_cache_idx: int,
    ) -> mx.array:
        ctx = get_context()
        if ctx is None:
            raise RuntimeError(
                "Gemma4 MTP assistant forward requires a paged attention context"
            )

        residual = hidden_states
        hidden_states = layer.input_layernorm(residual)
        hidden_states, _ = sdpa_forward(
            layer.self_attn,
            hidden_states,
            ctx,
            target_kv_cache,
            target_cache_idx,
            read_existing_kv=True,
        )
        hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states = hidden_states + residual
        residual = hidden_states

        if getattr(layer, "enable_moe", False):
            raise NotImplementedError(
                "Gemma4 MTP assistant MoE layers are not supported on Metal"
            )

        hidden_states = layer.pre_feedforward_layernorm(hidden_states)
        hidden_states = layer.mlp(hidden_states)
        hidden_states = layer.post_feedforward_layernorm(hidden_states)
        hidden_states = hidden_states + residual

        layer_scalar = getattr(layer, "layer_scalar", None)
        if layer_scalar is not None:
            hidden_states = hidden_states * layer_scalar
        return hidden_states

    @staticmethod
    def _required_positive_int(config: dict[str, Any], key: str) -> int:
        value = config.get(key)
        if value is None:
            raise ValueError(f"Gemma4 MTP assistant requires {key}")
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"Gemma4 MTP assistant {key} must be an integer")
        if value <= 0:
            raise ValueError(f"Gemma4 MTP assistant requires {key} > 0")
        return value
