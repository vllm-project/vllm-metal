# SPDX-License-Identifier: Apache-2.0
"""MLX model classes for Gemma4 MTP assistant checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models import gemma4_text
from mlx_lm.models.base import BaseModelArgs

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

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "Gemma4 MTP assistant forward is not wired on Metal yet."
        )

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
