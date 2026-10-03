# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The DeepSpec Authors
# Copyright (c) 2026 vLLM Metal contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Qwen3 DSpark checkpoint forward and checkpoint-owned prediction heads.

Adapted from deepseek-ai/DeepSpec (deepspec/modeling/dspark/qwen3/modeling.py,
markov_head.py, and common.py) at 005e03b81cec38b7da6399833d609ee89a2587f2.
The upstream MIT copyright and permission notice are retained above.
The full-context Qwen3 backbone is shared with DFlash. DSpark owns its embedding
and output head and predicts from slot zero, with sequential Markov corrections.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from vllm_metal.v1.dflash import DFlashConfig, DFlashModel
from vllm_metal.v1.draft_checkpoint import COMMON_DRAFT_OPTIONS, load_draft_weights


@dataclass(frozen=True)
class DSparkConfig:
    backbone: DFlashConfig
    markov_rank: int
    enable_confidence_head: bool
    confidence_head_with_markov: bool

    def __post_init__(self) -> None:
        if type(self.markov_rank) is not int or self.markov_rank <= 0:
            raise ValueError("DSpark markov_rank must be a positive integer")
        for name in ("enable_confidence_head", "confidence_head_with_markov"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"DSpark {name} must be a boolean")
        taps = self.backbone.target_layer_ids
        if any(a >= b for a, b in zip(taps, taps[1:], strict=False)):
            raise ValueError("DSpark target_layer_ids must be strictly increasing")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> DSparkConfig:
        if (
            raw.get("architectures") != ["Qwen3DSparkModel"]
            or raw.get("model_type") != "qwen3"
        ):
            raise ValueError("Expected a standalone Qwen3DSparkModel checkpoint")
        if raw.get("markov_head_type") != "vanilla":
            raise ValueError("DSpark requires an explicit vanilla markov_head_type")
        supported = {
            **COMMON_DRAFT_OPTIONS,
            "partial_rotary_factor": 1.0,
            "tie_word_embeddings": False,
            "dflash_query_causal": False,
            "log_snr_conditioning": False,
            "enable_qwen35_gated_q_proj": False,
        }
        for name, expected in supported.items():
            if raw.get(name, expected) != expected:
                raise ValueError(f"Unsupported DSpark {name}: {raw[name]!r}")
        rope = raw.get("rope_parameters")
        if rope is None:
            rope = {}
        if not isinstance(rope, dict) or set(rope) - {"rope_type", "rope_theta"}:
            raise ValueError("DSpark requires full default RoPE")
        if rope.get("rope_type", "default") != "default":
            raise ValueError("DSpark requires full default RoPE")
        if (
            "rope_theta" in raw
            and "rope_theta" in rope
            and raw["rope_theta"] != rope["rope_theta"]
        ):
            raise ValueError("Conflicting DSpark rope_theta values")
        try:
            values = {
                f.name: raw[f.name]
                for f in fields(DFlashConfig)
                if f.name != "rope_theta"
            }
            values["target_layer_ids"] = tuple(values["target_layer_ids"])
            values["rope_theta"] = rope.get("rope_theta", raw.get("rope_theta"))
            config = cls(
                backbone=DFlashConfig(**values),
                markov_rank=raw["markov_rank"],
                enable_confidence_head=raw["enable_confidence_head"],
                confidence_head_with_markov=raw.get(
                    "confidence_head_with_markov", False
                ),
            )
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"Incomplete DSpark checkpoint configuration: {exc}"
            ) from exc
        if (
            raw.get(
                "layer_types", ["full_attention"] * config.backbone.num_hidden_layers
            )
            != ["full_attention"] * config.backbone.num_hidden_layers
        ):
            raise ValueError("Only full-attention DSpark layers are supported")
        if config.enable_confidence_head and "confidence_head_with_markov" not in raw:
            raise ValueError("DSpark requires confidence_head_with_markov")
        return config


class _MarkovHead(nn.Module):
    def __init__(self, vocab_size: int, rank: int) -> None:
        super().__init__()
        self.markov_w1 = nn.Embedding(vocab_size, rank)
        self.markov_w2 = nn.Linear(rank, vocab_size, bias=False)


class _ConfidenceHead(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, 1)

    def __call__(self, features: mx.array) -> mx.array:
        return self.proj(features).squeeze(-1).astype(mx.float32)


class DSparkModel(nn.Module):
    """Stateless greedy DSpark forward with checkpoint-owned prediction heads."""

    def __init__(self, config: DSparkConfig) -> None:
        super().__init__()
        self.config = config
        cfg = config.backbone
        self.backbone = DFlashModel(cfg)
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.markov_head = _MarkovHead(cfg.vocab_size, config.markov_rank)
        self.confidence_head = (
            _ConfidenceHead(
                cfg.hidden_size
                + (config.markov_rank if config.confidence_head_with_markov else 0)
            )
            if config.enable_confidence_head
            else None
        )

    def block_hidden(
        self, anchors: mx.array, features: Sequence[mx.array], *, num_draft_tokens: int
    ) -> mx.array:
        """Return all K block states, including the anchor's prediction at slot 0.

        Inputs are one anchor and K-1 masks. Features cover the preceding prefix
        (excluding the anchor), using the shared HF target-capture convention.
        External token IDs must first pass validate_anchors outside compilation.
        """
        if any(f.dtype != self.embed_tokens.weight.dtype for f in features):
            raise ValueError("DSpark features must use the checkpoint precision")
        return self.backbone(self.block_embeddings(anchors, num_draft_tokens), features)

    def block_embeddings(self, anchors: mx.array, num_draft_tokens: int) -> mx.array:
        """Embed the anchor and K-1 masks for dense or paged block attention."""
        cfg = self.config.backbone
        if (
            type(num_draft_tokens) is not int
            or not 1 <= num_draft_tokens <= cfg.block_size
        ):
            raise ValueError("DSpark requires 1 <= num_draft_tokens <= block_size")
        self.backbone.validate_anchor_metadata(anchors)
        inputs = mx.concatenate(
            [
                anchors.astype(mx.int64)[:, None],
                mx.full(
                    (anchors.shape[0], num_draft_tokens - 1),
                    cfg.mask_token_id,
                    dtype=mx.int64,
                ),
            ],
            axis=1,
        )
        return self.embed_tokens(inputs)

    def greedy_proposal(
        self, hidden: mx.array, anchors: mx.array
    ) -> tuple[mx.array, mx.array, mx.array | None]:
        """Return token IDs, corrected logits, and optional raw confidence logits.

        Each correction and confidence prediction uses the preceding token:
        the target anchor first, then the actual preceding draft prediction.
        Confidence is not calibrated here and does not truncate the block.
        """
        self.backbone.validate_anchor_metadata(anchors)
        self.backbone.validate_embeddings(hidden, 0)
        if hidden.shape[0] != anchors.shape[0]:
            raise ValueError("DSpark requires one anchor per block")
        if hidden.dtype != self.lm_head.weight.dtype:
            raise ValueError("DSpark block states must use the checkpoint precision")
        logits = self.lm_head(hidden)
        previous = anchors.astype(mx.int64)
        tokens, corrected, previous_embeddings = [], [], []
        for i in range(hidden.shape[1]):
            embedding = self.markov_head.markov_w1(previous)
            previous_embeddings.append(embedding)
            step_logits = logits[:, i] + self.markov_head.markov_w2(embedding)
            previous = mx.argmax(step_logits, axis=-1)
            tokens.append(previous)
            corrected.append(step_logits)
        confidence = None
        if self.confidence_head is not None:
            inputs = hidden
            if self.config.confidence_head_with_markov:
                inputs = mx.concatenate(
                    [
                        hidden,
                        mx.stack(previous_embeddings, axis=1),
                    ],
                    axis=-1,
                )
            confidence = self.confidence_head(inputs)
        return mx.stack(tokens, axis=1), mx.stack(corrected, axis=1), confidence

    def draft(
        self, anchors: mx.array, features: Sequence[mx.array], *, num_draft_tokens: int
    ) -> tuple[mx.array, mx.array, mx.array | None]:
        hidden = self.block_hidden(anchors, features, num_draft_tokens=num_draft_tokens)
        return self.greedy_proposal(hidden, anchors)

    def validate_anchors(self, anchors: mx.array) -> None:
        self.backbone.validate_anchors(anchors)


def load_dspark(path: str | Path, *, target_config: Mapping[str, Any]) -> DSparkModel:
    """Load a local unsharded checkpoint without changing its trained precision."""
    path = Path(path)
    config = DSparkConfig.from_dict(json.loads((path / "config.json").read_text()))
    config.backbone.validate_target(target_config)
    model = DSparkModel(config)
    load_draft_weights(model, path, model_name="DSpark", strip_prefix="backbone.")
    return model
