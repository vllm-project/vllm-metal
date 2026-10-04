# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Z Lab
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
"""Native Qwen3 DFlash block forward for checkpoint qualification.

Adapted from z-lab/dflash's model_mlx.py at 07ebd93db9f472af339b644bb70221ad8428328a.
This stateless forward recomputes context K/V; it does not allocate a serving
cache or register a vLLM speculative method. Paged cache integration is separate.
The caller owns the target embedding, output projection, and captured features.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, cast

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.qwen3 import MLP

from vllm_metal.patches.aux_hidden_states import AuxHiddenStateCapture, _Output
from vllm_metal.v1.draft_checkpoint import COMMON_DRAFT_OPTIONS, load_draft_weights


@dataclass(frozen=True)
class DFlashConfig:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    num_target_layers: int
    max_position_embeddings: int
    block_size: int
    mask_token_id: int
    target_layer_ids: tuple[int, ...]
    rms_norm_eps: float
    rope_theta: float

    def __post_init__(self) -> None:
        for field in fields(self):
            if field.name in {
                "target_layer_ids",
                "mask_token_id",
                "rms_norm_eps",
                "rope_theta",
            }:
                continue
            value = getattr(self, field.name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"DFlash {field.name} must be a positive integer")
        if self.num_attention_heads % self.num_key_value_heads or self.head_dim % 2:
            raise ValueError("DFlash requires divisible GQA heads and an even head_dim")
        if self.block_size < 2 or self.block_size > self.max_position_embeddings:
            raise ValueError(
                "DFlash block_size must fit the context and include an anchor"
            )
        if (
            type(self.mask_token_id) is not int
            or not 0 <= self.mask_token_id < self.vocab_size
        ):
            raise ValueError("DFlash mask_token_id must be in the target vocabulary")
        if not self.target_layer_ids or any(
            type(i) is not int or not 0 <= i < self.num_target_layers
            for i in self.target_layer_ids
        ):
            raise ValueError("DFlash target_layer_ids must name target decoder outputs")
        for name in ("rms_norm_eps", "rope_theta"):
            value = getattr(self, name)
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"DFlash {name} must be finite and positive")

    @property
    def capture_layer_ids(self) -> tuple[int, ...]:
        """Bridge indices; DFlashTargetCapture normalizes the final-layer tap."""
        return tuple(i + 1 for i in self.target_layer_ids)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> DFlashConfig:
        if (
            raw.get("architectures") != ["DFlashDraftModel"]
            or raw.get("model_type") != "qwen3"
        ):
            raise ValueError("Expected a z-lab Qwen3 DFlashDraftModel checkpoint")
        supported = {**COMMON_DRAFT_OPTIONS, "rope_parameters": None}
        for name, expected in supported.items():
            if raw.get(name, expected) != expected:
                raise ValueError(f"Unsupported DFlash {name}: {raw[name]!r}")
        draft = raw.get("dflash_config")
        if not isinstance(draft, dict):
            raise ValueError("DFlash checkpoint requires dflash_config")
        for name, expected in {
            "input_embedding_scale": 1.0,
            "output_multiplier": 1.0,
            "final_logit_softcapping": None,
            "sample_from_anchor": False,
        }.items():
            if draft.get(name, expected) != expected:
                raise ValueError(f"Unsupported DFlash {name}")
        try:
            values = {
                f.name: raw[f.name]
                for f in fields(cls)
                if f.name not in {"target_layer_ids", "mask_token_id", "block_size"}
            }
            values.update(
                target_layer_ids=tuple(draft["target_layer_ids"]),
                mask_token_id=draft["mask_token_id"],
                block_size=draft.get("block_size", raw.get("block_size")),
            )
            config = cls(**values)
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"Incomplete DFlash checkpoint configuration: {exc}"
            ) from exc
        if (
            "block_size" in draft
            and "block_size" in raw
            and draft["block_size"] != raw["block_size"]
        ):
            raise ValueError("Conflicting DFlash block_size values")
        if (
            raw.get("layer_types", ["full_attention"] * config.num_hidden_layers)
            != ["full_attention"] * config.num_hidden_layers
        ):
            raise ValueError("Only full-attention DFlash layers are supported")
        return config

    def validate_target(self, target: Mapping[str, Any]) -> None:
        """Check structural compatibility; the checkpoint still determines pairing."""
        expected = {
            "model_type": "qwen3",
            "hidden_size": self.hidden_size,
            "vocab_size": self.vocab_size,
            "num_hidden_layers": self.num_target_layers,
        }
        for name, value in expected.items():
            if target.get(name) != value:
                raise ValueError(f"DFlash target {name} must be {value!r}")


class DFlashTargetCapture(AuxHiddenStateCapture):
    """Adapt native Qwen3 capture to DFlash's HF hidden-state tuple contract.

    The shared bridge always observes pre-norm decoder outputs. HF replaces
    its last hidden-state entry with the final normalized output, so only a
    checkpoint tap at target layer N-1 needs the target's final norm here.
    The target owns its parameters; captured features remain call-scoped.
    """

    def __init__(self, target: nn.Module, config: DFlashConfig) -> None:
        config.validate_target(vars(target.args))
        super().__init__(target, config.capture_layer_ids)
        self._final_layer_id = config.num_target_layers
        self._final_norm = target.model.norm

    def run(
        self, forward: Callable[..., _Output], *args: Any, **kwargs: Any
    ) -> tuple[_Output, tuple[mx.array, ...]]:
        output, features = super().run(forward, *args, **kwargs)
        return output, tuple(
            self._final_norm(feature) if i == self._final_layer_id else feature
            for i, feature in zip(self.layer_ids, features, strict=True)
        )


class _Attention(nn.Module):
    def __init__(self, config: DFlashConfig) -> None:
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.scale = config.head_dim**-0.5
        self.q_proj = nn.Linear(
            config.hidden_size, self.n_heads * config.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.n_kv_heads * config.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.n_kv_heads * config.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.n_heads * config.head_dim, config.hidden_size, bias=False
        )
        self.q_norm = nn.RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(config.head_dim, eps=config.rms_norm_eps)

    def project_context(
        self, context: mx.array, rope: nn.RoPE
    ) -> tuple[mx.array, mx.array]:
        batch, length, _ = context.shape
        keys = self.k_norm(
            self.k_proj(context).reshape(batch, length, self.n_kv_heads, -1)
        ).transpose(0, 2, 1, 3)
        values = (
            self.v_proj(context)
            .reshape(batch, length, self.n_kv_heads, -1)
            .transpose(0, 2, 1, 3)
        )
        return rope(keys), values

    def project_block(
        self,
        x: mx.array,
        rope: nn.RoPE,
        offset: int | mx.array,
    ) -> tuple[mx.array, mx.array, mx.array]:
        """Project the same Q/K/V for full-context and scheduler-paged drafting."""
        batch, width, _ = x.shape
        q = self.q_norm(
            self.q_proj(x).reshape(batch, width, self.n_heads, -1)
        ).transpose(0, 2, 1, 3)
        bk = self.k_norm(
            self.k_proj(x).reshape(batch, width, self.n_kv_heads, -1)
        ).transpose(0, 2, 1, 3)
        bv = (
            self.v_proj(x)
            .reshape(batch, width, self.n_kv_heads, -1)
            .transpose(0, 2, 1, 3)
        )
        return rope(q, offset=offset), rope(bk, offset=offset), bv

    def __call__(
        self,
        x: mx.array,
        context: tuple[mx.array, mx.array],
        rope: nn.RoPE,
        context_length: mx.array | None,
        mask: mx.array | None,
    ) -> mx.array:
        batch, width, _ = x.shape
        ck, cv = context
        length = ck.shape[2]
        offset = length if context_length is None else context_length
        q, bk, bv = self.project_block(x, rope, offset)
        keys = mx.concatenate([ck, bk], axis=2)
        values = mx.concatenate([cv, bv], axis=2)
        if context_length is not None:
            # Keep valid keys in their unpadded order: [context, block, padding].
            # A gap before the block changes SDPA's reduction order and can
            # amplify BF16 rounding through the trained draft's later layers.
            start = context_length.reshape(1)
            keys = mx.slice_update(keys, bk, start, axes=(2,))
            values = mx.slice_update(values, bv, start, axes=(2,))
        # Every block query sees the entire committed context AND draft block.
        # This is bidirectional block attention, not the target's causal mask.
        output = mx.fast.scaled_dot_product_attention(
            q,
            keys,
            values,
            scale=self.scale,
            mask=mask,
        )
        return self.o_proj(output.transpose(0, 2, 1, 3).reshape(batch, width, -1))


class _DecoderLayer(nn.Module):
    def __init__(self, config: DFlashConfig) -> None:
        super().__init__()
        self.self_attn = _Attention(config)
        self.mlp = MLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        context: tuple[mx.array, mx.array],
        rope: nn.RoPE,
        context_length: mx.array | None,
        mask: mx.array | None,
    ) -> mx.array:
        h = x + self.self_attn(
            self.input_layernorm(x), context, rope, context_length, mask
        )
        return h + self.mlp(self.post_attention_layernorm(h))


class DFlashModel(nn.Module):
    """Full-context block forward, with no persistent KV or borrowed parameters."""

    def __init__(self, config: DFlashConfig) -> None:
        super().__init__()
        self.config = config
        self.fc = nn.Linear(
            len(config.target_layer_ids) * config.hidden_size,
            config.hidden_size,
            bias=False,
        )
        self.hidden_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layers = [_DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rope = nn.RoPE(config.head_dim, traditional=False, base=config.rope_theta)

    def __call__(
        self,
        embeddings: mx.array,
        features: Sequence[mx.array],
        *,
        logits_start: int = 0,
    ) -> mx.array:
        """Return normalized block states from logits_start onward.

        Features from DFlashTargetCapture cover the full prefix starting at zero.
        Each feature is [batch, context length, hidden size], in capture order.
        There is no padding: every row in a call has the same context/block size.
        """
        self.validate_embeddings(embeddings, logits_start)
        self._validate_features(features, embeddings.shape[0], embeddings.shape[1])
        return self._block_forward(
            embeddings, self._project_context(features), logits_start=logits_start
        )

    def _validate_features(
        self, features: Sequence[mx.array], batch: int, width: int
    ) -> int:
        if len(features) != len(self.config.target_layer_ids):
            raise ValueError(
                "DFlash requires one feature tensor per configured target layer"
            )
        shape = features[0].shape
        if (
            len(shape) != 3
            or shape[0] != batch
            or shape[1] < 1
            or shape[2] != self.config.hidden_size
            or any(f.shape != shape for f in features)
        ):
            raise ValueError(
                "DFlash features must share [batch, context length, hidden_size]"
            )
        if shape[1] + width > self.config.max_position_embeddings:
            raise ValueError("DFlash context and block exceed max_position_embeddings")
        return shape[1]

    def validate_embeddings(self, embeddings: mx.array, logits_start: int) -> None:
        """Check block-state metadata and the selected suffix without evaluating it."""
        if (
            embeddings.ndim != 3
            or embeddings.shape[0] < 1
            or embeddings.shape[2] != self.config.hidden_size
        ):
            raise ValueError(
                "DFlash embeddings must have shape [batch, block, hidden_size]"
            )
        if not 1 <= embeddings.shape[1] <= self.config.block_size:
            raise ValueError("DFlash block exceeds the trained block_size")
        if type(logits_start) is not int or not 0 <= logits_start < embeddings.shape[1]:
            raise ValueError("DFlash logits_start must select a nonempty block suffix")

    def _project_context(
        self, features: Sequence[mx.array]
    ) -> tuple[tuple[mx.array, mx.array], ...]:
        context = self.hidden_norm(self.fc(mx.concatenate(features, axis=-1)))
        return tuple(
            layer.self_attn.project_context(context, self.rope) for layer in self.layers
        )

    def _block_forward(
        self,
        embeddings: mx.array,
        contexts: Sequence[tuple[mx.array, mx.array]],
        *,
        logits_start: int,
        context_length: mx.array | None = None,
    ) -> mx.array:
        length = contexts[0][0].shape[2]
        mask = None
        if context_length is not None:
            # All prefix and block keys precede the masked padding suffix.
            mask = mx.arange(length + embeddings.shape[1]) < (
                context_length + embeddings.shape[1]
            )
        h = embeddings
        for layer, context in zip(self.layers, contexts, strict=True):
            h = layer(h, context, self.rope, context_length, mask)
        # Slice before normalization, as in the reference. Normalization produces
        # contiguous rows for the borrowed head; a strided BF16 slice can select
        # a different quantized projection kernel for batched one-token drafts.
        return self.norm(h[:, logits_start:])

    def draft_logits(
        self,
        anchors: mx.array,
        features: Sequence[mx.array],
        *,
        num_draft_tokens: int,
        embed: Callable[[mx.array], mx.array],
        project: Callable[[mx.array], mx.array],
    ) -> mx.array:
        """Return slots 1..K without reading token values back to the host.

        Anchors must be valid target token IDs, e.g. produced by its sampler.
        For external inputs, call validate_anchors at the input boundary,
        outside the compiled/repeated drafting forward.
        """
        embeddings = self._draft_embeddings(anchors, num_draft_tokens, embed)
        hidden = self(embeddings, features, logits_start=1)
        return self._project_logits(hidden, project, anchors.shape[0], num_draft_tokens)

    def _draft_embeddings(
        self,
        anchors: mx.array,
        num_draft_tokens: int,
        embed: Callable[[mx.array], mx.array],
    ) -> mx.array:
        self._validate_num_draft_tokens(num_draft_tokens)
        self.validate_anchor_metadata(anchors)
        anchors = anchors.astype(mx.int64)
        masks = mx.full(
            (anchors.shape[0], num_draft_tokens),
            self.config.mask_token_id,
            dtype=mx.int64,
        )
        inputs = mx.concatenate([anchors[:, None], masks], axis=1)
        embeddings = embed(inputs)
        self.validate_embeddings(embeddings, 1)
        return embeddings

    def _validate_num_draft_tokens(
        self, num_draft_tokens: int, extra_slots: int = 1
    ) -> None:
        if (
            type(num_draft_tokens) is not int
            or not 1 <= num_draft_tokens <= self.config.block_size - extra_slots
        ):
            raise ValueError("num_draft_tokens must fit the trained block_size")

    def _project_logits(
        self,
        hidden: mx.array,
        project: Callable[[mx.array], mx.array],
        batch: int,
        num_draft_tokens: int,
    ) -> mx.array:
        logits = project(hidden)
        if logits.shape != (batch, num_draft_tokens, self.config.vocab_size):
            raise ValueError("DFlash target projection has an incompatible vocabulary")
        return logits

    def compile_draft(
        self,
        *,
        num_draft_tokens: int,
        embed: Callable[[mx.array], mx.array],
        project: Callable[[mx.array], mx.array],
        context_bucket_size: int = 256,
    ) -> Callable[[mx.array, Sequence[mx.array]], mx.array]:
        """Build once, then reuse for growing full-prefix features.

        The returned callable projects the real prefix and pads its K/V outside
        the compiled block forward. Batch size, dtype and context bucket
        determine its input shapes; the true prefix
        length is a device scalar used for masking and draft RoPE positions.
        All rows still require the same prefix length. Keep model weights and
        borrowed projections fixed for the lifetime of this callable.
        """
        if type(context_bucket_size) is not int or context_bucket_size < 1:
            raise ValueError("DFlash context_bucket_size must be a positive integer")
        self._validate_num_draft_tokens(num_draft_tokens)
        width = num_draft_tokens + 1

        def forward(anchors, contexts, context_length):
            embeddings = self._draft_embeddings(anchors, num_draft_tokens, embed)
            hidden = self._block_forward(
                embeddings, contexts, logits_start=1, context_length=context_length
            )
            return self._project_logits(
                hidden, project, anchors.shape[0], num_draft_tokens
            )

        compiled = mx.compile(forward)

        def draft(anchors: mx.array, features: Sequence[mx.array]) -> mx.array:
            self.validate_anchor_metadata(anchors)
            length = self._validate_features(features, anchors.shape[0], width)
            bucket = min(
                ((length + context_bucket_size - 1) // context_bucket_size)
                * context_bucket_size,
                self.config.max_position_embeddings - width,
            )
            # MLX 0.32.1 changes vector SDPA kernels/partition counts at
            # power-of-two KV lengths starting at 1024. Do not let padding
            # cross those boundaries (including their exact-length case):
            # changing the reduction can amplify BF16 rounding in the draft.
            span = length + width
            boundary = max(1024, 1 << (span - 1).bit_length())
            bucket = min(bucket, boundary - (span != boundary) - width)
            # Project the real prefix before padding. Padding feature rows
            # can select a different GEMM reduction and change BF16 values.
            # These shape-dependent projections stay outside the compiled
            # block graph; only their padded K/V enter that reusable graph.
            contexts = self._project_context(features)
            padded = tuple(
                tuple(
                    mx.pad(value, ((0, 0), (0, 0), (0, bucket - length), (0, 0)))
                    for value in context
                )
                for context in contexts
            )
            return compiled(anchors, padded, mx.array(length, dtype=mx.int32))

        return draft

    @staticmethod
    def validate_anchor_metadata(anchors: mx.array) -> None:
        """Check shape/dtype without reading token values; safe during compilation."""
        if (
            anchors.ndim != 1
            or anchors.size < 1
            or not mx.issubdtype(anchors.dtype, mx.integer)
        ):
            raise ValueError("DFlash anchors must be a nonempty integer vector")

    def validate_anchors(self, anchors: mx.array) -> None:
        """Synchronously validate external token IDs before entering the draft loop."""
        self.validate_anchor_metadata(anchors)
        # Python integers avoid narrowing the vocabulary bound. Check wide IDs
        # before draft_logits converts them, so overflow cannot hide invalid IDs.
        min_anchor = cast(int, anchors.min().item())
        max_anchor = cast(int, anchors.max().item())
        if min_anchor < 0 or max_anchor >= self.config.vocab_size:
            raise ValueError("DFlash anchor token is outside the target vocabulary")


def load_dflash(path: str | Path, *, target_config: Mapping[str, Any]) -> DFlashModel:
    """Validate the target before loading a local, single-file z-lab checkpoint."""
    path = Path(path)
    config = DFlashConfig.from_dict(json.loads((path / "config.json").read_text()))
    config.validate_target(target_config)
    model = DFlashModel(config)
    load_draft_weights(model, path, model_name="DFlash")
    return model
