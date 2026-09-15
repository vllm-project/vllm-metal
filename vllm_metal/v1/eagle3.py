# SPDX-License-Identifier: Apache-2.0
"""MLX EAGLE3 heads with Llama-style draft attention and MLPs.

The checkpoint describes the draft architecture independently of the target.
Red Hat's Qwen3 heads, for example, use a Llama layer without Q/K norms.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.llama import MLP, Attention, ModelArgs


@dataclass(frozen=True)
class Eagle3Config:
    layer: ModelArgs
    draft_vocab_size: int
    target_hidden_size: int
    aux_layer_ids: tuple[int, ...]
    norm_before_residual: bool = True
    norm_before_fc: bool = False
    norm_output: bool = False

    @classmethod
    def from_dict(cls, config: dict[str, Any], target: dict[str, Any]) -> Eagle3Config:
        # Red Hat nests the decoder config; original EAGLE checkpoints store
        # it at the top level and use a different residual-normalization default.
        raw_layer = config.get("transformer_layer_config", config)
        layer = ModelArgs.from_dict(raw_layer)
        if (
            raw_layer["model_type"] != "llama"
            or layer.num_hidden_layers != 1
            or layer.sliding_window is not None
        ):
            raise NotImplementedError(
                "Metal EAGLE3 requires one full-attention Llama draft layer"
            )
        if config.get("fc_norm"):
            raise NotImplementedError(
                "Metal EAGLE3 does not support fc_norm checkpoints"
            )
        num_layers = int(target["num_hidden_layers"])
        ids = tuple(
            config.get("eagle_aux_hidden_state_layer_ids")
            or config.get("eagle_config", {}).get("eagle_aux_hidden_state_layer_ids")
            or (2, num_layers // 2, num_layers - 3)
        )
        if len(ids) != 3:
            raise ValueError("EAGLE3 requires three target hidden-state indices")
        target_hidden = int(config.get("target_hidden_size") or layer.hidden_size)
        if (target_hidden, layer.vocab_size) != (
            target["hidden_size"],
            target["vocab_size"],
        ):
            raise ValueError(
                "EAGLE3 hidden size or vocabulary does not match the target"
            )
        return cls(
            layer=layer,
            draft_vocab_size=int(config["draft_vocab_size"]),
            target_hidden_size=target_hidden,
            aux_layer_ids=ids,
            norm_before_residual=bool(
                config.get("norm_before_residual", "transformer_layer_config" in config)
            ),
            norm_before_fc=bool(config.get("norm_before_fc", False)),
            norm_output=bool(config.get("norm_output", False)),
        )


class Eagle3Layer(nn.Module):
    def __init__(self, config: Eagle3Config):
        super().__init__()
        args = config.layer
        self.norm_before_residual = config.norm_before_residual
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.hidden_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.self_attn = Attention(args)
        for name in ("q_proj", "k_proj", "v_proj"):
            projection = getattr(self.self_attn, name)
            setattr(
                self.self_attn,
                name,
                nn.Linear(
                    2 * args.hidden_size,
                    projection.weight.shape[0],
                    bias=args.attention_bias,
                ),
            )
        self.mlp = MLP(args)

    def __call__(
        self, embeddings: mx.array, hidden: mx.array, *, mask: Any, cache: Any
    ) -> mx.array:
        normalized = self.hidden_norm(hidden)
        residual = normalized if self.norm_before_residual else hidden
        inputs = mx.concatenate((self.input_layernorm(embeddings), normalized), axis=-1)
        hidden = residual + self.self_attn(inputs, mask=mask, cache=cache)
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


class Eagle3Model(nn.Module):
    def __init__(self, config: Eagle3Config):
        super().__init__()
        self.config = config
        args = config.layer
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.fc = nn.Linear(3 * config.target_hidden_size, args.hidden_size, bias=False)
        if config.norm_before_fc:
            self.input_norm = nn.RMSNorm(
                3 * config.target_hidden_size, eps=args.rms_norm_eps
            )
        self.layers = [Eagle3Layer(config)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.lm_head = nn.Linear(args.hidden_size, config.draft_vocab_size, bias=False)
        self.d2t = mx.zeros((config.draft_vocab_size,), dtype=mx.int32)

    def combine_hidden_states(self, hidden: mx.array) -> mx.array:
        if self.config.norm_before_fc:
            hidden = self.input_norm(hidden)
        return self.fc(hidden)

    def __call__(
        self, tokens: mx.array, hidden: mx.array, *, mask: Any = None, cache: Any = None
    ) -> tuple[mx.array, mx.array]:
        """Consume already-fused features; return head input and recurrence state."""
        hidden = self.layers[0](
            self.embed_tokens(tokens), hidden, mask=mask, cache=cache
        )
        normalized = self.norm(hidden)
        return normalized, normalized if self.config.norm_output else hidden

    def top_tokens(self, hidden: mx.array) -> mx.array:
        # Map the winning reduced-vocabulary ID without expanding to target logits.
        ids = mx.argmax(self.lm_head(hidden), axis=-1)
        return (ids + self.d2t[ids]).astype(mx.int32)

    @classmethod
    def load(
        cls,
        model_name: str,
        target_args: dict[str, Any],
        dtype: mx.Dtype,
        revision: str | None = None,
        target_embedding: nn.Embedding | None = None,
    ) -> Eagle3Model:
        path = Path(model_name)
        if not path.is_dir():
            from huggingface_hub import snapshot_download

            path = Path(
                snapshot_download(
                    model_name,
                    revision=revision,
                    allow_patterns=[
                        "config.json",
                        "*.safetensors",
                        "pytorch_model.bin",
                    ],
                )
            )
        raw_config = json.loads((path / "config.json").read_text())
        config = Eagle3Config.from_dict(raw_config, target_args)
        model = cls(config)
        weights: dict[str, mx.array] = {}
        for shard in sorted(path.glob("*.safetensors")):
            weights.update(mx.load(str(shard)))
        if not weights and (path / "pytorch_model.bin").is_file():
            import torch

            from vllm_metal.pytorch_backend.tensor_bridge import torch_to_mlx

            state = torch.load(
                path / "pytorch_model.bin", map_location="cpu", weights_only=True
            )
            weights = {name: torch_to_mlx(value) for name, value in state.items()}
        weights = {
            name.replace("midlayer.", "layers.0."): value
            for name, value in weights.items()
        }
        weights.pop("t2d", None)
        if "embed_tokens.weight" not in weights:
            if not isinstance(target_embedding, nn.Embedding):
                raise ValueError(
                    "This EAGLE3 checkpoint requires the target's dense embedding"
                )
            # Share immutable weights, not a mutable target module or KV state.
            weights["embed_tokens.weight"] = target_embedding.weight
        quantization = raw_config.get("quantization")
        if quantization is not None:
            # Follow the saved tensors, so dense embeddings can coexist with
            # quantized projections without changing the input-token contract.
            nn.quantize(
                model,
                **quantization,
                class_predicate=lambda name, _: f"{name}.scales" in weights,
            )
        weights = {
            name: value.astype(mx.int32)
            if name == "d2t"
            else value.astype(dtype)
            if mx.issubdtype(value.dtype, mx.floating)
            else value
            for name, value in weights.items()
        }
        model.load_weights(list(weights.items()), strict=True)
        mx.eval(model.parameters())
        return model
