# SPDX-License-Identifier: Apache-2.0
"""Laya typed-decision encoder and option-marker pooling."""

import math

import mlx.core as mx
import mlx.nn as nn
import torch
from transformers import AutoTokenizer
from vllm.tasks import PoolingTask

from vllm_metal.pytorch_backend.tensor_bridge import mlx_to_torch
from vllm_metal.v1.pooling.backends.encoder.models.loading import (
    encoder_model_path,
    load_encoder_weights,
)
from vllm_metal.v1.pooling.backends.encoder.models.modernbert import ModernBertBackbone
from vllm_metal.v1.pooling.backends.encoder.runtime import MetalEncoderPoolingBackend
from vllm_metal.v1.pooling.contract import (
    TOKEN_CLASSIFY_TASK,
    EncoderPoolingRequest,
    LoadedEncoderBackend,
)
from vllm_metal.v1.pooling.validation import PoolingConfigView


class HeadAttention(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.in_proj = nn.Linear(width, 3 * width)
        self.out_proj = nn.Linear(width, width)
        self.heads = max(1, width // 64)

    def __call__(self, x, mask):
        batch, length, width = x.shape
        q, k, v = [
            a.reshape(batch, length, self.heads, width // self.heads).transpose(
                0, 2, 1, 3
            )
            for a in mx.split(self.in_proj(x), 3, axis=-1)
        ]
        h = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=(width // self.heads) ** -0.5, mask=mask
        )
        return self.out_proj(h.transpose(0, 2, 1, 3).reshape(batch, length, width))


class HeadLayer(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.self_attn = HeadAttention(width)
        self.norm1 = nn.LayerNorm(width, eps=1e-5)
        self.norm2 = nn.LayerNorm(width, eps=1e-5)
        self.linear1 = nn.Linear(width, 4 * width)
        self.linear2 = nn.Linear(4 * width, width)

    def __call__(self, x, mask):
        x = x + self.self_attn(self.norm1(x), mask)
        return x + self.linear2(nn.relu(self.linear1(self.norm2(x))))


class TypedHead(nn.Module):
    def __init__(self, width, layers):
        super().__init__()
        self.layers = [HeadLayer(width) for _ in range(layers)]


class LayaModel(nn.Module):
    def __init__(self, encoder_config, agent_config):
        super().__init__()
        width = encoder_config["hidden_size"]
        self.encoder = ModernBertBackbone(encoder_config)
        self.qtype_token_ids = agent_config["qtype_token_ids"]
        self.n_act = len(agent_config.get("act_costs", {})) + 1
        self.type_emb = nn.Embedding(3, width)
        self.head = TypedHead(width, agent_config["head_layers"])
        self.scorer = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1)
        )
        self.act_head = nn.Sequential(
            nn.Linear(width + 4, 256),
            nn.GELU(),
            nn.Linear(256, len(agent_config.get("act_costs", {})) + 1),
        )

    def typed_hidden(self, hidden, mask, qtype):
        h = hidden + self.type_emb(qtype)[:, None, :]
        for layer in self.head.layers:
            h = layer(h, mask[:, None, None, :].astype(mx.bool_))
        return h

    def decisions(self, h, marker_pos, marker_mask):
        m = mx.take_along_axis(h, mx.maximum(marker_pos, 0)[:, :, None], axis=1)
        logits = self.scorer(m).squeeze(-1).astype(mx.float32)
        logits = mx.where(marker_mask, logits, -1e4)
        p = mx.softmax(logits, axis=-1)
        k = mx.maximum(marker_mask.sum(-1), 2).astype(mx.float32)
        entropy = -(p * mx.log(mx.maximum(p, 1e-9))).sum(-1) / mx.log(k)
        sorted_p = mx.sort(p, axis=-1)
        top1 = sorted_p[:, -1]
        top2 = sorted_p[:, -2] if p.shape[-1] >= 2 else mx.zeros_like(top1)
        features = mx.stack([top1, top1 - top2, entropy, k / 255], axis=-1)
        action_logits = self.act_head(
            mx.concatenate([h[:, 0].astype(mx.float32), features], axis=-1)
        )
        return logits, action_logits

    def __call__(self, input_ids, attention_mask):
        type_tokens = (
            input_ids[:, 1]
            if input_ids.shape[1] > 1
            else mx.full((input_ids.shape[0],), -1)
        )
        qtype = (
            (type_tokens[:, None] == mx.array(self.qtype_token_ids)[None, :])
            .astype(mx.int32)
            .argmax(-1)
        )
        return self.typed_hidden(
            self.encoder(input_ids, attention_mask), attention_mask, qtype
        )

    @staticmethod
    def sanitize(weights):
        mapped = {}
        for name, value in weights.items():
            if name == "temperature":
                continue  # Agent reads calibration from config, not this buffer.
            name = name.replace(".in_proj_weight", ".in_proj.weight").replace(
                ".in_proj_bias", ".in_proj.bias"
            )
            if name.startswith(("scorer.", "act_head.")):
                prefix, suffix = name.split(".", 1)
                name = f"{prefix}.layers.{suffix}"
            mapped[name] = value
        return mapped


def clamp_temperature(value):
    if isinstance(value, bool):
        return 1.0
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 1.0
    return min(5.0, max(0.5, value)) if math.isfinite(value) else 1.0


def temperature_for(config, qtype, options):
    bucket = (
        "2"
        if options <= 2
        else "3-5"
        if options <= 5
        else "6-10"
        if options <= 10
        else "11+"
    )
    key = f"{['choice', 'score', 'noul'][qtype]}:{bucket}"
    return clamp_temperature(
        config.get("temperature_by_options", {}).get(
            key, config.get("temperature", [1.0] * 3)[qtype]
        )
    )


class LayaPooler:
    def __init__(self, config: PoolingConfigView, model: LayaModel):
        self.config = config
        self.model = model
        self.agent_config = config.hf_config.laya_config

    def supported_tasks(self) -> tuple[PoolingTask, ...]:
        if (
            self.config.is_text_only
            and self.config.task in (None, TOKEN_CLASSIFY_TASK)
            and self.config.pooler_config.tok_pooling_type in (None, "ALL")
            and not self.config.chunked_processing_enabled
        ):
            return (TOKEN_CLASSIFY_TASK,)
        return ()

    def validate_params(self, pooling_params) -> None:
        if not self.supported_tasks() or pooling_params.task not in (
            None,
            TOKEN_CLASSIFY_TASK,
        ):
            raise NotImplementedError(
                "Metal Laya supports only task='token_classify' with ALL token pooling."
            )
        if pooling_params.dimensions is not None:
            raise NotImplementedError(
                "Metal Laya does not support truncated output dimensions."
            )

    def pool_one(
        self, hidden_states: mx.array, request: EncoderPoolingRequest
    ) -> torch.Tensor:
        self.validate_params(request.pooling_params)
        markers = [
            i
            for i, token in enumerate(request.token_ids)
            if token == self.agent_config["mask_token_id"]
        ]
        if not markers:
            return torch.empty((0, 1 + self.model.n_act), dtype=torch.float32)
        type_token = request.token_ids[1] if len(request.token_ids) > 1 else -1
        types = self.agent_config["qtype_token_ids"]
        qtype = types.index(type_token) if type_token in types else 0
        logits, action_logits = self.model.decisions(
            hidden_states,
            mx.array([markers]),
            mx.ones((1, len(markers)), dtype=mx.bool_),
        )
        if request.pooling_params.use_activation is not False:
            logits = mx.softmax(
                logits / temperature_for(self.agent_config, qtype, len(markers)),
                axis=-1,
            )
            action_logits = mx.softmax(action_logits.astype(mx.float32), axis=-1)
        actions = mx.broadcast_to(
            action_logits[:, None, :], (1, len(markers), self.model.n_act)
        )
        output = mx.concatenate([logits[:, :, None], actions], axis=-1)[0]
        return mlx_to_torch(output, device="cpu").detach().clone()


def supports_laya_encoder(model_config) -> bool:
    return "LayaForDecision" in (model_config.hf_config.architectures or ())


def load_laya_backend(model_config) -> LoadedEncoderBackend:
    config = model_config.hf_config.to_dict()
    if model_config.quantization is not None or config.get("quantization_config"):
        raise NotImplementedError("Metal Laya does not support quantization yet.")
    if model_config.dtype != torch.float32:
        raise NotImplementedError("Metal Laya currently requires dtype=float32.")
    agent_config = config["laya_config"]
    if (
        len(agent_config["qtype_token_ids"]) != 3
        or len(set(agent_config["qtype_token_ids"])) != 3
    ):
        raise ValueError("Laya requires three distinct question-type token IDs.")
    model = LayaModel(config, agent_config)
    weights = model.sanitize(load_encoder_weights(encoder_model_path(model_config)))
    model.load_weights(
        [(name, value.astype(mx.float32)) for name, value in weights.items()],
        strict=True,
    )
    mx.eval(model.parameters())
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.tokenizer,
        revision=model_config.tokenizer_revision,
        trust_remote_code=model_config.trust_remote_code,
    )
    view = PoolingConfigView(model_config)
    return LoadedEncoderBackend(
        model,
        tokenizer,
        config,
        MetalEncoderPoolingBackend(view, model, LayaPooler(view, model)),
    )
