# SPDX-License-Identifier: Apache-2.0
"""Bidirectional ModernBERT backbone for Laya encoder pooling."""

import math

import mlx.core as mx
import mlx.nn as nn


def layer_norm(config):
    return nn.LayerNorm(
        config["hidden_size"], eps=config["norm_eps"], bias=config["norm_bias"]
    )


def rope(x, theta):
    dim = x.shape[-1]
    freq = 1 / (theta ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
    angle = mx.arange(x.shape[-2], dtype=mx.float32)[:, None] * freq[None, :]
    angle = mx.concatenate([angle, angle], axis=-1)
    first, second = mx.split(x, 2, axis=-1)
    rotated = mx.concatenate([-second, first], axis=-1)
    return x * mx.cos(angle) + rotated * mx.sin(angle)


class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tok_embeddings = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.norm = layer_norm(config)

    def __call__(self, ids):
        return self.norm(self.tok_embeddings(ids))


class Attention(nn.Module):
    def __init__(self, config, index):
        super().__init__()
        width = config["hidden_size"]
        self.Wqkv = nn.Linear(width, 3 * width, bias=config["attention_bias"])
        self.Wo = nn.Linear(width, width, bias=config["attention_bias"])
        self.heads = config["num_attention_heads"]
        self.layer_type = config["layer_types"][index]
        self.theta = config["rope_parameters"][self.layer_type]["rope_theta"]

    def __call__(self, x, mask):
        batch, length, width = x.shape
        qkv = self.Wqkv(x).reshape(batch, length, 3, self.heads, width // self.heads)
        q, k, v = [qkv[:, :, i].transpose(0, 2, 1, 3) for i in range(3)]
        q, k = rope(q, self.theta), rope(k, self.theta)
        scale = 1 / math.sqrt(width // self.heads)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
        return self.Wo(out.transpose(0, 2, 1, 3).reshape(batch, length, width))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        width, intermediate = config["hidden_size"], config["intermediate_size"]
        self.Wi = nn.Linear(width, intermediate * 2, bias=config["mlp_bias"])
        self.Wo = nn.Linear(intermediate, width, bias=config["mlp_bias"])

    def __call__(self, x):
        value, gate = mx.split(self.Wi(x), 2, axis=-1)
        return self.Wo(nn.gelu(value) * gate)


class Layer(nn.Module):
    def __init__(self, config, index):
        super().__init__()
        if index:
            self.attn_norm = layer_norm(config)
        self.attn = Attention(config, index)
        self.mlp_norm = layer_norm(config)
        self.mlp = MLP(config)

    def __call__(self, x, mask):
        x = x + self.attn(self.attn_norm(x) if "attn_norm" in self else x, mask)
        return x + self.mlp(self.mlp_norm(x))


class ModernBertBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config["hidden_activation"] != "gelu":
            raise NotImplementedError("Metal ModernBERT supports GELU only")
        if any(v["rope_type"] != "default" for v in config["rope_parameters"].values()):
            raise NotImplementedError("Metal ModernBERT supports default RoPE only")
        self.embeddings = Embeddings(config)
        self.layers = [Layer(config, i) for i in range(config["num_hidden_layers"])]
        self.final_norm = layer_norm(config)
        self.window = config["local_attention"] // 2

    def __call__(self, input_ids, attention_mask):
        length = input_ids.shape[1]
        positions = mx.arange(length)
        valid_keys = attention_mask[:, None, None, :].astype(mx.bool_)
        local = mx.abs(positions[:, None] - positions[None, :]) <= self.window
        masks = {
            "full_attention": valid_keys,
            "sliding_attention": valid_keys & local[None, None, :, :],
        }
        x = self.embeddings(input_ids)
        for layer in self.layers:
            x = layer(x, masks[layer.attn.layer_type])
        return self.final_norm(x)
