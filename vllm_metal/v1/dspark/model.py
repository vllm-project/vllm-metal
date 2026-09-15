# SPDX-License-Identifier: MIT
# Copyright (c) 2026 erahim3
"""DSpark drafter in MLX — Gemma-4 and Qwen3 families.

Faithful port of the DeepSpec inference path. The EAGLE-style cross-attention is shared:
Q comes from the draft block, K/V from concat([fused_target_context, block]), with the
context K/V cached per layer (CtxCache). Family differences (norm layout, rope, MLP act,
v handling, logit softcap) are config-driven. Module attribute names match the HF
checkpoints so weights load 1:1.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
from mlx_vlm.models.rope_utils import initialize_rope

from .config import DSparkConfig


class RMSNormNoScale(nn.Module):
    """RMSNorm with no learnable weight (Gemma-4 v_norm)."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, None, self.eps)


def _act(name: str):
    return nn.silu if name == "silu" else nn.gelu_approx


class MLP(nn.Module):
    def __init__(self, config: DSparkConfig):
        super().__init__()
        h, i = config.hidden_size, config.intermediate_size
        self.gate_proj = nn.Linear(h, i, bias=False)
        self.up_proj = nn.Linear(h, i, bias=False)
        self.down_proj = nn.Linear(i, h, bias=False)
        self.act = _act(config.mlp_activation)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


class CtxCache:
    """Per-layer cache of the target context's projected K/V (roped K, normed/raw V).

    Only the committed prefix is exposed to attention. Storage grows in chunks
    within the optional hard capacity and is reused across appends/rollback.
    """

    __slots__ = ("_keys", "_values", "_length", "capacity")

    def __init__(self, capacity: int | None = None):
        if capacity is not None and capacity <= 0:
            raise ValueError("DSpark context capacity must be positive")
        self._keys = None
        self._values = None
        self._length = 0
        self.capacity = capacity

    @property
    def k(self):
        return None if self._keys is None else self._keys[:, :, : self.length]

    @property
    def v(self):
        return None if self._values is None else self._values[:, :, : self.length]

    @property
    def allocated_bytes(self) -> int:
        if self._keys is None:
            return 0
        return self._keys.nbytes + self._values.nbytes

    def append(self, k: mx.array, v: mx.array) -> None:
        if k.ndim != 4 or v.ndim != 4 or k.shape[:3] != v.shape[:3]:
            raise ValueError(
                "context K/V must cover matching batch, head and token axes"
            )
        start = self.length
        end = start + k.shape[2]
        if self.capacity is not None and end > self.capacity:
            raise ValueError("DSpark context append exceeds reserved capacity")
        if self._keys is not None:
            for old, new in ((self._keys, k), (self._values, v)):
                if (
                    old.shape[:2] != new.shape[:2]
                    or old.shape[-1] != new.shape[-1]
                    or old.dtype != new.dtype
                ):
                    raise ValueError("DSpark context append changes geometry or dtype")
        allocated = 0 if self._keys is None else self._keys.shape[2]
        if end > allocated:
            from .memory import CONTEXT_ALIGNMENT

            length = (
                (end + CONTEXT_ALIGNMENT - 1) // CONTEXT_ALIGNMENT * CONTEXT_ALIGNMENT
            )
            if self.capacity is not None:
                length = min(length, self.capacity)
            shape = (*k.shape[:2], length - allocated)
            keys = mx.zeros((*shape, k.shape[-1]), dtype=k.dtype)
            values = mx.zeros((*shape, v.shape[-1]), dtype=v.dtype)
            self._keys = (
                keys
                if self._keys is None
                else mx.concatenate([self._keys, keys], axis=2)
            )
            self._values = (
                values
                if self._values is None
                else mx.concatenate([self._values, values], axis=2)
            )
        if end > start:
            self._keys[:, :, start:end] = k
            self._values[:, :, start:end] = v
        self._length = end

    def trim_to(self, length: int) -> None:
        """Retain a physical prefix; its keys keep their absolute RoPE positions."""
        if not 0 <= length <= self.length:
            raise ValueError("context trim must retain an existing prefix")
        self._length = length

    @property
    def length(self) -> int:
        if self._keys is None:
            if self._values is not None or self._length:
                raise RuntimeError("DSpark context has values without keys")
            return 0
        if (
            self._values is None
            or self._keys.shape[:3] != self._values.shape[:3]
            or not 0 <= self._length <= self._keys.shape[2]
        ):
            raise RuntimeError("DSpark context K/V coverage disagrees")
        return self._length


class DSparkAttention(nn.Module):
    """Cross-attention: Q from the draft block, K/V from [target_context, block]."""

    def __init__(self, config: DSparkConfig):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.head_dim = config.attn_head_dim
        self.k_eq_v = config.attention_k_eq_v
        self.n_kv_heads = config.n_kv_heads
        self.scale = config.scaling
        self.use_v_norm = config.use_v_norm
        self.gated = config.gated_q_proj

        h = config.hidden_size
        b = config.attention_bias
        # gated (qwen3_5): q_proj emits [q ‖ gate] interleaved per head (2× out-features),
        # split per head like mlx-lm's Qwen3NextAttention so the checkpoint loads 1:1.
        self.q_proj = nn.Linear(
            h, self.n_heads * self.head_dim * (2 if self.gated else 1), bias=b
        )
        self.k_proj = nn.Linear(h, self.n_kv_heads * self.head_dim, bias=b)
        if not self.k_eq_v:
            self.v_proj = nn.Linear(h, self.n_kv_heads * self.head_dim, bias=b)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, h, bias=b)

        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        if self.use_v_norm:
            self.v_norm = RMSNormNoScale(eps=config.rms_norm_eps)

        self.rope = initialize_rope(
            dims=config.rope_dims or self.head_dim,
            base=config.rope_theta,
            traditional=False,
            scaling_config=config.rope_parameters,
        )

    def _kv(self, x: mx.array):
        """Project x -> (roped+normed K, V). k_eq_v shares k_proj for V."""
        batch_size, seq_len, _ = x.shape
        kp = (
            self.k_proj(x)
            .reshape(batch_size, seq_len, self.n_kv_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        k = self.k_norm(kp)
        if self.k_eq_v:
            v = self.v_norm(kp)
        else:
            v = (
                self.v_proj(x)
                .reshape(batch_size, seq_len, self.n_kv_heads, self.head_dim)
                .transpose(0, 2, 1, 3)
            )
            if self.use_v_norm:
                v = self.v_norm(v)
        return k, v

    def update_ctx(self, fused_new: mx.array, ctx_offset: int, cache: CtxCache) -> None:
        k, v = self._kv(fused_new)
        cache.append(self.rope(k, offset=ctx_offset), v)  # V is not roped

    def attend(self, hidden: mx.array, block_offset, cache, mask=None) -> mx.array:
        """``block_offset`` may be an int (single sequence) or a per-row ``[B]`` array (batched
        drafting — rows sit at different context lengths). ``mask`` is None for the single-seq
        path (block attends the whole context + all block positions) or a ``[B, 1, k, Lctx+k]``
        boolean mask that hides each row's context padding when ``cache.k/v`` are a batched buffer."""
        batch_size, q_len, _ = hidden.shape
        q = self.q_proj(hidden).reshape(batch_size, q_len, self.n_heads, -1)
        gate = None
        if self.gated:
            q, gate = mx.split(q, 2, axis=-1)  # per-head [q ‖ gate]
            gate = gate.reshape(batch_size, q_len, -1)  # gate is not q-normed
        q = self.rope(self.q_norm(q).transpose(0, 2, 1, 3), offset=block_offset)

        k_blk, v_blk = self._kv(hidden)
        k_blk = self.rope(k_blk, offset=block_offset)
        k = mx.concatenate([cache.k, k_blk], axis=2)
        v = mx.concatenate([cache.v, v_blk], axis=2)

        # SDPA broadcasts GQA/MQA heads internally; keep K/V at their native
        # head count instead of copying the whole context to every query head.
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(batch_size, q_len, -1)
        if gate is not None:
            out = out * mx.sigmoid(gate)
        return self.o_proj(out)


class DSparkDecoderLayer(nn.Module):
    def __init__(self, config: DSparkConfig):
        super().__init__()
        eps = config.rms_norm_eps
        self.norm_style = config.norm_style
        self.self_attn = DSparkAttention(config)
        self.mlp = MLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=eps)
        if self.norm_style == "gemma":
            self.pre_feedforward_layernorm = nn.RMSNorm(config.hidden_size, eps=eps)
            self.post_feedforward_layernorm = nn.RMSNorm(config.hidden_size, eps=eps)
            self.layer_scalar = mx.ones((1,))

    def __call__(self, hidden, block_offset, cache, mask=None):
        if self.norm_style == "gemma":
            residual = hidden
            h = self.input_layernorm(hidden)
            h = self.self_attn.attend(h, block_offset, cache, mask)
            h = self.post_attention_layernorm(h)
            h = residual + h
            residual = h
            h = self.pre_feedforward_layernorm(h)
            h = self.mlp(h)
            h = self.post_feedforward_layernorm(h)
            h = residual + h
            return h * self.layer_scalar
        # qwen / llama 2-norm
        residual = hidden
        h = self.input_layernorm(hidden)
        h = self.self_attn.attend(h, block_offset, cache, mask)
        h = residual + h
        residual = h
        h = self.post_attention_layernorm(h)
        h = self.mlp(h)
        return residual + h


class VanillaMarkov(nn.Module):
    """Rank-256 previous-token correction: logits += w2(w1[prev_token])."""

    def __init__(self, config: DSparkConfig):
        super().__init__()
        self.markov_w1 = nn.Embedding(config.vocab_size, config.markov_rank)
        self.markov_w2 = nn.Linear(config.markov_rank, config.vocab_size, bias=False)

    def prev_embeddings(self, token_ids: mx.array) -> mx.array:
        return self.markov_w1(token_ids)

    def step_bias(self, token_ids: mx.array) -> mx.array:
        return self.markov_w2(self.markov_w1(token_ids))


class ConfidenceHead(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, 1)

    def __call__(self, features: mx.array) -> mx.array:
        return self.proj(features).squeeze(-1)


LOG_SNR_FREQS = (
    128  # sinusoidal feature count of the GIDD LogSnrEmbed (fixed by training)
)


def log_snr_features(
    block_size: int, min_log_snr: float, max_log_snr: float
) -> mx.array:
    """``[block_size, 128]`` sinusoidal features for the fixed inference-time log-SNR
    pattern (PrismML dspark.cpp): the anchor at block position 0 is "clean" (max_log_snr),
    every masked position is "fully noised" (min_log_snr); each value maps to
    ``t = (snr - min) / (max - min) * 1000`` and is featurized as
    ``[sin(t·f_i), cos(t·f_i)]`` with ``f_i = 10000^(-i/64)`` — the sin half first,
    matching the reference layout exactly."""
    import math

    half = LOG_SNR_FREQS // 2
    pos = mx.arange(block_size)
    log_snr = mx.where(pos == 0, max_log_snr, min_log_snr).astype(mx.float32)
    t = (log_snr - min_log_snr) / (max_log_snr - min_log_snr) * 1000.0
    freq = mx.exp(-math.log(10000.0) * mx.arange(half).astype(mx.float32) / half)
    angle = t[:, None] * freq[None, :]
    return mx.concatenate([mx.sin(angle), mx.cos(angle)], axis=-1)


class LogSnrEmbed(nn.Module):
    """GIDD log-SNR conditioning MLP (Bonsai drafters): 128 sinusoidal features →
    fc1 → silu → fc2 → an additive embedding on the draft block. Since the inference
    pattern is a pure function of the block position, the drafter caches the resulting
    ``[1, block_size, H]`` addend after the first call."""

    def __init__(self, config: DSparkConfig):
        super().__init__()
        self.fc1 = nn.Linear(LOG_SNR_FREQS, config.hidden_size)
        self.fc2 = nn.Linear(config.hidden_size, config.hidden_size)

    def __call__(self, feat: mx.array) -> mx.array:
        return self.fc2(nn.silu(self.fc1(feat)))


class DSparkDrafter(nn.Module):
    def __init__(self, config: DSparkConfig):
        super().__init__()
        self.config = config
        self.block_size = config.block_size
        self.mask_token_id = config.mask_token_id
        self.embed_scale = (
            (float(config.hidden_size) ** 0.5) if config.family == "gemma4" else 1.0
        )
        self.softcap = config.final_logit_softcapping

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.fc = nn.Linear(
            len(config.target_layer_ids) * config.hidden_size,
            config.hidden_size,
            bias=False,
        )
        self.hidden_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layers = [
            DSparkDecoderLayer(config) for _ in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.markov_head = VanillaMarkov(config) if config.markov_rank > 0 else None
        self.confidence_head = None
        if config.enable_confidence_head:
            in_dim = config.hidden_size + (
                config.markov_rank if config.confidence_head_with_markov else 0
            )
            self.confidence_head = ConfidenceHead(in_dim)
        self.log_snr_embed = (
            LogSnrEmbed(config) if config.log_snr_conditioning else None
        )
        self._snr_addend = None  # lazily-built [1, block_size, H] constant

    def embed(self, ids: mx.array) -> mx.array:
        e = self.embed_tokens(ids) * self.embed_scale
        if self.log_snr_embed is not None:
            if self._snr_addend is None:
                feat = log_snr_features(
                    self.block_size, self.config.min_log_snr, self.config.max_log_snr
                )
                self._snr_addend = self.log_snr_embed(feat.astype(e.dtype))[None]
            # ids is the draft block ([1, block_size]); slicing keeps shorter probes valid.
            e = e + self._snr_addend[:, : e.shape[1], :]
        return e

    def fuse_target(self, target_hidden_cat: mx.array) -> mx.array:
        return self.hidden_norm(self.fc(target_hidden_cat))

    def make_ctx_cache(self, capacity: int | None = None) -> list[CtxCache]:
        return [CtxCache(capacity) for _ in self.layers]

    def update_context(self, target_hidden_cat, ctx_offset, ctx_caches) -> None:
        # Use the drafter's qualified compute precision. Mixing BF16 draft
        # weights with FP16 target features can otherwise promote persistent
        # context to FP32 and invalidate the memory plan.
        fused = self.fuse_target(
            target_hidden_cat.astype(self.hidden_norm.weight.dtype)
        )
        for layer, cache in zip(self.layers, ctx_caches, strict=True):
            layer.self_attn.update_ctx(fused, ctx_offset, cache)

    def backbone(
        self, noise_embedding, block_offset, ctx_caches, mask=None
    ) -> mx.array:
        h = noise_embedding
        for layer, cache in zip(self.layers, ctx_caches, strict=True):
            h = layer(h, block_offset, cache, mask)
        return self.norm(h)

    def compute_logits(self, hidden: mx.array) -> mx.array:
        logits = self.lm_head(hidden)
        if self.softcap is not None:
            logits = mx.tanh(logits / self.softcap) * self.softcap
        return logits

    def confidence_logits(self, block_hidden, prev_token_ids):
        if self.confidence_head is None:
            return None
        if self.config.confidence_head_with_markov:
            feats = mx.concatenate(
                [block_hidden, self.markov_head.prev_embeddings(prev_token_ids)],
                axis=-1,
            )
        else:
            feats = block_hidden
        return self.confidence_head(feats)
