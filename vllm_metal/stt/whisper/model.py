# SPDX-License-Identifier: Apache-2.0
"""Whisper model implementation for MLX.

Encoder-decoder architecture for speech recognition, supporting both
HuggingFace and MLX weight formats.
"""

from __future__ import annotations

import math
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from vllm_metal.stt.runtime import STTRuntimeAdapter

from .config import WhisperConfig

# ===========================================================================
# Attention & transformer blocks
# ===========================================================================


class MultiHeadAttention(nn.Module):
    """Multi-head attention layer."""

    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = nn.Linear(n_state, n_state)
        self.key = nn.Linear(n_state, n_state, bias=False)
        self.value = nn.Linear(n_state, n_state)
        self.out = nn.Linear(n_state, n_state)

    def __call__(
        self,
        x: mx.array,
        xa: mx.array | None = None,
        mask: mx.array | None = None,
        kv_cache: tuple[mx.array, mx.array] | None = None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array], mx.array]:
        q = self.query(x)

        if xa is None:
            k = self.key(x)
            v = self.value(x)
            if kv_cache is not None:
                k = mx.concatenate([kv_cache[0], k], axis=1)
                v = mx.concatenate([kv_cache[1], v], axis=1)
        elif kv_cache is None:
            k = self.key(xa)
            v = self.value(xa)
        else:
            k, v = kv_cache

        wv, qk = self._qkv_attention(q, k, v, mask)
        return self.out(wv), (k, v), qk

    def _qkv_attention(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        mask: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25

        q = q.reshape(*q.shape[:2], self.n_head, -1).transpose(0, 2, 1, 3) * scale
        k = k.reshape(*k.shape[:2], self.n_head, -1).transpose(0, 2, 3, 1) * scale
        v = v.reshape(*v.shape[:2], self.n_head, -1).transpose(0, 2, 1, 3)

        qk = q @ k
        if mask is not None:
            k_len = k.shape[3]
            offset = k_len - n_ctx
            qk = qk + mask[offset : offset + n_ctx, :k_len]

        w = mx.softmax(qk, axis=-1, precise=True)
        out = (w @ v).transpose(0, 2, 1, 3)
        out = out.reshape(n_batch, n_ctx, n_state)
        return out, qk


class ResidualAttentionBlock(nn.Module):
    """Transformer block with residual connections."""

    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
        super().__init__()
        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = nn.LayerNorm(n_state)

        self.cross_attn = (
            MultiHeadAttention(n_state, n_head) if cross_attention else None
        )
        self.cross_attn_ln = nn.LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp1 = nn.Linear(n_state, n_mlp)
        self.mlp2 = nn.Linear(n_mlp, n_state)
        self.mlp_ln = nn.LayerNorm(n_state)

    def __call__(
        self,
        x: mx.array,
        xa: mx.array | None = None,
        mask: mx.array | None = None,
        kv_cache: tuple | None = None,
    ) -> tuple[mx.array, tuple, mx.array | None]:
        kv, cross_kv = kv_cache if kv_cache else (None, None)
        y, kv, _ = self.attn(self.attn_ln(x), mask=mask, kv_cache=kv)
        x = x + y

        cross_qk = None
        if self.cross_attn is not None:
            y, cross_kv, cross_qk = self.cross_attn(
                self.cross_attn_ln(x), xa, kv_cache=cross_kv
            )
            x = x + y

        x = x + self.mlp2(nn.gelu(self.mlp1(self.mlp_ln(x))))
        return x, (kv, cross_kv), cross_qk


# ===========================================================================
# Encoder / Decoder
# ===========================================================================


class AudioEncoder(nn.Module):
    """Whisper audio encoder.

    Two 1-D convolutions followed by sinusoidal positional embeddings
    and a stack of transformer blocks.
    """

    def __init__(
        self,
        n_mels: int,
        n_ctx: int,
        n_state: int,
        n_head: int,
        n_layer: int,
        dtype: mx.Dtype = mx.float16,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self._positional_embedding = self.sinusoids(n_ctx, n_state).astype(dtype)
        self.blocks = [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        self.ln_post = nn.LayerNorm(n_state)

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.gelu(self.conv1(x))
        x = nn.gelu(self.conv2(x))

        assert x.shape[1:] == self._positional_embedding.shape, (
            f"Incorrect audio shape: {x.shape[1:]} vs {self._positional_embedding.shape}"
        )
        x = x + self._positional_embedding

        for block in self.blocks:
            x, _, _ = block(x)

        x = self.ln_post(x)
        return x

    @staticmethod
    def sinusoids(length: int, channels: int, max_timescale: int = 10000) -> mx.array:
        """Generate sinusoidal positional embeddings."""
        assert channels % 2 == 0
        log_timescale_increment = math.log(max_timescale) / (channels // 2 - 1)
        inv_timescales = mx.exp(-log_timescale_increment * mx.arange(channels // 2))
        scaled_time = mx.arange(length)[:, None] * inv_timescales[None, :]
        return mx.concatenate([mx.sin(scaled_time), mx.cos(scaled_time)], axis=1)


class TextDecoder(nn.Module):
    """Whisper text decoder.

    Token + positional embeddings followed by transformer blocks with
    cross-attention to the encoder output.
    """

    def __init__(
        self,
        n_vocab: int,
        n_ctx: int,
        n_state: int,
        n_head: int,
        n_layer: int,
        dtype: mx.Dtype = mx.float16,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = mx.zeros((n_ctx, n_state))
        self.blocks = [
            ResidualAttentionBlock(n_state, n_head, cross_attention=True)
            for _ in range(n_layer)
        ]
        self.ln = nn.LayerNorm(n_state)
        self._mask = nn.MultiHeadAttention.create_additive_causal_mask(n_ctx).astype(
            dtype
        )

    def __call__(
        self,
        x: mx.array,
        xa: mx.array,
        kv_cache: list[Any] | None = None,
    ) -> tuple[mx.array, list[Any], list[Any]]:
        offset = kv_cache[0][0][0].shape[1] if kv_cache else 0
        x = (
            self.token_embedding(x)
            + self.positional_embedding[offset : offset + x.shape[-1]]
        )

        kv_cache_out: list[Any] = (
            kv_cache if kv_cache is not None else [None] * len(self.blocks)
        )
        cross_qk: list[Any] = [None] * len(self.blocks)

        for i, block in enumerate(self.blocks):
            x, kv_cache_out[i], cross_qk[i] = block(
                x, xa, mask=self._mask, kv_cache=kv_cache_out[i]
            )

        x = self.ln(x)
        return self.token_embedding.as_linear(x), kv_cache_out, cross_qk


# ===========================================================================
# Full model
# ===========================================================================


class WhisperModel(nn.Module):
    """Whisper speech recognition model.

    Combines :class:`AudioEncoder` and :class:`TextDecoder` into a single
    module with weight sanitization for HuggingFace compatibility.
    """

    model_type = "whisper"

    def __init__(self, config: WhisperConfig, dtype: mx.Dtype = mx.float16):
        super().__init__()
        self.config = config
        self.dtype = dtype

        self.encoder = AudioEncoder(
            config.n_mels,
            config.n_audio_ctx,
            config.n_audio_state,
            config.n_audio_head,
            config.n_audio_layer,
            dtype,
        )
        self.decoder = TextDecoder(
            config.n_vocab,
            config.n_text_ctx,
            config.n_text_state,
            config.n_text_head,
            config.n_text_layer,
            dtype,
        )

        # Alignment heads for word timestamps
        all_heads = np.zeros((config.n_text_layer, config.n_text_head), dtype=bool)
        all_heads[config.n_text_layer // 2 :] = True
        self._alignment_heads = mx.array(np.asarray(all_heads.nonzero()).T)

    def create_runtime_adapter(self, model_path: str) -> STTRuntimeAdapter:
        """Create the model-owned runtime adapter used by the vLLM runner."""
        # Local import: avoid import-time cycles (adapter imports transcriber).
        from .adapter import WhisperRuntimeAdapter

        return WhisperRuntimeAdapter(self, model_path)

    def encode(self, mel: mx.array) -> mx.array:
        """Encode audio mel spectrogram.

        Args:
            mel: Mel spectrogram of shape ``(batch, n_frames, n_mels)``.

        Returns:
            Encoder hidden states.
        """
        return self.encoder(mel)

    def decode(
        self,
        tokens: mx.array,
        audio_features: mx.array,
        kv_cache: list | None = None,
    ) -> tuple[mx.array, list]:
        """Decode tokens given audio features.

        Args:
            tokens: Token IDs of shape ``(batch, seq_len)``.
            audio_features: Encoder output.
            kv_cache: Optional cached key-value pairs.

        Returns:
            Tuple of (logits, updated kv_cache).
        """
        logits, kv_cache, _ = self.decoder(tokens, audio_features, kv_cache)
        return logits, kv_cache

    def __call__(self, mel: mx.array, tokens: mx.array) -> mx.array:
        """Forward pass: mel spectrogram + tokens -> logits."""
        return self.decoder(tokens, self.encoder(mel))[0]

    @property
    def is_multilingual(self) -> bool:
        return self.config.n_vocab >= 51865

    @property
    def num_languages(self) -> int:
        return self.config.n_vocab - 51765 - int(self.is_multilingual)

    # HuggingFace -> MLX weight key mapping.
    # Each tuple is (old_substring, new_substring).  ``None`` means skip.
    _HF_KEY_MAP = [
        ("encoder.embed_positions.weight", None),
        ("decoder.embed_positions.weight", "decoder.positional_embedding"),
        ("encoder.layer_norm.", "encoder.ln_post."),
        ("decoder.layer_norm.", "decoder.ln."),
        ("encoder.layers.", "encoder.blocks."),
        ("decoder.layers.", "decoder.blocks."),
        (".self_attn_layer_norm.", ".attn_ln."),
        (".final_layer_norm.", ".mlp_ln."),
        (".encoder_attn_layer_norm.", ".cross_attn_ln."),
        (".fc1.", ".mlp1."),
        (".fc2.", ".mlp2."),
        (".self_attn.q_proj.", ".attn.query."),
        (".self_attn.k_proj.", ".attn.key."),
        (".self_attn.v_proj.", ".attn.value."),
        (".self_attn.out_proj.", ".attn.out."),
        (".encoder_attn.q_proj.", ".cross_attn.query."),
        (".encoder_attn.k_proj.", ".cross_attn.key."),
        (".encoder_attn.v_proj.", ".cross_attn.value."),
        (".encoder_attn.out_proj.", ".cross_attn.out."),
        ("decoder.embed_tokens.", "decoder.token_embedding."),
    ]

    def sanitize(self, weights: dict) -> dict:
        """Sanitize weight keys and dtypes for MLX compatibility.

        Handles HuggingFace-format keys (``model.encoder.layers.*``) by
        renaming them to the MLX layout, transposes Conv1d weights, and
        casts to the model dtype.

        Args:
            weights: Raw weight dictionary.

        Returns:
            Sanitized weight dictionary ready for ``load_weights``.
        """
        is_hf_format = any(k.startswith("model.") for k in weights)

        sanitized = {}
        for k, v in weights.items():
            if is_hf_format:
                if k.startswith("model."):
                    k = k[6:]

                skip = False
                for old, new in self._HF_KEY_MAP:
                    if old in k:
                        if new is None:
                            skip = True
                            break
                        k = k.replace(old, new)

                if skip:
                    continue

                # Transpose Conv1d weights: HF (out, in, kernel) -> MLX (out, kernel, in)
                if "conv1.weight" in k or "conv2.weight" in k:
                    if v.ndim == 3:
                        v = v.transpose(0, 2, 1)

            if v.dtype != self.dtype and v.dtype != mx.uint32:
                v = v.astype(self.dtype)

            sanitized[k] = v
        return sanitized
