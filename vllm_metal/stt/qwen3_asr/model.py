# SPDX-License-Identifier: Apache-2.0
"""Qwen3-ASR model implementation for MLX.

Conv2d audio encoder + Qwen3 causal LM decoder for speech recognition.
Audio embeddings are injected into the token sequence at audio_pad positions.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from vllm_metal.stt.runtime import STTRuntimeAdapter

from .config import Qwen3ASRAudioConfig, Qwen3ASRConfig, Qwen3ASRTextConfig

# ===========================================================================
# Audio Encoder Components
# ===========================================================================


class AudioEncoderAttention(nn.Module):
    """Multi-head attention for the audio encoder."""

    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        b, seq, _ = x.shape
        scale = self.head_dim**-0.5

        q = (
            self.q_proj(x)
            .reshape(b, seq, self.n_head, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        k = (
            self.k_proj(x)
            .reshape(b, seq, self.n_head, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        v = (
            self.v_proj(x)
            .reshape(b, seq, self.n_head, self.head_dim)
            .transpose(0, 2, 1, 3)
        )

        w = (q * scale) @ k.transpose(0, 1, 3, 2)
        if mask is not None:
            w = w + mask
        w = mx.softmax(w, axis=-1, precise=True)
        out = (w @ v).transpose(0, 2, 1, 3).reshape(b, seq, -1)
        return self.out_proj(out)


class AudioEncoderLayer(nn.Module):
    """Pre-Norm transformer block: LN→Attn→LN→FFN(GELU)."""

    def __init__(self, d_model: int, n_head: int, ffn_dim: int):
        super().__init__()
        self.self_attn = AudioEncoderAttention(d_model, n_head)
        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, d_model)
        self.final_layer_norm = nn.LayerNorm(d_model)

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        r = self.self_attn(self.self_attn_layer_norm(x), mask=mask)
        x = x + r
        r = self.fc2(nn.gelu(self.fc1(self.final_layer_norm(x))))
        x = x + r
        return x


class AudioEncoder(nn.Module):
    """Qwen3-ASR audio encoder: 3x Conv2d + transformer layers + output proj."""

    def __init__(self, config: Qwen3ASRAudioConfig, dtype: mx.Dtype = mx.float16):
        super().__init__()
        self._config = config
        self._dtype = dtype

        # Conv2d layers for mel downsampling
        # MLX Conv2d uses NHWC layout
        self.conv2d1 = nn.Conv2d(
            1, config.downsample_hidden_size, 3, stride=2, padding=1
        )
        self.conv2d2 = nn.Conv2d(
            config.downsample_hidden_size,
            config.downsample_hidden_size,
            3,
            stride=2,
            padding=1,
        )
        self.conv2d3 = nn.Conv2d(
            config.downsample_hidden_size,
            config.downsample_hidden_size,
            3,
            stride=2,
            padding=1,
        )

        # Compute frequency dimension after 3x stride-2 on n_mels
        freq_out = config.num_mel_bins
        for _ in range(3):
            freq_out = (freq_out + 2 * 1 - 3) // 2 + 1
        conv_out_dim = config.downsample_hidden_size * freq_out

        self.conv_out = nn.Linear(conv_out_dim, config.d_model, bias=False)

        # Positional embedding (sinusoidal, non-learned)
        self._positional_embedding = self.sinusoidal_position_embedding(
            config.max_source_positions, config.d_model, dtype
        )

        # Transformer encoder layers
        self.layers = [
            AudioEncoderLayer(
                config.d_model, config.encoder_attention_heads, config.encoder_ffn_dim
            )
            for _ in range(config.encoder_layers)
        ]

        # Output projection
        self.ln_post = nn.LayerNorm(config.d_model)
        self.proj1 = nn.Linear(config.d_model, config.d_model)
        self.proj2 = nn.Linear(config.d_model, config.output_dim)

    def __call__(self, mel: mx.array) -> mx.array:
        """Encode mel spectrogram to audio embeddings.

        Args:
            mel: Shape ``(n_mels, time)`` or ``(batch, n_mels, time)``.

        Returns:
            Audio embeddings of shape ``(total_frames, output_dim)``.
        """
        if mel.ndim == 2:
            mel = mel[None, ...]
        _, n_mels, time = mel.shape

        chunk_size = self._config.n_window * 2
        n_full = time // chunk_size
        remainder = time % chunk_size
        n_chunks = n_full + (1 if remainder > 0 else 0)

        if n_chunks == 0:
            return mx.zeros((0, self._config.output_dim))

        # Build padded chunks: (n_chunks, n_mels, chunk_size)
        chunk_list = []
        chunk_lengths = []
        for i in range(n_full):
            chunk_list.append(mel[0, :, i * chunk_size : (i + 1) * chunk_size])
            chunk_lengths.append(chunk_size)
        if remainder > 0:
            chunk = mel[0, :, n_full * chunk_size :]
            pad = mx.zeros((n_mels, chunk_size - remainder), dtype=mel.dtype)
            chunk = mx.concatenate([chunk, pad], axis=1)
            chunk_list.append(chunk)
            chunk_lengths.append(remainder)

        padded = mx.stack(chunk_list)  # (n_chunks, n_mels, chunk_size)

        # NHWC for Conv2d: (n_chunks, n_mels, chunk_size, 1)
        x = padded[..., None]
        x = nn.gelu(self.conv2d1(x))
        x = nn.gelu(self.conv2d2(x))
        x = nn.gelu(self.conv2d3(x))

        # x: (n_chunks, freq_out, time_out, channels)  [MLX NHWC]
        # -> (n_chunks, time_out, channels * freq_out)
        # Must match PyTorch's (b,c,f,t).permute(0,3,1,2).view(b,t,c*f)
        b, f, t, c = x.shape
        x = x.transpose(0, 2, 3, 1).reshape(b, t, c * f)

        # Linear projection
        x = self.conv_out(x)  # (n_chunks, time_out, d_model)

        # Add positional embedding (same positions for each chunk)
        pos = self._positional_embedding[: x.shape[1], :][None, :]
        x = x + pos.astype(x.dtype)

        # Compute valid lengths after CNN per chunk
        cnn_lengths = [
            Qwen3ASRAudioConfig.cnn_output_length(cl) for cl in chunk_lengths
        ]

        # Extract valid frames per chunk
        valid = [x[i, : cnn_lengths[i], :] for i in range(n_chunks)]

        # Group into inference windows and process through transformer
        chunks_per_window = self._config.n_window_infer // chunk_size
        processed = []
        for w_start in range(0, n_chunks, chunks_per_window):
            w_end = min(w_start + chunks_per_window, n_chunks)
            window = mx.concatenate(valid[w_start:w_end], axis=0)  # (frames, d_model)
            window = window[None, ...]  # (1, frames, d_model)
            for layer in self.layers:
                window = layer(window)
            processed.append(window[0])

        hidden = mx.concatenate(processed, axis=0)  # (total_frames, d_model)

        # Output projection
        hidden = self.ln_post(hidden)
        hidden = nn.gelu(self.proj1(hidden))
        hidden = self.proj2(hidden)
        return hidden  # (total_frames, output_dim)

    @staticmethod
    def sinusoidal_position_embedding(
        max_len: int, d_model: int, dtype: mx.Dtype = mx.float32
    ) -> mx.array:
        """Generate sinusoidal positional embeddings (non-learned)."""
        assert d_model % 2 == 0
        half = d_model // 2
        log_timescale = math.log(10000.0) / (half - 1)
        inv_timescales = mx.exp(-log_timescale * mx.arange(half))
        positions = mx.arange(max_len)[:, None] * inv_timescales[None, :]
        return mx.concatenate([mx.sin(positions), mx.cos(positions)], axis=1).astype(
            dtype
        )


# ===========================================================================
# Text Decoder (Qwen3 LM) Components
# ===========================================================================


class Qwen3RMSNorm(nn.Module):
    """RMS normalization."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((hidden_size,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.weight, self.eps)


class Qwen3RotaryEmbedding(nn.Module):
    """Rotary position embedding with interleaved application."""

    def __init__(self, head_dim: int, rope_theta: float = 1000000.0):
        super().__init__()
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        # Precompute inverse frequencies
        inv_freq = 1.0 / (
            rope_theta ** (mx.arange(0, head_dim, 2).astype(mx.float32) / head_dim)
        )
        self._inv_freq = inv_freq

    def __call__(self, x: mx.array, offset: int = 0) -> mx.array:
        """Apply rotary embedding to x.

        Args:
            x: (batch, n_heads, seq_len, head_dim)
            offset: position offset for cached decoding.
        """
        seq_len = x.shape[2]
        positions = mx.arange(offset, offset + seq_len, dtype=mx.float32)
        freqs = positions[:, None] * self._inv_freq[None, :]  # (seq, head_dim/2)
        # Interleaved: apply cos/sin to consecutive pairs
        cos_f = mx.cos(freqs).astype(x.dtype)
        sin_f = mx.sin(freqs).astype(x.dtype)

        # Reshape for interleaved application
        # x pairs: (x0, x1), (x2, x3), ...
        x1 = x[..., 0::2]  # even indices
        x2 = x[..., 1::2]  # odd indices

        # Rotate
        o1 = x1 * cos_f - x2 * sin_f
        o2 = x1 * sin_f + x2 * cos_f

        # Interleave back: stack pairs then reshape
        return mx.stack([o1, o2], axis=-1).reshape(x.shape)


class Qwen3Attention(nn.Module):
    """Grouped-query attention with QK normalization."""

    def __init__(self, config: Qwen3ASRTextConfig, dtype: mx.Dtype = mx.float16):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.n_rep = self.n_heads // self.n_kv_heads

        self.q_proj = nn.Linear(
            config.hidden_size, self.n_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.n_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.n_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim, config.hidden_size, bias=False
        )

        # QK normalization (per head_dim RMSNorm)
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.rope = Qwen3RotaryEmbedding(self.head_dim, config.rope_theta)
        self.scale = self.head_dim**-0.5

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: tuple[mx.array, mx.array] | None = None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array]]:
        b, seq, _ = x.shape

        q = self.q_proj(x).reshape(b, seq, self.n_heads, self.head_dim)
        k = self.k_proj(x).reshape(b, seq, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(b, seq, self.n_kv_heads, self.head_dim)

        # QK normalization
        q = self.q_norm(q)
        k = self.k_norm(k)

        # (B, heads, L, head_dim)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # Apply RoPE
        offset = cache[0].shape[2] if cache is not None else 0
        q = self.rope(q, offset=offset)
        k = self.rope(k, offset=offset)

        # KV cache
        if cache is not None:
            k = mx.concatenate([cache[0], k], axis=2)
            v = mx.concatenate([cache[1], v], axis=2)
        new_cache = (k, v)

        # GQA: expand KV heads
        if self.n_rep > 1:
            k = mx.repeat(k, self.n_rep, axis=1)
            v = mx.repeat(v, self.n_rep, axis=1)

        # Attention
        w = (q * self.scale) @ k.transpose(0, 1, 3, 2)
        if mask is not None:
            w = w + mask
        w = mx.softmax(w, axis=-1, precise=True)
        out = (w @ v).transpose(0, 2, 1, 3).reshape(b, seq, -1)
        return self.o_proj(out), new_cache


class Qwen3MLP(nn.Module):
    """SiLU gated MLP: gate(SiLU) * up → down."""

    def __init__(self, config: Qwen3ASRTextConfig):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3DecoderLayer(nn.Module):
    """Qwen3 transformer decoder layer."""

    def __init__(self, config: Qwen3ASRTextConfig, dtype: mx.Dtype = mx.float16):
        super().__init__()
        self.self_attn = Qwen3Attention(config, dtype)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(
            config.hidden_size, config.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: tuple[mx.array, mx.array] | None = None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array]]:
        r, new_cache = self.self_attn(self.input_layernorm(x), mask=mask, cache=cache)
        x = x + r
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, new_cache


class Qwen3LM(nn.Module):
    """Qwen3 language model: embeddings + decoder layers + lm_head."""

    def __init__(self, config: Qwen3ASRTextConfig, dtype: mx.Dtype = mx.float16):
        super().__init__()
        self.config = config
        self._dtype = dtype
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            Qwen3DecoderLayer(config, dtype) for _ in range(config.num_hidden_layers)
        ]
        self.norm = Qwen3RMSNorm(config.hidden_size, config.rms_norm_eps)
        if config.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def embed(self, token_ids: mx.array) -> mx.array:
        """Convert token IDs to embeddings."""
        return self.embed_tokens(token_ids)

    def forward_embeds(
        self,
        embeds: mx.array,
        cache: list[tuple[mx.array, mx.array] | None] | None = None,
    ) -> tuple[mx.array, list[tuple[mx.array, mx.array]]]:
        """Forward pass from embeddings. Returns (logits, cache)."""
        _, seq, _ = embeds.shape
        h = embeds

        # Build causal mask
        if cache is not None and cache[0] is not None:
            offset = cache[0][0].shape[2]
        else:
            offset = 0
        total_len = offset + seq
        if seq > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(total_len)
            mask = mask.astype(h.dtype)
            mask = mask[-seq:, :total_len]
        else:
            mask = None

        if cache is None:
            cache = [None] * len(self.layers)

        new_cache = []
        for i, layer in enumerate(self.layers):
            h, layer_cache = layer(h, mask=mask, cache=cache[i])
            new_cache.append(layer_cache)

        h = self.norm(h)
        if self.lm_head is not None:
            logits = self.lm_head(h)
        else:
            logits = self.embed_tokens.as_linear(h)
        return logits, new_cache


# ===========================================================================
# Full Model
# ===========================================================================


class Qwen3ASRModel(nn.Module):
    """Qwen3-ASR: audio encoder + Qwen3 language model.

    Inference pipeline:
    1. Encode mel → audio embeddings
    2. Embed prompt tokens, inject audio embeddings at audio_pad positions
    3. Prefill through LM → logits + KV cache
    4. Autoregressive decode until EOS
    """

    model_type = "qwen3_asr"

    def __init__(self, config: Qwen3ASRConfig, dtype: mx.Dtype = mx.float16):
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.audio_tower = AudioEncoder(config.audio_config, dtype)
        self.language_model = Qwen3LM(config.text_config, dtype)

    def encode(self, mel: mx.array) -> mx.array:
        """Encode mel spectrogram to audio embeddings.

        Args:
            mel: ``(n_mels, time)`` or ``(batch, n_mels, time)``.

        Returns:
            Audio embeddings ``(total_frames, output_dim)``.
        """
        return self.audio_tower(mel)

    def prefill(
        self,
        token_ids: mx.array,
        audio_embeddings: mx.array,
    ) -> tuple[mx.array, list]:
        """Forward pass with audio embedding injection.

        Args:
            token_ids: (1, seq_len) token IDs with audio_pad placeholders.
            audio_embeddings: (n_audio_frames, hidden_size) from encode().

        Returns:
            (logits, cache) tuple.
        """
        # Embed all tokens
        embeds = self.language_model.embed(token_ids)  # (1, seq_len, hidden)

        # Find audio_pad positions and inject audio embeddings
        token_list = token_ids[0].tolist()
        audio_pad_id = self.config.audio_token_id
        audio_positions = [i for i, t in enumerate(token_list) if t == audio_pad_id]

        if audio_positions and audio_embeddings.shape[0] > 0:
            n_inject = min(len(audio_positions), audio_embeddings.shape[0])
            # Build updated embeddings by replacing audio_pad positions
            parts = []
            prev = 0
            for idx in range(n_inject):
                pos = audio_positions[idx]
                if pos > prev:
                    parts.append(embeds[0, prev:pos, :])
                parts.append(audio_embeddings[idx : idx + 1].astype(embeds.dtype))
                prev = pos + 1
            if prev < embeds.shape[1]:
                parts.append(embeds[0, prev:, :])
            embeds = mx.concatenate(parts, axis=0)[None, ...]

        return self.language_model.forward_embeds(embeds)

    def decode_step(
        self,
        token_id: mx.array,
        cache: list,
    ) -> tuple[mx.array, list]:
        """Single autoregressive decode step.

        Args:
            token_id: (1, 1) token ID.
            cache: KV cache from prefill or previous step.

        Returns:
            (logits, updated_cache) tuple.
        """
        embeds = self.language_model.embed(token_id)
        return self.language_model.forward_embeds(embeds, cache)

    def create_runtime_adapter(self, model_path: str) -> STTRuntimeAdapter:
        """Create the model-owned runtime adapter used by the vLLM runner."""
        # Local import: avoid import-time cycles (adapter imports transcriber).
        from .adapter import Qwen3ASRRuntimeAdapter

        return Qwen3ASRRuntimeAdapter(self, model_path)

    def sanitize(self, weights: dict) -> dict:
        """Map HF weight keys to MLX model structure.

        Handles:
        - Strip ``thinker.`` prefix
        - ``thinker.model.*`` → ``language_model.*``
        - ``thinker.lm_head.*`` → ``language_model.lm_head.*``
        - Conv2d transpose: NCHW → NHWC
        - Dtype casting
        """
        sanitized = {}
        for k, v in weights.items():
            new_k = k

            # Strip thinker. prefix
            if new_k.startswith("thinker."):
                new_k = new_k[len("thinker.") :]

            # Map model.* → language_model.*
            if new_k.startswith("model."):
                new_k = "language_model." + new_k[len("model.") :]

            # Map lm_head.* → language_model.lm_head.*
            if new_k.startswith("lm_head."):
                # Skip lm_head when weights are tied to embed_tokens
                if self.config.text_config.tie_word_embeddings:
                    continue
                new_k = "language_model." + new_k

            # Conv2d weight transpose: PyTorch NCHW (O,I,H,W) → MLX NHWC (O,H,W,I)
            if "conv2d" in new_k and "weight" in new_k and v.ndim == 4:
                v = v.transpose(0, 2, 3, 1)

            # Cast dtype (skip quantization scales)
            if v.dtype != self.dtype and v.dtype != mx.uint32:
                v = v.astype(self.dtype)

            sanitized[new_k] = v
        return sanitized
