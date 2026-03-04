# SPDX-License-Identifier: Apache-2.0
"""Tests for Whisper model: weight sanitization and decoder cache paths."""

from __future__ import annotations

import mlx.core as mx
import pytest

from vllm_metal.stt.whisper import WhisperConfig, WhisperModel


@pytest.fixture()
def model():
    """Create a minimal WhisperModel for testing."""
    return WhisperModel(WhisperConfig(), dtype=mx.float16)


# ===========================================================================
# Weight sanitization
# ===========================================================================


class TestWeightSanitize:
    """Tests for WhisperModel.sanitize() weight mapping."""

    def test_sanitize_hf_key_rename(self, model) -> None:
        """HuggingFace keys should be renamed to MLX format."""
        weights = {
            "model.encoder.layers.0.self_attn.q_proj.weight": mx.zeros((512, 512)),
        }
        sanitized = model.sanitize(weights)
        assert "encoder.blocks.0.attn.query.weight" in sanitized
        assert "model.encoder.layers.0.self_attn.q_proj.weight" not in sanitized

    def test_sanitize_skips_encoder_positions(self, model) -> None:
        """encoder.embed_positions should be skipped (None mapping)."""
        weights = {
            "model.encoder.embed_positions.weight": mx.zeros((1500, 512)),
            "model.decoder.embed_tokens.weight": mx.zeros((51865, 512)),
        }
        sanitized = model.sanitize(weights)
        assert "encoder.embed_positions.weight" not in sanitized
        assert "decoder.token_embedding.weight" in sanitized

    def test_sanitize_transposes_conv_weights(self, model) -> None:
        """Conv1d weights should be transposed from HF format."""
        hf_conv = mx.zeros((512, 80, 3))
        weights = {"model.encoder.conv1.weight": hf_conv}
        sanitized = model.sanitize(weights)
        assert sanitized["encoder.conv1.weight"].shape == (512, 3, 80)

    def test_sanitize_preserves_mlx_format(self, model) -> None:
        """Already-MLX-format weights pass through unchanged."""
        weights = {
            "encoder.blocks.0.attn.query.weight": mx.zeros((512, 512)),
        }
        sanitized = model.sanitize(weights)
        assert "encoder.blocks.0.attn.query.weight" in sanitized

    def test_sanitize_casts_dtype(self, model) -> None:
        """Weights should be cast to model dtype."""
        weights = {"encoder.ln_post.weight": mx.ones((512,), dtype=mx.float32)}
        sanitized = model.sanitize(weights)
        assert sanitized["encoder.ln_post.weight"].dtype == mx.float16


# ===========================================================================
# Decoder cache paths
# ===========================================================================


class TestDecoderCachePaths:
    """Tests for decoder self-attention with and without KV cache."""

    @pytest.fixture()
    def tiny_model(self):
        """Create a tiny model for fast decode tests.

        n_audio_ctx must equal input_frames // 2 because conv2 has stride=2.
        We use input frames = 20 -> conv2 output = 10 -> n_audio_ctx = 10.
        """
        config = WhisperConfig(
            n_mels=80,
            n_audio_ctx=10,
            n_audio_state=64,
            n_audio_head=2,
            n_audio_layer=1,
            n_vocab=100,
            n_text_ctx=32,
            n_text_state=64,
            n_text_head=2,
            n_text_layer=1,
        )
        return WhisperModel(config, dtype=mx.float32)

    def test_prefill_without_cache(self, tiny_model) -> None:
        """Prefill (no cache) should produce logits without error."""
        mel = mx.random.normal((1, 20, 80))
        tokens = mx.array([[1, 2, 3]])

        audio_features = tiny_model.encode(mel)
        logits, kv_cache = tiny_model.decode(tokens, audio_features)

        assert logits.shape == (1, 3, 100)
        assert kv_cache is not None
        assert len(kv_cache) == 1  # 1 layer

    def test_cached_decode_single_token(self, tiny_model) -> None:
        """Decode a single token with cache should work."""
        mel = mx.random.normal((1, 20, 80))
        tokens_prefill = mx.array([[1, 2, 3]])

        audio_features = tiny_model.encode(mel)
        _, kv_cache = tiny_model.decode(tokens_prefill, audio_features)

        # Decode 1 new token
        next_token = mx.array([[4]])
        logits, kv_cache2 = tiny_model.decode(next_token, audio_features, kv_cache)

        assert logits.shape == (1, 1, 100)
        # Self-attn cache k should now have 4 tokens (3 prefill + 1 new)
        assert kv_cache2[0][0][0].shape[1] == 4

    def test_cached_decode_multiple_tokens(self, tiny_model) -> None:
        """Decode q_len > 1 with cache — the mask bug repro case."""
        mel = mx.random.normal((1, 20, 80))
        tokens_prefill = mx.array([[1, 2, 3]])

        audio_features = tiny_model.encode(mel)
        _, kv_cache = tiny_model.decode(tokens_prefill, audio_features)

        # Decode 2 tokens at once with cache (q_len=2, k_len=5)
        next_tokens = mx.array([[4, 5]])
        logits, kv_cache2 = tiny_model.decode(next_tokens, audio_features, kv_cache)

        assert logits.shape == (1, 2, 100)
        assert kv_cache2[0][0][0].shape[1] == 5

    def test_cached_vs_full_decode_match(self, tiny_model) -> None:
        """Cached decode should produce same logits as full non-cached decode."""
        mx.random.seed(42)
        mel = mx.random.normal((1, 20, 80))
        all_tokens = mx.array([[1, 2, 3, 4, 5]])

        audio_features = tiny_model.encode(mel)

        # Full decode without cache
        logits_full, _ = tiny_model.decode(all_tokens, audio_features)

        # Incremental: prefill 3, then decode 2
        tokens_prefill = mx.array([[1, 2, 3]])
        _, kv_cache = tiny_model.decode(tokens_prefill, audio_features)

        next_tokens = mx.array([[4, 5]])
        logits_cached, _ = tiny_model.decode(next_tokens, audio_features, kv_cache)

        # Last 2 logits should match
        mx.eval(logits_full, logits_cached)
        diff = mx.abs(logits_full[:, 3:, :] - logits_cached).max().item()
        assert diff < 1e-4, f"Cached decode diverged from full decode: max diff={diff}"
