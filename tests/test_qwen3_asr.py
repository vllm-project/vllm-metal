# SPDX-License-Identifier: Apache-2.0
"""Tests for Qwen3-ASR model: config, encoder shapes, weight sanitization."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock

import mlx.core as mx
import pytest

from vllm_metal.stt.config import is_stt_model
from vllm_metal.stt.loader import load_model
from vllm_metal.stt.qwen3_asr.config import (
    Qwen3ASRAudioConfig,
    Qwen3ASRConfig,
    Qwen3ASRTextConfig,
)
from vllm_metal.stt.qwen3_asr.model import (
    AudioEncoder,
    Qwen3ASRModel,
    Qwen3Attention,
    Qwen3LM,
)
from vllm_metal.stt.qwen3_asr.transcriber import Qwen3ASRTranscriber

# ===========================================================================
# Configuration
# ===========================================================================


class TestQwen3ASRConfig:
    """Tests for Qwen3ASRConfig.from_dict with 0.6B config."""

    def test_from_dict_basic(self) -> None:
        """Config should be parsed from nested thinker_config dict."""
        d = {
            "model_type": "qwen3_asr",
            "thinker_config": {
                "audio_config": {
                    "d_model": 896,
                    "num_mel_bins": 128,
                    "encoder_layers": 18,
                    "encoder_attention_heads": 14,
                    "encoder_ffn_dim": 3584,
                    "downsample_hidden_size": 480,
                    "output_dim": 1024,
                    "max_source_positions": 1500,
                    "n_window": 50,
                    "n_window_infer": 800,
                },
                "text_config": {
                    "hidden_size": 1024,
                    "num_hidden_layers": 28,
                    "num_attention_heads": 16,
                    "num_key_value_heads": 8,
                    "head_dim": 128,
                    "intermediate_size": 3072,
                    "vocab_size": 151936,
                    "rms_norm_eps": 1e-6,
                    "rope_theta": 1000000.0,
                    "tie_word_embeddings": True,
                },
                "audio_token_id": 151676,
                "audio_start_token_id": 151669,
                "audio_end_token_id": 151670,
            },
        }
        config = Qwen3ASRConfig.from_dict(d)
        assert config.audio_config.d_model == 896
        assert config.audio_config.encoder_layers == 18
        assert config.audio_config.num_mel_bins == 128
        assert config.audio_config.n_window == 50
        assert config.text_config.hidden_size == 1024
        assert config.text_config.num_hidden_layers == 28
        assert config.text_config.num_attention_heads == 16
        assert config.text_config.num_key_value_heads == 8
        assert config.audio_token_id == 151676
        assert config.n_mels == 128
        assert config.n_audio_ctx == 1500

    def test_defaults(self) -> None:
        """Default config should have 0.6B model values."""
        config = Qwen3ASRConfig()
        assert config.audio_config.d_model == 896
        assert config.text_config.vocab_size == 151936
        assert config.eos_token_id == 151643


# ===========================================================================
# CNN output lengths
# ===========================================================================


class TestCNNOutputLengths:
    """Tests for Qwen3ASRAudioConfig shape helpers."""

    def test_single_conv_stride(self) -> None:
        """3x Conv2d stride-2 on 100 frames → 13 output frames."""
        assert Qwen3ASRAudioConfig.cnn_output_length(100) == 13

    def test_small_inputs(self) -> None:
        """Edge cases for small input lengths."""
        assert Qwen3ASRAudioConfig.cnn_output_length(1) == 1
        assert Qwen3ASRAudioConfig.cnn_output_length(2) == 1
        # 3 -> 2 -> 1 -> 1 after 3x stride-2
        assert Qwen3ASRAudioConfig.cnn_output_length(3) == 1

    def test_feat_extract_full_chunks(self) -> None:
        """Full chunks of 100 frames each produce 13 frames per chunk."""
        cfg = Qwen3ASRAudioConfig()
        assert cfg.feat_extract_output_length(100) == 13
        assert cfg.feat_extract_output_length(200) == 26
        assert cfg.feat_extract_output_length(300) == 39

    def test_feat_extract_with_remainder(self) -> None:
        """Partial chunk adds its CNN output to full chunks."""
        # 150 = 1 full chunk (13) + 50 remainder
        cfg = Qwen3ASRAudioConfig()
        remainder_out = Qwen3ASRAudioConfig.cnn_output_length(50)
        assert cfg.feat_extract_output_length(150) == 13 + remainder_out

    def test_feat_extract_3000_frames(self) -> None:
        """30 seconds at 16kHz/hop160 = 3000 frames → 30 * 13 = 390."""
        assert Qwen3ASRAudioConfig().feat_extract_output_length(3000) == 390


# ===========================================================================
# Audio Encoder shapes
# ===========================================================================


class TestAudioEncoderShapes:
    """Tests for AudioEncoder output dimensions."""

    @pytest.fixture()
    def tiny_encoder(self):
        """Create a tiny audio encoder for shape tests."""
        config = Qwen3ASRAudioConfig(
            num_mel_bins=16,
            d_model=32,
            encoder_layers=1,
            encoder_attention_heads=2,
            encoder_ffn_dim=64,
            downsample_hidden_size=8,
            output_dim=48,
            max_source_positions=100,
            n_window=50,
            n_window_infer=800,
        )
        return AudioEncoder(config, dtype=mx.float32)

    def test_single_chunk(self, tiny_encoder) -> None:
        """Input shorter than chunk_size should produce correct output."""
        mel = mx.random.normal((16, 80))  # 80 < 100 frames
        out = tiny_encoder(mel)
        mx.eval(out)
        expected_frames = Qwen3ASRAudioConfig.cnn_output_length(80)
        assert out.shape == (expected_frames, 48)

    def test_exact_chunk(self, tiny_encoder) -> None:
        """Input exactly one chunk (100 frames) → 13 output frames."""
        mel = mx.random.normal((16, 100))
        out = tiny_encoder(mel)
        mx.eval(out)
        assert out.shape == (13, 48)

    def test_multiple_chunks(self, tiny_encoder) -> None:
        """Two full chunks → 26 output frames."""
        mel = mx.random.normal((16, 200))
        out = tiny_encoder(mel)
        mx.eval(out)
        assert out.shape == (26, 48)

    def test_with_batch_dim(self, tiny_encoder) -> None:
        """3D input (batch, n_mels, time) should also work."""
        mel = mx.random.normal((1, 16, 100))
        out = tiny_encoder(mel)
        mx.eval(out)
        assert out.shape == (13, 48)


# ===========================================================================
# Qwen3 Attention
# ===========================================================================


class TestQwen3Attention:
    """Tests for GQA with QK normalization."""

    def test_head_counts(self) -> None:
        """GQA should have correct head counts."""
        config = Qwen3ASRTextConfig(
            hidden_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            intermediate_size=128,
            vocab_size=100,
        )
        attn = Qwen3Attention(config, dtype=mx.float32)
        assert attn.n_heads == 4
        assert attn.n_kv_heads == 2
        assert attn.n_rep == 2

    def test_output_shape(self) -> None:
        """Attention output should match input shape."""
        config = Qwen3ASRTextConfig(
            hidden_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            intermediate_size=128,
            vocab_size=100,
        )
        attn = Qwen3Attention(config, dtype=mx.float32)
        x = mx.random.normal((1, 5, 64))
        out, cache = attn(x)
        mx.eval(out)
        assert out.shape == (1, 5, 64)
        assert cache[0].shape == (1, 2, 5, 16)  # k: (B, n_kv_heads, L, head_dim)

    def test_cached_decode(self) -> None:
        """Cached decode should extend KV cache."""
        config = Qwen3ASRTextConfig(
            hidden_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            intermediate_size=128,
            vocab_size=100,
        )
        attn = Qwen3Attention(config, dtype=mx.float32)
        x = mx.random.normal((1, 5, 64))
        _, cache = attn(x)

        x2 = mx.random.normal((1, 1, 64))
        out2, cache2 = attn(x2, cache=cache)
        mx.eval(out2)
        assert out2.shape == (1, 1, 64)
        assert cache2[0].shape == (1, 2, 6, 16)  # 5 + 1 = 6


# ===========================================================================
# Weight sanitization
# ===========================================================================


class TestWeightSanitize:
    """Tests for Qwen3ASRModel.sanitize() weight mapping."""

    @pytest.fixture()
    def model(self):
        """Create minimal model for sanitize tests."""
        config = Qwen3ASRConfig(
            audio_config=Qwen3ASRAudioConfig(
                num_mel_bins=16,
                d_model=32,
                encoder_layers=1,
                encoder_attention_heads=2,
                encoder_ffn_dim=64,
                downsample_hidden_size=8,
                output_dim=48,
            ),
            text_config=Qwen3ASRTextConfig(
                hidden_size=48,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=24,
                intermediate_size=96,
                vocab_size=100,
            ),
        )
        return Qwen3ASRModel(config, dtype=mx.float32)

    def test_strips_thinker_prefix(self, model) -> None:
        """'thinker.' prefix should be stripped."""
        weights = {
            "thinker.audio_tower.ln_post.weight": mx.zeros((32,)),
        }
        sanitized = model.sanitize(weights)
        assert "audio_tower.ln_post.weight" in sanitized
        assert "thinker.audio_tower.ln_post.weight" not in sanitized

    def test_maps_model_to_language_model(self, model) -> None:
        """'thinker.model.*' should map to 'language_model.*'."""
        weights = {
            "thinker.model.embed_tokens.weight": mx.zeros((100, 48)),
        }
        sanitized = model.sanitize(weights)
        assert "language_model.embed_tokens.weight" in sanitized

    def test_skips_lm_head_when_tied(self, model) -> None:
        """lm_head weights should be skipped when tie_word_embeddings=True."""
        weights = {
            "thinker.lm_head.weight": mx.zeros((100, 48)),
        }
        sanitized = model.sanitize(weights)
        assert "language_model.lm_head.weight" not in sanitized

    def test_maps_lm_head_when_untied(self) -> None:
        """'thinker.lm_head.*' should map to 'language_model.lm_head.*' when untied."""
        config = Qwen3ASRConfig(
            text_config=Qwen3ASRTextConfig(
                hidden_size=48,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=24,
                intermediate_size=96,
                vocab_size=100,
                tie_word_embeddings=False,
            ),
        )
        untied_model = Qwen3ASRModel(config, dtype=mx.float32)
        weights = {
            "thinker.lm_head.weight": mx.zeros((100, 48)),
        }
        sanitized = untied_model.sanitize(weights)
        assert "language_model.lm_head.weight" in sanitized

    def test_transposes_conv2d_weights(self, model) -> None:
        """Conv2d weights should be transposed: NCHW → NHWC."""
        # PyTorch: (out, in, kH, kW)
        weights = {
            "thinker.audio_tower.conv2d1.weight": mx.zeros((8, 1, 3, 3)),
        }
        sanitized = model.sanitize(weights)
        # MLX: (out, kH, kW, in)
        assert sanitized["audio_tower.conv2d1.weight"].shape == (8, 3, 3, 1)

    def test_text_layer_key_mapping(self, model) -> None:
        """Text decoder layer weights should map correctly."""
        weights = {
            "thinker.model.layers.0.self_attn.q_proj.weight": mx.zeros((48, 48)),
            "thinker.model.layers.0.input_layernorm.weight": mx.zeros((48,)),
            "thinker.model.layers.0.mlp.gate_proj.weight": mx.zeros((96, 48)),
        }
        sanitized = model.sanitize(weights)
        assert "language_model.layers.0.self_attn.q_proj.weight" in sanitized
        assert "language_model.layers.0.input_layernorm.weight" in sanitized
        assert "language_model.layers.0.mlp.gate_proj.weight" in sanitized

    def test_casts_dtype(self, model) -> None:
        """Weights should be cast to model dtype."""
        weights = {
            "thinker.audio_tower.ln_post.weight": mx.ones((32,), dtype=mx.bfloat16),
        }
        sanitized = model.sanitize(weights)
        assert sanitized["audio_tower.ln_post.weight"].dtype == mx.float32


# ===========================================================================
# Qwen3 LM forward
# ===========================================================================


class TestQwen3LM:
    """Tests for Qwen3LM forward pass."""

    @pytest.fixture()
    def tiny_lm(self):
        config = Qwen3ASRTextConfig(
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            intermediate_size=128,
            vocab_size=100,
        )
        return Qwen3LM(config, dtype=mx.float32)

    def test_prefill(self, tiny_lm) -> None:
        """Prefill should produce logits of correct shape."""
        embeds = mx.random.normal((1, 5, 64))
        logits, cache = tiny_lm.forward_embeds(embeds)
        mx.eval(logits)
        assert logits.shape == (1, 5, 100)
        assert len(cache) == 2  # 2 layers

    def test_decode_step(self, tiny_lm) -> None:
        """Decode step with cache should produce single-token logits."""
        embeds = mx.random.normal((1, 5, 64))
        _, cache = tiny_lm.forward_embeds(embeds)

        step_embeds = mx.random.normal((1, 1, 64))
        logits, cache2 = tiny_lm.forward_embeds(step_embeds, cache)
        mx.eval(logits)
        assert logits.shape == (1, 1, 100)


# ===========================================================================
# Full model
# ===========================================================================


class TestQwen3ASRModel:
    """Tests for the full Qwen3ASRModel."""

    @pytest.fixture()
    def tiny_model(self):
        config = Qwen3ASRConfig(
            audio_config=Qwen3ASRAudioConfig(
                num_mel_bins=16,
                d_model=32,
                encoder_layers=1,
                encoder_attention_heads=2,
                encoder_ffn_dim=64,
                downsample_hidden_size=8,
                output_dim=48,
                max_source_positions=100,
                n_window=50,
                n_window_infer=800,
            ),
            text_config=Qwen3ASRTextConfig(
                hidden_size=48,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=24,
                intermediate_size=96,
                vocab_size=100,
            ),
            audio_token_id=99,
            eos_token_id=0,
        )
        return Qwen3ASRModel(config, dtype=mx.float32)

    def test_model_type(self, tiny_model) -> None:
        assert tiny_model.model_type == "qwen3_asr"

    def test_encode(self, tiny_model) -> None:
        """Encode should produce audio embeddings."""
        mel = mx.random.normal((16, 100))
        embeddings = tiny_model.encode(mel)
        mx.eval(embeddings)
        assert embeddings.shape == (13, 48)  # 100 frames → 13 after CNN

    def test_prefill_and_decode(self, tiny_model) -> None:
        """Prefill + decode step should work end-to-end."""
        mel = mx.random.normal((16, 100))
        audio_emb = tiny_model.encode(mel)
        mx.eval(audio_emb)

        # Build a simple prompt with audio_pad tokens
        # [start_tok, audio_pad * 13, end_tok, other_tok]
        n_audio = audio_emb.shape[0]
        token_ids = mx.array([[1] + [99] * n_audio + [2, 3]], dtype=mx.int32)

        logits, cache = tiny_model.prefill(token_ids, audio_emb)
        mx.eval(logits)
        assert logits.shape[0] == 1
        assert logits.shape[1] == token_ids.shape[1]
        assert logits.shape[2] == 100  # vocab_size

        # Decode step
        next_tok = mx.array([[5]], dtype=mx.int32)
        logits2, cache2 = tiny_model.decode_step(next_tok, cache)
        mx.eval(logits2)
        assert logits2.shape == (1, 1, 100)


# ===========================================================================
# Post-process output
# ===========================================================================


class TestPostProcessOutput:
    """Tests for Qwen3ASRTranscriber.post_process_output."""

    def test_strips_asr_text_tag(self) -> None:
        text = "language english<asr_text>Hello world"
        assert Qwen3ASRTranscriber.post_process_output(text) == "Hello world"

    def test_no_tag_passthrough(self) -> None:
        text = "Hello world"
        assert Qwen3ASRTranscriber.post_process_output(text) == "Hello world"

    def test_empty_string(self) -> None:
        assert Qwen3ASRTranscriber.post_process_output("") == ""


# ===========================================================================
# Config detection
# ===========================================================================


class TestPostProcessOutputTruncation:
    """Tests for special token truncation in post_process_output."""

    def test_truncates_at_im_end(self) -> None:
        text = "language english<asr_text>Hello world<|im_end|>"
        assert Qwen3ASRTranscriber.post_process_output(text) == "Hello world"

    def test_truncates_at_endoftext(self) -> None:
        text = "language english<asr_text>Hello world<|endoftext|>"
        assert Qwen3ASRTranscriber.post_process_output(text) == "Hello world"

    def test_truncates_at_im_start(self) -> None:
        text = "language english<asr_text>Hello world<|im_start|>system"
        assert Qwen3ASRTranscriber.post_process_output(text) == "Hello world"

    def test_multiple_asr_text_uses_last(self) -> None:
        text = "language english<asr_text>wrong<asr_text>Hello world<|im_end|>"
        assert Qwen3ASRTranscriber.post_process_output(text) == "Hello world"

    def test_strips_whitespace(self) -> None:
        text = "language english<asr_text>  Hello world  <|im_end|>"
        assert Qwen3ASRTranscriber.post_process_output(text) == "Hello world"


# ===========================================================================
# Build prompt tokens
# ===========================================================================


class TestBuildPromptTokens:
    """Tests for Qwen3ASRTranscriber.build_prompt_tokens structure."""

    @pytest.fixture()
    def transcriber(self, tmp_path):
        """Create a transcriber with a mock tokenizer for prompt tests."""
        config = Qwen3ASRConfig(
            audio_token_id=99,
            audio_start_token_id=97,
            audio_end_token_id=98,
            eos_token_id=0,
        )
        model = MagicMock()
        model.config = config

        # Inject mock tokenizer with deterministic encode
        mock_tok = MagicMock()
        _encode_map = {
            "<|im_start|>": [10],
            "<|im_end|>": [11],
            "user\n": [20],
            "assistant\n": [30],
            "\n": [40],
        }
        mock_tok.encode = MagicMock(
            side_effect=lambda s, add_special_tokens=False: _encode_map.get(s, [0])
        )
        t = Qwen3ASRTranscriber(model, tokenizer=mock_tok)
        return t

    def test_audio_pad_count_matches_frames(self, transcriber) -> None:
        """Number of audio_pad tokens should equal n_audio_frames."""
        # Act
        prompt = transcriber.build_prompt_tokens(50)

        # Assert
        audio_pad_count = prompt.count(99)  # audio_token_id
        assert audio_pad_count == 50

    def test_audio_pad_count_zero(self, transcriber) -> None:
        """Zero audio frames should produce no audio_pad tokens."""
        # Act
        prompt = transcriber.build_prompt_tokens(0)

        # Assert
        assert prompt.count(99) == 0

    def test_prompt_contains_structural_tokens(self, transcriber) -> None:
        """Prompt should contain audio_start, audio_end, im_start, user, assistant."""
        # Act
        prompt = transcriber.build_prompt_tokens(5)

        # Assert
        assert 97 in prompt  # audio_start
        assert 98 in prompt  # audio_end
        assert 10 in prompt  # im_start
        assert 20 in prompt  # user
        assert 30 in prompt  # assistant

    def test_prompt_structure_order(self, transcriber) -> None:
        """Audio tokens should be between audio_start and audio_end."""
        # Act
        prompt = transcriber.build_prompt_tokens(3)

        # Assert
        start_idx = prompt.index(97)  # audio_start
        end_idx = prompt.index(98)  # audio_end
        for i, tok in enumerate(prompt):
            if tok == 99:
                assert start_idx < i < end_idx


# ===========================================================================
# Config detection
# ===========================================================================


class TestConfigDetection:
    """Tests for is_stt_model with Qwen3-ASR config."""

    def test_qwen3_asr_detected(self, tmp_path) -> None:
        """qwen3_asr model_type should be detected as STT."""
        config = {"model_type": "qwen3_asr"}
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert is_stt_model(str(tmp_path)) is True


# ===========================================================================
# Slow tests (require real model)
# ===========================================================================


@pytest.mark.slow
class TestModelLoad:
    """Tests that load the real Qwen3-ASR-0.6B model.

    Set QWEN3_ASR_MODEL_PATH and QWEN3_ASR_AUDIO_PATH env vars to run.
    """

    @pytest.fixture(autouse=True)
    def _model_path(self):
        path = os.environ.get("QWEN3_ASR_MODEL_PATH")
        if not path:
            pytest.skip("QWEN3_ASR_MODEL_PATH not set")
        if not Path(path).exists():
            pytest.skip(f"Model path does not exist: {path}")
        self._MODEL_PATH = path

    def test_load_model(self) -> None:
        """Should load model without errors."""
        model = load_model(self._MODEL_PATH)
        assert model.model_type == "qwen3_asr"

    def test_encode_dummy_mel(self) -> None:
        """Should encode a dummy mel spectrogram."""
        model = load_model(self._MODEL_PATH)
        mel = mx.random.normal((128, 300))
        embeddings = model.encode(mel)
        mx.eval(embeddings)
        expected = model.config.audio_config.feat_extract_output_length(300)
        assert embeddings.shape == (expected, 1024)

    def test_greedy_decode(self) -> None:
        """Should encode + decode a real audio file using WhisperFeatureExtractor."""
        import numpy as np
        from transformers import WhisperFeatureExtractor

        from vllm_metal.stt.audio import load_audio

        audio_path = os.environ.get("QWEN3_ASR_AUDIO_PATH")
        if not audio_path or not Path(audio_path).exists():
            pytest.skip("QWEN3_ASR_AUDIO_PATH not set or file not found")

        model = load_model(self._MODEL_PATH)
        transcriber = Qwen3ASRTranscriber(model, model_path=self._MODEL_PATH)

        # Use WhisperFeatureExtractor (same as real pipeline)
        feature_extractor = WhisperFeatureExtractor(
            feature_size=128, sampling_rate=16000
        )
        audio = load_audio(audio_path)
        audio_np = np.array(audio.tolist(), dtype=np.float32)
        features = feature_extractor(audio_np, sampling_rate=16000, return_tensors="np")
        mel = mx.array(features["input_features"][0])  # (128, time)

        # Encode
        audio_emb = model.encode(mel)
        mx.eval(audio_emb)

        # Build prompt
        n_audio = audio_emb.shape[0]
        prompt = transcriber.build_prompt_tokens(n_audio)

        # Decode
        tokens = transcriber.greedy_decode_tokens(audio_emb, prompt, max_tokens=100)
        assert isinstance(tokens, list)
        assert len(tokens) > 0

        # Decode to text
        text = transcriber.tokenizer.decode(tokens, skip_special_tokens=True)
        text = Qwen3ASRTranscriber.post_process_output(text)
        assert isinstance(text, str)
        assert len(text) > 0
