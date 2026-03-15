# SPDX-License-Identifier: Apache-2.0
"""Qwen3-ASR configuration (MLX-free).

Keep this module free of MLX imports so vLLM compat code can import config and
shape helpers during planning/registration without pulling in the model stack.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Qwen3ASRAudioConfig:
    """Audio encoder configuration."""

    num_mel_bins: int = 128
    d_model: int = 896
    encoder_layers: int = 18
    encoder_attention_heads: int = 14
    encoder_ffn_dim: int = 3584
    downsample_hidden_size: int = 480
    output_dim: int = 1024
    max_source_positions: int = 1500
    n_window: int = 50
    n_window_infer: int = 800
    activation_function: str = "gelu"

    @staticmethod
    def cnn_output_length(num_frames: int) -> int:
        """Return time length after 3x Conv2d(stride=2) downsampling."""
        length = num_frames
        for _ in range(3):
            length = (length - 1) // 2 + 1
        return int(length)

    def feat_extract_output_length(self, num_mel_frames: int) -> int:
        """Return number of audio tokens produced from a mel with N time frames."""
        chunk_size = self.n_window * 2
        frames_per_full_chunk = self.cnn_output_length(chunk_size)
        full_chunks, remainder = divmod(num_mel_frames, chunk_size)
        if remainder == 0:
            return int(full_chunks * frames_per_full_chunk)
        return int(
            full_chunks * frames_per_full_chunk + self.cnn_output_length(remainder)
        )


@dataclass
class Qwen3ASRTextConfig:
    """Text decoder (Qwen3 LM) configuration."""

    hidden_size: int = 1024
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 3072
    vocab_size: int = 151936
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    tie_word_embeddings: bool = True


@dataclass
class Qwen3ASRConfig:
    """Top-level Qwen3-ASR model configuration."""

    audio_config: Qwen3ASRAudioConfig = field(default_factory=Qwen3ASRAudioConfig)
    text_config: Qwen3ASRTextConfig = field(default_factory=Qwen3ASRTextConfig)
    audio_token_id: int = 151676
    audio_start_token_id: int = 151669
    audio_end_token_id: int = 151670
    eos_token_id: int = 151643
    # Compatibility with Whisper interface for load_model dispatching
    n_mels: int = 128
    n_audio_ctx: int = 1500

    @classmethod
    def from_dict(cls, d: dict) -> Qwen3ASRConfig:
        """Create config from config.json dictionary."""
        thinker = d.get("thinker_config", d)

        audio_dict = thinker.get("audio_config", {})
        audio_cfg = Qwen3ASRAudioConfig(
            num_mel_bins=audio_dict.get("num_mel_bins", 128),
            d_model=audio_dict.get("d_model", 896),
            encoder_layers=audio_dict.get("encoder_layers", 18),
            encoder_attention_heads=audio_dict.get("encoder_attention_heads", 14),
            encoder_ffn_dim=audio_dict.get("encoder_ffn_dim", 3584),
            downsample_hidden_size=audio_dict.get("downsample_hidden_size", 480),
            output_dim=audio_dict.get("output_dim", 1024),
            max_source_positions=audio_dict.get("max_source_positions", 1500),
            n_window=audio_dict.get("n_window", 50),
            n_window_infer=audio_dict.get("n_window_infer", 800),
            activation_function=audio_dict.get("activation_function", "gelu"),
        )

        text_dict = thinker.get("text_config", {})
        text_cfg = Qwen3ASRTextConfig(
            hidden_size=text_dict.get("hidden_size", 1024),
            num_hidden_layers=text_dict.get("num_hidden_layers", 28),
            num_attention_heads=text_dict.get("num_attention_heads", 16),
            num_key_value_heads=text_dict.get("num_key_value_heads", 8),
            head_dim=text_dict.get("head_dim", 128),
            intermediate_size=text_dict.get("intermediate_size", 3072),
            vocab_size=text_dict.get("vocab_size", 151936),
            rms_norm_eps=text_dict.get("rms_norm_eps", 1e-6),
            rope_theta=text_dict.get("rope_theta", 1000000.0),
            tie_word_embeddings=text_dict.get("tie_word_embeddings", True),
        )

        return cls(
            audio_config=audio_cfg,
            text_config=text_cfg,
            audio_token_id=thinker.get("audio_token_id", 151676),
            audio_start_token_id=thinker.get("audio_start_token_id", 151669),
            audio_end_token_id=thinker.get("audio_end_token_id", 151670),
            n_mels=audio_cfg.num_mel_bins,
            n_audio_ctx=audio_cfg.max_source_positions,
        )
