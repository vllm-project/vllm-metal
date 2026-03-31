# SPDX-License-Identifier: Apache-2.0
"""Qwen3-ASR configuration (MLX-free).

Keep this module free of MLX imports so vLLM compat code can import config and
shape helpers during planning/registration without pulling in the model stack.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from vllm.transformers_utils.configs.qwen3_asr import (
    Qwen3ASRConfig as VllmQwen3ASRConfig,
)

# Maximum decode tokens for Qwen3-ASR decode loop.
QWEN3_ASR_MAX_DECODE_TOKENS = 1024


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
    eos_token_id: int = 151643
    # Compatibility with Whisper interface for load_model dispatching
    n_mels: int = 128
    n_audio_ctx: int = 1500

    @classmethod
    def _from_vllm_config(cls, config: VllmQwen3ASRConfig) -> Qwen3ASRConfig:
        """Adapt the upstream vLLM/HF config into the local MLX model config."""
        thinker = config.thinker_config
        audio = thinker.audio_config
        text = thinker.text_config

        audio_cfg = Qwen3ASRAudioConfig(
            num_mel_bins=audio.num_mel_bins,
            d_model=audio.d_model,
            encoder_layers=audio.encoder_layers,
            encoder_attention_heads=audio.encoder_attention_heads,
            encoder_ffn_dim=audio.encoder_ffn_dim,
            downsample_hidden_size=audio.downsample_hidden_size,
            output_dim=audio.output_dim,
            max_source_positions=audio.max_source_positions,
            n_window=audio.n_window,
            n_window_infer=audio.n_window_infer,
            activation_function=audio.activation_function,
        )
        text_cfg = Qwen3ASRTextConfig(
            hidden_size=text.hidden_size,
            num_hidden_layers=text.num_hidden_layers,
            num_attention_heads=text.num_attention_heads,
            num_key_value_heads=text.num_key_value_heads,
            head_dim=text.head_dim,
            intermediate_size=text.intermediate_size,
            vocab_size=text.vocab_size,
            rms_norm_eps=text.rms_norm_eps,
            rope_theta=text.rope_theta,
            tie_word_embeddings=text.tie_word_embeddings,
        )
        config_kwargs = {
            "audio_config": audio_cfg,
            "text_config": text_cfg,
            "audio_token_id": thinker.audio_token_id,
            "n_mels": audio_cfg.num_mel_bins,
            "n_audio_ctx": audio_cfg.max_source_positions,
        }
        # The nested text config may omit eos_token_id entirely depending on how
        # upstream constructs Qwen3-ASR configs. Preserve the local model
        # default unless upstream exposes an explicit value here.
        eos_token_id = getattr(text, "eos_token_id", None)
        if eos_token_id is not None:
            config_kwargs["eos_token_id"] = eos_token_id
        return cls(**config_kwargs)

    @classmethod
    def from_dict(cls, d: dict) -> Qwen3ASRConfig:
        """Create config from config.json using the upstream schema owner."""
        return cls._from_vllm_config(VllmQwen3ASRConfig.from_dict(d))
