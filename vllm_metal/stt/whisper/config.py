# SPDX-License-Identifier: Apache-2.0
"""Whisper model configuration.

Keep this module MLX-free so config consumers can import it without
pulling in the full model stack.
"""

from __future__ import annotations

from dataclasses import dataclass

# Maximum decode tokens for Whisper models (matches Whisper context window).
WHISPER_MAX_DECODE_TOKENS = 448


@dataclass
class WhisperConfig:
    """Whisper model configuration.

    Supports construction from both HuggingFace and MLX config formats
    via :meth:`from_dict`.
    """

    n_mels: int = 80
    n_audio_ctx: int = 1500
    n_audio_state: int = 512
    n_audio_head: int = 8
    n_audio_layer: int = 6
    n_vocab: int = 51865
    n_text_ctx: int = 448
    n_text_state: int = 512
    n_text_head: int = 8
    n_text_layer: int = 6

    @classmethod
    def from_dict(cls, config: dict) -> WhisperConfig:
        """Create config from a dictionary.

        Automatically detects HuggingFace format (``d_model``,
        ``encoder_layers``, etc.) and maps to MLX field names.

        Args:
            config: Raw config dictionary from ``config.json``.

        Returns:
            Populated :class:`WhisperConfig` instance.
        """
        config = config.copy()

        # HuggingFace format
        if "d_model" in config or "encoder_layers" in config:
            return cls(
                n_mels=config.get("num_mel_bins", 80),
                n_audio_ctx=config.get("max_source_positions", 1500),
                n_audio_state=config.get("d_model", 512),
                n_audio_head=config.get("encoder_attention_heads", 8),
                n_audio_layer=config.get("encoder_layers", 6),
                n_vocab=config.get("vocab_size", 51865),
                n_text_ctx=config.get("max_target_positions", 448),
                n_text_state=config.get("d_model", 512),
                n_text_head=config.get("decoder_attention_heads", 8),
                n_text_layer=config.get("decoder_layers", 6),
            )

        # MLX format — filter to known fields only
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in config.items() if k in known_fields}
        return cls(**filtered)
