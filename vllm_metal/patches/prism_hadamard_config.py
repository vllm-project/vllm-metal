# SPDX-License-Identifier: Apache-2.0
"""Transformers config compatibility for Prism Hadamard-rotated Qwen3.5 packs."""

from __future__ import annotations

from typing import Any

from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config


class PrismHadamardQwen35Config(Qwen3_5Config):
    """Qwen3.5 config alias for ``model_type: prism_hadamard_qwen35`` checkpoints.

    PrismML's Ternary-Bonsai-2 MLX packs carry Hadamard-rotated 2-bit Qwen3.5
    weights (loaded natively by mlx-vlm and mlx-lm) under a custom
    ``model_type`` and omit the top-level ``architectures`` field. Populate the
    matching Qwen3.5 architecture so vLLM can inspect hybrid and multimodal
    metadata without remote code.
    """

    model_type = "prism_hadamard_qwen35"

    def __post_init__(self, **kwargs: Any) -> None:
        has_vision = self.vision_config is not None
        super().__post_init__(**kwargs)
        if not self.architectures:
            self.architectures = [
                "Qwen3_5ForConditionalGeneration"
                if has_vision
                else "Qwen3_5ForCausalLM"
            ]
