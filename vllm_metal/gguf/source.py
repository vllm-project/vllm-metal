# SPDX-License-Identifier: Apache-2.0
"""Resolved GGUF load identity carried from vLLM config to the loader."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class GGUFLoadSource:
    """Local GGUF weights plus companion config/tokenizer sources."""

    weights_path: str
    config_dir: str
    tokenizer_dir: str

    @classmethod
    def from_model_config(cls, model_config: Any) -> GGUFLoadSource | None:
        if model_config.quantization != "gguf":
            return None

        weights_path = model_config.model_weights
        if not cls.is_weights_path(weights_path):
            raise ValueError(
                "GGUF model_config must carry the local .gguf weights path in "
                f"model_weights; got {weights_path!r}."
            )

        config_dir = model_config.model
        return cls(
            weights_path=weights_path,
            config_dir=config_dir,
            tokenizer_dir=model_config.tokenizer or config_dir,
        )

    @staticmethod
    def is_weights_path(value: str) -> bool:
        return value.endswith(".gguf")
