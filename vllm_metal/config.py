# SPDX-License-Identifier: Apache-2.0
"""Configuration for vLLM Metal plugin via environment variables."""

import os
from dataclasses import dataclass
from typing import Literal


@dataclass
class MetalConfig:
    """Configuration for vLLM Metal plugin."""

    memory_fraction: float
    use_mlx: bool
    mlx_device: Literal["gpu", "cpu"]
    block_size: int
    debug: bool

    @classmethod
    def from_env(cls) -> "MetalConfig":
        """Load configuration from environment variables."""
        return cls(
            memory_fraction=float(os.environ.get("VLLM_METAL_MEMORY_FRACTION", "0.9")),
            use_mlx=os.environ.get("VLLM_METAL_USE_MLX", "1") == "1",
            mlx_device=os.environ.get("VLLM_MLX_DEVICE", "gpu"),  # type: ignore[arg-type]
            block_size=int(os.environ.get("VLLM_METAL_BLOCK_SIZE", "16")),
            debug=os.environ.get("VLLM_METAL_DEBUG", "0") == "1",
        )


# Global config instance
_config: MetalConfig | None = None


def get_config() -> MetalConfig:
    """Get the global Metal configuration."""
    global _config
    if _config is None:
        _config = MetalConfig.from_env()
    return _config


def reset_config() -> None:
    """Reset the global config (useful for testing)."""
    global _config
    _config = None
