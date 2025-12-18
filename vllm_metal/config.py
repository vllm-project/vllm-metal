# SPDX-License-Identifier: Apache-2.0
"""Configuration classes for vLLM Metal backend."""

from dataclasses import dataclass, field
from typing import Literal

from vllm_metal.envs import (
    VLLM_METAL_BLOCK_SIZE,
    VLLM_METAL_DEBUG,
    VLLM_METAL_MAX_BATCH_SIZE,
    VLLM_METAL_MEMORY_FRACTION,
    VLLM_METAL_USE_MLX,
    VLLM_MLX_DEVICE,
    VLLM_MLX_LAZY_EVAL,
)


@dataclass
class MetalConfig:
    """Configuration for Metal backend.

    This configuration controls Metal-specific behavior including
    MLX settings, memory management, and performance tuning.
    """

    # Memory settings
    memory_fraction: float = field(default_factory=lambda: VLLM_METAL_MEMORY_FRACTION)
    block_size: int = field(default_factory=lambda: VLLM_METAL_BLOCK_SIZE)
    max_batch_size: int = field(default_factory=lambda: VLLM_METAL_MAX_BATCH_SIZE)

    # MLX settings
    use_mlx: bool = field(default_factory=lambda: VLLM_METAL_USE_MLX)
    mlx_device: Literal["gpu", "cpu"] = field(default_factory=lambda: VLLM_MLX_DEVICE)  # type: ignore
    mlx_lazy_eval: bool = field(default_factory=lambda: VLLM_MLX_LAZY_EVAL)

    # Debug settings
    debug: bool = field(default_factory=lambda: VLLM_METAL_DEBUG)

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not 0.0 < self.memory_fraction <= 1.0:
            raise ValueError(
                f"memory_fraction must be in (0, 1], got {self.memory_fraction}"
            )

        if self.block_size not in (8, 16, 32):
            raise ValueError(
                f"block_size must be 8, 16, or 32, got {self.block_size}"
            )

        if self.max_batch_size < 1:
            raise ValueError(
                f"max_batch_size must be positive, got {self.max_batch_size}"
            )


# Global config instance (lazy initialized)
_global_config: MetalConfig | None = None


def get_metal_config() -> MetalConfig:
    """Get the global Metal configuration.

    Returns:
        The global MetalConfig instance.
    """
    global _global_config
    if _global_config is None:
        _global_config = MetalConfig()
    return _global_config


def set_metal_config(config: MetalConfig) -> None:
    """Set the global Metal configuration.

    Args:
        config: The MetalConfig instance to set as global.
    """
    global _global_config
    _global_config = config
