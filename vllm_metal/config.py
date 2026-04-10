# SPDX-License-Identifier: Apache-2.0
"""Configuration for vLLM Metal plugin via environment variables."""

from dataclasses import dataclass
from typing import Literal

import vllm_metal.envs as envs

# Sentinel value indicating auto memory calculation
AUTO_MEMORY_FRACTION = -1.0

# Paged attention: placeholder overhead for activations, framework, OS, etc.
# Will be replaced by a profiling pass in a future PR.
PAGED_ATTENTION_OVERHEAD_BYTES = 800 * 1024 * 1024  # 800 MB

# Default memory fraction when user leaves VLLM_METAL_MEMORY_FRACTION as "auto"
# but enables paged attention (auto is for the MLX path).
PAGED_ATTENTION_DEFAULT_MEMORY_FRACTION = 0.9

# Minimum blocks required for paged attention to be usable.
PAGED_ATTENTION_MIN_BLOCKS = 16


@dataclass
class MetalConfig:
    """Configuration for vLLM Metal plugin."""

    memory_fraction: float  # -1.0 means "auto" (calculate minimal needed)
    use_mlx: bool
    mlx_device: Literal["gpu", "cpu"]
    block_size: int
    debug: bool
    use_paged_attention: bool = True

    def __post_init__(self) -> None:
        if self.block_size <= 0:
            msg = (
                f"Invalid VLLM_METAL_BLOCK_SIZE={self.block_size}. "
                "This controls tokens per KV cache block and must be a positive "
                "integer (>0)."
            )
            raise ValueError(msg)

        if not self.use_paged_attention and not self.is_auto_memory:
            raise ValueError(
                f"VLLM_METAL_MEMORY_FRACTION={self.memory_fraction} is only "
                "supported with paged attention (the default). "
                "The MLX KV cache path (VLLM_METAL_USE_PAGED_ATTENTION=0) "
                "requires VLLM_METAL_MEMORY_FRACTION=auto."
            )

        if self.use_paged_attention and not self.is_auto_memory:
            if not (0 < self.memory_fraction <= 1):
                raise ValueError(
                    f"Invalid VLLM_METAL_MEMORY_FRACTION={self.memory_fraction}. "
                    "Must be a finite value in (0, 1] when paged attention is enabled."
                )

    @property
    def is_auto_memory(self) -> bool:
        """Check if memory fraction is set to auto mode."""
        return self.memory_fraction == AUTO_MEMORY_FRACTION

    @classmethod
    def from_env(cls) -> "MetalConfig":
        """Load configuration from environment variables."""
        memory_fraction_str = envs.VLLM_METAL_MEMORY_FRACTION
        if memory_fraction_str.lower() == "auto":
            memory_fraction = AUTO_MEMORY_FRACTION
        else:
            try:
                memory_fraction = float(memory_fraction_str)
            except ValueError as e:
                raise ValueError(
                    f"Invalid VLLM_METAL_MEMORY_FRACTION={memory_fraction_str!r}. "
                    "Must be 'auto' or a numeric value in (0, 1]."
                ) from e

        block_size_str = envs.VLLM_METAL_BLOCK_SIZE
        try:
            block_size = int(block_size_str)
        except ValueError as e:
            raise ValueError(
                f"Invalid VLLM_METAL_BLOCK_SIZE={block_size_str!r}. "
                "Must be a positive integer."
            ) from e

        return cls(
            memory_fraction=memory_fraction,
            use_mlx=envs.VLLM_METAL_USE_MLX,
            mlx_device=envs.VLLM_MLX_DEVICE,  # type: ignore[arg-type]
            block_size=block_size,
            debug=envs.VLLM_METAL_DEBUG,
            use_paged_attention=envs.VLLM_METAL_USE_PAGED_ATTENTION,
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
