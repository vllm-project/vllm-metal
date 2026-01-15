# SPDX-License-Identifier: Apache-2.0
"""Configuration for vLLM Metal plugin via environment variables."""

import os
from dataclasses import dataclass
from typing import Literal

# Sentinel value indicating auto memory calculation
AUTO_MEMORY_FRACTION = -1.0

# Auto memory estimation heuristics.
#
# These heuristics intentionally over-estimate the minimum unified memory needed
# to run the model + a minimal KV cache, to account for transient allocations and
# fragmentation.
AUTO_MEMORY_OVERHEAD_FACTOR = 1.2

# Extra slack on the minimum number of KV blocks derived from `max_model_len` to
# avoid under-allocation due to rounding and other small overheads.
AUTO_MEMORY_MIN_BLOCKS_BUFFER_FACTOR = 1.1


@dataclass
class MetalConfig:
    """Configuration for vLLM Metal plugin."""

    memory_fraction: float  # -1.0 means "auto" (calculate minimal needed)
    use_mlx: bool
    mlx_device: Literal["gpu", "cpu"]
    block_size: int
    debug: bool

    def __post_init__(self) -> None:
        if self.block_size <= 0:
            msg = (
                f"Invalid VLLM_METAL_BLOCK_SIZE={self.block_size}. "
                "This controls tokens per KV cache block and must be a positive "
                "integer (>0)."
            )
            raise ValueError(msg)

    @property
    def is_auto_memory(self) -> bool:
        """Check if memory fraction is set to auto mode."""
        return self.memory_fraction == AUTO_MEMORY_FRACTION

    @classmethod
    def from_env(cls) -> "MetalConfig":
        """Load configuration from environment variables."""
        memory_fraction_str = os.environ.get("VLLM_METAL_MEMORY_FRACTION", "auto")
        if memory_fraction_str.lower() == "auto":
            memory_fraction = AUTO_MEMORY_FRACTION
        else:
            memory_fraction = float(memory_fraction_str)

        return cls(
            memory_fraction=memory_fraction,
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
