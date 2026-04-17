# SPDX-License-Identifier: Apache-2.0
"""Configuration for vLLM Metal plugin via environment variables."""

from dataclasses import dataclass
from typing import Literal

import vllm_metal.envs as envs

# Sentinel value indicating auto memory calculation
AUTO_MEMORY_FRACTION = -1.0

# Default memory fraction when user leaves VLLM_METAL_MEMORY_FRACTION as "auto"
# but enables paged attention (auto is for the MLX path).
PAGED_ATTENTION_DEFAULT_MEMORY_FRACTION = 0.9

# Minimum blocks required for paged attention to be usable.
PAGED_ATTENTION_MIN_BLOCKS = 16

# Valid key quantization types for TurboQuant (mirrors QUANT_PARAMS in turboquant.py).
# Kept here as a plain set so config can be imported without MLX.
TURBOQUANT_VALID_K_QUANTS: frozenset[str] = frozenset(
    {"q8_0", "int8", "uint8", "q5_0", "q4_0", "int4", "uint4", "int2", "uint2"}
)

# Valid value quantization types for TurboQuant.
# V uses Lloyd-Max quantization with FWHT rotation.
TURBOQUANT_VALID_V_QUANTS: frozenset[str] = frozenset(
    {"q2_0", "q3_0", "q4_0", "q5_0", "q8_0"}
)


@dataclass
class MetalConfig:
    """Configuration for vLLM Metal plugin."""

    memory_fraction: float  # -1.0 means "auto" (calculate minimal needed)
    use_mlx: bool
    mlx_device: Literal["gpu", "cpu"]
    block_size: int
    debug: bool
    use_paged_attention: bool = True
    turboquant: bool = False  # Enable TurboQuant KV cache compression
    k_quant: str = "q8_0"  # Key quantization type: q8_0, q4_0, int8, uint8, etc.
    v_quant: str = "q3_0"  # Value quantization type: q2_0, q3_0, q4_0, q5_0 (Lloyd-Max)

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

        self._validate_turboquant()

    def _validate_turboquant(self) -> None:
        """Validate TurboQuant configuration."""
        if self.turboquant:
            if not self.use_paged_attention:
                raise ValueError(
                    "turboquant requires paged attention. "
                    "TurboQuant KV cache compression only works with paged attention."
                )
            if self.k_quant not in TURBOQUANT_VALID_K_QUANTS:
                available = ", ".join(sorted(TURBOQUANT_VALID_K_QUANTS))
                raise ValueError(
                    f"Invalid k_quant={self.k_quant!r}. "
                    f"Available quantization types: {available}"
                )
            if self.v_quant not in TURBOQUANT_VALID_V_QUANTS:
                available = ", ".join(sorted(TURBOQUANT_VALID_V_QUANTS))
                raise ValueError(
                    f"Invalid v_quant={self.v_quant!r}. "
                    f"Available quantization types: {available}"
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

        # TurboQuant config is set via --additional-config, not env vars.
        # See MetalPlatform.check_and_update_config() for how it's applied.
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
