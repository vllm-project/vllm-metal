# SPDX-License-Identifier: Apache-2.0
"""Configuration for vLLM Metal plugin via environment variables."""

from dataclasses import dataclass
from typing import Literal

import vllm_metal.envs as envs

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

MultimodalMode = Literal["auto", "multimodal-native", "text-only"]
VALID_MULTIMODAL_MODES: frozenset[MultimodalMode] = frozenset(
    {"auto", "multimodal-native", "text-only"}
)


@dataclass
class MetalConfig:
    """Configuration for vLLM Metal plugin."""

    mlx_device: Literal["gpu", "cpu"]
    multimodal_mode: MultimodalMode = "auto"
    turboquant: bool = False  # Enable TurboQuant KV cache compression
    k_quant: str = "q8_0"  # Key quantization type: q8_0, q4_0, int8, uint8, etc.
    v_quant: str = "q3_0"  # Value quantization type: q2_0, q3_0, q4_0, q5_0 (Lloyd-Max)

    def __post_init__(self) -> None:
        if self.multimodal_mode not in VALID_MULTIMODAL_MODES:
            available = ", ".join(sorted(VALID_MULTIMODAL_MODES))
            raise ValueError(
                f"Invalid VLLM_METAL_MULTIMODAL_MODE={self.multimodal_mode!r}. "
                f"Available modes: {available}."
            )

        self._validate_turboquant()

    def _validate_turboquant(self) -> None:
        """Validate TurboQuant configuration."""
        if self.turboquant:
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

    @classmethod
    def from_env(cls) -> "MetalConfig":
        """Load configuration from environment variables."""
        # TurboQuant config is set via --additional-config, not env vars.
        # See MetalPlatform.check_and_update_config() for how it's applied.
        return cls(
            mlx_device=envs.VLLM_MLX_DEVICE,  # type: ignore[arg-type]
            multimodal_mode=envs.VLLM_METAL_MULTIMODAL_MODE,  # type: ignore[arg-type]
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
