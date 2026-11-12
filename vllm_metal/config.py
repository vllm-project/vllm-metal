# SPDX-License-Identifier: Apache-2.0
"""Configuration for vLLM Metal plugin via environment variables."""

from dataclasses import dataclass
from typing import Any, Literal, cast

import vllm_metal.envs as envs

# Minimum blocks required for paged attention to be usable.
PAGED_ATTENTION_MIN_BLOCKS = 16

# Per-runner DSpark weight conversion, selected through additional_config.
DSPARK_DRAFT_QUANTIZATION_KEY = "dspark_draft_quantization"
DSPARK_DRAFT_QUANTIZATION_Q4 = "q4"
DSPARK_Q4_GROUP_SIZE = 64

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
    cast(tuple[MultimodalMode, ...], envs.MULTIMODAL_MODES)
)


@dataclass
class MetalConfig:
    """Configuration for vLLM Metal plugin."""

    mlx_device: Literal["gpu", "cpu"]
    multimodal_mode: MultimodalMode = "auto"
    turboquant: bool = False  # Enable TurboQuant KV cache compression
    k_quant: str = "q8_0"  # Key quantization type: q8_0, q4_0, int8, uint8, etc.
    v_quant: str = "q3_0"  # Value quantization type: q2_0, q3_0, q4_0, q5_0 (Lloyd-Max)
    # >0 streams MoE experts from the checkpoint through a cache of this size.
    moe_expert_cache_bytes: int = 0

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

    def apply_turboquant(self, additional_config: Any) -> None:
        """Read the TurboQuant keys from ``--additional-config``.

        The dict is caller-provided JSON, so typos would otherwise leave
        TurboQuant off (or on with default quantization) without a word.
        """
        if not isinstance(additional_config, dict):
            return

        canonical = {"turboquant", "k_quant", "v_quant"}
        normalized = {
            "turboquant": "turboquant",
            "kquant": "k_quant",
            "vquant": "v_quant",
        }
        for key in additional_config:
            slug = key.lower().replace("_", "").replace("-", "").rstrip("s")
            if slug in normalized and key not in canonical:
                raise ValueError(
                    f"Unknown --additional-config key {key!r}; did you mean "
                    f"{normalized[slug]!r}? TurboQuant keys: {sorted(canonical)}"
                )

        if "turboquant" in additional_config and not isinstance(
            additional_config["turboquant"], bool
        ):
            raise ValueError(
                "turboquant must be a JSON boolean, got "
                f"{additional_config['turboquant']!r}"
            )

        if not additional_config.get("turboquant"):
            extra = {"k_quant", "v_quant"} & additional_config.keys()
            if extra:
                raise ValueError(f"{', '.join(sorted(extra))} requires turboquant=true")
            return

        self.turboquant = True
        self.k_quant = additional_config.get("k_quant", "q8_0")
        self.v_quant = additional_config.get("v_quant", "q3_0")
        self._validate_turboquant()

    def apply_moe_expert_offload(self, additional_config: Any) -> None:
        """Read ``{"moe_expert_offload": {"cache_gb": <float>}}`` from --additional-config."""
        if not isinstance(additional_config, dict):
            return
        options = additional_config.get("moe_expert_offload")
        if options is None:
            return
        if not isinstance(options, dict) or set(options) != {"cache_gb"}:
            raise ValueError(
                "moe_expert_offload takes exactly one key, cache_gb, e.g. "
                '--additional-config \'{"moe_expert_offload": {"cache_gb": 2}}\''
            )
        cache_gb = options["cache_gb"]
        if isinstance(cache_gb, bool) or not isinstance(cache_gb, int | float):
            raise ValueError(
                f"moe_expert_offload.cache_gb must be a number, got {cache_gb!r}"
            )
        if cache_gb <= 0:
            raise ValueError(
                f"moe_expert_offload.cache_gb must be positive, got {cache_gb!r}"
            )
        self.moe_expert_cache_bytes = int(cache_gb * 2**30)

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
