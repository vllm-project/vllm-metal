# SPDX-License-Identifier: Apache-2.0
"""Tests for vLLM Metal configuration."""

import pytest

import vllm_metal.envs as envs
from vllm_metal.config import (
    MetalConfig,
    get_config,
    reset_config,
)


class TestMetalConfig:
    """Tests for MetalConfig class."""

    @pytest.fixture(autouse=True)
    def _reset(self, monkeypatch):
        """Reset config singleton before and after each test."""
        for var in envs.environment_variables:
            monkeypatch.delenv(var, raising=False)
        reset_config()
        yield
        reset_config()

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = MetalConfig.from_env()

        assert config.mlx_device == "gpu"
        assert config.multimodal_mode == "auto"

    def test_custom_config_from_env(self, monkeypatch) -> None:
        """Test configuration from environment variables."""
        monkeypatch.setenv("VLLM_MLX_DEVICE", "cpu")
        monkeypatch.setenv("VLLM_METAL_MULTIMODAL_MODE", "multimodal-native")

        config = MetalConfig.from_env()

        assert config.mlx_device == "cpu"
        assert config.multimodal_mode == "multimodal-native"

    def test_get_config_singleton(self) -> None:
        """Test that get_config returns a singleton."""
        config1 = get_config()
        config2 = get_config()

        assert config1 is config2

    def test_reset_config(self) -> None:
        """Test that reset_config clears the singleton."""
        config1 = get_config()
        reset_config()
        config2 = get_config()

        # After reset, we get a new config instance
        # (but with same values since env vars haven't changed)
        assert config1 is not config2

    def test_turboquant_defaults(self) -> None:
        """Test default TurboQuant config values."""
        config = MetalConfig.from_env()
        assert config.turboquant is False
        assert config.k_quant == "q8_0"
        assert config.v_quant == "q3_0"

    @pytest.mark.parametrize("mode", ["text-only-compat", "vlm"])
    def test_invalid_multimodal_mode_rejected(self, mode: str) -> None:
        with pytest.raises(ValueError, match="Invalid VLLM_METAL_MULTIMODAL_MODE"):
            MetalConfig(
                mlx_device="gpu",
                multimodal_mode=mode,  # type: ignore[arg-type]
            )

    def test_turboquant_invalid_k_quant_rejected(self) -> None:
        """Test that invalid k_quant values are rejected."""
        with pytest.raises(ValueError, match="Invalid k_quant"):
            MetalConfig(
                mlx_device="gpu",
                turboquant=True,
                k_quant="fp16",
            )

    def test_turboquant_invalid_v_quant_rejected(self) -> None:
        """Test that invalid v_quant values are rejected."""
        with pytest.raises(ValueError, match="Invalid v_quant"):
            MetalConfig(
                mlx_device="gpu",
                turboquant=True,
                k_quant="q8_0",
                v_quant="fp16",
            )

    def test_text_only_multimodal_mode_is_accepted(self, monkeypatch) -> None:
        monkeypatch.setenv("VLLM_METAL_MULTIMODAL_MODE", "text-only")
        reset_config()
        try:
            assert get_config().multimodal_mode == "text-only"
        finally:
            reset_config()
