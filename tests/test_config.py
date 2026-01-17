# SPDX-License-Identifier: Apache-2.0
"""Tests for vLLM Metal configuration."""

import os

import pytest

from vllm_metal.config import (
    AUTO_MEMORY_FRACTION,
    MetalConfig,
    get_config,
    reset_config,
)


class TestMetalConfig:
    """Tests for MetalConfig class."""

    def setup_method(self) -> None:
        """Reset config before each test."""
        reset_config()
        # Clear environment variables
        for var in [
            "VLLM_METAL_MEMORY_FRACTION",
            "VLLM_METAL_USE_MLX",
            "VLLM_MLX_DEVICE",
            "VLLM_METAL_BLOCK_SIZE",
            "VLLM_METAL_DEBUG",
        ]:
            os.environ.pop(var, None)

    def teardown_method(self) -> None:
        """Reset config after each test."""
        reset_config()
        for var in [
            "VLLM_METAL_MEMORY_FRACTION",
            "VLLM_METAL_USE_MLX",
            "VLLM_MLX_DEVICE",
            "VLLM_METAL_BLOCK_SIZE",
            "VLLM_METAL_DEBUG",
        ]:
            os.environ.pop(var, None)

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = MetalConfig.from_env()

        assert config.memory_fraction == AUTO_MEMORY_FRACTION
        assert config.is_auto_memory is True
        assert config.use_mlx is True
        assert config.mlx_device == "gpu"
        assert config.block_size == 16
        assert config.debug is False

    def test_custom_config_from_env(self) -> None:
        """Test configuration from environment variables."""
        os.environ["VLLM_METAL_MEMORY_FRACTION"] = "0.75"
        os.environ["VLLM_METAL_USE_MLX"] = "0"
        os.environ["VLLM_MLX_DEVICE"] = "cpu"
        os.environ["VLLM_METAL_BLOCK_SIZE"] = "32"
        os.environ["VLLM_METAL_DEBUG"] = "1"

        config = MetalConfig.from_env()

        assert config.memory_fraction == 0.75
        assert config.use_mlx is False
        assert config.mlx_device == "cpu"
        assert config.block_size == 32
        assert config.debug is True

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

    def test_auto_memory_fraction(self) -> None:
        """Test that 'auto' is parsed as AUTO_MEMORY_FRACTION."""
        os.environ["VLLM_METAL_MEMORY_FRACTION"] = "auto"

        config = MetalConfig.from_env()

        assert config.memory_fraction == AUTO_MEMORY_FRACTION
        assert config.is_auto_memory is True

    def test_auto_memory_fraction_case_insensitive(self) -> None:
        """Test that 'AUTO' and 'Auto' are also accepted."""
        for value in ["AUTO", "Auto", "AuTo"]:
            reset_config()
            os.environ["VLLM_METAL_MEMORY_FRACTION"] = value

            config = MetalConfig.from_env()

            assert config.memory_fraction == AUTO_MEMORY_FRACTION
            assert config.is_auto_memory is True

    def test_is_auto_memory_false_for_numeric(self) -> None:
        """Test that is_auto_memory is False for numeric values."""
        os.environ["VLLM_METAL_MEMORY_FRACTION"] = "0.5"

        config = MetalConfig.from_env()

        assert config.memory_fraction == 0.5
        assert config.is_auto_memory is False

    def test_block_size_must_be_positive(self) -> None:
        for value in ["0", "-1"]:
            reset_config()
            os.environ["VLLM_METAL_BLOCK_SIZE"] = value
            with pytest.raises(ValueError, match="Invalid VLLM_METAL_BLOCK_SIZE"):
                MetalConfig.from_env()
