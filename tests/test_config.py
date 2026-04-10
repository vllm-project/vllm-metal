# SPDX-License-Identifier: Apache-2.0
"""Tests for vLLM Metal configuration."""

import pytest

import vllm_metal.envs as envs
from vllm_metal.config import (
    AUTO_MEMORY_FRACTION,
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

        assert config.memory_fraction == AUTO_MEMORY_FRACTION
        assert config.is_auto_memory is True
        assert config.use_mlx is True
        assert config.mlx_device == "gpu"
        assert config.block_size == 16
        assert config.debug is False
        assert config.use_paged_attention is True

    def test_custom_config_from_env(self, monkeypatch) -> None:
        """Test configuration from environment variables."""
        monkeypatch.setenv("VLLM_METAL_MEMORY_FRACTION", "0.75")
        monkeypatch.setenv("VLLM_METAL_USE_MLX", "0")
        monkeypatch.setenv("VLLM_MLX_DEVICE", "cpu")
        monkeypatch.setenv("VLLM_METAL_BLOCK_SIZE", "32")
        monkeypatch.setenv("VLLM_METAL_DEBUG", "1")
        monkeypatch.setenv("VLLM_METAL_USE_PAGED_ATTENTION", "1")

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

    def test_auto_memory_fraction(self, monkeypatch) -> None:
        """Test that 'auto' is parsed as AUTO_MEMORY_FRACTION."""
        monkeypatch.setenv("VLLM_METAL_MEMORY_FRACTION", "auto")

        config = MetalConfig.from_env()

        assert config.memory_fraction == AUTO_MEMORY_FRACTION
        assert config.is_auto_memory is True

    def test_auto_memory_fraction_case_insensitive(self, monkeypatch) -> None:
        """Test that 'AUTO' and 'Auto' are also accepted."""
        for value in ["AUTO", "Auto", "AuTo"]:
            reset_config()
            monkeypatch.setenv("VLLM_METAL_MEMORY_FRACTION", value)

            config = MetalConfig.from_env()

            assert config.memory_fraction == AUTO_MEMORY_FRACTION
            assert config.is_auto_memory is True

    def test_is_auto_memory_false_for_numeric(self, monkeypatch) -> None:
        """Test that is_auto_memory is False for numeric values."""
        monkeypatch.setenv("VLLM_METAL_MEMORY_FRACTION", "0.5")

        config = MetalConfig.from_env()

        assert config.memory_fraction == 0.5
        assert config.is_auto_memory is False

    def test_explicit_fraction_requires_paged_attention(self) -> None:
        """Test that explicit memory fraction without paged attention is rejected."""
        with pytest.raises(ValueError, match="only supported with paged attention"):
            MetalConfig(
                memory_fraction=0.7,
                use_mlx=False,
                mlx_device="gpu",
                block_size=16,
                debug=False,
                use_paged_attention=False,
            )

    def test_block_size_must_be_positive(self, monkeypatch) -> None:
        for value in ["0", "-1"]:
            reset_config()
            monkeypatch.setenv("VLLM_METAL_BLOCK_SIZE", value)
            with pytest.raises(ValueError, match="Invalid VLLM_METAL_BLOCK_SIZE"):
                MetalConfig.from_env()

    def test_fraction_above_one_rejected(self) -> None:
        with pytest.raises(ValueError, match="Invalid VLLM_METAL_MEMORY_FRACTION"):
            MetalConfig(
                memory_fraction=1.5,
                use_mlx=False,
                mlx_device="gpu",
                block_size=16,
                debug=False,
                use_paged_attention=True,
            )

    def test_fraction_zero_or_negative_rejected(self) -> None:
        for fraction in [0.0, -0.1]:
            with pytest.raises(ValueError, match="Invalid VLLM_METAL_MEMORY_FRACTION"):
                MetalConfig(
                    memory_fraction=fraction,
                    use_mlx=False,
                    mlx_device="gpu",
                    block_size=16,
                    debug=False,
                    use_paged_attention=True,
                )
