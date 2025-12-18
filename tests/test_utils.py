# SPDX-License-Identifier: Apache-2.0
"""Tests for utility functions."""

import pytest


class TestUtilityFunctions:
    """Tests for vllm_metal.utils functions."""

    def test_is_apple_silicon(self):
        """Test is_apple_silicon detection."""
        import platform
        from vllm_metal.utils import is_apple_silicon

        result = is_apple_silicon()
        assert isinstance(result, bool)

        # Verify consistency with platform info
        expected = (
            platform.system() == "Darwin"
            and platform.machine() in ("arm64", "aarch64")
        )
        assert result == expected

    @pytest.mark.apple_silicon
    def test_get_apple_chip_name(self):
        """Test chip name detection on Apple Silicon."""
        from vllm_metal.utils import get_apple_chip_name

        name = get_apple_chip_name()
        assert isinstance(name, str)
        assert len(name) > 0
        # Should contain "Apple" for Apple Silicon
        assert "Apple" in name or "M1" in name or "M2" in name or "M3" in name or "M4" in name

    def test_get_apple_chip_name_non_apple(self):
        """Test chip name on non-Apple platforms."""
        from vllm_metal.utils import get_apple_chip_name, is_apple_silicon

        if not is_apple_silicon():
            name = get_apple_chip_name()
            assert name == "Not Apple Silicon"

    @pytest.mark.apple_silicon
    def test_get_metal_device_info(self):
        """Test Metal device info retrieval."""
        from vllm_metal.utils import get_metal_device_info

        info = get_metal_device_info()

        assert isinstance(info, dict)
        assert "name" in info
        assert "metal_available" in info
        assert "total_memory" in info

        assert info["metal_available"] is True
        assert info["total_memory"] > 0

    @pytest.mark.apple_silicon
    def test_get_metal_memory_info(self):
        """Test Metal memory info retrieval."""
        from vllm_metal.utils import get_metal_memory_info

        allocated, total = get_metal_memory_info()

        assert isinstance(allocated, int)
        assert isinstance(total, int)
        assert total > 0
        assert allocated >= 0
        assert allocated <= total

    @pytest.mark.apple_silicon
    def test_check_metal_availability(self):
        """Test Metal availability check."""
        from vllm_metal.utils import check_metal_availability

        available, error = check_metal_availability()

        assert isinstance(available, bool)
        if available:
            assert error is None
        else:
            assert isinstance(error, str)

    @pytest.mark.apple_silicon
    @pytest.mark.mlx
    def test_metal_operations(self):
        """Test Metal cache clearing and synchronization."""
        from vllm_metal.utils import metal_empty_cache, metal_synchronize

        # These should not raise errors
        metal_empty_cache()
        metal_synchronize()

    def test_get_supported_dtypes(self):
        """Test supported dtypes retrieval."""
        from vllm_metal.utils import get_supported_dtypes

        dtypes = get_supported_dtypes()

        assert isinstance(dtypes, set)
        assert "float32" in dtypes
        assert "float16" in dtypes
        assert "bfloat16" in dtypes
        assert "int32" in dtypes


class TestEnvironmentVariables:
    """Tests for environment variable handling."""

    def test_default_memory_fraction(self):
        """Test default memory fraction."""
        from vllm_metal.envs import VLLM_METAL_MEMORY_FRACTION

        assert isinstance(VLLM_METAL_MEMORY_FRACTION, float)
        assert 0.0 < VLLM_METAL_MEMORY_FRACTION <= 1.0

    def test_default_use_mlx(self):
        """Test default MLX usage flag."""
        from vllm_metal.envs import VLLM_METAL_USE_MLX

        assert isinstance(VLLM_METAL_USE_MLX, bool)
        assert VLLM_METAL_USE_MLX is True  # Default should be True

    def test_default_block_size(self):
        """Test default block size."""
        from vllm_metal.envs import VLLM_METAL_BLOCK_SIZE

        assert isinstance(VLLM_METAL_BLOCK_SIZE, int)
        assert VLLM_METAL_BLOCK_SIZE in (8, 16, 32)


class TestConfig:
    """Tests for Metal configuration."""

    def test_metal_config_defaults(self):
        """Test MetalConfig default values."""
        from vllm_metal.config import MetalConfig

        config = MetalConfig()

        assert 0.0 < config.memory_fraction <= 1.0
        assert config.block_size in (8, 16, 32)
        assert config.max_batch_size > 0
        assert config.use_mlx is True
        assert config.mlx_device in ("gpu", "cpu")

    def test_metal_config_validation(self):
        """Test MetalConfig validation."""
        from vllm_metal.config import MetalConfig

        # Invalid memory fraction
        with pytest.raises(ValueError):
            MetalConfig(memory_fraction=0.0)

        with pytest.raises(ValueError):
            MetalConfig(memory_fraction=1.5)

        # Invalid block size
        with pytest.raises(ValueError):
            MetalConfig(block_size=64)

        # Invalid batch size
        with pytest.raises(ValueError):
            MetalConfig(max_batch_size=0)

    def test_global_config(self):
        """Test global config getter/setter."""
        from vllm_metal.config import get_metal_config, set_metal_config, MetalConfig

        # Get default config
        config = get_metal_config()
        assert isinstance(config, MetalConfig)

        # Set new config
        new_config = MetalConfig(memory_fraction=0.8)
        set_metal_config(new_config)

        # Verify it was set
        retrieved = get_metal_config()
        assert retrieved.memory_fraction == 0.8
