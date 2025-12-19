# SPDX-License-Identifier: Apache-2.0
"""Tests for Metal platform."""

import platform

import pytest

from vllm_metal.platform import MetalPlatform


class TestMetalPlatform:
    """Tests for MetalPlatform class."""

    def test_device_name(self) -> None:
        """Test device name retrieval."""
        name = MetalPlatform.get_device_name()
        assert "Apple Silicon" in name

    def test_device_count(self) -> None:
        """Test device count."""
        count = MetalPlatform.get_device_count()
        assert count == 1

    def test_current_device(self) -> None:
        """Test current device."""
        device = MetalPlatform.current_device()
        assert device == 0

    def test_set_device_valid(self) -> None:
        """Test setting valid device."""
        MetalPlatform.set_device(0)  # Should not raise

    def test_set_device_invalid(self) -> None:
        """Test setting invalid device."""
        with pytest.raises(ValueError, match="only supports device 0"):
            MetalPlatform.set_device(1)

    def test_device_capability(self) -> None:
        """Test device capability."""
        major, minor = MetalPlatform.get_device_capability()
        assert isinstance(major, int)
        assert isinstance(minor, int)

    def test_memory_info(self) -> None:
        """Test memory information."""
        total = MetalPlatform.get_device_total_memory()
        available = MetalPlatform.get_device_available_memory()

        assert total > 0
        assert available > 0
        assert available <= total

    @pytest.mark.skipif(
        platform.machine() != "arm64" or platform.system() != "Darwin",
        reason="Only runs on Apple Silicon",
    )
    def test_is_available(self) -> None:
        """Test platform availability on Apple Silicon."""
        assert MetalPlatform.is_available() is True

    def test_torch_device(self) -> None:
        """Test PyTorch device retrieval."""

        device = MetalPlatform.get_torch_device()
        assert device.type in ("mps", "cpu")

    def test_verify_quantization_supported(self) -> None:
        """Test supported quantization methods."""
        # These should not raise
        MetalPlatform.verify_quantization("none")
        MetalPlatform.verify_quantization(None)
        MetalPlatform.verify_quantization("fp16")
        MetalPlatform.verify_quantization("bfloat16")

    def test_verify_quantization_unsupported(self) -> None:
        """Test unsupported quantization methods."""
        with pytest.raises(ValueError, match="does not support"):
            MetalPlatform.verify_quantization("int8")

        with pytest.raises(ValueError, match="does not support"):
            MetalPlatform.verify_quantization("awq")
