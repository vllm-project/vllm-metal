# SPDX-License-Identifier: Apache-2.0
"""Tests for Metal platform."""

import pytest


class TestMetalPlatform:
    """Tests for MetalPlatform class."""

    def test_register_returns_platform_path(self):
        """Test that register() returns platform class path on Apple Silicon."""
        from vllm_metal import register
        from vllm_metal.utils import is_apple_silicon

        result = register()

        if is_apple_silicon():
            assert result == "vllm_metal.platform.MetalPlatform"
        else:
            assert result is None

    @pytest.mark.apple_silicon
    def test_platform_device_name(self):
        """Test that platform returns correct device name."""
        from vllm_metal.platform import MetalPlatform

        assert MetalPlatform.device_name == "mps"
        assert MetalPlatform.device_type == "mps"

    @pytest.mark.apple_silicon
    def test_platform_get_device_name(self):
        """Test get_device_name returns chip name."""
        from vllm_metal.platform import MetalPlatform

        name = MetalPlatform.get_device_name()
        assert isinstance(name, str)
        assert len(name) > 0

    @pytest.mark.apple_silicon
    def test_platform_get_total_memory(self):
        """Test get_device_total_memory returns reasonable value."""
        from vllm_metal.platform import MetalPlatform

        memory = MetalPlatform.get_device_total_memory()
        assert memory > 0
        # At least 1GB
        assert memory > 1e9

    @pytest.mark.apple_silicon
    def test_platform_mem_get_info(self):
        """Test mem_get_info returns free and total memory."""
        from vllm_metal.platform import MetalPlatform

        free, total = MetalPlatform.mem_get_info()
        assert total > 0
        assert free >= 0
        assert free <= total

    @pytest.mark.apple_silicon
    def test_platform_supports_dtype(self):
        """Test dtype support checking."""
        import torch
        from vllm_metal.platform import MetalPlatform

        assert MetalPlatform.check_if_supports_dtype(torch.float32)
        assert MetalPlatform.check_if_supports_dtype(torch.float16)
        assert MetalPlatform.check_if_supports_dtype(torch.bfloat16)

    @pytest.mark.apple_silicon
    def test_platform_no_fp8_support(self):
        """Test that FP8 is not supported."""
        from vllm_metal.platform import MetalPlatform

        assert not MetalPlatform.supports_fp8()

    @pytest.mark.apple_silicon
    def test_platform_no_distributed(self):
        """Test that distributed is not supported."""
        from vllm_metal.platform import MetalPlatform

        assert MetalPlatform.get_device_communicator_cls() is None

    @pytest.mark.apple_silicon
    def test_platform_attention_backend(self):
        """Test attention backend selection."""
        from vllm_metal.platform import MetalPlatform

        backend_cls = MetalPlatform.get_attn_backend_cls(
            selected_backend=None,
            head_size=64,
            dtype=None,
            kv_cache_dtype=None,
            block_size=16,
            use_mla=False,
            has_sink=False,
            use_sparse=False,
        )
        assert backend_cls == "vllm_metal.attention.backend.MetalAttentionBackend"
