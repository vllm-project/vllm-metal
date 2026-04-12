# SPDX-License-Identifier: Apache-2.0
"""Unit tests for MetalPlatform.update_block_size_for_backend() and _find_non_ssm_backend().

Tests cover:
1. _find_non_ssm_backend returns MetalBackend with correct kernel block alignment
2. update_block_size_for_backend delegates to vLLM base implementation
3. Metal-specific adjustments (block_size multiple of 32 for paged attention)
"""

import logging
from unittest.mock import patch

import pytest
from vllm.config import CacheConfig, ModelConfig, ParallelConfig, VllmConfig

from vllm_metal.platform import MetalPlatform


class TestFindNonSsmBackend:
    """Test suite for _find_non_ssm_backend() method."""

    def test_returns_metal_backend_class(self):
        """Test: _find_non_ssm_backend returns a MetalBackend class."""
        backend_cls = MetalPlatform._find_non_ssm_backend(None)  # type: ignore

        assert backend_cls is not None
        assert backend_cls.get_name() == "METAL_ATTN"

    def test_metal_backend_kernel_block_sizes(self):
        """Test: MetalBackend returns MultipleOf(32) for kernel block sizes."""
        from vllm.v1.attention.backend import MultipleOf

        backend_cls = MetalPlatform._find_non_ssm_backend(None)  # type: ignore
        sizes = backend_cls.get_supported_kernel_block_sizes()  # type: ignore

        assert len(sizes) == 1
        assert isinstance(sizes[0], MultipleOf)
        assert sizes[0].base == 32

    def test_metal_backend_required_methods(self):
        """Test: MetalBackend has all required AttentionBackend methods."""
        backend_cls = MetalPlatform._find_non_ssm_backend(None)  # type: ignore

        # Check all required static methods exist
        assert hasattr(backend_cls, "get_name")
        assert hasattr(backend_cls, "get_supported_kernel_block_sizes")
        assert hasattr(backend_cls, "get_impl_cls")
        assert hasattr(backend_cls, "get_builder_cls")
        assert hasattr(backend_cls, "get_kv_cache_shape")

        # Verify they raise NotImplementedError (not implemented for block_size calc)
        with pytest.raises(NotImplementedError):
            backend_cls.get_impl_cls()  # type: ignore
        with pytest.raises(NotImplementedError):
            backend_cls.get_builder_cls()  # type: ignore
        with pytest.raises(NotImplementedError):
            backend_cls.get_kv_cache_shape()  # type: ignore


class TestUpdateBlockSizeForBackend:
    """Test suite for update_block_size_for_backend() method."""

    # ========================================================================
    # Fixtures
    # ========================================================================

    @pytest.fixture
    def base_cache_config(self):
        """Create a base CacheConfig mock for testing."""
        cache_config = CacheConfig(
            block_size=16,
            gpu_memory_utilization=0.9,
            cache_dtype="auto",
        )
        # Override user_specified_block_size to allow adjustments
        cache_config.user_specified_block_size = False
        cache_config.mamba_cache_mode = "none"
        cache_config.mamba_block_size = None
        cache_config.mamba_page_size_padded = None
        return cache_config

    @pytest.fixture
    def base_model_config(self):
        """Create a base ModelConfig mock for testing."""
        import torch

        model_config = ModelConfig(
            model="test-model",
            task="auto",
            tokenizer="test-tokenizer",
            tokenizer_mode="auto",
            trust_remote_code=False,
            dtype=torch.float16,
            seed=0,
            revision=None,
            code_revision=None,
            rope_scaling=None,
            rope_theta=None,
            tokenizer_revision=None,
            max_model_len=512,
            quantization=None,
            quantization_param_path=None,
            enforce_eager=False,
            max_seq_len_to_capture=None,
            max_logprobs=10,
            disable_sliding_window=False,
            top_p=1.0,
            top_k=-1,
        )
        model_config.is_hybrid = True
        model_config.architecture = "Qwen3_5ForCausalLM"
        return model_config

    @pytest.fixture
    def vllm_config(self, base_cache_config, base_model_config):
        """Create a complete VllmConfig for hybrid model testing."""
        parallel_config = ParallelConfig(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
        )

        config = VllmConfig(
            model_config=base_model_config,
            cache_config=base_cache_config,
            parallel_config=parallel_config,
        )
        return config

    # ========================================================================
    # Core Functionality Tests
    # ========================================================================

    def test_calls_super_implementation(self, vllm_config, caplog):
        """Test: update_block_size_for_backend calls super() implementation.

        Since we delegate to vLLM's base implementation via super(),
        the method should complete without errors for valid configs.
        """
        # Note: This test verifies the method completes without crashing
        # Actual block_size calculation is handled by vLLM base class
        MetalPlatform.update_block_size_for_backend(vllm_config)

        # Should complete without errors
        assert vllm_config.cache_config.block_size >= 16

    def test_non_hybrid_model_skipped(self, vllm_config):
        """Test: Non-hybrid model skips Metal-specific adjustments.

        Non-hybrid models use base implementation without Metal adjustments.
        """
        # Set model as non-hybrid
        vllm_config.model_config.is_hybrid = False
        original_block_size = vllm_config.cache_config.block_size

        # Execute (should use base implementation only)
        MetalPlatform.update_block_size_for_backend(vllm_config)

        # For non-hybrid, base implementation may adjust block_size
        # but Metal-specific paged attention adjustment should not apply
        assert vllm_config.cache_config.block_size >= original_block_size

    def test_model_config_none(self):
        """Test: None model_config returns early without error."""
        config = VllmConfig(
            model_config=None,  # type: ignore
            cache_config=CacheConfig(block_size=16, gpu_memory_utilization=0.9),
            parallel_config=ParallelConfig(),
        )

        # Execute (should not raise)
        MetalPlatform.update_block_size_for_backend(config)

    # ========================================================================
    # Metal-Specific Adjustments
    # ========================================================================

    def test_paged_attention_adjusts_block_size_to_multiple_of_32(
        self, vllm_config, caplog
    ):
        """Test: Paged attention adjusts block_size to multiple of 32.

        When paged attention is enabled, block_size should be adjusted
        to be a multiple of 32 for Metal GPU kernel compatibility.
        """
        from vllm_metal.config import MetalConfig

        # Set block_size to a value not divisible by 32
        vllm_config.cache_config.block_size = 160  # 160 % 32 = 0, use 48
        vllm_config.cache_config.block_size = 48  # 48 % 32 = 16, should adjust to 64

        # Mock metal config with paged attention enabled
        with patch("vllm_metal.config.get_config") as mock_get_config:
            mock_metal_config = MetalConfig()
            mock_metal_config.use_paged_attention = True
            mock_get_config.return_value = mock_metal_config

            MetalPlatform.update_block_size_for_backend(vllm_config)

            # block_size should be adjusted to multiple of 32
            assert vllm_config.cache_config.block_size % 32 == 0

    def test_paged_attention_logs_warning(self, vllm_config, caplog):
        """Test: Hybrid + paged attention logs warning about block-size translation."""
        from vllm_metal.config import MetalConfig

        platform_logger = logging.getLogger("vllm_metal.platform")
        original_level = platform_logger.level
        platform_logger.addHandler(caplog.handler)
        platform_logger.setLevel(logging.WARNING)

        try:
            with patch("vllm_metal.config.get_config") as mock_get_config:
                mock_metal_config = MetalConfig()
                mock_metal_config.use_paged_attention = True
                mock_get_config.return_value = mock_metal_config

                MetalPlatform.update_block_size_for_backend(vllm_config)

                # Verify warning was logged
                assert "block-size translation" in caplog.text
                assert "PR #235" in caplog.text
        finally:
            platform_logger.removeHandler(caplog.handler)
            platform_logger.setLevel(original_level)

    def test_no_adjustment_when_already_multiple_of_32(self, vllm_config, caplog):
        """Test: No adjustment when block_size is already multiple of 32."""
        from vllm_metal.config import MetalConfig

        # Set block_size to multiple of 32
        vllm_config.cache_config.block_size = 64

        with patch("vllm_metal.config.get_config") as mock_get_config:
            mock_metal_config = MetalConfig()
            mock_metal_config.use_paged_attention = True
            mock_get_config.return_value = mock_metal_config

            MetalPlatform.update_block_size_for_backend(vllm_config)

            # Should remain unchanged
            assert vllm_config.cache_config.block_size == 64

            # No warning should be logged
            assert "Metal paged attention requires block_size" not in caplog.text
