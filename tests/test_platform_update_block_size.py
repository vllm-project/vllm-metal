# SPDX-License-Identifier: Apache-2.0
"""Unit tests for MetalPlatform.update_block_size_for_backend().

Tests cover:
1. Success cases: hybrid models, non-hybrid models
2. Failure cases: model resolution failure, invalid config, etc.
"""

from unittest.mock import MagicMock, patch

import pytest
from vllm.config import CacheConfig, ModelConfig, ParallelConfig, VllmConfig

from vllm_metal.platform import MetalPlatform


class TestUpdateBlockSizeForBackend:
    """Test suite for update_block_size_for_backend() method."""

    # ========================================================================
    # Fixtures
    # ========================================================================

    @pytest.fixture
    def base_cache_config(self):
        """Create a base CacheConfig mock for testing."""
        cache_config = MagicMock(spec=CacheConfig)
        cache_config.block_size = 16
        cache_config.user_specified_block_size = (
            False  # Allow block_size to be adjusted
        )
        cache_config.gpu_memory_utilization = 0.9
        cache_config.cache_dtype = "auto"
        cache_config.mamba_cache_mode = "none"
        cache_config.mamba_block_size = None
        cache_config.mamba_page_size_padded = None
        return cache_config

    @pytest.fixture
    def base_model_config(self):
        """Create a base ModelConfig mock for testing."""
        import torch

        model_config = MagicMock(spec=ModelConfig)
        model_config.is_hybrid = True
        model_config.architecture = "Qwen3_5ForCausalLM"
        model_config.dtype = torch.float16  # Use torch.dtype instead of string
        model_config.max_model_len = 512
        model_config.get_num_kv_heads.return_value = 8
        model_config.get_head_size.return_value = 128
        return model_config

    @pytest.fixture
    def vllm_config(self, base_cache_config, base_model_config):
        """Create a complete VllmConfig for hybrid model testing."""
        parallel_config = MagicMock(spec=ParallelConfig)
        parallel_config.tensor_parallel_size = 1
        parallel_config.pipeline_parallel_size = 1

        config = MagicMock(spec=VllmConfig)
        config.model_config = base_model_config
        config.cache_config = base_cache_config
        config.parallel_config = parallel_config
        return config

    @pytest.fixture
    def mock_mamba_state(self):
        """Create mock mamba state shape and dtype.

        Use small shapes that result in reasonable block_size calculations.
        For Qwen3.5-0.8B:
        - conv_shape: (max_seqs, conv_kernel-1, conv_dim)
        - recurrent_shape: (max_seqs, num_v_heads, value_head_dim, key_head_dim)

        Using max_seqs=1 keeps the page size small for testing.
        """
        import torch

        return {
            "shape": (
                (1, 3, 2048),  # conv_shape (max_seqs=1)
                (1, 8, 128, 128),  # recurrent_shape (max_seqs=1)
            ),
            "dtype": (
                torch.float32,  # conv_dtype
                torch.float32,  # recurrent_dtype
            ),
        }

    # ========================================================================
    # Success Cases
    # ========================================================================

    def test_hybrid_model_success(self, vllm_config, mock_mamba_state):
        """Test: Hybrid model successfully sets mamba_page_size_padded.

        This is the main success path - hybrid model with valid config.
        """
        with patch("vllm.model_executor.models.ModelRegistry") as mock_registry:
            # Setup mock
            mock_model_cls = MagicMock()
            mock_model_cls.get_mamba_state_shape_from_config.return_value = (
                mock_mamba_state["shape"]
            )
            mock_model_cls.get_mamba_state_dtype_from_config.return_value = (
                mock_mamba_state["dtype"]
            )
            mock_registry.resolve_model_cls.return_value = (mock_model_cls, None)

            # Execute
            MetalPlatform.update_block_size_for_backend(vllm_config)

            # Verify
            cache_config = vllm_config.cache_config
            assert cache_config.mamba_page_size_padded is not None, (
                "mamba_page_size_padded should be set for hybrid model"
            )
            assert cache_config.mamba_page_size_padded > 0, (
                "mamba_page_size_padded should be positive"
            )
            assert cache_config.block_size >= 16, (
                "block_size should be >= original value"
            )

    def test_hybrid_model_block_size_already_sufficient(
        self, vllm_config, mock_mamba_state
    ):
        """Test: Hybrid model with already-sufficient block_size.

        When block_size is already large enough, no padding is needed.
        """
        # Set a very large block_size upfront
        vllm_config.cache_config.block_size = 256

        with patch("vllm.model_executor.models.ModelRegistry") as mock_registry:
            mock_model_cls = MagicMock()
            mock_model_cls.get_mamba_state_shape_from_config.return_value = (
                mock_mamba_state["shape"]
            )
            mock_model_cls.get_mamba_state_dtype_from_config.return_value = (
                mock_mamba_state["dtype"]
            )
            mock_registry.resolve_model_cls.return_value = (mock_model_cls, None)

            # Execute
            MetalPlatform.update_block_size_for_backend(vllm_config)

            # Verify: block_size should remain unchanged
            assert vllm_config.cache_config.block_size == 256

    def test_non_hybrid_model_skipped(self, vllm_config):
        """Test: Non-hybrid model skips the update entirely.

        Non-hybrid models don't need mamba_page_size_padded.
        """
        # Set model as non-hybrid
        vllm_config.model_config.is_hybrid = False

        original_block_size = vllm_config.cache_config.block_size

        # Execute (should return early)
        MetalPlatform.update_block_size_for_backend(vllm_config)

        # Verify: no changes
        assert vllm_config.cache_config.block_size == original_block_size
        assert vllm_config.cache_config.mamba_page_size_padded is None

    # ========================================================================
    # Failure Cases - Early Return (No Exception)
    # ========================================================================

    def test_model_config_none(self):
        """Test: None model_config returns early without error.

        This can happen during initialization edge cases.
        """
        config = MagicMock(spec=VllmConfig)
        config.model_config = None
        config.cache_config = MagicMock()

        # Execute (should not raise)
        MetalPlatform.update_block_size_for_backend(config)

        # Verify: no changes to cache_config
        # (method should return early)

    # ========================================================================
    # Failure Cases - Raise Exceptions
    # ========================================================================

    def test_model_resolution_failure(self, vllm_config):
        """Test: Model class resolution failure raises exception.

        This happens when the model architecture is not registered.
        """
        with patch("vllm.model_executor.models.ModelRegistry") as mock_registry:
            # Setup mock to raise exception
            mock_registry.resolve_model_cls.side_effect = ValueError(
                "Model architecture 'Qwen3_5ForCausalLM' not found"
            )

            # Execute and verify exception
            with pytest.raises(ValueError) as exc_info:
                MetalPlatform.update_block_size_for_backend(vllm_config)

            # Verify exception message
            assert "not found" in str(exc_info.value).lower()

    def test_get_mamba_state_shape_failure(self, vllm_config):
        """Test: get_mamba_state_shape_from_config failure raises exception.

        This happens when model class doesn't have the required method.
        """
        with patch("vllm.model_executor.models.ModelRegistry") as mock_registry:
            mock_model_cls = MagicMock()
            mock_model_cls.get_mamba_state_shape_from_config.side_effect = (
                AttributeError("Model has no get_mamba_state_shape_from_config")
            )
            mock_registry.resolve_model_cls.return_value = (mock_model_cls, None)

            # Execute and verify exception
            with pytest.raises(AttributeError) as exc_info:
                MetalPlatform.update_block_size_for_backend(vllm_config)

            # Verify exception message
            assert "get_mamba_state_shape_from_config" in str(exc_info.value)

    def test_get_mamba_state_dtype_failure(self, vllm_config, mock_mamba_state):
        """Test: get_mamba_state_dtype_from_config failure raises exception.

        This happens when model class doesn't have the dtype method.
        """
        with patch("vllm.model_executor.models.ModelRegistry") as mock_registry:
            mock_model_cls = MagicMock()
            mock_model_cls.get_mamba_state_shape_from_config.return_value = (
                mock_mamba_state["shape"]
            )
            mock_model_cls.get_mamba_state_dtype_from_config.side_effect = (
                AttributeError("Model has no get_mamba_state_dtype_from_config")
            )
            mock_registry.resolve_model_cls.return_value = (mock_model_cls, None)

            # Execute and verify exception
            with pytest.raises(AttributeError) as exc_info:
                MetalPlatform.update_block_size_for_backend(vllm_config)

            # Verify exception message
            assert "get_mamba_state_dtype_from_config" in str(exc_info.value)

    def test_mamba_page_size_zero(self, vllm_config):
        """Test: Zero mamba_page_size raises exception.

        This happens when state shape calculation results in zero.
        """
        import torch

        with patch("vllm.model_executor.models.ModelRegistry") as mock_registry:
            mock_model_cls = MagicMock()
            # Return zero-sized shape
            mock_model_cls.get_mamba_state_shape_from_config.return_value = (
                (0, 0, 0),
                (0, 0, 0, 0),
            )
            mock_model_cls.get_mamba_state_dtype_from_config.return_value = (
                torch.float32,
                torch.float32,
            )
            mock_registry.resolve_model_cls.return_value = (mock_model_cls, None)

            # Execute and verify exception
            with pytest.raises(ValueError) as exc_info:
                MetalPlatform.update_block_size_for_backend(vllm_config)

            # Verify exception message
            assert "zero" in str(exc_info.value).lower()

    def test_invalid_architecture(self, vllm_config):
        """Test: Invalid architecture raises exception.

        This happens when the architecture string is malformed.
        """
        vllm_config.model_config.architecture = "InvalidArchitecture_123"

        with patch("vllm.model_executor.models.ModelRegistry") as mock_registry:
            mock_registry.resolve_model_cls.side_effect = KeyError(
                "Unknown architecture: InvalidArchitecture_123"
            )

            # Execute and verify exception
            with pytest.raises(KeyError) as exc_info:
                MetalPlatform.update_block_size_for_backend(vllm_config)

            # Verify exception message
            assert "InvalidArchitecture" in str(exc_info.value)

    # ========================================================================
    # Edge Cases
    # ========================================================================

    def test_block_size_increased_to_minimum(self, vllm_config, mock_mamba_state):
        """Test: block_size is increased to minimum required value.

        When original block_size is too small, it should be increased.
        """
        # Set very small block_size
        vllm_config.cache_config.block_size = 1

        with patch("vllm.model_executor.models.ModelRegistry") as mock_registry:
            mock_model_cls = MagicMock()
            mock_model_cls.get_mamba_state_shape_from_config.return_value = (
                mock_mamba_state["shape"]
            )
            mock_model_cls.get_mamba_state_dtype_from_config.return_value = (
                mock_mamba_state["dtype"]
            )
            mock_registry.resolve_model_cls.return_value = (mock_model_cls, None)

            # Execute
            MetalPlatform.update_block_size_for_backend(vllm_config)

            # Verify: block_size should be increased
            assert vllm_config.cache_config.block_size > 1
            # Should be at least 32 (kernel_block_alignment_size)
            assert vllm_config.cache_config.block_size >= 32

    def test_mamba_cache_mode_align(self, vllm_config, mock_mamba_state):
        """Test: mamba_block_size is synced when mamba_cache_mode='align'.

        This tests the align mode specific logic.
        """
        vllm_config.cache_config.mamba_cache_mode = "align"

        with patch("vllm.model_executor.models.ModelRegistry") as mock_registry:
            mock_model_cls = MagicMock()
            mock_model_cls.get_mamba_state_shape_from_config.return_value = (
                mock_mamba_state["shape"]
            )
            mock_model_cls.get_mamba_state_dtype_from_config.return_value = (
                mock_mamba_state["dtype"]
            )
            mock_registry.resolve_model_cls.return_value = (mock_model_cls, None)

            # Execute
            MetalPlatform.update_block_size_for_backend(vllm_config)

            # Verify: mamba_block_size should equal block_size
            assert vllm_config.cache_config.mamba_block_size == (
                vllm_config.cache_config.block_size
            )
