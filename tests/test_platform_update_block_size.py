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

        When block_size is already large enough, block_size should not be reduced.
        Note: mamba_page_size_padded may still be set if attn_page_size > mamba_page_size.
        """
        # Set a very large block_size upfront
        vllm_config.cache_config.block_size = 256
        # Set cache_dtype to ensure consistent page size calculation
        vllm_config.cache_config.cache_dtype = "auto"

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

            # Verify: block_size should remain unchanged at 256
            # (it may be adjusted slightly due to alignment requirements)
            assert vllm_config.cache_config.block_size >= 256, (
                "block_size should not decrease"
            )

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

    def test_hybrid_with_paged_attention_raises_error(
        self, vllm_config, mock_mamba_state
    ):
        """Test: Hybrid model + paged attention raises ValueError.

        Metal paged attention kernels only support block_size in {8, 16, 32},
        but hybrid models require block_size=160. This configuration is
        unsupported and should raise a clear error message.
        """
        with (
            patch("vllm.model_executor.models.ModelRegistry") as mock_registry,
            patch("vllm_metal.config.get_config") as mock_get_config,
        ):
            mock_model_cls = MagicMock()
            mock_model_cls.get_mamba_state_shape_from_config.return_value = (
                mock_mamba_state["shape"]
            )
            mock_model_cls.get_mamba_state_dtype_from_config.return_value = (
                mock_mamba_state["dtype"]
            )
            mock_registry.resolve_model_cls.return_value = (mock_model_cls, None)

            # Mock metal config with paged attention enabled
            mock_metal_config = MagicMock()
            mock_metal_config.use_paged_attention = True
            mock_get_config.return_value = mock_metal_config

            # Execute and verify exception
            with pytest.raises(ValueError) as exc_info:
                MetalPlatform.update_block_size_for_backend(vllm_config)

            # Verify exception message contains helpful guidance
            error_msg = str(exc_info.value)
            assert "Hybrid models" in error_msg
            assert "not supported with paged attention" in error_msg
            assert "block_size in {8, 16, 32}" in error_msg
            assert "VLLM_METAL_USE_PAGED_ATTENTION=1" in error_msg


# ============================================================================
# MLA Model Tests
# ============================================================================


class TestMLAModels:
    """Test suite for MLA (Multi-Token Latent Attention) model support."""

    @pytest.fixture
    def mla_cache_config(self):
        """Create a CacheConfig mock for MLA models."""
        cache_config = MagicMock(spec=CacheConfig)
        cache_config.block_size = 16
        cache_config.user_specified_block_size = False
        cache_config.gpu_memory_utilization = 0.9
        cache_config.cache_dtype = "auto"
        cache_config.mamba_cache_mode = "none"
        cache_config.mamba_block_size = None
        cache_config.mamba_page_size_padded = None
        return cache_config

    @pytest.fixture
    def mla_model_config(self):
        """Create a ModelConfig mock for MLA models (e.g., DeepSeek)."""
        import torch

        model_config = MagicMock(spec=ModelConfig)
        model_config.is_hybrid = True
        model_config.use_mla = True  # MLA flag
        model_config.is_deepseek_mla = True
        model_config.architecture = "DeepSeekV2ForCausalLM"
        model_config.dtype = torch.float16
        model_config.max_model_len = 512
        model_config.get_num_kv_heads.return_value = 8
        model_config.get_head_size.return_value = 128
        return model_config

    @pytest.fixture
    def mla_vllm_config(self, mla_cache_config, mla_model_config):
        """Create a complete VllmConfig for MLA hybrid model testing."""
        parallel_config = MagicMock(spec=ParallelConfig)
        parallel_config.tensor_parallel_size = 1
        parallel_config.pipeline_parallel_size = 1

        config = MagicMock(spec=VllmConfig)
        config.model_config = mla_model_config
        config.cache_config = mla_cache_config
        config.parallel_config = parallel_config
        return config

    @pytest.fixture
    def mock_mla_mamba_state(self):
        """Create mock mamba state shape and dtype for MLA models.

        Using shapes that result in different page sizes for MLA vs FullAttention.
        MLA has different KV head dimensions which affects page_size_bytes.
        """
        import torch

        return {
            "shape": (
                (1, 3, 2048),  # conv_shape (max_seqs=1)
                (1, 4, 256, 128),  # recurrent_shape - MLA uses different head dims
            ),
            "dtype": (
                torch.float32,  # conv_dtype
                torch.float32,  # recurrent_dtype
            ),
        }

    def test_mla_hybrid_model_uses_mla_spec(
        self, mla_vllm_config, mock_mla_mamba_state
    ):
        """Test: MLA + Hybrid model uses MLAAttentionSpec (not FullAttentionSpec).

        This test verifies that MLA models use MLAAttentionSpec for page size
        calculation by checking that the implementation checks model_config.use_mla.

        Expected behavior:
        - Check model_config.use_mla == True
        - Use MLAAttentionSpec (which has different page_size calculation)
        """
        with patch("vllm.model_executor.models.ModelRegistry") as mock_registry:
            mock_model_cls = MagicMock()
            mock_model_cls.get_mamba_state_shape_from_config.return_value = (
                mock_mla_mamba_state["shape"]
            )
            mock_model_cls.get_mamba_state_dtype_from_config.return_value = (
                mock_mla_mamba_state["dtype"]
            )
            mock_registry.resolve_model_cls.return_value = (mock_model_cls, None)

            # Mock to track which Spec class is used
            # Patch at the vllm.v1.kv_cache_interface level where they're imported from
            with (
                patch("vllm.v1.kv_cache_interface.MLAAttentionSpec") as mock_mla_spec,
                patch("vllm.v1.kv_cache_interface.FullAttentionSpec") as mock_full_spec,
            ):
                # Setup mock return values
                mock_mla_spec_instance = MagicMock()
                mock_mla_spec_instance.page_size_bytes = 4096  # MLA page size
                mock_mla_spec.return_value = mock_mla_spec_instance

                mock_full_spec_instance = MagicMock()
                mock_full_spec_instance.page_size_bytes = (
                    2048  # Different FullAttention page size
                )
                mock_full_spec.return_value = mock_full_spec_instance

                # Execute
                MetalPlatform.update_block_size_for_backend(mla_vllm_config)

                # Verify: MLAAttentionSpec should be used for MLA models
                assert mock_mla_spec.called, (
                    "MLAAttentionSpec should be used for MLA models (use_mla=True)"
                )
                assert not mock_full_spec.called, (
                    "FullAttentionSpec should NOT be used for MLA models"
                )

    def test_mla_non_hybrid_skipped(self, mla_vllm_config):
        """Test: Pure MLA model (non-hybrid) skips the update.

        When use_mla=True but is_hybrid=False, the method should return early
        without modifying cache_config.

        Expected behavior:
        - is_hybrid = False triggers early return
        - cache_config remains unchanged
        """
        mla_vllm_config.model_config.is_hybrid = False

        original_block_size = mla_vllm_config.cache_config.block_size
        original_mamba_page_size_padded = (
            mla_vllm_config.cache_config.mamba_page_size_padded
        )

        # Execute
        MetalPlatform.update_block_size_for_backend(mla_vllm_config)

        # Verify: no changes
        assert mla_vllm_config.cache_config.block_size == original_block_size
        assert (
            mla_vllm_config.cache_config.mamba_page_size_padded
            == original_mamba_page_size_padded
        )

    @pytest.mark.parametrize("cache_dtype", ["bfloat16", "float16"])
    def test_mla_with_cache_dtype(
        self, mla_vllm_config, mock_mla_mamba_state, cache_dtype
    ):
        """Test: MLA model with different cache_dtype values.

        This test verifies that cache_config.cache_dtype is properly handled
        when computing page sizes for MLA models.

        Expected behavior:
        - cache_dtype is converted to torch.dtype correctly
        - MLAAttentionSpec uses the correct dtype
        - mamba_page_size_padded is set correctly
        """
        mla_vllm_config.cache_config.cache_dtype = cache_dtype

        with patch("vllm.model_executor.models.ModelRegistry") as mock_registry:
            mock_model_cls = MagicMock()
            mock_model_cls.get_mamba_state_shape_from_config.return_value = (
                mock_mla_mamba_state["shape"]
            )
            mock_model_cls.get_mamba_state_dtype_from_config.return_value = (
                mock_mla_mamba_state["dtype"]
            )
            mock_registry.resolve_model_cls.return_value = (mock_model_cls, None)

            # Execute (should not raise)
            MetalPlatform.update_block_size_for_backend(mla_vllm_config)

            # Verify
            cache_config = mla_vllm_config.cache_config
            assert cache_config.mamba_page_size_padded is not None, (
                f"mamba_page_size_padded should be set for cache_dtype={cache_dtype}"
            )
