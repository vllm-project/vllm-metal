# SPDX-License-Identifier: Apache-2.0
"""Tests for MLX tensor bridge."""

import pytest
import numpy as np


@pytest.mark.mlx
class TestTensorBridge:
    """Tests for PyTorch-MLX tensor bridge."""

    def test_to_mlx_from_torch(self, torch_device):
        """Test converting PyTorch tensor to MLX array."""
        import torch
        import mlx.core as mx
        from vllm_metal.mlx import to_mlx

        # Create PyTorch tensor
        torch_tensor = torch.randn(4, 8, device="cpu", dtype=torch.float32)

        # Convert to MLX
        mlx_array = to_mlx(torch_tensor)

        assert isinstance(mlx_array, mx.array)
        assert mlx_array.shape == (4, 8)
        assert mlx_array.dtype == mx.float32

    def test_to_torch_from_mlx(self, torch_device):
        """Test converting MLX array to PyTorch tensor."""
        import torch
        import mlx.core as mx
        from vllm_metal.mlx import to_torch

        # Create MLX array
        mlx_array = mx.random.normal(shape=(4, 8))

        # Convert to PyTorch
        torch_tensor = to_torch(mlx_array, device="cpu")

        assert isinstance(torch_tensor, torch.Tensor)
        assert torch_tensor.shape == (4, 8)

    def test_roundtrip_conversion(self):
        """Test roundtrip conversion preserves values."""
        import torch
        import mlx.core as mx
        from vllm_metal.mlx import to_mlx, to_torch

        # Create original tensor
        original = torch.randn(4, 8, dtype=torch.float32)

        # Convert to MLX and back
        mlx_array = to_mlx(original)
        recovered = to_torch(mlx_array, device="cpu", dtype=torch.float32)

        # Check values are close
        np.testing.assert_allclose(
            original.numpy(),
            recovered.numpy(),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_float16_conversion(self):
        """Test float16 conversion."""
        import torch
        import mlx.core as mx
        from vllm_metal.mlx import to_mlx, to_torch

        # Create float16 tensor
        original = torch.randn(4, 8, dtype=torch.float16)

        # Convert to MLX
        mlx_array = to_mlx(original)
        assert mlx_array.dtype == mx.float16

        # Convert back
        recovered = to_torch(mlx_array, device="cpu", dtype=torch.float16)
        assert recovered.dtype == torch.float16

    def test_bfloat16_conversion(self):
        """Test bfloat16 conversion."""
        import torch
        import mlx.core as mx
        from vllm_metal.mlx import to_mlx, to_torch

        # Create bfloat16 tensor
        original = torch.randn(4, 8, dtype=torch.bfloat16)

        # Convert to MLX
        mlx_array = to_mlx(original)
        assert mlx_array.dtype == mx.bfloat16

        # Convert back
        recovered = to_torch(mlx_array, device="cpu", dtype=torch.bfloat16)
        assert recovered.dtype == torch.bfloat16

    def test_tensor_bridge_context_manager(self):
        """Test TensorBridge context manager."""
        import torch
        from vllm_metal.mlx import TensorBridge

        original = torch.randn(4, 8, dtype=torch.float32)

        with TensorBridge(default_torch_device="cpu") as bridge:
            mlx_array = bridge.to_mlx(original)
            recovered = bridge.to_torch(mlx_array)

            assert recovered.shape == original.shape

    def test_dtype_mapping(self):
        """Test dtype conversion functions."""
        import torch
        import mlx.core as mx
        from vllm_metal.mlx.tensor_bridge import get_mlx_dtype, get_torch_dtype

        # Test PyTorch to MLX
        assert get_mlx_dtype(torch.float32) == mx.float32
        assert get_mlx_dtype(torch.float16) == mx.float16
        assert get_mlx_dtype(torch.int64) == mx.int64

        # Test MLX to PyTorch
        assert get_torch_dtype(mx.float32) == torch.float32
        assert get_torch_dtype(mx.float16) == torch.float16
        assert get_torch_dtype(mx.int64) == torch.int64
