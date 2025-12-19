# SPDX-License-Identifier: Apache-2.0
"""Tests for tensor bridge between MLX and PyTorch."""

import mlx.core as mx
import numpy as np
import torch

from vllm_metal.pytorch_backend.tensor_bridge import (
    MLX_TO_TORCH_DTYPE,
    TORCH_TO_MLX_DTYPE,
    get_torch_device,
    mlx_to_torch,
    sync_mlx,
    sync_torch,
    torch_to_mlx,
)


class TestDtypeMappings:
    """Tests for dtype mappings."""

    def test_mlx_to_torch_dtype_mapping(self) -> None:
        """Test MLX to PyTorch dtype mapping."""
        assert MLX_TO_TORCH_DTYPE[mx.float32] == torch.float32
        assert MLX_TO_TORCH_DTYPE[mx.float16] == torch.float16
        assert MLX_TO_TORCH_DTYPE[mx.int32] == torch.int32
        assert MLX_TO_TORCH_DTYPE[mx.int64] == torch.int64
        assert MLX_TO_TORCH_DTYPE[mx.bool_] == torch.bool

    def test_torch_to_mlx_dtype_mapping(self) -> None:
        """Test PyTorch to MLX dtype mapping."""
        assert TORCH_TO_MLX_DTYPE[torch.float32] == mx.float32
        assert TORCH_TO_MLX_DTYPE[torch.float16] == mx.float16
        assert TORCH_TO_MLX_DTYPE[torch.int32] == mx.int32
        assert TORCH_TO_MLX_DTYPE[torch.int64] == mx.int64
        assert TORCH_TO_MLX_DTYPE[torch.bool] == mx.bool_


class TestTorchDevice:
    """Tests for PyTorch device selection."""

    def test_get_torch_device(self) -> None:
        """Test PyTorch device retrieval."""
        device = get_torch_device()
        # Should be MPS on Apple Silicon or CPU otherwise
        assert device.type in ("mps", "cpu")


class TestTensorConversion:
    """Tests for tensor conversion between MLX and PyTorch."""

    def test_torch_to_mlx_float32(self) -> None:
        """Test PyTorch to MLX conversion for float32."""
        torch_tensor = torch.randn(2, 3, dtype=torch.float32)
        mlx_array = torch_to_mlx(torch_tensor)
        mx.eval(mlx_array)

        assert mlx_array.shape == (2, 3)
        assert mlx_array.dtype == mx.float32
        np.testing.assert_allclose(np.array(mlx_array), torch_tensor.numpy(), rtol=1e-5)

    def test_torch_to_mlx_float16(self) -> None:
        """Test PyTorch to MLX conversion for float16."""
        torch_tensor = torch.randn(2, 3, dtype=torch.float16)
        mlx_array = torch_to_mlx(torch_tensor)
        mx.eval(mlx_array)

        assert mlx_array.shape == (2, 3)
        assert mlx_array.dtype == mx.float16

    def test_torch_to_mlx_int32(self) -> None:
        """Test PyTorch to MLX conversion for int32."""
        torch_tensor = torch.randint(0, 100, (2, 3), dtype=torch.int32)
        mlx_array = torch_to_mlx(torch_tensor)
        mx.eval(mlx_array)

        assert mlx_array.shape == (2, 3)
        assert mlx_array.dtype == mx.int32
        np.testing.assert_array_equal(np.array(mlx_array), torch_tensor.numpy())

    def test_mlx_to_torch_float32(self) -> None:
        """Test MLX to PyTorch conversion for float32."""
        mlx_array = mx.random.normal((2, 3))
        mx.eval(mlx_array)

        torch_tensor = mlx_to_torch(mlx_array, device="cpu")

        assert torch_tensor.shape == (2, 3)
        assert torch_tensor.dtype == torch.float32
        np.testing.assert_allclose(torch_tensor.numpy(), np.array(mlx_array), rtol=1e-5)

    def test_mlx_to_torch_int32(self) -> None:
        """Test MLX to PyTorch conversion for int32."""
        mlx_array = mx.array([[1, 2, 3], [4, 5, 6]], dtype=mx.int32)
        mx.eval(mlx_array)

        torch_tensor = mlx_to_torch(mlx_array, device="cpu")

        assert torch_tensor.shape == (2, 3)
        assert torch_tensor.dtype == torch.int32
        np.testing.assert_array_equal(torch_tensor.numpy(), np.array(mlx_array))

    def test_round_trip_conversion(self) -> None:
        """Test round-trip conversion preserves values."""
        # PyTorch -> MLX -> PyTorch
        original = torch.randn(4, 5, dtype=torch.float32)
        mlx_array = torch_to_mlx(original)
        mx.eval(mlx_array)
        result = mlx_to_torch(mlx_array, device="cpu")

        np.testing.assert_allclose(result.numpy(), original.numpy(), rtol=1e-5)

    def test_mlx_to_torch_default_device(self) -> None:
        """Test MLX to PyTorch with default device."""
        mlx_array = mx.array([1.0, 2.0, 3.0])
        mx.eval(mlx_array)

        torch_tensor = mlx_to_torch(mlx_array)

        assert torch_tensor.shape == (3,)
        # Device should be MPS or CPU depending on availability
        assert torch_tensor.device.type in ("mps", "cpu")


class TestSynchronization:
    """Tests for synchronization functions."""

    def test_sync_mlx(self) -> None:
        """Test MLX synchronization."""
        # Should not raise
        sync_mlx()

    def test_sync_torch(self) -> None:
        """Test PyTorch synchronization."""
        # Should not raise
        sync_torch()
