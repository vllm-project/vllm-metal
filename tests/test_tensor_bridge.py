# SPDX-License-Identifier: Apache-2.0
"""Tests for tensor bridge between MLX and PyTorch."""

import gc

import mlx.core as mx
import numpy as np
import pytest
import torch

from vllm_metal.tensor_bridge import (
    MLX_TO_TORCH_DTYPE,
    TORCH_TO_MLX_DTYPE,
    get_torch_device,
    mlx_to_torch,
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

    def test_torch_to_mlx_bfloat16(self) -> None:
        """Test PyTorch to MLX conversion for bfloat16."""
        torch_tensor = torch.randn(2, 3, dtype=torch.bfloat16)
        mlx_array = torch_to_mlx(torch_tensor)
        mx.eval(mlx_array)

        assert mlx_array.shape == (2, 3)
        assert mlx_array.dtype == mx.bfloat16

        # Compare as float32 since numpy doesn't support bfloat16.
        mlx_f32 = mlx_array.astype(mx.float32)
        mx.eval(mlx_f32)
        np.testing.assert_allclose(
            np.array(mlx_f32),
            torch_tensor.float().numpy(),
            rtol=1e-2,
            atol=1e-2,
        )

        # Encoder checkpoints can be BF16; MLX must retain their Torch storage.
        torch_tensor.fill_(3)
        assert mlx_array.tolist() == [[3.0] * 3] * 2
        del torch_tensor
        gc.collect()
        assert (mlx_array + 1).tolist() == [[4.0] * 3] * 2

    def test_torch_to_mlx_copy_owns_its_storage(self) -> None:
        """A copied import neither sees nor makes writes to the source."""
        source = torch.linspace(0, 1, 8, dtype=torch.bfloat16)
        original = source.clone()

        array = torch_to_mlx(source, copy=True)
        source.fill_(3)
        assert array.tolist() == original.float().tolist()

        # MLX may hand an input buffer to an op's output once the graph holds
        # the last reference to it; a copy's buffer belongs to MLX alone.
        source.copy_(original)
        normalized = 2 * (torch_to_mlx(source, copy=True) - 0.5)
        mx.eval(normalized)
        assert torch.equal(source, original)

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

    def test_mlx_to_torch_bfloat16(self) -> None:
        """Test MLX to PyTorch conversion for bfloat16."""
        mlx_array = mx.array([[1.0, 2.0], [3.0, 4.0]], dtype=mx.bfloat16)
        mx.eval(mlx_array)

        torch_tensor = mlx_to_torch(mlx_array, device="cpu")

        assert torch_tensor.shape == (2, 2)
        assert torch_tensor.dtype == torch.bfloat16
        # Compare as float32 since numpy doesn't support bfloat16
        torch.testing.assert_close(
            torch_tensor.float(),
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        )

    def test_mlx_to_torch_view(self) -> None:
        """MLX views (e.g. slices) should be convertible to torch."""
        mlx_array = mx.random.normal((2, 3, 4))
        mlx_view = mlx_array[:, -1, :]
        mx.eval(mlx_view)

        torch_tensor = mlx_to_torch(mlx_view, device="cpu")

        assert torch_tensor.shape == (2, 4)
        assert torch_tensor.dtype == torch.float32
        assert torch_tensor.stride() == (12, 1)
        np.testing.assert_allclose(torch_tensor.numpy(), np.array(mlx_view), rtol=1e-5)

        torch_tensor.zero_()
        assert mlx_view.tolist() == [[0.0] * 4] * 2
        del mlx_array, mlx_view
        gc.collect()
        assert torch_tensor.tolist() == [[0.0] * 4] * 2

    def test_round_trip_conversion(self) -> None:
        """Test round-trip conversion preserves values."""
        # PyTorch -> MLX -> PyTorch
        original = torch.randn(4, 5, dtype=torch.float32)
        mlx_array = torch_to_mlx(original)
        mx.eval(mlx_array)
        result = mlx_to_torch(mlx_array, device="cpu")

        assert result.data_ptr() == original.data_ptr()
        np.testing.assert_allclose(result.numpy(), original.numpy(), rtol=1e-5)

    def test_mlx_to_torch_default_device(self) -> None:
        """Test MLX to PyTorch with default device."""
        mlx_array = mx.array([1.0, 2.0, 3.0])
        mx.eval(mlx_array)

        torch_tensor = mlx_to_torch(mlx_array)

        assert torch_tensor.shape == (3,)
        # Device should be MPS or CPU depending on availability
        assert torch_tensor.device.type in ("mps", "cpu")

    @pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS required")
    def test_torch_to_mlx_shares_mps_input(self) -> None:
        source = torch.arange(8, device="mps", dtype=torch.float32)
        array = torch_to_mlx(source)
        assert array.tolist() == list(range(8))
        source.zero_()
        torch.mps.synchronize()
        assert array.tolist() == [0.0] * 8

    def test_negative_strides_are_rejected(self) -> None:
        array = mx.arange(8, dtype=mx.float32)[::-2]
        with pytest.raises(ValueError, match="negative strides"):
            mlx_to_torch(array, device="cpu")
