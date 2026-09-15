# SPDX-License-Identifier: Apache-2.0
"""Tests for tensor bridge between MLX and PyTorch."""

import gc

import mlx.core as mx
import numpy as np
import pytest
import torch

from vllm_metal.pytorch_backend.tensor_bridge import (
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
        np.testing.assert_allclose(torch_tensor.numpy(), np.array(mlx_view), rtol=1e-5)

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


@pytest.fixture(params=["cpu", "mps"])
def torch_device(request: pytest.FixtureRequest) -> str:
    if request.param == "mps" and not torch.backends.mps.is_available():
        pytest.skip("PyTorch MPS is required")
    return request.param


class TestDLPackSharing:
    @pytest.mark.parametrize(
        "dtype",
        [torch.float32, torch.float16, torch.bfloat16, torch.int32, torch.bool],
    )
    def test_import_shares_offset_strided_view(
        self, torch_device: str, dtype: torch.dtype
    ) -> None:
        source = torch.arange(40, device=torch_device, dtype=torch.float32)
        view = source.to(dtype).reshape(5, 8)[1::2, 1::2].T
        array = torch_to_mlx(view)
        assert array.tolist() == view.tolist()
        assert array.dtype == TORCH_TO_MLX_DTYPE[dtype]

        round_trip = mlx_to_torch(array, device=torch_device)
        assert round_trip.device.type == torch_device
        assert round_trip.stride() == view.stride()
        assert round_trip.data_ptr() == view.data_ptr()

        view.zero_()
        if torch_device == "mps":
            torch.mps.synchronize()
        assert array.tolist() == view.tolist()

    @pytest.mark.parametrize(
        "dtype", [mx.float32, mx.float16, mx.bfloat16, mx.int32, mx.bool_]
    )
    def test_export_shares_offset_strided_view(
        self, torch_device: str, dtype: mx.Dtype
    ) -> None:
        array = mx.arange(40).astype(dtype).reshape(5, 8)[1::2, 1::2].T
        tensor = mlx_to_torch(array, device=torch_device)
        assert tensor.tolist() == array.tolist()
        assert tensor.dtype == MLX_TO_TORCH_DTYPE[dtype]
        assert tensor.device.type == torch_device
        assert tensor.stride() == (2, 16)

        tensor.zero_()
        if torch_device == "mps":
            torch.mps.synchronize()
        assert array.tolist() == tensor.tolist()

    def test_import_keeps_source_alive(self, torch_device: str) -> None:
        source = torch.arange(16, device=torch_device, dtype=torch.float32)[3:12:2]
        array = torch_to_mlx(source)
        del source
        gc.collect()
        assert (array + 1).tolist() == [4, 6, 8, 10, 12]

    def test_export_keeps_source_alive(self, torch_device: str) -> None:
        array = mx.arange(16, dtype=mx.float32)[3:12:2]
        tensor = mlx_to_torch(array, device=torch_device)
        del array
        gc.collect()
        assert (tensor + 1).tolist() == [4, 6, 8, 10, 12]

    def test_import_detaches_autograd(self, torch_device: str) -> None:
        source = torch.tensor([1.0, 2.0], device=torch_device, requires_grad=True)
        array = torch_to_mlx(source)
        assert array.tolist() == [1.0, 2.0]
        assert source.requires_grad

    @pytest.mark.parametrize("shape", [(), (0,), (2, 0)])
    def test_scalar_and_empty_round_trip(
        self, torch_device: str, shape: tuple[int, ...]
    ) -> None:
        source = torch.zeros(shape, device=torch_device, dtype=torch.bfloat16)
        result = mlx_to_torch(torch_to_mlx(source), device=torch_device)
        assert result.shape == source.shape
        assert result.dtype == source.dtype
        assert result.device.type == torch_device
        assert result.tolist() == source.tolist()

    @pytest.mark.parametrize("layout", ["negative", "broadcast"])
    def test_materialized_views_are_writable(
        self, torch_device: str, layout: str
    ) -> None:
        array = (
            mx.arange(8, dtype=mx.float32)[::-2]
            if layout == "negative"
            else mx.broadcast_to(mx.array(2.0), (4, 3))
        )
        expected = array.tolist()
        tensor = mlx_to_torch(array, device=torch_device)
        assert tensor.is_contiguous()
        tensor.add_(1)
        if torch_device == "mps":
            torch.mps.synchronize()
        assert tensor.tolist() == (np.array(expected) + 1).tolist()
        assert array.tolist() == expected
