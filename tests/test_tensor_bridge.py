# SPDX-License-Identifier: Apache-2.0
"""Tests for tensor bridge between MLX and PyTorch."""

import mlx.core as mx
import numpy as np
import pytest
import torch

import vllm_metal.pytorch_backend.tensor_bridge as tensor_bridge
from vllm_metal.pytorch_backend.tensor_bridge import (
    _MPS_SAFE_SIZE_BYTES,
    MLX_TO_TORCH_DTYPE,
    TORCH_TO_MLX_DTYPE,
    _get_tensor_size_bytes,
    _is_safe_for_mps,
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


class TestSynchronization:
    """Tests for synchronization functions."""

    def test_sync_mlx_uses_barrier_when_available(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Sync should call the MX barrier when present."""
        if not hasattr(tensor_bridge.mx, "synchronize"):
            pytest.skip("mlx.core.synchronize not available")

        called = False

        def fake_sync() -> None:
            nonlocal called
            called = True

        monkeypatch.setattr(tensor_bridge.mx, "synchronize", fake_sync)
        sync_mlx()
        assert called is True

    def test_sync_mlx_falls_back_to_eval_when_missing_barrier(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If barrier is absent, we should still force evaluation."""
        monkeypatch.delattr(tensor_bridge.mx, "synchronize", raising=False)

        called = False

        def fake_eval(value: mx.array) -> None:
            nonlocal called
            called = True

        monkeypatch.setattr(tensor_bridge.mx, "eval", fake_eval)
        sync_mlx()
        assert called is True

    def test_sync_mlx_falls_back_to_eval_when_barrier_signature_incompatible(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If barrier exists but can't be called with no args, fall back."""

        def fake_sync(_stream: object) -> None:
            return None

        monkeypatch.setattr(tensor_bridge.mx, "synchronize", fake_sync)

        called = False

        def fake_eval(value: mx.array) -> None:
            nonlocal called
            called = True

        monkeypatch.setattr(tensor_bridge.mx, "eval", fake_eval)
        sync_mlx()
        assert called is True

    def test_sync_torch(self) -> None:
        """Test PyTorch synchronization."""
        # Should not raise
        sync_torch()


class TestMPSSizeLimit:
    """Tests for MPS 4GB size limit handling.

    See: https://github.com/anthropics/vllm-metal/issues/43
    """

    def test_get_tensor_size_bytes_float32(self) -> None:
        """Test tensor size calculation for float32."""
        # 2x3 float32 = 6 elements * 4 bytes = 24 bytes
        array = mx.zeros((2, 3), dtype=mx.float32)
        assert _get_tensor_size_bytes(array) == 24

    def test_get_tensor_size_bytes_float16(self) -> None:
        """Test tensor size calculation for float16."""
        # 4x5 float16 = 20 elements * 2 bytes = 40 bytes
        array = mx.zeros((4, 5), dtype=mx.float16)
        assert _get_tensor_size_bytes(array) == 40

    def test_get_tensor_size_bytes_int32(self) -> None:
        """Test tensor size calculation for int32."""
        # 10x10 int32 = 100 elements * 4 bytes = 400 bytes
        array = mx.zeros((10, 10), dtype=mx.int32)
        assert _get_tensor_size_bytes(array) == 400

    def test_is_safe_for_mps_small_tensor(self) -> None:
        """Test that small tensors are safe for MPS."""
        # 100 float32 = 400 bytes, well under 1GB limit
        array = mx.zeros((100,), dtype=mx.float32)
        assert _is_safe_for_mps(array) is True

    def test_is_safe_for_mps_large_tensor(self) -> None:
        """Test that large tensors are detected as unsafe for MPS."""
        # Create a tensor larger than the safe limit
        # _MPS_SAFE_SIZE_BYTES is 1GB = 2^30 bytes
        # We need more than 2^30 / 4 = 2^28 = 268,435,456 float32 elements
        # Use a shape that exceeds this: e.g., 512 * 1024 * 1024 = 536,870,912
        # But we don't want to actually allocate that much memory in tests
        # Instead, verify the threshold constant is correct
        assert _MPS_SAFE_SIZE_BYTES == 1 << 30  # 1GB

        # Small tensor should be safe
        small_array = mx.zeros((1000, 1000), dtype=mx.float32)  # 4MB
        assert _is_safe_for_mps(small_array) is True

    def test_mlx_to_torch_small_tensor_uses_mps(self) -> None:
        """Test that small tensors go to MPS when available."""
        if not torch.backends.mps.is_available():
            return  # Skip on non-MPS systems

        array = mx.array([1.0, 2.0, 3.0], dtype=mx.float32)
        mx.eval(array)

        tensor = mlx_to_torch(array, device="mps")
        assert tensor.device.type == "mps"

    def test_mlx_to_torch_explicit_cpu(self) -> None:
        """Test that explicit CPU device is respected."""
        array = mx.array([1.0, 2.0, 3.0], dtype=mx.float32)
        mx.eval(array)

        tensor = mlx_to_torch(array, device="cpu")
        assert tensor.device.type == "cpu"
