# SPDX-License-Identifier: Apache-2.0
"""Tensor bridge between PyTorch and MLX.

This module provides efficient conversion between PyTorch tensors and
MLX arrays, leveraging Apple Silicon's unified memory architecture.
"""

from typing import Union

import mlx.core as mx
import numpy as np
import torch

# Type alias for tensor-like objects
TensorLike = Union[torch.Tensor, mx.array, np.ndarray]


def to_mlx(tensor: TensorLike) -> mx.array:
    """Convert a tensor to MLX array.

    This function converts PyTorch tensors, NumPy arrays, or MLX arrays
    to MLX arrays. On Apple Silicon with unified memory, the conversion
    is efficient as data doesn't need to be copied between CPU and GPU.

    Args:
        tensor: Input tensor (PyTorch, NumPy, or MLX).

    Returns:
        MLX array.
    """
    if isinstance(tensor, mx.array):
        return tensor

    if isinstance(tensor, torch.Tensor):
        # Move to CPU if on MPS (unified memory, so this is fast)
        if tensor.device.type == "mps":
            tensor = tensor.cpu()

        # Convert via NumPy (most reliable path)
        # Note: bfloat16 requires special handling
        if tensor.dtype == torch.bfloat16:
            # MLX supports bfloat16 natively
            np_array = tensor.view(torch.int16).numpy()
            return mx.array(np_array).view(mx.bfloat16)
        else:
            return mx.array(tensor.numpy())

    if isinstance(tensor, np.ndarray):
        return mx.array(tensor)

    raise TypeError(f"Cannot convert {type(tensor)} to MLX array")


def to_torch(
    array: TensorLike,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Convert an MLX array to PyTorch tensor.

    Args:
        array: Input array (MLX, NumPy, or PyTorch).
        device: Target PyTorch device ("cpu" or "mps").
        dtype: Optional target dtype.

    Returns:
        PyTorch tensor.
    """
    if isinstance(array, torch.Tensor):
        tensor = array
    elif isinstance(array, mx.array):
        # Force evaluation if lazy
        mx.eval(array)

        # Handle bfloat16 specially
        if array.dtype == mx.bfloat16:
            # Convert via int16 view
            int_array = array.view(mx.int16)
            np_array = np.array(int_array)
            tensor = torch.from_numpy(np_array).view(torch.bfloat16)
        else:
            np_array = np.array(array)
            tensor = torch.from_numpy(np_array)
    elif isinstance(array, np.ndarray):
        tensor = torch.from_numpy(array)
    else:
        raise TypeError(f"Cannot convert {type(array)} to PyTorch tensor")

    # Move to target device
    if device != "cpu":
        tensor = tensor.to(device)

    # Convert dtype if specified
    if dtype is not None and tensor.dtype != dtype:
        tensor = tensor.to(dtype)

    return tensor


class TensorBridge:
    """Context manager for PyTorch-MLX tensor bridge.

    This class provides a convenient way to work with both frameworks,
    automatically handling conversions at the boundaries.

    Example:
        with TensorBridge() as bridge:
            mlx_q = bridge.to_mlx(pytorch_q)
            mlx_result = mx.fast.scaled_dot_product_attention(mlx_q, ...)
            pytorch_result = bridge.to_torch(mlx_result, device="mps")
    """

    def __init__(self, default_torch_device: str = "cpu"):
        """Initialize the tensor bridge.

        Args:
            default_torch_device: Default device for PyTorch tensors.
        """
        self.default_torch_device = default_torch_device
        self._converted_tensors: list[mx.array] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Clear references to allow garbage collection
        self._converted_tensors.clear()
        return False

    def to_mlx(self, tensor: TensorLike) -> mx.array:
        """Convert tensor to MLX array."""
        result = to_mlx(tensor)
        self._converted_tensors.append(result)
        return result

    def to_torch(
        self,
        array: TensorLike,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Convert MLX array to PyTorch tensor."""
        if device is None:
            device = self.default_torch_device
        return to_torch(array, device=device, dtype=dtype)


# Convenience functions for common dtype conversions
def get_mlx_dtype(torch_dtype: torch.dtype) -> mx.Dtype:
    """Convert PyTorch dtype to MLX dtype.

    Args:
        torch_dtype: PyTorch data type.

    Returns:
        Corresponding MLX data type.
    """
    dtype_map = {
        torch.float32: mx.float32,
        torch.float16: mx.float16,
        torch.bfloat16: mx.bfloat16,
        torch.int32: mx.int32,
        torch.int64: mx.int64,
        torch.int16: mx.int16,
        torch.int8: mx.int8,
        torch.uint8: mx.uint8,
        torch.bool: mx.bool_,
    }
    if torch_dtype not in dtype_map:
        raise ValueError(f"Unsupported dtype: {torch_dtype}")
    return dtype_map[torch_dtype]


def get_torch_dtype(mlx_dtype: mx.Dtype) -> torch.dtype:
    """Convert MLX dtype to PyTorch dtype.

    Args:
        mlx_dtype: MLX data type.

    Returns:
        Corresponding PyTorch data type.
    """
    dtype_map = {
        mx.float32: torch.float32,
        mx.float16: torch.float16,
        mx.bfloat16: torch.bfloat16,
        mx.int32: torch.int32,
        mx.int64: torch.int64,
        mx.int16: torch.int16,
        mx.int8: torch.int8,
        mx.uint8: torch.uint8,
        mx.bool_: torch.bool,
    }
    if mlx_dtype not in dtype_map:
        raise ValueError(f"Unsupported dtype: {mlx_dtype}")
    return dtype_map[mlx_dtype]
