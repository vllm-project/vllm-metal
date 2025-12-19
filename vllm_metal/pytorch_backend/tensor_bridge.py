# SPDX-License-Identifier: Apache-2.0
"""Tensor bridge between MLX and PyTorch.

Provides zero-copy conversion when possible using Apple Silicon's unified memory.
"""

from typing import Literal

import mlx.core as mx
import numpy as np
import torch

# MLX to PyTorch dtype mapping
MLX_TO_TORCH_DTYPE: dict[mx.Dtype, torch.dtype] = {
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

# PyTorch to MLX dtype mapping
TORCH_TO_MLX_DTYPE: dict[torch.dtype, mx.Dtype] = {
    v: k for k, v in MLX_TO_TORCH_DTYPE.items()
}


def get_torch_device() -> torch.device:
    """Get the PyTorch device for Metal/MPS.

    Returns:
        torch.device for MPS if available, else CPU
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def torch_to_mlx(tensor: torch.Tensor) -> mx.array:
    """Convert PyTorch tensor to MLX array.

    Uses numpy as an intermediate to enable zero-copy on unified memory.

    Args:
        tensor: PyTorch tensor (can be on any device)

    Returns:
        MLX array with the same data
    """
    # Move to CPU if on MPS for numpy conversion
    if tensor.device.type == "mps":
        tensor = tensor.cpu()

    # Convert via numpy for zero-copy on unified memory
    np_array = tensor.detach().numpy()
    return mx.array(np_array)


def mlx_to_torch(
    array: mx.array,
    device: torch.device | Literal["mps", "cpu"] | None = None,
) -> torch.Tensor:
    """Convert MLX array to PyTorch tensor.

    Uses numpy as an intermediate to enable zero-copy on unified memory.

    Args:
        array: MLX array
        device: Target PyTorch device (default: MPS if available)

    Returns:
        PyTorch tensor with the same data
    """
    if device is None:
        device = get_torch_device()
    elif isinstance(device, str):
        device = torch.device(device)

    # Evaluate any pending MLX operations
    mx.eval(array)

    # Convert via numpy
    np_array = np.array(array)
    tensor = torch.from_numpy(np_array)

    # Move to target device
    if device.type != "cpu":
        tensor = tensor.to(device)

    return tensor


def sync_mlx() -> None:
    """Synchronize MLX operations.

    Call this before converting MLX arrays to ensure all operations complete.
    """
    mx.eval([])


def sync_torch() -> None:
    """Synchronize PyTorch MPS operations.

    Call this before converting PyTorch tensors to ensure all operations complete.
    """
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
