# SPDX-License-Identifier: Apache-2.0
"""Tensor bridge between MLX and PyTorch using DLPack."""

from typing import Literal

import mlx.core as mx
import torch

# These mappings are also used by cache allocation and model loading.
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
TORCH_TO_MLX_DTYPE: dict[torch.dtype, mx.Dtype] = {
    v: k for k, v in MLX_TO_TORCH_DTYPE.items()
}


def get_torch_device() -> torch.device:
    """Get the PyTorch device for Metal/MPS, or CPU if MPS is unavailable."""
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def torch_to_mlx(tensor: torch.Tensor) -> mx.array:
    """Import a detached tensor, sharing its storage when possible.

    MPS writes are synchronized before MLX can read them. The source must not
    be mutated while MLX is using the shared data.
    """
    tensor = tensor.detach()
    if tensor.device.type == "mps":
        torch.mps.synchronize()
    elif tensor.device.type != "cpu":
        tensor = tensor.cpu()
    return mx.from_dlpack(tensor)


def mlx_to_torch(
    array: mx.array,
    device: torch.device | Literal["mps", "cpu"] | None = None,
) -> torch.Tensor:
    """Export an evaluated array, sharing storage on CPU and MPS.

    Reversed and broadcast views are materialized for a writable Torch layout.
    Other views retain their strides. Torch writes affect the shared MLX data;
    synchronize MPS writes before reading that data from MLX again.
    """
    if device is None:
        device = get_torch_device()
    else:
        device = torch.device(device)

    # PyTorch aborts on negative strides and cannot update broadcast views in place.
    strides = memoryview(array).strides
    if any(
        stride < 0 or (stride == 0 and size > 1)
        for size, stride in zip(array.shape, strides, strict=True)
    ):
        array = mx.contiguous(array)

    # Request CPU storage explicitly: importing as MPS and then calling .cpu()
    # would copy, even though both frameworks can access the same Metal buffer.
    dl_device = (8, 0) if device.type == "mps" else (1, 0)
    tensor = torch.from_dlpack(array.__dlpack__(dl_device=dl_device))
    if device.type not in ("cpu", "mps"):
        tensor = tensor.to(device)
    return tensor
