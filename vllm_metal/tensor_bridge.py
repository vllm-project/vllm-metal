# SPDX-License-Identifier: Apache-2.0
"""Share tensor storage between MLX and PyTorch using DLPack."""

from typing import Literal, cast

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


def torch_to_mlx(tensor: torch.Tensor, *, copy: bool = False) -> mx.array:
    """Share a detached tensor's storage with MLX, or copy it with ``copy=True``.

    Raises if the tensor is unsupported. MPS writes are synchronized before
    MLX can read them. When shared, the source must not be mutated while MLX
    is using the data. Sharing also runs the other way: once a lazy graph
    holds the last reference to a shared array, MLX may donate its buffer to
    an op's output and write into the source tensor. ``copy=True`` gives MLX
    its own buffer, for tensors the caller keeps using or does not own.
    """
    if tensor.device.type == "mps":
        torch.mps.synchronize()
    return mx.from_dlpack(tensor.detach(), copy=copy)


def mlx_to_torch(
    array: mx.array,
    device: torch.device | Literal["mps", "cpu"] | None = None,
) -> torch.Tensor:
    """Share evaluated MLX storage with Torch on CPU or MPS.

    Views retain their strides; negative strides are unsupported. Torch writes
    affect the shared MLX data, including aliases in broadcast views. Synchronize
    MPS writes before reading that data from MLX again.
    """
    if device is None:
        device = get_torch_device()
    else:
        device = torch.device(device)
    if device.type not in ("cpu", "mps"):
        raise ValueError("The tensor bridge supports only CPU and MPS exports")

    # PyTorch aborts instead of raising when it imports negative strides.
    strides = cast(tuple[int, ...], memoryview(array).strides)
    if any(stride < 0 for stride in strides):
        raise ValueError("Cannot share an MLX array with negative strides with Torch")

    # Request CPU storage explicitly: importing as MPS and then calling .cpu()
    # would copy, even though both frameworks can access the same Metal buffer.
    dl_device = (8, 0) if device.type == "mps" else (1, 0)
    return torch.from_dlpack(array.__dlpack__(dl_device=dl_device, copy=False))
