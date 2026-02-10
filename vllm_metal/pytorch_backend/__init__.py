# SPDX-License-Identifier: Apache-2.0
"""PyTorch backend for model loading and tensor interop."""

from vllm_metal.pytorch_backend.tensor_bridge import (
    get_torch_device,
    mlx_to_torch,
    torch_to_mlx,
)

__all__ = [
    "mlx_to_torch",
    "torch_to_mlx",
    "get_torch_device",
]
