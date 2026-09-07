# SPDX-License-Identifier: Apache-2.0
"""PyTorch backend for model loading and tensor interop."""

__all__ = [
    "mlx_to_torch",
    "torch_to_mlx",
    "get_torch_device",
]


def __getattr__(name):
    if name in __all__:
        from vllm_metal.pytorch_backend import tensor_bridge

        return getattr(tensor_bridge, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
