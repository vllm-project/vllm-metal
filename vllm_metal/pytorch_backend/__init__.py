# SPDX-License-Identifier: Apache-2.0
"""PyTorch backend with optional, lazily imported MLX interoperability."""

__all__ = [
    "mlx_to_torch",
    "torch_to_mlx",
    "get_torch_device",
]


def __getattr__(name):
    # Importing the MPS sampler itself only needs Torch and vLLM. Preserve the
    # public bridge exports without loading MLX for every PyTorch submodule.
    if name in __all__:
        from importlib import import_module

        return getattr(import_module(f"{__name__}.tensor_bridge"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
