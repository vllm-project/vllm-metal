# SPDX-License-Identifier: Apache-2.0
"""Experimental GGUF support for vllm-metal (MLX-native quantized execution).

Re-exports are lazy: ``vllm_metal.gguf.vllm_integration`` is imported by every
vLLM run through the ``vllm.general_plugins`` entry point, and a default
install (no ``gguf`` extra) must not pull the loader's optional ``gguf``
dependency on that path.
"""

from typing import Any

__all__ = ["GGUFLoadError", "GGUFModelLoader"]


def __getattr__(name: str) -> Any:
    if name == "GGUFLoadError":
        from vllm_metal.gguf.adapter import GGUFLoadError

        return GGUFLoadError
    if name == "GGUFModelLoader":
        from vllm_metal.gguf.loader import GGUFModelLoader

        return GGUFModelLoader
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
