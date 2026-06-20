# SPDX-License-Identifier: Apache-2.0
"""Experimental GGUF support for vllm-metal (MLX-native quantized execution)."""

from vllm_metal.gguf.adapter import GGUFLoadError
from vllm_metal.gguf.loader import GGUFModelLoader

__all__ = ["GGUFLoadError", "GGUFModelLoader"]
