# SPDX-License-Identifier: Apache-2.0
"""
vLLM Metal Backend - Hardware plugin for Apple Silicon

This module provides Metal backend support for vLLM using MLX as the
primary compute backend, enabling high-performance LLM inference on
Apple Silicon devices (M1/M2/M3/M4/M5 series).

The plugin unifies MLX and PyTorch under a single lowering path:
- MLX: Primary compute backend for GPU operations (attention, norms, etc.)
- PyTorch: Model loading, weight conversion, tensor interface compatibility
"""

import os

# Metal requires V2 model runner which properly handles prefill_token_ids
# in the scheduler. This must be set BEFORE vllm.envs is imported anywhere.
os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "1")

# Metal/MLX contexts cannot survive fork(), so use spawn multiprocessing.
# This must be set before any multiprocessing is used.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

__version__ = "0.1.0"


def register() -> str | None:
    """Register the Metal platform with vLLM.

    This function is called by vLLM's plugin system via the entry point
    defined in pyproject.toml. It returns the fully qualified class name
    of MetalPlatform if running on Apple Silicon, or None otherwise.

    Returns:
        The fully qualified class name of MetalPlatform, or None if
        not running on Apple Silicon.
    """
    from vllm_metal.utils import is_apple_silicon

    if not is_apple_silicon():
        return None

    return "vllm_metal.platform.MetalPlatform"


def register_ops() -> None:
    """Register Metal-specific operations with vLLM.

    This function is called by vLLM's plugin system to register
    MLX-based operation implementations. Operations are registered
    lazily to avoid importing MLX until actually needed.
    """
    from vllm_metal.ops import register_metal_ops

    register_metal_ops()
