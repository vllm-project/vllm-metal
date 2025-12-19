# SPDX-License-Identifier: Apache-2.0
"""vLLM Metal Plugin - High-performance LLM inference on Apple Silicon.

This plugin enables vLLM to run on Apple Silicon Macs using MLX as the
primary compute backend, with PyTorch for model loading and interoperability.
"""

from vllm_metal.config import MetalConfig, get_config, reset_config
from vllm_metal.model_runner import MetalModelRunner
from vllm_metal.platform import MetalPlatform
from vllm_metal.worker import MetalWorker

__version__ = "0.1.0"

__all__ = [
    "MetalConfig",
    "MetalPlatform",
    "MetalWorker",
    "MetalModelRunner",
    "get_config",
    "reset_config",
    "register",
    "register_ops",
]


def register() -> str | None:
    """Register the Metal platform plugin with vLLM.

    This is the entry point for vLLM's platform plugin system.

    Returns:
        Fully qualified class name if platform is available, None otherwise
    """
    if MetalPlatform.is_available():
        return "vllm_metal.platform.MetalPlatform"
    return None


def register_ops() -> None:
    """Register Metal operations with vLLM.

    This is the entry point for vLLM's general plugin system.
    Currently a no-op as operations are handled internally.
    """
    # Operations are registered implicitly through the MLX backend
    pass
