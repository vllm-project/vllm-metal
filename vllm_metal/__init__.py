# SPDX-License-Identifier: Apache-2.0
"""vLLM Metal Plugin - High-performance LLM inference on Apple Silicon.

This plugin enables vLLM to run on Apple Silicon Macs using MLX as the
primary compute backend, with PyTorch for model loading and interoperability.
"""

import logging
import os
import sys

__version__ = "0.1.0"

logger = logging.getLogger(__name__)


def _apply_macos_defaults() -> None:
    """Apply safe defaults for macOS when using the Metal plugin.

    vLLM's v1 engine launches a worker process. When the start method is `fork`,
    macOS can crash the child process if the parent has imported libraries that
    touched the Objective-C runtime (commonly surfaced as
    `objc_initializeAfterForkError`).

    Defaulting to `spawn` avoids forking a partially-initialized runtime.
    """
    if sys.platform != "darwin":
        return
    if os.environ.get("VLLM_WORKER_MULTIPROC_METHOD") is not None:
        return

    # macOS fork-safety:
    # `fork()` with an initialized Objective-C runtime is unsafe and can crash in
    # the child process (commonly observed via `objc_initializeAfterForkError`).
    # Using `spawn` starts a fresh interpreter and avoids inheriting this state.
    # See: https://www.sealiesoftware.com/blog/archive/2017/6/5/Objective-C_and_fork_in_macOS_1013.html
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    logger.debug(
        "macOS detected + Metal plugin active: defaulting VLLM_WORKER_MULTIPROC_METHOD "
        "to 'spawn' to avoid Objective-C runtime fork-safety crashes. "
        "Set VLLM_WORKER_MULTIPROC_METHOD explicitly to override."
    )


# Lazy imports to avoid loading vLLM dependencies when just importing the Rust extension
def __getattr__(name):
    """Lazy import module components."""
    if name == "MetalConfig":
        from vllm_metal.config import MetalConfig

        return MetalConfig
    elif name == "get_config":
        from vllm_metal.config import get_config

        return get_config
    elif name == "reset_config":
        from vllm_metal.config import reset_config

        return reset_config
    elif name == "MetalPlatform":
        from vllm_metal.platform import MetalPlatform

        return MetalPlatform
    elif name == "register":
        return _register
    elif name == "register_ops":
        return _register_ops
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "MetalConfig",
    "MetalPlatform",
    "get_config",
    "reset_config",
    "register",
    "register_ops",
]


def _register() -> str | None:
    """Register the Metal platform plugin with vLLM.

    This is the entry point for vLLM's platform plugin system.

    Returns:
        Fully qualified class name if platform is available, None otherwise
    """
    _apply_macos_defaults()

    from vllm_metal.platform import MetalPlatform

    if MetalPlatform.is_available():
        return "vllm_metal.platform.MetalPlatform"
    return None


def _register_ops() -> None:
    """Register Metal operations with vLLM.

    This is the entry point for vLLM's general plugin system.
    Currently a no-op as operations are handled internally.
    """
    # Operations are registered implicitly through the MLX backend
    pass
