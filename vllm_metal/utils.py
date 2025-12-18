# SPDX-License-Identifier: Apache-2.0
"""Utility functions for vLLM Metal backend."""

import platform
import subprocess
from functools import lru_cache
from typing import Any


@lru_cache(maxsize=1)
def is_apple_silicon() -> bool:
    """Check if running on Apple Silicon.

    Returns:
        True if running on Apple Silicon (M1/M2/M3/M4/M5), False otherwise.
    """
    if platform.system() != "Darwin":
        return False

    # Check for ARM64 architecture (Apple Silicon)
    machine = platform.machine()
    return machine in ("arm64", "aarch64")


@lru_cache(maxsize=1)
def get_apple_chip_name() -> str:
    """Get the Apple chip name (e.g., 'Apple M1 Pro').

    Returns:
        The chip name string, or 'Unknown Apple Silicon' if not determinable.
    """
    if not is_apple_silicon():
        return "Not Apple Silicon"

    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "Unknown Apple Silicon"


@lru_cache(maxsize=1)
def get_metal_device_info() -> dict[str, Any]:
    """Get Metal device information.

    Returns:
        Dictionary containing:
        - name: Device name (chip name)
        - metal_available: Whether Metal is available
        - total_memory: Total unified memory in bytes
        - max_threads_per_threadgroup: Maximum threads per threadgroup
    """
    import psutil

    info: dict[str, Any] = {
        "name": get_apple_chip_name(),
        "metal_available": False,
        "total_memory": 0,
        "max_threads_per_threadgroup": 1024,  # Default for Apple Silicon
    }

    if not is_apple_silicon():
        return info

    # Check if Metal is available via PyTorch
    try:
        import torch

        info["metal_available"] = torch.backends.mps.is_available()
    except ImportError:
        pass

    # Also check MLX availability
    try:
        import mlx.core as mx

        info["metal_available"] = True
        info["mlx_available"] = True
        info["mlx_default_device"] = str(mx.default_device())
    except ImportError:
        info["mlx_available"] = False

    # Get total system memory (unified memory on Apple Silicon)
    info["total_memory"] = psutil.virtual_memory().total

    return info


def get_metal_memory_info() -> tuple[int, int]:
    """Get Metal memory usage information.

    On Apple Silicon, GPU uses unified memory shared with CPU.
    This returns approximate allocation based on MLX/PyTorch tracking.

    Returns:
        Tuple of (allocated_bytes, total_bytes).
    """
    import psutil

    from vllm_metal.envs import VLLM_METAL_MEMORY_FRACTION

    total = psutil.virtual_memory().total
    available_total = int(total * VLLM_METAL_MEMORY_FRACTION)

    # Try to get MLX memory stats
    try:
        import mlx.core as mx

        # MLX tracks peak memory usage
        allocated = mx.get_peak_memory()
        return allocated, available_total
    except (ImportError, AttributeError):
        pass

    # Fall back to PyTorch MPS if available
    try:
        import torch

        if torch.backends.mps.is_available():
            # MPS doesn't have great memory tracking, estimate from system
            allocated = torch.mps.current_allocated_memory()
            return allocated, available_total
    except (ImportError, AttributeError):
        pass

    # No tracking available, return 0 allocated
    return 0, available_total


def metal_empty_cache() -> None:
    """Clear Metal memory caches."""
    # Clear MLX cache
    try:
        import mlx.core as mx

        mx.metal.clear_cache()
    except (ImportError, AttributeError):
        pass

    # Clear PyTorch MPS cache
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except (ImportError, AttributeError):
        pass


def metal_synchronize() -> None:
    """Synchronize Metal operations."""
    # Synchronize MLX
    try:
        import mlx.core as mx

        mx.eval([])  # Force evaluation of pending operations
    except (ImportError, AttributeError):
        pass

    # Synchronize PyTorch MPS
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.synchronize()
    except (ImportError, AttributeError):
        pass


def check_metal_availability() -> tuple[bool, str | None]:
    """Check if Metal backend is available and functional.

    Returns:
        Tuple of (is_available, error_message).
        If available, error_message is None.
    """
    if not is_apple_silicon():
        return False, "Not running on Apple Silicon"

    # Check MLX availability (required)
    try:
        import mlx.core as mx

        # Try a simple operation
        x = mx.array([1.0, 2.0, 3.0])
        _ = mx.sum(x)
    except ImportError:
        return False, "MLX is not installed. Install with: pip install mlx mlx-lm"
    except Exception as e:
        return False, f"MLX error: {e}"

    # Check PyTorch MPS availability (optional but recommended)
    try:
        import torch

        if not torch.backends.mps.is_available():
            # Warning but not an error - MLX is primary backend
            pass
    except ImportError:
        pass

    return True, None


@lru_cache(maxsize=1)
def get_supported_dtypes() -> set[str]:
    """Get supported data types for Metal backend.

    Returns:
        Set of supported dtype strings.
    """
    return {
        "float32",
        "float16",
        "bfloat16",  # Supported on M1+ chips
        "int32",
        "int64",
        "int16",
        "int8",
        "uint8",
        "bool",
    }
