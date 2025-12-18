# SPDX-License-Identifier: Apache-2.0
"""MLX device management for vLLM Metal backend."""

import mlx.core as mx

from vllm_metal.envs import VLLM_MLX_DEVICE


def get_mlx_device() -> mx.Device:
    """Get the MLX device to use for computation.

    Returns:
        The MLX device (GPU or CPU) based on configuration.
    """
    if VLLM_MLX_DEVICE == "cpu":
        return mx.cpu
    return mx.gpu


def set_mlx_device(device: str = "gpu") -> None:
    """Set the default MLX device.

    Args:
        device: Either "gpu" or "cpu".
    """
    if device == "cpu":
        mx.set_default_device(mx.cpu)
    else:
        mx.set_default_device(mx.gpu)


def mlx_synchronize() -> None:
    """Synchronize MLX operations.

    Forces evaluation of all pending lazy operations.
    """
    mx.eval([])


def mlx_clear_cache() -> None:
    """Clear MLX memory cache."""
    try:
        mx.metal.clear_cache()
    except AttributeError:
        # Older MLX versions may not have this
        pass


def get_mlx_memory_info() -> dict:
    """Get MLX memory information.

    Returns:
        Dictionary with memory statistics.
    """
    try:
        return {
            "peak_memory": mx.get_peak_memory(),
            "active_memory": mx.get_active_memory(),
            "cache_memory": mx.get_cache_memory(),
        }
    except AttributeError:
        return {
            "peak_memory": 0,
            "active_memory": 0,
            "cache_memory": 0,
        }


def reset_mlx_peak_memory() -> None:
    """Reset MLX peak memory tracking."""
    try:
        mx.metal.reset_peak_memory()
    except AttributeError:
        pass
