# SPDX-License-Identifier: Apache-2.0
"""Environment variable definitions for the vLLM Metal plugin.

This module is the single source of truth for all ``VLLM_METAL_*`` (and
``VLLM_MLX_*``) environment variables.  It mirrors the lazy-evaluation
pattern used by ``vllm/envs.py``: each variable is read from
``os.environ`` on access via ``__getattr__``, so values are never stale
and ``monkeypatch.setenv`` works in tests without extra resets.

During plugin registration (``vllm_metal._register``), the
``environment_variables`` dict is merged into
``vllm.envs.environment_variables`` so that ``validate_environ()``
recognises our variables and does not emit spurious "Unknown vLLM
environment variable" warnings.
"""

import os
from collections.abc import Callable
from typing import Any

# Sentinel value indicating auto memory calculation (matches config.py).
_AUTO_MEMORY_FRACTION: float = -1.0


def _parse_memory_fraction() -> float:
    """Parse VLLM_METAL_MEMORY_FRACTION, returning -1.0 for 'auto'."""
    raw = os.getenv("VLLM_METAL_MEMORY_FRACTION", "auto")
    return _AUTO_MEMORY_FRACTION if raw.lower() == "auto" else float(raw)


environment_variables: dict[str, Callable[[], Any]] = {
    # Fraction of unified memory to use.  "auto" (the default) means the
    # plugin calculates the minimal amount needed at startup.
    "VLLM_METAL_MEMORY_FRACTION":
        _parse_memory_fraction,
    # Whether to use MLX as the compute backend (default True).
    "VLLM_METAL_USE_MLX":
        lambda: os.getenv("VLLM_METAL_USE_MLX", "1") == "1",
    # MLX device type: "gpu" (default) or "cpu".
    "VLLM_MLX_DEVICE":
        lambda: os.getenv("VLLM_MLX_DEVICE", "gpu"),
    # Tokens per KV-cache block (default 16).
    "VLLM_METAL_BLOCK_SIZE":
        lambda: int(os.getenv("VLLM_METAL_BLOCK_SIZE", "16")),
    # Enable verbose debug logging (default False).
    "VLLM_METAL_DEBUG":
        lambda: os.getenv("VLLM_METAL_DEBUG", "0") == "1",
    # Use native Metal paged attention (default True).
    "VLLM_METAL_USE_PAGED_ATTENTION":
        lambda: os.getenv("VLLM_METAL_USE_PAGED_ATTENTION", "1") == "1",
    # Enable content-hash prefix caching (presence-based: set to any
    # value to enable, unset to disable).
    "VLLM_METAL_PREFIX_CACHE":
        lambda: "VLLM_METAL_PREFIX_CACHE" in os.environ,
    # Fraction of MLX working set for the prefix cache (raw string;
    # the consumer in model_runner.py validates and applies a default).
    "VLLM_METAL_PREFIX_CACHE_FRACTION":
        lambda: os.getenv("VLLM_METAL_PREFIX_CACHE_FRACTION", ""),
    # Custom cache directory for ModelScope downloads (None if unset).
    "VLLM_METAL_MODELSCOPE_CACHE":
        lambda: os.getenv("VLLM_METAL_MODELSCOPE_CACHE"),
}


def __getattr__(name: str) -> Any:
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return list(environment_variables.keys())


def is_set(name: str) -> bool:
    """Check if an environment variable is explicitly set in os.environ."""
    if name in environment_variables:
        return name in os.environ
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
