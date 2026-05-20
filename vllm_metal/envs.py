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
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    VLLM_METAL_MEMORY_FRACTION: str = "auto"
    VLLM_METAL_USE_MLX: bool = True
    VLLM_MLX_DEVICE: str = "gpu"
    VLLM_METAL_DEBUG: bool = False
    VLLM_METAL_USE_PAGED_ATTENTION: bool = True
    VLLM_METAL_KV_SHARING_FAST_PREFILL: bool = True
    VLLM_METAL_MULTIMODAL_MODE: str = "auto"
    VLLM_METAL_PREFIX_CACHE: bool = False
    VLLM_METAL_PREFIX_CACHE_FRACTION: str = ""
    VLLM_METAL_MODELSCOPE_CACHE: str | None = None
    VLLM_METAL_GDN_LAZY_KERNELS: bool = True
    VLLM_METAL_MLA_KERNEL: bool = False

environment_variables: dict[str, Callable[[], Any]] = {
    # Fraction of unified memory to use.  "auto" (the default) means the
    # plugin calculates the minimal amount needed at startup.
    # Returns the raw string; config.py handles "auto" → sentinel conversion.
    "VLLM_METAL_MEMORY_FRACTION": lambda: os.getenv(
        "VLLM_METAL_MEMORY_FRACTION", "auto"
    ),
    # Whether to use MLX as the compute backend (default True).
    "VLLM_METAL_USE_MLX": lambda: os.getenv("VLLM_METAL_USE_MLX", "1") == "1",
    # MLX device type: "gpu" (default) or "cpu".
    "VLLM_MLX_DEVICE": lambda: os.getenv("VLLM_MLX_DEVICE", "gpu"),
    # Enable verbose debug logging (default False).
    "VLLM_METAL_DEBUG": lambda: os.getenv("VLLM_METAL_DEBUG", "0") == "1",
    # Use native Metal paged attention (default True).
    "VLLM_METAL_USE_PAGED_ATTENTION": lambda: (
        os.getenv("VLLM_METAL_USE_PAGED_ATTENTION", "1") == "1"
    ),
    # Experimental YOCO/KV-sharing fast prefill. Default on for eligible
    # paged-attention models.
    "VLLM_METAL_KV_SHARING_FAST_PREFILL": lambda: (
        os.getenv("VLLM_METAL_KV_SHARING_FAST_PREFILL", "1") == "1"
    ),
    # Multimodal serving mode:
    # - "auto": known-incompatible multimodal checkpoints fall back to the
    #   text-only compatibility path.
    # - "text-only-compat": force known-safe multimodal checkpoints onto the
    #   text-only compatibility path.
    # - "multimodal-native": keep native multimodal loading enabled.
    "VLLM_METAL_MULTIMODAL_MODE": lambda: os.getenv(
        "VLLM_METAL_MULTIMODAL_MODE", "auto"
    ),
    # Enable content-hash prefix caching (presence-based: set to any
    # value to enable, unset to disable).
    "VLLM_METAL_PREFIX_CACHE": lambda: "VLLM_METAL_PREFIX_CACHE" in os.environ,
    # Fraction of MLX working set for the prefix cache (raw string;
    # the consumer in model_runner.py validates and applies a default).
    "VLLM_METAL_PREFIX_CACHE_FRACTION": lambda: os.getenv(
        "VLLM_METAL_PREFIX_CACHE_FRACTION", ""
    ),
    # Custom cache directory for ModelScope downloads (None if unset).
    "VLLM_METAL_MODELSCOPE_CACHE": lambda: os.getenv("VLLM_METAL_MODELSCOPE_CACHE"),
    # Enable lazy GDN kernels by default.
    # Set to "0" to force the eager conv / C++ recurrent fallback path.
    "VLLM_METAL_GDN_LAZY_KERNELS": lambda: (
        os.getenv("VLLM_METAL_GDN_LAZY_KERNELS", "1") == "1"
    ),
    # Experimental MLA Metal decode kernel (RFC #360). Off by default —
    # the MLA wrapper uses the MLX SDPA per-request slow path unless
    # this opt-in is set. Set to "1" to route absorbed-MLA decode
    # through the single-pass Metal kernel when the workload matches
    # the kernel's instantiated specialization (kv_lora_rank=512,
    # qk_rope_head_dim=64, block_size ∈ {16, 32}, fp16/bf16,
    # decode-only).
    "VLLM_METAL_MLA_KERNEL": lambda: os.getenv("VLLM_METAL_MLA_KERNEL", "0") == "1",
}


def __getattr__(name: str) -> Any:
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    # Mirrors vllm/envs.py; enables tab-completion and introspection.
    return list(environment_variables.keys())
