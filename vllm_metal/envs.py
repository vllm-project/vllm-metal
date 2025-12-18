# SPDX-License-Identifier: Apache-2.0
"""Environment variables for vLLM Metal backend."""

import os

# Memory fraction of unified memory to use for GPU operations.
# Apple Silicon uses unified memory, so this controls the total
# allocation available to the KV cache and model weights.
VLLM_METAL_MEMORY_FRACTION: float = float(
    os.environ.get("VLLM_METAL_MEMORY_FRACTION", "0.9")
)

# Whether to use MLX for compute operations.
# Set to "0" to fall back to PyTorch MPS (much slower).
VLLM_METAL_USE_MLX: bool = os.environ.get("VLLM_METAL_USE_MLX", "1") == "1"

# MLX default device: "gpu" or "cpu"
# GPU is the default and recommended for Apple Silicon.
VLLM_MLX_DEVICE: str = os.environ.get("VLLM_MLX_DEVICE", "gpu")

# Enable MLX lazy evaluation optimization.
# When enabled, MLX builds computation graphs and executes them lazily.
VLLM_MLX_LAZY_EVAL: bool = os.environ.get("VLLM_MLX_LAZY_EVAL", "1") == "1"

# Block size for KV cache.
# Smaller blocks allow finer-grained memory management but increase overhead.
VLLM_METAL_BLOCK_SIZE: int = int(os.environ.get("VLLM_METAL_BLOCK_SIZE", "16"))

# Enable debug logging for Metal backend.
VLLM_METAL_DEBUG: bool = os.environ.get("VLLM_METAL_DEBUG", "0") == "1"

# Maximum number of sequences to batch together.
# Lower values reduce memory but may decrease throughput.
VLLM_METAL_MAX_BATCH_SIZE: int = int(
    os.environ.get("VLLM_METAL_MAX_BATCH_SIZE", "256")
)
