# SPDX-License-Identifier: Apache-2.0
"""MLX backend for vLLM Metal - primary compute backend."""

from vllm_metal.mlx_backend.cache import KVCache, PagedKVCache
from vllm_metal.mlx_backend.ops import (
    attention,
    rms_norm,
    rotary_embedding,
    silu_and_mul,
)

__all__ = [
    "attention",
    "rms_norm",
    "rotary_embedding",
    "silu_and_mul",
    "KVCache",
    "PagedKVCache",
]
