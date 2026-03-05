# SPDX-License-Identifier: Apache-2.0
"""MLX-backed paged KV cache for native Metal paged attention.

Stores per-layer key/value caches as MLX arrays in the layout expected by
``reshape_and_cache`` and ``paged_attention_v1``:

- key_cache:   [num_blocks, num_kv_heads, head_dim // x, block_size, x]
               where x = 16 // element_size (8 for float16)
- value_cache: [num_blocks, num_kv_heads, head_dim, block_size]

Block allocation is managed externally by the scheduler's KV cache manager.
"""

from __future__ import annotations

import mlx.core as mx

# mx.Dtype → element size in bytes
_DTYPE_SIZE = {
    mx.float16: 2,
    mx.bfloat16: 2,
    mx.float32: 4,
}


class MPSPagedKVCache:
    """Per-layer MLX arrays for native Metal paged attention."""

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        num_blocks: int,
        block_size: int,
        dtype: mx.Dtype = mx.float16,
    ) -> None:
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.dtype = dtype

        element_size = _DTYPE_SIZE.get(dtype)
        if element_size is None:
            raise ValueError(f"Unsupported dtype for paged KV cache: {dtype}")
        self.x = 16 // element_size  # 8 for float16, 4 for float32

        if head_dim % self.x != 0:
            raise ValueError(
                f"head_dim ({head_dim}) must be divisible by x ({self.x}) "
                f"for the 5-D key cache layout [num_blocks, num_kv_heads, "
                f"head_dim // x, block_size, x]"
            )

        # Per-layer caches
        self.key_caches: list[mx.array] = []
        self.value_caches: list[mx.array] = []
        for _ in range(num_layers):
            self.key_caches.append(
                mx.zeros(
                    (num_blocks, num_kv_heads, head_dim // self.x, block_size, self.x),
                    dtype=dtype,
                )
            )
            self.value_caches.append(
                mx.zeros(
                    (num_blocks, num_kv_heads, head_dim, block_size),
                    dtype=dtype,
                )
            )

        # Force allocation so Metal buffers exist before kernel dispatch
        mx.eval(*self.key_caches, *self.value_caches)
