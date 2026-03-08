# SPDX-License-Identifier: Apache-2.0
"""MLX-backed paged KV cache for native Metal paged attention.

Stores per-layer key/value caches as MLX arrays in the layout expected by
``reshape_and_cache`` and ``paged_attention_v1``:

- key_cache:   [num_blocks, block_size, num_kv_heads, head_dim]
- value_cache: [num_blocks, block_size, num_kv_heads, head_dim]

Both caches use the same token-contiguous layout where each token's
KV vector is stored contiguously.  This simplifies indexing and is
compatible with variable-length / FlashInfer-style paged attention.

Block allocation is managed externally by the scheduler's KV cache manager.
"""

from __future__ import annotations

import mlx.core as mx


class MetalPagedKVCache:
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

        if dtype not in (mx.float16, mx.bfloat16, mx.float32):
            raise ValueError(f"Unsupported dtype for paged KV cache: {dtype}")

        # Per-layer caches — unified layout for both K and V
        self.key_caches: list[mx.array] = []
        self.value_caches: list[mx.array] = []
        for _ in range(num_layers):
            self.key_caches.append(
                mx.zeros(
                    (num_blocks, block_size, num_kv_heads, head_dim),
                    dtype=dtype,
                )
            )
            self.value_caches.append(
                mx.zeros(
                    (num_blocks, block_size, num_kv_heads, head_dim),
                    dtype=dtype,
                )
            )

        # Force allocation so Metal buffers exist before kernel dispatch
        mx.eval(*self.key_caches, *self.value_caches)
