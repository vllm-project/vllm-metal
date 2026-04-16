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
    """Per-layer MLX arrays for native Metal paged attention.

    Supports heterogeneous per-layer shapes: when ``kv_heads_per_layer``
    and ``head_dim_per_layer`` are provided, each cache layer is allocated
    with its own ``(num_kv_heads, head_dim)`` pair.  When omitted, all
    layers share the scalar ``num_kv_heads`` / ``head_dim`` (backward
    compat for MLA, Hybrid, and uniform MHA models).
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        num_blocks: int,
        block_size: int,
        dtype: mx.Dtype = mx.float16,
        *,
        kv_heads_per_layer: list[int] | None = None,
        head_dim_per_layer: list[int] | None = None,
    ) -> None:
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.dtype = dtype

        self.kv_heads_per_layer = kv_heads_per_layer or [num_kv_heads] * num_layers
        self.head_dim_per_layer = head_dim_per_layer or [head_dim] * num_layers

        # Canonical scalars for warm_up_paged_cache (layer-0 shape)
        self.num_kv_heads = self.kv_heads_per_layer[0]
        self.head_dim = self.head_dim_per_layer[0]

        if len(self.kv_heads_per_layer) != num_layers:
            raise ValueError(
                f"kv_heads_per_layer length {len(self.kv_heads_per_layer)} "
                f"!= num_layers {num_layers}"
            )
        if len(self.head_dim_per_layer) != num_layers:
            raise ValueError(
                f"head_dim_per_layer length {len(self.head_dim_per_layer)} "
                f"!= num_layers {num_layers}"
            )

        if dtype not in (mx.float16, mx.bfloat16, mx.float32):
            raise ValueError(f"Unsupported dtype for paged KV cache: {dtype}")

        # Per-layer caches — each layer sized by its own (kv_heads, head_dim)
        self.key_caches: list[mx.array] = []
        self.value_caches: list[mx.array] = []
        for i in range(num_layers):
            shape = (
                num_blocks,
                block_size,
                self.kv_heads_per_layer[i],
                self.head_dim_per_layer[i],
            )
            self.key_caches.append(mx.zeros(shape, dtype=dtype))
            self.value_caches.append(mx.zeros(shape, dtype=dtype))

        # Force allocation so Metal buffers exist before kernel dispatch
        mx.eval(*self.key_caches, *self.value_caches)
