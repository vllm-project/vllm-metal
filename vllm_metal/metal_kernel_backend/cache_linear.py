# SPDX-License-Identifier: Apache-2.0
"""Block-managed recurrent state cache for linear attention (GDN) layers.

Stores per-layer recurrent state in the same block-indexed layout as
``MetalPagedKVCache`` so that the vLLM scheduler's block management
(allocation, preemption, release) works uniformly across attention types.

Layout per linear layer:
  - conv_state:      [num_blocks, conv_kernel - 1, conv_dim]
  - recurrent_state: [num_blocks, num_v_heads, value_head_dim, key_head_dim]

Each request occupies one block (state is fixed-size, unlike SDPA KV which
may span multiple blocks).  Block allocation is managed externally by the
vLLM scheduler — this class only stores the tensors.
"""

from __future__ import annotations

import mlx.core as mx


class LinearAttentionCache:
    """Per-layer MLX arrays for GDN linear attention recurrent state."""

    def __init__(
        self,
        *,
        num_layers: int,
        num_blocks: int,
        conv_kernel_dim: int,
        conv_dim: int,
        num_v_heads: int,
        value_head_dim: int,
        key_head_dim: int,
        dtype: mx.Dtype = mx.float16,
    ) -> None:
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.conv_kernel_dim = conv_kernel_dim
        self.conv_dim = conv_dim
        self.num_v_heads = num_v_heads
        self.value_head_dim = value_head_dim
        self.key_head_dim = key_head_dim
        self.dtype = dtype

        conv_state_shape = (num_blocks, conv_kernel_dim - 1, conv_dim)
        recurrent_shape = (num_blocks, num_v_heads, value_head_dim, key_head_dim)

        self.conv_states: list[mx.array] = []
        self.recurrent_states: list[mx.array] = []
        for _ in range(num_layers):
            self.conv_states.append(mx.zeros(conv_state_shape, dtype=dtype))
            self.recurrent_states.append(mx.zeros(recurrent_shape, dtype=dtype))

        mx.eval(*self.conv_states, *self.recurrent_states)

    def bytes_per_block(self) -> int:
        """Total bytes for one block across all linear layers."""
        dtype_size = self.dtype.size
        conv_bytes = (self.conv_kernel_dim - 1) * self.conv_dim * dtype_size
        recurrent_bytes = (
            self.num_v_heads * self.value_head_dim * self.key_head_dim * dtype_size
        )
        return self.num_layers * (conv_bytes + recurrent_bytes)
