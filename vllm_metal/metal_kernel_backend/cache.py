# SPDX-License-Identifier: Apache-2.0
"""MPS-backed paged KV cache for the HF Metal kernel.

Stores per-layer key/value caches as PyTorch MPS tensors in the layout
expected by ``reshape_and_cache`` and ``paged_attention_v1``:

- key_cache:   [num_blocks, num_kv_heads, head_dim // x, block_size, x]
               where x = 16 // element_size (8 for float16)
- value_cache: [num_blocks, num_kv_heads, head_dim, block_size]

Block allocation is handled by the Rust ``BlockAllocator``.
"""

from __future__ import annotations

import torch

from vllm_metal._rs import BlockAllocator


class MPSPagedKVCache:
    """Per-layer MPS tensors for the HF paged-attention kernel."""

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        num_blocks: int,
        block_size: int,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.dtype = dtype

        self._allocator = BlockAllocator(num_blocks)

        # x factor for key cache layout
        element_size = torch.tensor([], dtype=dtype).element_size()
        self.x = 16 // element_size  # 8 for float16, 4 for float32

        # Per-layer caches
        self.key_caches: list[torch.Tensor] = []
        self.value_caches: list[torch.Tensor] = []
        for _ in range(num_layers):
            self.key_caches.append(
                torch.zeros(
                    num_blocks,
                    num_kv_heads,
                    head_dim // self.x,
                    block_size,
                    self.x,
                    dtype=dtype,
                    device="mps",
                )
            )
            self.value_caches.append(
                torch.zeros(
                    num_blocks,
                    num_kv_heads,
                    head_dim,
                    block_size,
                    dtype=dtype,
                    device="mps",
                )
            )

        # Scale tensors (identity scaling)
        self.k_scale_tensor = torch.tensor(1.0, dtype=torch.float32, device="mps")
        self.v_scale_tensor = torch.tensor(1.0, dtype=torch.float32, device="mps")

    def allocate_blocks(self, seq_id: int, num_blocks: int) -> list[int]:
        """Allocate *num_blocks* blocks for *seq_id*.

        Returns list of block indices.
        """
        return self._allocator.allocate_blocks(seq_id, num_blocks)

    def free_sequence(self, seq_id: int) -> None:
        """Free all blocks belonging to *seq_id*."""
        self._allocator.free_sequence(seq_id)

    def has_sequence(self, seq_id: int) -> bool:
        """Check whether *seq_id* has allocated blocks."""
        return self._allocator.has_sequence(seq_id)

    @property
    def num_free_blocks(self) -> int:
        return self._allocator.num_free_blocks
