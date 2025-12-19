# SPDX-License-Identifier: Apache-2.0
"""KV Cache implementation for MLX backend."""

from dataclasses import dataclass

import mlx.core as mx


@dataclass
class CacheConfig:
    """Configuration for KV cache."""

    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    max_seq_len: int
    block_size: int
    dtype: mx.Dtype = mx.float16


class KVCache:
    """Simple KV cache for single-sequence inference."""

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int,
        dtype: mx.Dtype = mx.float16,
    ):
        """Initialize KV cache.

        Args:
            num_layers: Number of transformer layers
            num_kv_heads: Number of key-value heads
            head_dim: Dimension of each head
            max_seq_len: Maximum sequence length
            dtype: Data type for cache
        """
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.dtype = dtype

        # Initialize cache tensors
        # Shape: (num_layers, max_seq_len, num_kv_heads, head_dim)
        self.key_cache = mx.zeros(
            (num_layers, max_seq_len, num_kv_heads, head_dim),
            dtype=dtype,
        )
        self.value_cache = mx.zeros(
            (num_layers, max_seq_len, num_kv_heads, head_dim),
            dtype=dtype,
        )
        self.seq_len = 0

    def update(
        self,
        layer_idx: int,
        key: mx.array,
        value: mx.array,
        positions: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """Update cache with new key-value pairs and return full cache.

        Args:
            layer_idx: Index of the transformer layer
            key: New key tensor of shape (batch, seq_len, num_kv_heads, head_dim)
            value: New value tensor of shape (batch, seq_len, num_kv_heads, head_dim)
            positions: Position indices

        Returns:
            Tuple of (cached_keys, cached_values) up to current position
        """
        # Squeeze batch dimension for single-sequence
        key = key.squeeze(0)  # (seq_len, num_kv_heads, head_dim)
        value = value.squeeze(0)

        # Get positions
        start_pos = int(positions[0, 0].item())
        end_pos = start_pos + key.shape[0]

        # Update cache
        self.key_cache[layer_idx, start_pos:end_pos] = key
        self.value_cache[layer_idx, start_pos:end_pos] = value
        self.seq_len = max(self.seq_len, end_pos)

        # Return cached values up to current position
        cached_keys = self.key_cache[layer_idx, :end_pos][None, ...]
        cached_values = self.value_cache[layer_idx, :end_pos][None, ...]

        return cached_keys, cached_values

    def get(
        self, layer_idx: int, end_pos: int | None = None
    ) -> tuple[mx.array, mx.array]:
        """Get cached key-value pairs.

        Args:
            layer_idx: Index of the transformer layer
            end_pos: End position (default: current seq_len)

        Returns:
            Tuple of (cached_keys, cached_values)
        """
        if end_pos is None:
            end_pos = self.seq_len

        return (
            self.key_cache[layer_idx, :end_pos][None, ...],
            self.value_cache[layer_idx, :end_pos][None, ...],
        )

    def reset(self) -> None:
        """Reset the cache."""
        self.key_cache = mx.zeros_like(self.key_cache)
        self.value_cache = mx.zeros_like(self.value_cache)
        self.seq_len = 0


class PagedKVCache:
    """Paged KV cache for batched inference with variable sequence lengths."""

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        num_blocks: int,
        block_size: int,
        dtype: mx.Dtype = mx.float16,
    ):
        """Initialize paged KV cache.

        Args:
            num_layers: Number of transformer layers
            num_kv_heads: Number of key-value heads
            head_dim: Dimension of each head
            num_blocks: Total number of blocks in cache
            block_size: Number of tokens per block
            dtype: Data type for cache
        """
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.dtype = dtype

        # Block pool: (num_blocks, num_layers, 2, block_size, num_kv_heads, head_dim)
        # where 2 = key + value
        self.block_pool = mx.zeros(
            (num_blocks, num_layers, 2, block_size, num_kv_heads, head_dim),
            dtype=dtype,
        )

        # Track which blocks are free
        self.free_blocks: list[int] = list(range(num_blocks))
        # Map sequence_id -> list of block indices
        self.sequence_blocks: dict[int, list[int]] = {}

    def allocate_blocks(self, seq_id: int, num_blocks: int) -> list[int]:
        """Allocate blocks for a sequence.

        Args:
            seq_id: Sequence identifier
            num_blocks: Number of blocks to allocate

        Returns:
            List of allocated block indices

        Raises:
            RuntimeError: If not enough free blocks
        """
        if len(self.free_blocks) < num_blocks:
            msg = f"Not enough free blocks: need {num_blocks}, have {len(self.free_blocks)}"
            raise RuntimeError(msg)

        allocated = []
        for _ in range(num_blocks):
            block_idx = self.free_blocks.pop(0)
            allocated.append(block_idx)

        self.sequence_blocks[seq_id] = self.sequence_blocks.get(seq_id, []) + allocated
        return allocated

    def free_sequence(self, seq_id: int) -> None:
        """Free all blocks for a sequence.

        Args:
            seq_id: Sequence identifier
        """
        if seq_id in self.sequence_blocks:
            self.free_blocks.extend(self.sequence_blocks[seq_id])
            del self.sequence_blocks[seq_id]

    def update_block(
        self,
        block_idx: int,
        layer_idx: int,
        key: mx.array,
        value: mx.array,
        slot_offset: int,
    ) -> None:
        """Update a specific block with key-value pairs.

        Args:
            block_idx: Index of the block
            layer_idx: Index of the transformer layer
            key: Key tensor of shape (num_tokens, num_kv_heads, head_dim)
            value: Value tensor of shape (num_tokens, num_kv_heads, head_dim)
            slot_offset: Starting slot within the block
        """
        num_tokens = key.shape[0]
        end_slot = slot_offset + num_tokens

        # Update key (index 0) and value (index 1)
        self.block_pool[block_idx, layer_idx, 0, slot_offset:end_slot] = key
        self.block_pool[block_idx, layer_idx, 1, slot_offset:end_slot] = value

    def get_sequence_kv(
        self, seq_id: int, layer_idx: int, seq_len: int
    ) -> tuple[mx.array, mx.array]:
        """Get key-value pairs for a sequence.

        Args:
            seq_id: Sequence identifier
            layer_idx: Index of the transformer layer
            seq_len: Current sequence length

        Returns:
            Tuple of (keys, values) of shape (seq_len, num_kv_heads, head_dim)
        """
        blocks = self.sequence_blocks.get(seq_id, [])
        if not blocks:
            return (
                mx.zeros((0, self.num_kv_heads, self.head_dim), dtype=self.dtype),
                mx.zeros((0, self.num_kv_heads, self.head_dim), dtype=self.dtype),
            )

        keys = []
        values = []
        remaining = seq_len

        for block_idx in blocks:
            tokens_in_block = min(remaining, self.block_size)
            keys.append(self.block_pool[block_idx, layer_idx, 0, :tokens_in_block])
            values.append(self.block_pool[block_idx, layer_idx, 1, :tokens_in_block])
            remaining -= tokens_in_block
            if remaining <= 0:
                break

        return mx.concatenate(keys, axis=0), mx.concatenate(values, axis=0)

    @property
    def num_free_blocks(self) -> int:
        """Return number of free blocks."""
        return len(self.free_blocks)
