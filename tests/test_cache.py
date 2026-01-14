# SPDX-License-Identifier: Apache-2.0
"""Tests for KV cache implementations."""

import mlx.core as mx
import pytest

from vllm_metal.mlx_backend.cache import KVCache, PagedKVCache


class TestKVCache:
    """Tests for simple KV cache."""

    def test_cache_initialization(self) -> None:
        """Test cache initialization."""
        cache = KVCache(
            num_layers=4,
            num_kv_heads=8,
            head_dim=64,
            max_seq_len=512,
        )

        assert cache.num_layers == 4
        assert cache.num_kv_heads == 8
        assert cache.head_dim == 64
        assert cache.max_seq_len == 512
        assert cache.seq_len == 0

    def test_cache_update(self) -> None:
        """Test cache update and retrieval."""
        cache = KVCache(
            num_layers=2,
            num_kv_heads=4,
            head_dim=32,
            max_seq_len=128,
        )

        # Create test tensors
        batch = 1
        seq_len = 4
        key = mx.random.normal((batch, seq_len, 4, 32))
        value = mx.random.normal((batch, seq_len, 4, 32))
        positions = mx.arange(seq_len)[None, :]

        # Update layer 0
        cached_k, cached_v = cache.update(0, key, value, positions)
        mx.eval(cached_k, cached_v)

        assert cached_k.shape == (1, seq_len, 4, 32)
        assert cached_v.shape == (1, seq_len, 4, 32)
        assert cache.seq_len == seq_len

    def test_cache_incremental_update(self) -> None:
        """Test incremental cache updates."""
        cache = KVCache(
            num_layers=2,
            num_kv_heads=4,
            head_dim=32,
            max_seq_len=128,
        )

        # First update
        key1 = mx.random.normal((1, 4, 4, 32))
        value1 = mx.random.normal((1, 4, 4, 32))
        positions1 = mx.arange(4)[None, :]

        cache.update(0, key1, value1, positions1)
        assert cache.seq_len == 4

        # Second update (incremental)
        key2 = mx.random.normal((1, 1, 4, 32))
        value2 = mx.random.normal((1, 1, 4, 32))
        positions2 = mx.array([[4]])

        cached_k, cached_v = cache.update(0, key2, value2, positions2)
        mx.eval(cached_k, cached_v)

        assert cached_k.shape == (1, 5, 4, 32)
        assert cached_v.shape == (1, 5, 4, 32)
        assert cache.seq_len == 5

    def test_cache_reset(self) -> None:
        """Test cache reset."""
        cache = KVCache(
            num_layers=2,
            num_kv_heads=4,
            head_dim=32,
            max_seq_len=128,
        )

        # Add some data
        key = mx.random.normal((1, 4, 4, 32))
        value = mx.random.normal((1, 4, 4, 32))
        positions = mx.arange(4)[None, :]
        cache.update(0, key, value, positions)

        assert cache.seq_len == 4

        # Reset
        cache.reset()

        assert cache.seq_len == 0


class TestPagedKVCache:
    """Tests for paged KV cache."""

    def test_paged_cache_initialization(self) -> None:
        """Test paged cache initialization."""
        cache = PagedKVCache(
            num_layers=4,
            num_kv_heads=8,
            head_dim=64,
            num_blocks=100,
            block_size=16,
        )

        assert cache.num_layers == 4
        assert cache.num_kv_heads == 8
        assert cache.head_dim == 64
        assert cache.num_blocks == 100
        assert cache.block_size == 16
        assert cache.num_free_blocks == 100

    def test_block_allocation(self) -> None:
        """Test block allocation."""
        cache = PagedKVCache(
            num_layers=2,
            num_kv_heads=4,
            head_dim=32,
            num_blocks=10,
            block_size=16,
        )

        # Allocate blocks for sequence 0
        blocks = cache.allocate_blocks(seq_id=0, num_blocks=3)

        assert len(blocks) == 3
        assert cache.num_free_blocks == 7
        assert cache.has_sequence(0)

    def test_block_allocation_insufficient(self) -> None:
        """Test block allocation with insufficient blocks."""
        cache = PagedKVCache(
            num_layers=2,
            num_kv_heads=4,
            head_dim=32,
            num_blocks=5,
            block_size=16,
        )

        # Try to allocate more blocks than available
        with pytest.raises(RuntimeError, match="Not enough free blocks"):
            cache.allocate_blocks(seq_id=0, num_blocks=10)

    def test_sequence_free(self) -> None:
        """Test freeing sequence blocks."""
        cache = PagedKVCache(
            num_layers=2,
            num_kv_heads=4,
            head_dim=32,
            num_blocks=10,
            block_size=16,
        )

        # Allocate blocks
        cache.allocate_blocks(seq_id=0, num_blocks=3)
        cache.allocate_blocks(seq_id=1, num_blocks=2)

        assert cache.num_free_blocks == 5

        # Free sequence 0
        cache.free_sequence(seq_id=0)

        assert cache.num_free_blocks == 8
        assert not cache.has_sequence(0)
        assert cache.has_sequence(1)

    def test_cache_usage(self) -> None:
        """Test cache usage statistics."""
        cache = PagedKVCache(
            num_layers=2,
            num_kv_heads=4,
            head_dim=32,
            num_blocks=10,
            block_size=16,
        )

        # Initially no blocks used
        used, total, ratio = cache.get_cache_usage()
        assert used == 0
        assert total == 10
        assert ratio == 0.0

        # Allocate some blocks
        cache.allocate_blocks(seq_id=0, num_blocks=3)
        used, total, ratio = cache.get_cache_usage()
        assert used == 3
        assert total == 10
        assert ratio == 0.3

        # Allocate more
        cache.allocate_blocks(seq_id=1, num_blocks=2)
        used, total, ratio = cache.get_cache_usage()
        assert used == 5
        assert total == 10
        assert ratio == 0.5

        # Free and check
        cache.free_sequence(seq_id=0)
        used, total, ratio = cache.get_cache_usage()
        assert used == 2
        assert total == 10
        assert ratio == 0.2

    def test_block_update(self) -> None:
        """Test updating block contents."""
        cache = PagedKVCache(
            num_layers=2,
            num_kv_heads=4,
            head_dim=32,
            num_blocks=10,
            block_size=16,
        )

        blocks = cache.allocate_blocks(seq_id=0, num_blocks=1)
        block_idx = blocks[0]

        # Update block
        key = mx.random.normal((8, 4, 32))
        value = mx.random.normal((8, 4, 32))

        cache.update_block(
            block_idx=block_idx,
            layer_idx=0,
            key=key,
            value=value,
            slot_offset=0,
        )

        # Verify update
        cached_k, cached_v = cache.get_sequence_kv(seq_id=0, layer_idx=0, seq_len=8)
        mx.eval(cached_k, cached_v)

        assert cached_k.shape == (8, 4, 32)
        assert cached_v.shape == (8, 4, 32)
