# SPDX-License-Identifier: Apache-2.0
"""Tests for paged KV cache adapters."""

import mlx.core as mx

from vllm_metal.mlx_backend.cache import PagedKVCache
from vllm_metal.mlx_backend.paged_cache_adapter import (
    PagedBatchKVCacheAdapter,
    PagedKVCacheAdapter,
    PagedKVCacheAdapterList,
)


class TestPagedKVCacheAdapter:
    """Tests for single-layer paged cache adapter."""

    def test_adapter_initialization(self) -> None:
        """Test adapter initialization."""
        cache = PagedKVCache(
            num_layers=4,
            num_kv_heads=8,
            head_dim=64,
            num_blocks=100,
            block_size=16,
        )
        # Pre-allocate blocks for seq 0
        cache.allocate_blocks(seq_id=0, num_blocks=5)

        adapter = PagedKVCacheAdapter(
            paged_cache=cache,
            seq_id=0,
            layer_idx=0,
            block_size=16,
        )

        assert adapter.seq_id == 0
        assert adapter.layer_idx == 0
        assert adapter.block_size == 16
        assert adapter.offset == 0

    def test_update_and_fetch_basic(self) -> None:
        """Test basic update_and_fetch operation."""
        cache = PagedKVCache(
            num_layers=2,
            num_kv_heads=4,
            head_dim=32,
            num_blocks=10,
            block_size=16,
        )
        cache.allocate_blocks(seq_id=0, num_blocks=2)

        adapter = PagedKVCacheAdapter(
            paged_cache=cache,
            seq_id=0,
            layer_idx=0,
            block_size=16,
        )

        # Create input in mlx_lm format: (batch, heads, seq, dim)
        keys = mx.random.normal((1, 4, 8, 32))
        values = mx.random.normal((1, 4, 8, 32))

        # Update and fetch
        cached_k, cached_v = adapter.update_and_fetch(keys, values)
        mx.eval(cached_k, cached_v)

        assert cached_k.shape == (1, 4, 8, 32)
        assert cached_v.shape == (1, 4, 8, 32)
        assert adapter.offset == 8

    def test_update_and_fetch_incremental(self) -> None:
        """Test incremental decode updates (single token at a time)."""
        cache = PagedKVCache(
            num_layers=2,
            num_kv_heads=4,
            head_dim=32,
            num_blocks=10,
            block_size=16,
        )
        cache.allocate_blocks(seq_id=0, num_blocks=2)

        adapter = PagedKVCacheAdapter(
            paged_cache=cache,
            seq_id=0,
            layer_idx=0,
            block_size=16,
        )

        # First update: prefill with 8 tokens
        keys1 = mx.random.normal((1, 4, 8, 32))
        values1 = mx.random.normal((1, 4, 8, 32))
        adapter.update_and_fetch(keys1, values1)
        assert adapter.offset == 8

        # Second update: decode 1 token
        keys2 = mx.random.normal((1, 4, 1, 32))
        values2 = mx.random.normal((1, 4, 1, 32))
        cached_k, cached_v = adapter.update_and_fetch(keys2, values2)
        mx.eval(cached_k, cached_v)

        assert cached_k.shape == (1, 4, 9, 32)
        assert cached_v.shape == (1, 4, 9, 32)
        assert adapter.offset == 9

    def test_update_across_block_boundary(self) -> None:
        """Test updates that span multiple blocks."""
        block_size = 8
        cache = PagedKVCache(
            num_layers=2,
            num_kv_heads=4,
            head_dim=32,
            num_blocks=10,
            block_size=block_size,
        )
        cache.allocate_blocks(seq_id=0, num_blocks=3)

        adapter = PagedKVCacheAdapter(
            paged_cache=cache,
            seq_id=0,
            layer_idx=0,
            block_size=block_size,
        )

        # Update with 20 tokens (spans 3 blocks of size 8)
        keys = mx.random.normal((1, 4, 20, 32))
        values = mx.random.normal((1, 4, 20, 32))

        cached_k, cached_v = adapter.update_and_fetch(keys, values)
        mx.eval(cached_k, cached_v)

        assert cached_k.shape == (1, 4, 20, 32)
        assert cached_v.shape == (1, 4, 20, 32)
        assert adapter.offset == 20

    def test_state_property(self) -> None:
        """Test state property returns correct values."""
        cache = PagedKVCache(
            num_layers=2,
            num_kv_heads=4,
            head_dim=32,
            num_blocks=10,
            block_size=16,
        )
        cache.allocate_blocks(seq_id=0, num_blocks=2)

        adapter = PagedKVCacheAdapter(
            paged_cache=cache,
            seq_id=0,
            layer_idx=0,
            block_size=16,
        )

        # Before any updates, state should be empty
        k, v = adapter.state
        assert k.shape[0] == 0
        assert v.shape[0] == 0

        # After update
        keys = mx.random.normal((1, 4, 8, 32))
        values = mx.random.normal((1, 4, 8, 32))
        adapter.update_and_fetch(keys, values)

        k, v = adapter.state
        mx.eval(k, v)
        assert k.shape == (8, 4, 32)
        assert v.shape == (8, 4, 32)

    def test_lazy_block_allocation(self) -> None:
        """Test that blocks are allocated lazily as needed."""
        block_size = 4
        cache = PagedKVCache(
            num_layers=2,
            num_kv_heads=4,
            head_dim=32,
            num_blocks=10,
            block_size=block_size,
        )
        # Start with only 1 block
        cache.allocate_blocks(seq_id=0, num_blocks=1)
        initial_free = cache.num_free_blocks

        adapter = PagedKVCacheAdapter(
            paged_cache=cache,
            seq_id=0,
            layer_idx=0,
            block_size=block_size,
        )

        # First 4 tokens fit in initial block
        keys1 = mx.random.normal((1, 4, 4, 32))
        values1 = mx.random.normal((1, 4, 4, 32))
        adapter.update_and_fetch(keys1, values1)
        assert cache.num_free_blocks == initial_free  # No new allocation

        # Next token triggers new block allocation
        keys2 = mx.random.normal((1, 4, 1, 32))
        values2 = mx.random.normal((1, 4, 1, 32))
        adapter.update_and_fetch(keys2, values2)
        assert cache.num_free_blocks == initial_free - 1  # One new block allocated

    def test_external_block_ids(self) -> None:
        """Test adapter with externally provided block IDs (from scheduler)."""
        block_size = 8
        cache = PagedKVCache(
            num_layers=2,
            num_kv_heads=4,
            head_dim=32,
            num_blocks=10,
            block_size=block_size,
        )

        # Specify external block IDs (simulating scheduler allocation)
        external_block_ids = [3, 7, 1]  # Non-contiguous blocks

        adapter = PagedKVCacheAdapter(
            paged_cache=cache,
            seq_id=0,
            layer_idx=0,
            block_size=block_size,
            block_ids=external_block_ids,
        )

        # Update should use external blocks, not allocate new ones
        initial_free = cache.num_free_blocks
        keys = mx.random.normal((1, 4, 20, 32))  # 20 tokens = 3 blocks
        values = mx.random.normal((1, 4, 20, 32))

        cached_k, cached_v = adapter.update_and_fetch(keys, values)
        mx.eval(cached_k, cached_v)

        # Should not have allocated internally
        assert cache.num_free_blocks == initial_free
        assert cached_k.shape == (1, 4, 20, 32)
        assert adapter.offset == 20
        assert adapter.get_block_ids() == external_block_ids

    def test_append_block_ids(self) -> None:
        """Test appending new block IDs during decode."""
        block_size = 8
        cache = PagedKVCache(
            num_layers=2,
            num_kv_heads=4,
            head_dim=32,
            num_blocks=10,
            block_size=block_size,
        )

        # Start with 1 block
        external_block_ids = [5]

        adapter = PagedKVCacheAdapter(
            paged_cache=cache,
            seq_id=0,
            layer_idx=0,
            block_size=block_size,
            block_ids=external_block_ids,
        )

        # Fill first block
        keys1 = mx.random.normal((1, 4, 8, 32))
        values1 = mx.random.normal((1, 4, 8, 32))
        adapter.update_and_fetch(keys1, values1)
        assert adapter.offset == 8

        # Append new block IDs (simulating scheduler allocating more)
        adapter.append_block_ids([2])
        assert adapter.get_block_ids() == [5, 2]

        # Continue decoding into new block
        keys2 = mx.random.normal((1, 4, 4, 32))
        values2 = mx.random.normal((1, 4, 4, 32))
        cached_k, cached_v = adapter.update_and_fetch(keys2, values2)
        mx.eval(cached_k, cached_v)

        assert adapter.offset == 12
        assert cached_k.shape == (1, 4, 12, 32)


class TestPagedKVCacheAdapterList:
    """Tests for per-sequence adapter list."""

    def test_adapter_list_initialization(self) -> None:
        """Test adapter list initialization."""
        cache = PagedKVCache(
            num_layers=4,
            num_kv_heads=8,
            head_dim=64,
            num_blocks=100,
            block_size=16,
        )
        cache.allocate_blocks(seq_id=0, num_blocks=5)

        adapter_list = PagedKVCacheAdapterList(
            paged_cache=cache,
            seq_id=0,
            num_layers=4,
            block_size=16,
        )

        assert len(adapter_list) == 4
        assert adapter_list.seq_id == 0

    def test_layer_indexing(self) -> None:
        """Test per-layer indexing."""
        cache = PagedKVCache(
            num_layers=4,
            num_kv_heads=8,
            head_dim=64,
            num_blocks=100,
            block_size=16,
        )
        cache.allocate_blocks(seq_id=0, num_blocks=5)

        adapter_list = PagedKVCacheAdapterList(
            paged_cache=cache,
            seq_id=0,
            num_layers=4,
            block_size=16,
        )

        # Access each layer
        for i in range(4):
            adapter = adapter_list[i]
            assert adapter.layer_idx == i
            assert adapter.seq_id == 0

    def test_iteration(self) -> None:
        """Test iterating over layers."""
        cache = PagedKVCache(
            num_layers=4,
            num_kv_heads=8,
            head_dim=64,
            num_blocks=100,
            block_size=16,
        )
        cache.allocate_blocks(seq_id=0, num_blocks=5)

        adapter_list = PagedKVCacheAdapterList(
            paged_cache=cache,
            seq_id=0,
            num_layers=4,
            block_size=16,
        )

        layers = list(adapter_list)
        assert len(layers) == 4
        for i, adapter in enumerate(layers):
            assert adapter.layer_idx == i

    def test_free_releases_blocks(self) -> None:
        """Test that free() releases blocks back to pool."""
        cache = PagedKVCache(
            num_layers=4,
            num_kv_heads=8,
            head_dim=64,
            num_blocks=100,
            block_size=16,
        )
        initial_free = cache.num_free_blocks

        cache.allocate_blocks(seq_id=0, num_blocks=5)
        assert cache.num_free_blocks == initial_free - 5

        adapter_list = PagedKVCacheAdapterList(
            paged_cache=cache,
            seq_id=0,
            num_layers=4,
            block_size=16,
        )

        # Free should release blocks
        adapter_list.free()
        assert cache.num_free_blocks == initial_free
        assert not cache.has_sequence(0)

    def test_offset_property(self) -> None:
        """Test offset returns correct sequence length."""
        cache = PagedKVCache(
            num_layers=2,
            num_kv_heads=4,
            head_dim=32,
            num_blocks=10,
            block_size=16,
        )
        cache.allocate_blocks(seq_id=0, num_blocks=2)

        adapter_list = PagedKVCacheAdapterList(
            paged_cache=cache,
            seq_id=0,
            num_layers=2,
            block_size=16,
        )

        assert adapter_list.offset == 0

        # Update layer 0
        keys = mx.random.normal((1, 4, 8, 32))
        values = mx.random.normal((1, 4, 8, 32))
        adapter_list[0].update_and_fetch(keys, values)

        # Offset should reflect the update
        assert adapter_list[0].offset == 8


class TestPagedBatchKVCacheAdapter:
    """Tests for batched paged cache adapter."""

    def test_merge_single_sequence(self) -> None:
        """Test merging a single sequence."""
        cache = PagedKVCache(
            num_layers=2,
            num_kv_heads=4,
            head_dim=32,
            num_blocks=20,
            block_size=16,
        )
        cache.allocate_blocks(seq_id=0, num_blocks=2)

        adapter = PagedKVCacheAdapterList(
            paged_cache=cache,
            seq_id=0,
            num_layers=2,
            block_size=16,
        )

        # Pre-fill with some data
        keys = mx.random.normal((1, 4, 8, 32))
        values = mx.random.normal((1, 4, 8, 32))
        adapter[0].update_and_fetch(keys, values)
        adapter[1].update_and_fetch(keys, values)

        # Merge
        batch_layers = PagedBatchKVCacheAdapter.merge([adapter])

        assert len(batch_layers) == 2
        assert batch_layers[0].layer_idx == 0
        assert batch_layers[1].layer_idx == 1

    def test_merge_multiple_sequences(self) -> None:
        """Test merging multiple sequences."""
        cache = PagedKVCache(
            num_layers=2,
            num_kv_heads=4,
            head_dim=32,
            num_blocks=40,
            block_size=16,
        )

        # Create two sequences with different lengths
        cache.allocate_blocks(seq_id=0, num_blocks=2)
        cache.allocate_blocks(seq_id=1, num_blocks=2)

        adapter0 = PagedKVCacheAdapterList(
            paged_cache=cache,
            seq_id=0,
            num_layers=2,
            block_size=16,
        )
        adapter1 = PagedKVCacheAdapterList(
            paged_cache=cache,
            seq_id=1,
            num_layers=2,
            block_size=16,
        )

        # Prefill with different lengths
        keys0 = mx.random.normal((1, 4, 8, 32))
        values0 = mx.random.normal((1, 4, 8, 32))
        keys1 = mx.random.normal((1, 4, 12, 32))
        values1 = mx.random.normal((1, 4, 12, 32))

        for layer_idx in range(2):
            adapter0[layer_idx].update_and_fetch(keys0, values0)
            adapter1[layer_idx].update_and_fetch(keys1, values1)

        # Merge
        batch_layers = PagedBatchKVCacheAdapter.merge([adapter0, adapter1])

        assert len(batch_layers) == 2

    def test_batched_update_and_fetch(self) -> None:
        """Test batched update_and_fetch with left-padding."""
        cache = PagedKVCache(
            num_layers=2,
            num_kv_heads=4,
            head_dim=32,
            num_blocks=40,
            block_size=16,
        )

        cache.allocate_blocks(seq_id=0, num_blocks=2)
        cache.allocate_blocks(seq_id=1, num_blocks=2)

        adapter0 = PagedKVCacheAdapterList(
            paged_cache=cache,
            seq_id=0,
            num_layers=2,
            block_size=16,
        )
        adapter1 = PagedKVCacheAdapterList(
            paged_cache=cache,
            seq_id=1,
            num_layers=2,
            block_size=16,
        )

        # Prefill seq0 with 6 tokens, seq1 with 10 tokens
        keys0 = mx.random.normal((1, 4, 6, 32))
        values0 = mx.random.normal((1, 4, 6, 32))
        keys1 = mx.random.normal((1, 4, 10, 32))
        values1 = mx.random.normal((1, 4, 10, 32))

        for layer_idx in range(2):
            adapter0[layer_idx].update_and_fetch(keys0, values0)
            adapter1[layer_idx].update_and_fetch(keys1, values1)

        # Merge and do batched decode
        batch_layers = PagedBatchKVCacheAdapter.merge([adapter0, adapter1])

        # Batched decode: 2 sequences, 1 token each
        batch_keys = mx.random.normal((2, 4, 1, 32))
        batch_values = mx.random.normal((2, 4, 1, 32))

        cached_k, cached_v = batch_layers[0].update_and_fetch(batch_keys, batch_values)
        mx.eval(cached_k, cached_v)

        # After decode: seq0 has 7 tokens, seq1 has 11 tokens
        # Max length is 11, so output should be (2, 4, 11, 32) with left-padding
        assert cached_k.shape == (2, 4, 11, 32)
        assert cached_v.shape == (2, 4, 11, 32)

    def test_extract_sequence(self) -> None:
        """Test extracting a sequence from batch."""
        cache = PagedKVCache(
            num_layers=2,
            num_kv_heads=4,
            head_dim=32,
            num_blocks=40,
            block_size=16,
        )

        cache.allocate_blocks(seq_id=0, num_blocks=2)
        cache.allocate_blocks(seq_id=1, num_blocks=2)

        adapter0 = PagedKVCacheAdapterList(
            paged_cache=cache,
            seq_id=0,
            num_layers=2,
            block_size=16,
        )
        adapter1 = PagedKVCacheAdapterList(
            paged_cache=cache,
            seq_id=1,
            num_layers=2,
            block_size=16,
        )

        batch = PagedBatchKVCacheAdapter([adapter0, adapter1])

        # Extract should return original adapters
        extracted0 = batch.extract(0)
        extracted1 = batch.extract(1)

        assert extracted0 is adapter0
        assert extracted1 is adapter1

    def test_empty_merge(self) -> None:
        """Test merging empty list returns empty list."""
        batch_layers = PagedBatchKVCacheAdapter.merge([])
        assert batch_layers == []

    def test_left_padding_zeros(self) -> None:
        """Test that left-padding uses zeros."""
        cache = PagedKVCache(
            num_layers=1,
            num_kv_heads=2,
            head_dim=4,
            num_blocks=20,
            block_size=8,
        )

        cache.allocate_blocks(seq_id=0, num_blocks=1)
        cache.allocate_blocks(seq_id=1, num_blocks=1)

        adapter0 = PagedKVCacheAdapterList(
            paged_cache=cache,
            seq_id=0,
            num_layers=1,
            block_size=8,
        )
        adapter1 = PagedKVCacheAdapterList(
            paged_cache=cache,
            seq_id=1,
            num_layers=1,
            block_size=8,
        )

        # Seq0: 2 tokens, Seq1: 4 tokens
        keys0 = mx.ones((1, 2, 2, 4))
        values0 = mx.ones((1, 2, 2, 4))
        keys1 = mx.ones((1, 2, 4, 4)) * 2
        values1 = mx.ones((1, 2, 4, 4)) * 2

        adapter0[0].update_and_fetch(keys0, values0)
        adapter1[0].update_and_fetch(keys1, values1)

        batch_layers = PagedBatchKVCacheAdapter.merge([adapter0, adapter1])

        # Decode with 1 token each
        batch_keys = mx.ones((2, 2, 1, 4)) * 3
        batch_values = mx.ones((2, 2, 1, 4)) * 3

        cached_k, cached_v = batch_layers[0].update_and_fetch(batch_keys, batch_values)
        mx.eval(cached_k, cached_v)

        # Seq0 now has 3 tokens, seq1 has 5 tokens -> max len 5
        assert cached_k.shape == (2, 2, 5, 4)

        # First 2 positions of seq0 should be zero (left padding)
        # Check that padding positions are zeros
        seq0_k = cached_k[0]  # Shape: (2, 5, 4)
        padding_k = seq0_k[:, :2, :]  # First 2 positions
        mx.eval(padding_k)
        assert mx.all(padding_k == 0).item()
