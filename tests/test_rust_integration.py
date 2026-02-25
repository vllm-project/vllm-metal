#!/usr/bin/env python3
"""Test Rust extension integration with PagedKVCache."""

import time

import pytest

# Import from the bundled Rust extension
from vllm_metal._rs import BlockAllocator, InputPreparer, RequestStateManager


class TestRustBlockAllocator:
    """Test the Rust BlockAllocator directly."""

    def test_basic_allocation(self):
        """Test basic block allocation."""
        allocator = BlockAllocator(100)
        assert allocator.num_free_blocks == 100

        blocks = allocator.allocate_blocks("seq-0", 10)
        assert len(blocks) == 10
        assert allocator.num_free_blocks == 90
        assert blocks == list(range(10))

    def test_multiple_sequences(self):
        """Test allocation for multiple sequences."""
        allocator = BlockAllocator(100)

        blocks0 = allocator.allocate_blocks("seq-0", 5)
        blocks1 = allocator.allocate_blocks("seq-1", 5)
        blocks2 = allocator.allocate_blocks("seq-2", 5)

        assert len(blocks0) == 5
        assert len(blocks1) == 5
        assert len(blocks2) == 5
        assert allocator.num_free_blocks == 85

        # Verify no overlap
        all_blocks = set(blocks0 + blocks1 + blocks2)
        assert len(all_blocks) == 15

    def test_free_sequence(self):
        """Test freeing blocks for a sequence."""
        allocator = BlockAllocator(100)

        allocator.allocate_blocks("seq-0", 10)
        allocator.allocate_blocks("seq-1", 10)
        assert allocator.num_free_blocks == 80

        allocator.free_sequence("seq-0")
        assert allocator.num_free_blocks == 90

        allocator.free_sequence("seq-1")
        assert allocator.num_free_blocks == 100

    def test_get_sequence_blocks(self):
        """Test getting blocks for a sequence."""
        allocator = BlockAllocator(100)

        expected = allocator.allocate_blocks("seq-42", 7)
        actual = allocator.get_sequence_blocks("seq-42")

        assert actual == expected

    def test_allocation_error(self):
        """Test error when not enough blocks."""
        allocator = BlockAllocator(10)

        with pytest.raises(RuntimeError, match="Not enough free blocks"):
            allocator.allocate_blocks("seq-0", 20)

    def test_reset(self):
        """Test resetting the allocator."""
        allocator = BlockAllocator(100)

        allocator.allocate_blocks("seq-0", 50)
        allocator.allocate_blocks("seq-1", 30)
        assert allocator.num_free_blocks == 20

        allocator.reset()
        assert allocator.num_free_blocks == 100


class TestRustInputPreparer:
    """Test the Rust InputPreparer."""

    def test_prepare_prefill(self):
        """Test preparing prefill inputs."""
        import numpy as np

        preparer = InputPreparer()
        sequences = [
            (list(range(10)), True),  # 10 tokens, prefill
            (list(range(5)), True),  # 5 tokens, prefill
        ]

        input_ids, positions = preparer.prepare_numpy(sequences)

        assert isinstance(input_ids, np.ndarray)
        assert isinstance(positions, np.ndarray)
        assert len(input_ids) == 15  # 10 + 5
        assert len(positions) == 15

    def test_prepare_decode(self):
        """Test preparing decode inputs."""

        preparer = InputPreparer()
        sequences = [
            (list(range(100)), False),  # 100 tokens, decode (only last)
            (list(range(50)), False),  # 50 tokens, decode (only last)
        ]

        input_ids, positions = preparer.prepare_numpy(sequences)

        assert len(input_ids) == 2  # Only last token from each
        assert len(positions) == 2
        assert input_ids[0] == 99  # Last token of first sequence
        assert input_ids[1] == 49  # Last token of second sequence


class TestRustRequestStateManager:
    """Test the Rust RequestStateManager."""

    def test_add_and_get_request(self):
        """Test adding requests and getting tokens."""
        manager = RequestStateManager()
        manager.add_request("req-1", [1, 2, 3, 4, 5])
        manager.add_request("req-2", [10, 20, 30])

        assert manager.num_requests == 2
        assert manager.get_last_token("req-1") == 5
        assert manager.get_last_token("req-2") == 30
        assert manager.get_tokens("req-1") == [1, 2, 3, 4, 5]

    def test_batch_operations(self):
        """Test batch get and append operations."""
        manager = RequestStateManager()
        manager.add_request("req-1", [1, 2, 3])
        manager.add_request("req-2", [10, 20])
        manager.add_request("req-3", [100])

        # Batch get last tokens
        last_tokens = manager.get_last_tokens_batch(["req-1", "req-2", "req-3"])
        assert last_tokens == [3, 20, 100]

        # Batch append tokens
        manager.append_tokens_batch(["req-1", "req-2", "req-3"], [4, 21, 101])
        last_tokens = manager.get_last_tokens_batch(["req-1", "req-2", "req-3"])
        assert last_tokens == [4, 21, 101]

    def test_append_token(self):
        """Test appending tokens to a request."""
        manager = RequestStateManager()
        manager.add_request("req-1", [1, 2, 3])

        manager.append_token("req-1", 4)
        assert manager.get_last_token("req-1") == 4
        assert manager.get_tokens("req-1") == [1, 2, 3, 4]
        assert manager.get_generated_count("req-1") == 1

        manager.append_token("req-1", 5)
        assert manager.get_last_token("req-1") == 5
        assert manager.get_generated_count("req-1") == 2

    def test_remove_request(self):
        """Test removing requests."""
        manager = RequestStateManager()
        manager.add_request("req-1", [1, 2, 3])
        manager.add_request("req-2", [10, 20])

        assert manager.num_requests == 2
        assert manager.has_request("req-1")

        manager.remove_request("req-1")
        assert manager.num_requests == 1
        assert not manager.has_request("req-1")
        assert manager.has_request("req-2")

    def test_batch_remove(self):
        """Test batch removal of requests."""
        manager = RequestStateManager()
        manager.add_request("req-1", [1])
        manager.add_request("req-2", [2])
        manager.add_request("req-3", [3])

        manager.remove_requests_batch(["req-1", "req-3"])
        assert manager.num_requests == 1
        assert not manager.has_request("req-1")
        assert manager.has_request("req-2")
        assert not manager.has_request("req-3")

    def test_missing_request(self):
        """Test behavior with missing requests."""
        manager = RequestStateManager()
        assert manager.get_last_token("nonexistent") == 0
        assert manager.get_tokens("nonexistent") == []
        assert not manager.has_request("nonexistent")

    def test_clear(self):
        """Test clearing all requests."""
        manager = RequestStateManager()
        manager.add_request("req-1", [1, 2, 3])
        manager.add_request("req-2", [4, 5, 6])

        assert manager.num_requests == 2
        manager.clear()
        assert manager.num_requests == 0


@pytest.mark.slow
def test_performance_comparison():
    """Ensure Rust allocator stays >5x faster than naive Python baseline."""
    num_blocks = 10000
    num_seqs = 200
    blocks_per_seq = 4

    # Rust allocation
    start = time.perf_counter()
    for _ in range(100):
        allocator = BlockAllocator(num_blocks)
        for seq_id in range(num_seqs):
            allocator.allocate_blocks(f"seq-{seq_id}", blocks_per_seq)
    rust_time = time.perf_counter() - start

    # Python list.pop(0) allocation
    start = time.perf_counter()
    for _ in range(100):
        free_blocks = list(range(num_blocks))
        seq_blocks = {}
        for seq_id in range(num_seqs):
            allocated = []
            for _ in range(blocks_per_seq):
                if free_blocks:
                    allocated.append(free_blocks.pop(0))
            seq_blocks[seq_id] = allocated
    python_time = time.perf_counter() - start

    speedup = python_time / rust_time
    print(f"\nRust time: {rust_time:.4f}s")
    print(f"Python time: {python_time:.4f}s")
    print(f"Speedup: {speedup:.1f}x")

    # Rust should be significantly faster
    assert speedup > 5, f"Expected at least 5x speedup, got {speedup:.1f}x"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
