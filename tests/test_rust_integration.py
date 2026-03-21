#!/usr/bin/env python3
"""Test Rust extension integration with PagedKVCache."""

import pytest

try:
    from vllm_metal._rs import RequestStateManager
except ImportError:
    pytest.skip(
        "Rust extension not installed. Build/rebuild it with: "
        "uv pip install -e . --reinstall --no-deps",
        allow_module_level=True,
    )


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
