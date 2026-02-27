"""Helpers for probing optional Rust extensions in tests."""

from __future__ import annotations

import pytest


def rust_block_allocator_supports_string_seq_id() -> bool:
    """Return True if the installed Rust extension accepts string seq_id.

    vllm-metal's Rust `BlockAllocator` is intended to take a string `seq_id`.
    When developers pull new commits without rebuilding the extension, a stale
    local `.so` can have an incompatible signature (e.g. expecting an int),
    which otherwise produces confusing TypeError failures in tests.
    """

    try:
        from vllm_metal._rs import BlockAllocator
    except ImportError:
        return False

    try:
        allocator = BlockAllocator(1)
        # Allocate 0 blocks to avoid side effects while validating the signature.
        allocator.allocate_blocks("probe", 0)
        return True
    except (TypeError, AttributeError, RuntimeError, ValueError):
        return False


def require_rust_block_allocator_string_seq_id() -> None:
    """Skip tests when the local Rust extension is stale/incompatible."""

    if rust_block_allocator_supports_string_seq_id():
        return

    pytest.skip(
        "Rust extension BlockAllocator is missing or incompatible with this checkout "
        "(expected string seq_id). "
        "This usually means your local vllm-metal Rust extension is stale. "
        "Rebuild the local extension with: uv pip install -e . --reinstall --no-deps",
        allow_module_level=True,
    )
