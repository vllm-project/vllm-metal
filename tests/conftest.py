# SPDX-License-Identifier: Apache-2.0
"""Pytest configuration and fixtures for vLLM Metal tests."""

import pytest
import torch

# Check for Apple Silicon
IS_APPLE_SILICON = False
try:
    import platform
    IS_APPLE_SILICON = (
        platform.system() == "Darwin" and platform.machine() in ("arm64", "aarch64")
    )
except Exception:
    pass

# Check for MLX
HAS_MLX = False
try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    pass


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "apple_silicon: mark test as requiring Apple Silicon"
    )
    config.addinivalue_line(
        "markers", "mlx: mark test as requiring MLX"
    )


def pytest_collection_modifyitems(config, items):
    """Skip tests based on available hardware."""
    skip_apple_silicon = pytest.mark.skip(
        reason="Test requires Apple Silicon"
    )
    skip_mlx = pytest.mark.skip(reason="Test requires MLX")

    for item in items:
        if "apple_silicon" in item.keywords and not IS_APPLE_SILICON:
            item.add_marker(skip_apple_silicon)
        if "mlx" in item.keywords and not HAS_MLX:
            item.add_marker(skip_mlx)


@pytest.fixture
def mlx_device():
    """Get MLX device for tests."""
    if not HAS_MLX:
        pytest.skip("MLX not available")
    import mlx.core as mx
    return mx.gpu if IS_APPLE_SILICON else mx.cpu


@pytest.fixture
def torch_device():
    """Get PyTorch device for tests."""
    if IS_APPLE_SILICON and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@pytest.fixture
def sample_tensors(torch_device):
    """Create sample tensors for testing."""
    batch_size = 2
    seq_len = 16
    num_heads = 8
    head_size = 64

    query = torch.randn(
        batch_size, seq_len, num_heads, head_size,
        device=torch_device, dtype=torch.float16
    )
    key = torch.randn(
        batch_size, seq_len, num_heads, head_size,
        device=torch_device, dtype=torch.float16
    )
    value = torch.randn(
        batch_size, seq_len, num_heads, head_size,
        device=torch_device, dtype=torch.float16
    )

    return {
        "query": query,
        "key": key,
        "value": value,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "num_heads": num_heads,
        "head_size": head_size,
    }


@pytest.fixture
def kv_cache_tensors(torch_device):
    """Create KV cache tensors for testing."""
    num_blocks = 32
    block_size = 16
    num_kv_heads = 8
    head_size = 64

    key_cache = torch.zeros(
        num_blocks, block_size, num_kv_heads, head_size,
        device=torch_device, dtype=torch.float16
    )
    value_cache = torch.zeros(
        num_blocks, block_size, num_kv_heads, head_size,
        device=torch_device, dtype=torch.float16
    )

    return {
        "key_cache": key_cache,
        "value_cache": value_cache,
        "num_blocks": num_blocks,
        "block_size": block_size,
        "num_kv_heads": num_kv_heads,
        "head_size": head_size,
    }
