# SPDX-License-Identifier: Apache-2.0
"""Pytest configuration and fixtures for vLLM Metal tests."""

import platform

import pytest
import torch


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "metal: mark test as requiring Metal/MPS backend"
    )
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "server: mark test as requiring vLLM server")


def pytest_collection_modifyitems(config, items):
    """Skip tests based on platform and availability."""
    skip_metal = pytest.mark.skip(reason="Metal/MPS not available")
    skip_not_macos = pytest.mark.skip(reason="Not running on macOS")

    for item in items:
        if "metal" in item.keywords:
            # Check if we're on macOS
            if platform.system() != "Darwin":
                item.add_marker(skip_not_macos)
            # Check if MPS is available
            elif not torch.backends.mps.is_available():
                item.add_marker(skip_metal)


@pytest.fixture
def mps_device():
    """Fixture providing an MPS device if available."""
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    return torch.device("mps")


@pytest.fixture
def cpu_device():
    """Fixture providing a CPU device."""
    return torch.device("cpu")


@pytest.fixture
def sample_tensor(mps_device):
    """Fixture providing a sample tensor on MPS."""
    return torch.randn(4, 8, 16, device=mps_device, dtype=torch.float16)


@pytest.fixture
def metal_config():
    """Fixture providing a default Metal configuration."""
    from vllm_metal.config import MetalConfig

    return MetalConfig()
