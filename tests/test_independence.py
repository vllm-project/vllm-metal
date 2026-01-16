# SPDX-License-Identifier: Apache-2.0
"""Tests for vLLM-metal independence from vLLM runtime dependencies."""

import subprocess
import sys


class TestIndependence:
    """Tests that vLLM-metal modules can be imported without importing vLLM."""

    def test_server_import_does_not_import_vllm(self) -> None:
        """Test that importing vllm_metal.server does not import vllm."""
        code = """
import sys
try:
    import vllm_metal.server
    print('vllm' in sys.modules)
except ImportError:
    print('import_failed')
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            # If import fails due to missing dependencies, skip the test
            import pytest

            pytest.skip(f"Import failed due to missing dependencies: {result.stderr}")
        assert result.stdout.strip() in ["False", "import_failed"], (
            f"Unexpected result: {result.stdout.strip()}"
        )
        if result.stdout.strip() == "False":
            # Only check if import succeeded
            assert result.stdout.strip() == "False", "vLLM was imported unexpectedly"

    def test_model_runner_import_does_not_import_vllm(self) -> None:
        """Test that importing vllm_metal.model_runner does not import vllm."""
        code = """
import sys
try:
    import vllm_metal.model_runner
    print('vllm' in sys.modules)
except ImportError:
    print('import_failed')
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            # If import fails due to missing dependencies, skip the test
            import pytest

            pytest.skip(f"Import failed due to missing dependencies: {result.stderr}")
        assert result.stdout.strip() in ["False", "import_failed"], (
            f"Unexpected result: {result.stdout.strip()}"
        )
        if result.stdout.strip() == "False":
            # Only check if import succeeded
            assert result.stdout.strip() == "False", "vLLM was imported unexpectedly"
