# SPDX-License-Identifier: Apache-2.0
"""Tests for vllm_metal logger initialization."""

import logging

import pytest


@pytest.fixture(autouse=True)
def configure_vllm_metal_logging():
    """Ensure vllm_metal logging is configured before each test."""
    # Trigger _configure_logging by calling _register (or directly)
    import vllm_metal

    vllm_metal._configure_logging()
    yield


class TestLoggerConfiguration:
    """Test vllm_metal logger configuration."""

    def test_metal_logger_has_handlers(self):
        """Test vllm_metal logger has handlers from vllm."""
        from vllm_metal import logger

        assert len(logger.handlers) > 0, (
            "vllm_metal logger should have handlers (shared from vllm)"
        )

    def test_metal_logger_propagate_false(self):
        """Test vllm_metal logger does not propagate to root."""
        from vllm_metal import logger

        assert logger.propagate is False, (
            "vllm_metal logger should not propagate to avoid duplicate logs"
        )

    def test_metal_logger_level_matches_vllm(self):
        """Test vllm_metal logger level matches vllm logger."""
        from vllm_metal import logger

        vllm_logger = logging.getLogger("vllm")
        assert logger.level == vllm_logger.level, (
            f"vllm_metal logger level ({logger.level}) should match "
            f"vllm logger level ({vllm_logger.level})"
        )

    def test_metal_logger_respects_vllm_logging_level(self, monkeypatch):
        """Test vllm_metal respects VLLM_LOGGING_LEVEL environment variable."""
        # Force reconfigure with DEBUG level
        monkeypatch.setenv("VLLM_LOGGING_LEVEL", "DEBUG")

        # Re-import to trigger _configure_logging
        import vllm_metal

        vllm_metal._configure_logging()

        metal_logger = logging.getLogger("vllm_metal")
        assert metal_logger.level == logging.DEBUG, (
            f"vllm_metal should respect VLLM_LOGGING_LEVEL=DEBUG, "
            f"but got level {metal_logger.level}"
        )

    def test_submodule_logger_inherits_config(self):
        """Test submodules inherit proper logger configuration."""
        from vllm_metal.platform import logger as platform_logger
        from vllm_metal.utils import logger as utils_logger

        # Both should have effective level from vllm_metal
        assert (
            platform_logger.getEffectiveLevel() == logging.getLogger("vllm_metal").level
        )
        assert utils_logger.getEffectiveLevel() == logging.getLogger("vllm_metal").level


class TestLoggerEmission:
    """Test that logs are actually emitted through handlers."""

    def test_handler_is_stream_handler(self):
        """Test vllm_metal uses StreamHandler like vllm."""
        from logging import StreamHandler

        from vllm_metal import logger

        assert any(isinstance(h, StreamHandler) for h in logger.handlers), (
            "vllm_metal should use StreamHandler like vllm"
        )


class TestLoggerConsistency:
    """Test that all loggers use consistent initialization."""

    def test_all_loggers_use_init_logger(self):
        """Verify all logger definitions use init_logger pattern.

        This is a static check that ensures no module uses logging.getLogger().
        """
        import ast
        from pathlib import Path

        import vllm_metal

        package_dir = Path(vllm_metal.__file__).parent

        violations = []

        for py_file in package_dir.rglob("*.py"):
            try:
                content = py_file.read_text()
                tree = ast.parse(content)

                for node in ast.walk(tree):
                    if isinstance(node, ast.Assign):
                        for target in node.targets:
                            if isinstance(target, ast.Name) and target.id == "logger":
                                if isinstance(node.value, ast.Call):
                                    func = node.value.func
                                    if isinstance(func, ast.Attribute):
                                        if func.attr == "getLogger":
                                            violations.append(
                                                f"{py_file.relative_to(package_dir)}: "
                                                f"uses logging.getLogger() instead of init_logger()"
                                            )
            except (SyntaxError, UnicodeDecodeError):
                continue

        if violations:
            pytest.fail(
                "Found modules using logging.getLogger() instead of init_logger():\n"
                + "\n".join(f"  - {v}" for v in violations)
            )
