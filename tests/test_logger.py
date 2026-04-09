# SPDX-License-Identifier: Apache-2.0
"""Tests for vllm_metal logger initialization.

This module verifies that all vllm_metal loggers use init_logger()
and have the correct effective log level (INFO).
"""

import logging

import pytest


class TestLoggerInitialization:
    """Test that vllm_metal loggers are properly initialized."""

    @pytest.fixture(autouse=True)
    def setup_vllm_logging(self):
        """Setup vllm logging configuration before each test."""
        # Import vllm logger to configure logging
        from vllm.logger import init_logger as vllm_init_logger

        # This triggers vllm's logging configuration
        vllm_init_logger("vllm.test_setup")

    def test_vllm_metal_root_logger_level(self):
        """Test vllm_metal root logger has correct effective level."""
        from vllm_metal import logger

        effective_level = logger.getEffectiveLevel()
        assert effective_level <= logging.INFO, (
            f"vllm_metal logger effective level is {effective_level} "
            f"({logging.getLevelName(effective_level)}), expected INFO or lower"
        )

    def test_vllm_metal_platform_logger_level(self):
        """Test platform module logger has correct effective level."""
        from vllm_metal.platform import logger

        effective_level = logger.getEffectiveLevel()
        assert effective_level <= logging.INFO, (
            f"platform logger effective level is {effective_level}, expected INFO or lower"
        )

    def test_vllm_metal_utils_logger_level(self):
        """Test utils module logger has correct effective level."""
        from vllm_metal.utils import logger

        effective_level = logger.getEffectiveLevel()
        assert effective_level <= logging.INFO, (
            f"utils logger effective level is {effective_level}, expected INFO or lower"
        )

    def test_vllm_metal_compat_logger_level(self):
        """Test compat module logger has correct effective level."""
        from vllm_metal.compat import logger

        effective_level = logger.getEffectiveLevel()
        assert effective_level <= logging.INFO, (
            f"compat logger effective level is {effective_level}, expected INFO or lower"
        )

    def test_vllm_metal_metal_logger_level(self):
        """Test metal module logger has correct effective level."""
        from vllm_metal.metal import logger

        effective_level = logger.getEffectiveLevel()
        assert effective_level <= logging.INFO, (
            f"metal logger effective level is {effective_level}, expected INFO or lower"
        )

    def test_vllm_metal_metal_build_logger_level(self):
        """Test metal.build module logger has correct effective level."""
        from vllm_metal.metal.build import logger

        effective_level = logger.getEffectiveLevel()
        assert effective_level <= logging.INFO, (
            f"metal.build logger effective level is {effective_level}, expected INFO or lower"
        )

    def test_vllm_metal_tensor_bridge_logger_level(self):
        """Test pytorch_backend.tensor_bridge module logger has correct effective level."""
        from vllm_metal.pytorch_backend.tensor_bridge import logger

        effective_level = logger.getEffectiveLevel()
        assert effective_level <= logging.INFO, (
            f"tensor_bridge logger effective level is {effective_level}, expected INFO or lower"
        )

    def test_vllm_metal_stt_detection_logger_level(self):
        """Test stt.detection module logger has correct effective level."""
        from vllm_metal.stt.detection import logger

        effective_level = logger.getEffectiveLevel()
        assert effective_level <= logging.INFO, (
            f"stt.detection logger effective level is {effective_level}, expected INFO or lower"
        )

    def test_vllm_metal_stt_loader_logger_level(self):
        """Test stt.loader module logger has correct effective level."""
        from vllm_metal.stt.loader import logger

        effective_level = logger.getEffectiveLevel()
        assert effective_level <= logging.INFO, (
            f"stt.loader logger effective level is {effective_level}, expected INFO or lower"
        )

    def test_vllm_metal_stt_whisper_adapter_logger_level(self):
        """Test stt.whisper.adapter module logger has correct effective level."""
        from vllm_metal.stt.whisper.adapter import logger

        effective_level = logger.getEffectiveLevel()
        assert effective_level <= logging.INFO, (
            f"stt.whisper.adapter logger effective level is {effective_level}, expected INFO or lower"
        )

    def test_vllm_metal_stt_whisper_transcriber_logger_level(self):
        """Test stt.whisper.transcriber module logger has correct effective level."""
        from vllm_metal.stt.whisper.transcriber import logger

        effective_level = logger.getEffectiveLevel()
        assert effective_level <= logging.INFO, (
            f"stt.whisper.transcriber logger effective level is {effective_level}, expected INFO or lower"
        )


class TestLoggerOutput:
    """Test that logger output actually appears."""

    @pytest.fixture(autouse=True)
    def setup_vllm_logging(self):
        """Setup vllm logging configuration before each test."""
        from vllm.logger import init_logger as vllm_init_logger

        vllm_init_logger("vllm.test_setup")

    def test_logger_info_output(self, caplog):
        """Test that INFO level logs are captured."""
        from vllm_metal import logger

        with caplog.at_level(logging.INFO):
            logger.info("Test INFO message")
            assert "Test INFO message" in caplog.text

    def test_logger_warning_output(self, caplog):
        """Test that WARNING level logs are captured."""
        from vllm_metal import logger

        with caplog.at_level(logging.WARNING):
            logger.warning("Test WARNING message")
            assert "Test WARNING message" in caplog.text

    def test_logger_debug_output(self, caplog):
        """Test that DEBUG level logs depend on configuration."""
        from vllm_metal import logger

        # DEBUG may or may not appear depending on vllm config
        with caplog.at_level(logging.DEBUG):
            logger.debug("Test DEBUG message")
            # Just verify it doesn't crash
            # Actual output depends on vllm's logging configuration


class TestLoggerConsistency:
    """Test that all loggers use consistent initialization."""

    def test_all_loggers_use_init_logger(self):
        """Verify all logger definitions use init_logger pattern.

        This is a static check that ensures no module uses logging.getLogger().
        """
        import ast
        from pathlib import Path

        # Get the vllm_metal package directory
        import vllm_metal

        package_dir = Path(vllm_metal.__file__).parent

        violations = []

        # Walk through all Python files
        for py_file in package_dir.rglob("*.py"):
            # Skip __init__.py files that don't define loggers
            try:
                content = py_file.read_text()
                tree = ast.parse(content)

                # Look for logger assignments
                for node in ast.walk(tree):
                    if isinstance(node, ast.Assign):
                        for target in node.targets:
                            if isinstance(target, ast.Name) and target.id == "logger":
                                # Check if it's using logging.getLogger
                                if isinstance(node.value, ast.Call):
                                    func = node.value.func
                                    if isinstance(func, ast.Attribute):
                                        if func.attr == "getLogger":
                                            violations.append(
                                                f"{py_file.relative_to(package_dir)}: "
                                                f"uses logging.getLogger() instead of init_logger()"
                                            )
            except (SyntaxError, UnicodeDecodeError):
                # Skip files that can't be parsed
                continue

        if violations:
            pytest.fail(
                "Found modules using logging.getLogger() instead of init_logger():\n"
                + "\n".join(f"  - {v}" for v in violations)
            )
