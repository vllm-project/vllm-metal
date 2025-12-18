# SPDX-License-Identifier: Apache-2.0
"""Compatibility layer for vLLM imports.

This module provides version-compatible imports from vLLM to handle
API changes across different vLLM versions (0.12+).
"""

from typing import TYPE_CHECKING

# Core vLLM imports that are stable across versions
try:
    from vllm.config import VllmConfig
except ImportError:
    VllmConfig = None  # type: ignore[misc, assignment]

# Platform imports
try:
    from vllm.platforms import Platform, PlatformEnum
except ImportError:
    # Fallback for older versions
    try:
        from vllm.platforms.interface import Platform, PlatformEnum
    except ImportError:
        Platform = object  # type: ignore[misc, assignment]
        PlatformEnum = None  # type: ignore[misc, assignment]

# Logger import
try:
    from vllm.logger import init_logger
except ImportError:
    import logging

    def init_logger(name: str) -> logging.Logger:
        """Fallback logger initialization."""
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            )
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger


# Attention backend imports
try:
    from vllm.attention.backends.abstract import (
        AttentionBackend,
        AttentionImpl,
        AttentionMetadata,
        AttentionType,
    )
except ImportError:
    AttentionBackend = object  # type: ignore[misc, assignment]
    AttentionImpl = object  # type: ignore[misc, assignment]
    AttentionMetadata = object  # type: ignore[misc, assignment]
    AttentionType = None  # type: ignore[misc, assignment]

# Worker imports
try:
    from vllm.v1.worker.gpu_worker import Worker as GPUWorker
except ImportError:
    try:
        from vllm.worker.worker import Worker as GPUWorker
    except ImportError:
        GPUWorker = object  # type: ignore[misc, assignment]

# Model runner imports
try:
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
except ImportError:
    try:
        from vllm.worker.model_runner import ModelRunner as GPUModelRunner
    except ImportError:
        GPUModelRunner = object  # type: ignore[misc, assignment]

# Compilation config imports (for disabling CUDA graphs)
try:
    from vllm.config.compilation import CompilationMode, CUDAGraphMode
except ImportError:
    CompilationMode = None  # type: ignore[misc, assignment]
    CUDAGraphMode = None  # type: ignore[misc, assignment]

if TYPE_CHECKING:
    from vllm.config import VllmConfig as VllmConfigType

__all__ = [
    "VllmConfig",
    "Platform",
    "PlatformEnum",
    "init_logger",
    "AttentionBackend",
    "AttentionImpl",
    "AttentionMetadata",
    "AttentionType",
    "GPUWorker",
    "GPUModelRunner",
    "CompilationMode",
    "CUDAGraphMode",
]
