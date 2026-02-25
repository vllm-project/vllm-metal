# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

from vllm_metal.config import (
    AUTO_MEMORY_FRACTION,
    MetalConfig,
)
from vllm_metal.v1.worker import MetalWorker as V1MetalWorker


def _make_worker() -> V1MetalWorker:
    """Create a minimal V1MetalWorker for unit testing."""
    worker = V1MetalWorker.__new__(V1MetalWorker)
    worker.metal_config = MetalConfig(
        memory_fraction=AUTO_MEMORY_FRACTION,
        use_mlx=False,
        mlx_device="gpu",
        block_size=16,
        debug=False,
    )
    worker.model_config = SimpleNamespace(max_model_len=64)
    worker.model_runner = SimpleNamespace(
        num_layers=2,
        num_kv_heads=4,
        head_dim=8,
    )
    return worker


class TestAutoMemoryGuardrails:
    def test_determine_available_memory_returns_one_sequence_budget(self) -> None:
        worker = _make_worker()

        available = worker.determine_available_memory()
        assert available == worker._one_sequence_kv_bytes()
