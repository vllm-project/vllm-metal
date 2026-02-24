# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from types import SimpleNamespace

import psutil
import pytest

from vllm_metal.config import (
    AUTO_MEMORY_FRACTION,
    MetalConfig,
)
from vllm_metal.v1 import worker as v1_worker_module
from vllm_metal.v1.worker import MetalWorker as V1MetalWorker


def _make_worker(
    *,
    memory_fraction: float = AUTO_MEMORY_FRACTION,
    model_memory: int = 1000,
) -> V1MetalWorker:
    """Create a minimal V1MetalWorker for unit testing."""
    worker = V1MetalWorker.__new__(V1MetalWorker)
    worker.metal_config = MetalConfig(
        memory_fraction=memory_fraction,
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
    worker._get_model_memory_usage = lambda: model_memory
    return worker


class TestAutoMemoryGuardrails:
    def test_v1_worker_sets_mx_memory_limit_in_auto_mode_when_feasible(
        self, monkeypatch
    ) -> None:
        worker = _make_worker()

        monkeypatch.setattr(
            psutil,
            "virtual_memory",
            lambda: SimpleNamespace(total=10_000_000),
        )

        captured = SimpleNamespace(limit=None)

        def _fake_set_memory_limit(limit: int) -> None:
            captured.limit = limit

        monkeypatch.setattr(
            v1_worker_module.mx, "set_memory_limit", _fake_set_memory_limit
        )

        worker._set_auto_memory_limit()
        assert captured.limit is not None

        kv_bytes = worker._one_sequence_kv_bytes()
        expected = int((worker._get_model_memory_usage() + kv_bytes) * 1.2)
        assert captured.limit == expected

    def test_v1_worker_raises_when_minimal_needed_exceeds_total(
        self, monkeypatch
    ) -> None:
        worker = _make_worker(model_memory=10_000_000)

        monkeypatch.setattr(
            psutil,
            "virtual_memory",
            lambda: SimpleNamespace(total=1000),
        )

        with pytest.raises(ValueError, match="Auto memory mode"):
            worker._set_auto_memory_limit()

    def test_determine_available_memory_returns_one_sequence_budget(self) -> None:
        worker = _make_worker()

        available = worker.determine_available_memory()
        assert available == worker._one_sequence_kv_bytes()

    def test_explicit_fraction_warns_without_paged_attention(
        self, caplog
    ) -> None:
        worker = _make_worker(memory_fraction=0.7)

        with caplog.at_level(logging.WARNING):
            available = worker.determine_available_memory()

        assert available == worker._one_sequence_kv_bytes()
        assert "ignored without paged attention" in caplog.text
