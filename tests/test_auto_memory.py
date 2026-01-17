# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import psutil
import pytest

from vllm_metal.config import (
    AUTO_MEMORY_FRACTION,
    AUTO_MEMORY_MIN_BLOCKS_BUFFER_FACTOR,
    AUTO_MEMORY_OVERHEAD_FACTOR,
    MetalConfig,
)
from vllm_metal.v1 import worker as v1_worker_module
from vllm_metal.v1.worker import MetalWorker as V1MetalWorker
from vllm_metal.worker import MetalWorker as LegacyMetalWorker


class TestAutoMemoryGuardrails:
    def test_legacy_worker_returns_blocks_when_feasible(self, monkeypatch) -> None:
        worker = LegacyMetalWorker.__new__(LegacyMetalWorker)
        worker.config = MetalConfig(
            memory_fraction=AUTO_MEMORY_FRACTION,
            use_mlx=False,
            mlx_device="gpu",
            block_size=16,
            debug=False,
        )
        worker.vllm_config = None
        worker.model_runner = None
        worker._get_model_memory_usage = lambda: 500_000_000

        monkeypatch.setattr(
            psutil, "virtual_memory", lambda: SimpleNamespace(total=8_000_000_000)
        )

        num_gpu_blocks, num_cpu_blocks = worker.determine_num_available_blocks()
        assert num_cpu_blocks == 0
        assert num_gpu_blocks > 0

    def test_legacy_worker_raises_when_minimal_needed_exceeds_total(
        self, monkeypatch
    ) -> None:
        worker = LegacyMetalWorker.__new__(LegacyMetalWorker)
        worker.config = MetalConfig(
            memory_fraction=AUTO_MEMORY_FRACTION,
            use_mlx=False,
            mlx_device="gpu",
            block_size=16,
            debug=False,
        )
        worker.vllm_config = None
        worker.model_runner = None
        worker._get_model_memory_usage = lambda: 2_000_000_000

        monkeypatch.setattr(
            psutil, "virtual_memory", lambda: SimpleNamespace(total=1_000_000_000)
        )

        with pytest.raises(ValueError, match="Auto memory mode"):
            worker.determine_num_available_blocks()

    def test_v1_worker_sets_mx_memory_limit_in_auto_mode_when_feasible(
        self, monkeypatch
    ) -> None:
        worker = V1MetalWorker.__new__(V1MetalWorker)
        worker.metal_config = MetalConfig(
            memory_fraction=AUTO_MEMORY_FRACTION,
            use_mlx=False,
            mlx_device="gpu",
            block_size=16,
            debug=False,
        )
        worker.model_config = SimpleNamespace(max_model_len=2048)
        worker.get_cache_block_size_bytes = lambda: 4096
        worker._get_model_memory_usage = lambda: 500_000_000

        monkeypatch.setattr(
            psutil, "virtual_memory", lambda: SimpleNamespace(total=8_000_000_000)
        )

        captured = SimpleNamespace(limit=None)

        def _fake_set_memory_limit(limit: int) -> None:
            captured.limit = limit

        monkeypatch.setattr(
            v1_worker_module.mx, "set_memory_limit", _fake_set_memory_limit
        )

        worker._set_auto_memory_limit()
        assert captured.limit is not None

        min_blocks = (
            worker.model_config.max_model_len + worker.metal_config.block_size - 1
        ) // worker.metal_config.block_size
        min_blocks = int(min_blocks * AUTO_MEMORY_MIN_BLOCKS_BUFFER_FACTOR)
        kv_cache_memory = min_blocks * worker.get_cache_block_size_bytes()
        expected = int(
            (worker._get_model_memory_usage() + kv_cache_memory)
            * AUTO_MEMORY_OVERHEAD_FACTOR
        )
        assert captured.limit == expected

    def test_v1_worker_raises_when_minimal_needed_exceeds_total(
        self, monkeypatch
    ) -> None:
        worker = V1MetalWorker.__new__(V1MetalWorker)
        worker.metal_config = MetalConfig(
            memory_fraction=AUTO_MEMORY_FRACTION,
            use_mlx=False,
            mlx_device="gpu",
            block_size=16,
            debug=False,
        )
        worker.model_config = SimpleNamespace(max_model_len=2048)
        worker.get_cache_block_size_bytes = lambda: 4096
        worker._get_model_memory_usage = lambda: 2_000_000_000

        monkeypatch.setattr(
            psutil, "virtual_memory", lambda: SimpleNamespace(total=1_000_000_000)
        )

        with pytest.raises(ValueError, match="Auto memory mode"):
            worker.determine_available_memory()
