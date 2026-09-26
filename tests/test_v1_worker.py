# SPDX-License-Identifier: Apache-2.0
"""Tests for v1 MetalWorker STT boundary delegation."""

from __future__ import annotations

import re
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

pytest.importorskip("vllm", reason="vllm not installed")

import vllm.v1.worker.worker_base as worker_base  # noqa: E402
from vllm.config import CacheConfig, VllmConfig  # noqa: E402
from vllm.v1.attention.backends.utils import resolve_kv_cache_layout  # noqa: E402
from vllm.v1.kv_cache_interface import (  # noqa: E402
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheLayout,
    SlidingWindowSpec,
)

from tests.stub_runner import make_stub_runner  # noqa: E402
from vllm_metal.attention.caches.placement import KV_CACHE_LAYOUT  # noqa: E402
from vllm_metal.config import MetalConfig
from vllm_metal.stt.policy import STT_SCHED_AVAILABLE_BYTES  # noqa: E402
from vllm_metal.v1.cache_policy import (  # noqa: E402
    WorkerCachePlanner,
)
from vllm_metal.v1.worker import MetalWorker  # noqa: E402


class TestKVCacheLayoutRpcs:
    """The engine core resolves one KV layout from every worker's list."""

    _MIXED_ATTENTION_SPECS = (
        FullAttentionSpec(
            block_size=32, num_kv_heads=4, head_size=512, dtype=torch.bfloat16
        ),
        SlidingWindowSpec(
            block_size=16,
            num_kv_heads=16,
            head_size=256,
            dtype=torch.bfloat16,
            sliding_window=1024,
        ),
    )

    def test_engine_resolves_metal_page_order_without_backend_probe(
        self, monkeypatch
    ) -> None:
        worker = _make_worker(SimpleNamespace())
        monkeypatch.setattr(
            worker_base, "get_current_attn_backends", _refuse_backend_probe
        )
        vllm_config = VllmConfig()

        layout = resolve_kv_cache_layout(
            vllm_config,
            [worker.get_supported_kv_cache_layouts()],
            self._MIXED_ATTENTION_SPECS,
        )

        assert layout is KVCacheLayout.LBNHC
        assert vllm_config.cache_config.kv_cache_layout == KV_CACHE_LAYOUT

    def test_explicit_block_outermost_layout_fails_loud(self, monkeypatch) -> None:
        worker = _make_worker(SimpleNamespace())
        monkeypatch.setenv("VLLM_KV_CACHE_LAYOUT", "BLHNC")

        with pytest.raises(
            ValueError,
            match=re.escape(
                "VLLM_KV_CACHE_LAYOUT=BLHNC does not satisfy every supported set; "
                "valid layouts: ['LBNHC']."
            ),
        ):
            resolve_kv_cache_layout(
                VllmConfig(),
                [worker.get_supported_kv_cache_layouts()],
                self._MIXED_ATTENTION_SPECS,
            )

    def test_initialize_from_config_records_resolved_layout(self) -> None:
        runner = SimpleNamespace(initialize_kv_cache=MagicMock())
        worker = _make_worker(runner)
        worker.cache_config = CacheConfig()
        kv_cache_config = KVCacheConfig(
            num_blocks=1,
            kv_cache_tensors=[],
            kv_cache_groups=[],
            kv_cache_layout=KV_CACHE_LAYOUT,
        )

        worker.initialize_from_config(kv_cache_config)

        assert worker.cache_config.kv_cache_layout == KV_CACHE_LAYOUT
        runner.initialize_kv_cache.assert_called_once_with(kv_cache_config)


def _refuse_backend_probe(vllm_config: object) -> None:
    raise AssertionError("worker answered the layout RPC via the backend probe")


def _make_worker(model_runner: object) -> MetalWorker:
    worker = MetalWorker.__new__(MetalWorker)
    worker.model_runner = model_runner  # type: ignore[assignment]
    worker.metal_config = MetalConfig(mlx_device="gpu")
    worker.cache_config = SimpleNamespace(
        block_size=16,
        gpu_memory_utilization=0.92,
        num_gpu_blocks_override=None,
    )
    worker.vllm_config = SimpleNamespace(cache_config=worker.cache_config)
    return worker


class TestWorkerRunnerBoundaryDelegation:
    """Worker should honor model runner memory-reporting modes."""

    @pytest.mark.parametrize("revision", [None, "release-tag", "a" * 40])
    def test_init_device_preserves_stt_revision(self, monkeypatch, revision) -> None:
        from vllm_metal.v1 import worker as worker_module
        from vllm_metal.v1.stt_model_runner import STTModelRunner

        worker = _make_worker(None)
        worker.model_config = SimpleNamespace(
            model="org/model", revision=revision, seed=0
        )
        worker.vllm_config.model_config = worker.model_config
        worker.vllm_config.scheduler_config = SimpleNamespace()
        worker.parallel_config = SimpleNamespace(pipeline_parallel_size=1)
        worker.rank = worker.local_rank = 0
        worker.distributed_init_method = "unused"
        monkeypatch.setattr(worker_module, "set_wired_limit", lambda: None)
        monkeypatch.setattr(
            worker_module, "init_worker_distributed_environment", lambda *_: None
        )
        resolve = MagicMock(return_value="org/model")
        detect = MagicMock(return_value=True)
        monkeypatch.setattr("vllm_metal.utils.get_model_download_path", resolve)
        monkeypatch.setattr("vllm_metal.stt.detection.is_stt_model", detect)

        worker.init_device()

        assert isinstance(worker.model_runner, STTModelRunner)
        resolve.assert_called_once_with("org/model", revision=revision)
        detect.assert_called_once_with("org/model", revision=revision)

    def test_determine_available_memory_stt_nominal_mode(self) -> None:
        model_runner = SimpleNamespace(
            scheduler_memory_reporting_mode=MagicMock(return_value="stt_nominal"),
        )
        worker = _make_worker(model_runner)

        available = MetalWorker.determine_available_memory(worker)

        assert available == STT_SCHED_AVAILABLE_BYTES
        model_runner.scheduler_memory_reporting_mode.assert_called_once_with()

    def test_determine_available_memory_paged_capacity_mode(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        num_blocks = 8
        block_size_bytes = 16
        measured_overhead = 200 * 1024 * 1024
        model_runner = SimpleNamespace(
            scheduler_memory_reporting_mode=MagicMock(
                return_value="paged_attention_capacity"
            ),
            profile_run=MagicMock(return_value=measured_overhead),
            paged_attention_runtime=None,
        )
        worker = _make_worker(model_runner)
        worker.get_cache_block_size_bytes = MagicMock(return_value=block_size_bytes)

        def _fake_setup(*, overhead: int) -> None:
            model_runner.paged_attention_runtime = SimpleNamespace(
                num_blocks=lambda: num_blocks
            )

        setup_paged_attention = MagicMock(side_effect=_fake_setup)
        monkeypatch.setattr(
            WorkerCachePlanner,
            "setup_paged_attention",
            setup_paged_attention,
        )

        available = MetalWorker.determine_available_memory(worker)

        assert available == num_blocks * block_size_bytes
        model_runner.profile_run.assert_called_once_with()
        setup_paged_attention.assert_called_once_with(overhead=measured_overhead)
        worker.get_cache_block_size_bytes.assert_called_once_with()

    def _make_layout_budget_worker(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        gpu_memory_utilization: float,
        metal_limit: int = 10_000_000_000,
        model_memory: int = 2_000_000_000,
        overhead: int = 100_000_000,
    ) -> MetalWorker:
        model_runner = SimpleNamespace(
            scheduler_memory_reporting_mode=MagicMock(
                return_value="paged_attention_layout_budget"
            ),
            profile_run=MagicMock(return_value=overhead),
            draft_scratch_reserve_bytes=MagicMock(return_value=0),
        )
        worker = _make_worker(model_runner)
        worker.cache_config.gpu_memory_utilization = gpu_memory_utilization
        worker.get_cache_block_size_bytes = MagicMock(return_value=1)
        monkeypatch.setattr(
            WorkerCachePlanner, "_metal_limit_bytes", lambda self: metal_limit
        )
        monkeypatch.setattr(
            WorkerCachePlanner, "get_model_memory_usage", lambda self: model_memory
        )
        return worker

    def test_determine_available_memory_layout_budget_mode(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        worker = self._make_layout_budget_worker(
            monkeypatch, gpu_memory_utilization=0.5
        )

        available = MetalWorker.determine_available_memory(worker)

        # 10 GB * 0.5 - 2 GB weights - 0.1 GB overhead.
        assert available == 2_900_000_000
        worker.model_runner.profile_run.assert_called_once_with()

    def test_layout_budget_oom_reports_breakdown_and_mitigations(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A budget that cannot fit fails on Metal's own diagnostics.

        The upstream-storage path used to hand a negative budget straight
        to vLLM, whose generic error names none of the terms below.
        """
        worker = self._make_layout_budget_worker(
            monkeypatch, gpu_memory_utilization=0.15
        )

        with pytest.raises(ValueError) as exc_info:
            MetalWorker.determine_available_memory(worker)

        message = str(exc_info.value)
        assert "not enough Metal memory for KV cache" in message
        assert "fraction=0.15" in message
        assert "model_memory=2.00GB" in message
        assert "overhead=0.10GB" in message
        assert "kv_budget=-0.60GB" in message
        assert "increase --gpu-memory-utilization (currently 0.15)" in message

    def test_layout_budget_skips_dense_block_floor(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A small positive budget is reported, not rejected, on this path.

        The per-block size here is the dense estimate; vLLM chooses the real
        layout afterwards and a few dense blocks of budget can still yield a
        valid grouped layout (see the Gemma4 grouped-layout tests).
        """
        worker = self._make_layout_budget_worker(
            monkeypatch, gpu_memory_utilization=0.5
        )
        # 2.9 GB budget / 1 GB dense blocks = 2 blocks, under the 16-block
        # floor the capacity path enforces.
        worker.get_cache_block_size_bytes = MagicMock(return_value=1_000_000_000)

        assert MetalWorker.determine_available_memory(worker) == 2_900_000_000


class TestPagedAttentionPlanDiagnostics:
    def _make_planner(
        self,
        model_runner: object,
        *,
        gpu_memory_utilization: float,
        block_size: int = 16,
        per_block_bytes: int = 1,
    ) -> WorkerCachePlanner:
        worker = _make_worker(model_runner)
        worker.cache_config.block_size = block_size
        worker.cache_config.gpu_memory_utilization = gpu_memory_utilization
        worker.get_cache_block_size_bytes = MagicMock(return_value=per_block_bytes)
        return WorkerCachePlanner(worker)

    def test_oom_mitigation_names_gpu_memory_utilization(self, monkeypatch) -> None:
        runner = SimpleNamespace(
            is_hybrid=False,
            draft_scratch_reserve_bytes=MagicMock(return_value=0),
        )
        planner = self._make_planner(runner, gpu_memory_utilization=0.15)
        monkeypatch.setattr(
            WorkerCachePlanner,
            "_metal_limit_bytes",
            lambda self: 10_000_000_000,
        )
        monkeypatch.setattr(
            WorkerCachePlanner,
            "get_model_memory_usage",
            lambda self: 2_000_000_000,
        )

        plan = planner._paged_attention_plan(overhead=100_000_000)
        with pytest.raises(ValueError) as exc_info:
            planner._validate_paged_attention_plan(plan, require_min_blocks=True)

        message = str(exc_info.value)
        assert "increase --gpu-memory-utilization (currently 0.15)" in message

    def test_non_hybrid_oom_error_omits_gdn_reservation(self, monkeypatch) -> None:
        runner = SimpleNamespace(
            is_hybrid=False,
            draft_scratch_reserve_bytes=MagicMock(return_value=0),
        )
        planner = self._make_planner(runner, gpu_memory_utilization=0.1)
        monkeypatch.setattr(
            WorkerCachePlanner,
            "_metal_limit_bytes",
            lambda self: 10_000_000_000,
        )
        monkeypatch.setattr(
            WorkerCachePlanner,
            "get_model_memory_usage",
            lambda self: 2_000_000_000,
        )

        plan = planner._paged_attention_plan(overhead=100_000_000)
        with pytest.raises(ValueError) as exc_info:
            planner._validate_paged_attention_plan(plan, require_min_blocks=True)

        message = str(exc_info.value)
        assert "hybrid_gdn_state" not in message
        assert "kv_budget_before_hybrid" not in message
        assert "--max-num-seqs" not in message
        assert "kv_budget=-1.10GB" in message

    @pytest.mark.parametrize(
        "gpu_mem_util",
        [
            pytest.param(0.92, id="vllm_default"),
            pytest.param(0.5, id="vllm_flag"),
        ],
    )
    def test_memory_fraction_follows_gpu_memory_utilization(
        self, gpu_mem_util: float
    ) -> None:
        worker = _make_worker(SimpleNamespace(is_hybrid=False))
        worker.cache_config.gpu_memory_utilization = gpu_mem_util

        fraction = WorkerCachePlanner(worker)._memory_fraction()

        assert fraction == gpu_mem_util


class TestHybridPlanGuard:
    def test_hybrid_sizing_without_plan_rejects(self) -> None:
        # The typed config says hybrid; the stub leaves the plan at None.
        runner = make_stub_runner(is_hybrid=True)

        with pytest.raises(RuntimeError, match="no resolved hybrid_runtime_plan"):
            runner.build_paged_attention_runtime(block_size=16)
