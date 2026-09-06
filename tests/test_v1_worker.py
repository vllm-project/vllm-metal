# SPDX-License-Identifier: Apache-2.0
"""Tests for v1 MetalWorker STT boundary delegation."""

from __future__ import annotations

import re
from types import SimpleNamespace
from unittest.mock import MagicMock

import mlx.core as mx
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
from vllm_metal.attention.caches.mha_layout import KV_CACHE_LAYOUT  # noqa: E402
from vllm_metal.attention.runtime.families.gdn import build_gdn_hybrid_plan
from vllm_metal.config import AUTO_MEMORY_FRACTION, MetalConfig
from vllm_metal.stt.policy import STT_SCHED_AVAILABLE_BYTES  # noqa: E402
from vllm_metal.v1 import model_runner as mr  # noqa: E402
from vllm_metal.v1.cache_policy import (  # noqa: E402
    WorkerCachePlanner,
)
from vllm_metal.v1.worker import MetalWorker  # noqa: E402


class TestKVCacheLayoutRpcs:
    """The engine core resolves one KV layout from every worker's list."""

    _MIXED_MHA_SPECS = (
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
            self._MIXED_MHA_SPECS,
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
                self._MIXED_MHA_SPECS,
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
    worker.metal_config = MetalConfig(
        memory_fraction=AUTO_MEMORY_FRACTION,
        mlx_device="gpu",
    )
    worker.cache_config = SimpleNamespace(block_size=16, gpu_memory_utilization=0.92)
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


class TestPagedAttentionPlanDiagnostics:
    def _make_planner(
        self,
        model_runner: object,
        *,
        memory_fraction: float,
        block_size: int = 16,
        per_block_bytes: int = 1,
    ) -> WorkerCachePlanner:
        worker = _make_worker(model_runner)
        worker.cache_config.block_size = block_size
        worker.metal_config.memory_fraction = memory_fraction
        worker.get_cache_block_size_bytes = MagicMock(return_value=per_block_bytes)
        return WorkerCachePlanner(worker)

    def test_hybrid_oom_error_reports_lazy_gdn_state(self, monkeypatch) -> None:
        runner = SimpleNamespace(
            is_hybrid=True,
            scheduler_memory_reporting_mode=MagicMock(
                return_value="paged_attention_capacity"
            ),
            profile_run=MagicMock(return_value=3_900_000_000),
            validate_paged_attention_support=MagicMock(),
            scheduler_config=SimpleNamespace(max_num_seqs=2),
            cache_config=SimpleNamespace(mamba_cache_mode="none"),
            linear_cache_bytes_per_slot=MagicMock(return_value=64_400_000),
            draft_scratch_reserve_bytes=MagicMock(return_value=0),
        )
        worker = _make_worker(runner)
        worker.metal_config.memory_fraction = 0.5
        worker.get_cache_block_size_bytes = MagicMock(return_value=1)
        monkeypatch.setattr(
            WorkerCachePlanner,
            "_metal_limit_bytes",
            lambda self: 10_000_000_000,
        )
        monkeypatch.setattr(
            WorkerCachePlanner,
            "get_model_memory_usage",
            lambda self: 1_000_000_000,
        )

        with pytest.raises(ValueError) as exc_info:
            MetalWorker.determine_available_memory(worker)

        message = str(exc_info.value)
        assert "kv_budget_before_hybrid=0.10GB" in message
        assert "hybrid_gdn_state=lazy" in message
        assert (
            "growth_peak_reserve=0.19GB, 64.4MB/seq * peak_slots=3/max_num_seqs=2"
        ) in message
        assert "kv_budget=-0.09GB" in message
        assert "lower --max-num-seqs" in message
        assert "increase VLLM_METAL_MEMORY_FRACTION" in message
        runner.scheduler_memory_reporting_mode.assert_called_once_with()
        runner.profile_run.assert_called_once_with()
        runner.validate_paged_attention_support.assert_called_once_with()

    def test_hybrid_plan_reserves_bounded_gdn_growth_cushion(self, monkeypatch) -> None:
        runner = SimpleNamespace(
            is_hybrid=True,
            scheduler_config=SimpleNamespace(max_num_seqs=256),
            cache_config=SimpleNamespace(mamba_cache_mode="none"),
            linear_cache_bytes_per_slot=MagicMock(return_value=64_400_000),
            draft_scratch_reserve_bytes=MagicMock(return_value=0),
        )
        planner = self._make_planner(
            runner,
            memory_fraction=0.5,
            per_block_bytes=100_000_000,
        )
        monkeypatch.setattr(
            WorkerCachePlanner,
            "_metal_limit_bytes",
            lambda self: 10_000_000_000,
        )
        monkeypatch.setattr(
            WorkerCachePlanner,
            "get_model_memory_usage",
            lambda self: 1_000_000_000,
        )

        plan = planner._paged_attention_plan(overhead=500_000_000)

        assert plan.base_kv_budget == 3_500_000_000
        assert plan.hybrid_gdn_reservation.bytes_per_slot == 64_400_000
        assert plan.hybrid_gdn_reservation.reserved_slots == 3
        assert plan.hybrid_gdn_reservation.max_num_seqs == 256
        assert plan.hybrid_gdn_reservation.total_bytes == 193_200_000
        assert plan.kv_budget == 3_306_800_000
        assert plan.num_blocks == 33
        breakdown = plan.format_breakdown()
        assert "hybrid_gdn_state=lazy" in breakdown
        assert "kv_budget_before_hybrid=3.50GB" in breakdown
        assert (
            "growth_peak_reserve=0.19GB, 64.4MB/seq * peak_slots=3/max_num_seqs=256"
        ) in breakdown

    def test_hybrid_plan_reserves_one_peak_slot_for_single_sequence(
        self, monkeypatch
    ) -> None:
        runner = SimpleNamespace(
            is_hybrid=True,
            scheduler_config=SimpleNamespace(max_num_seqs=1),
            cache_config=SimpleNamespace(mamba_cache_mode="none"),
            linear_cache_bytes_per_slot=MagicMock(return_value=64_400_000),
            draft_scratch_reserve_bytes=MagicMock(return_value=0),
        )
        planner = self._make_planner(
            runner,
            memory_fraction=0.5,
            per_block_bytes=100_000_000,
        )
        monkeypatch.setattr(
            WorkerCachePlanner,
            "_metal_limit_bytes",
            lambda self: 10_000_000_000,
        )
        monkeypatch.setattr(
            WorkerCachePlanner,
            "get_model_memory_usage",
            lambda self: 1_000_000_000,
        )

        plan = planner._paged_attention_plan(overhead=500_000_000)

        assert plan.hybrid_gdn_reservation.reserved_slots == 1
        assert plan.hybrid_gdn_reservation.total_bytes == 64_400_000
        assert plan.num_blocks == 34

    def test_align_plan_budgets_one_old_physical_pool_for_growth(
        self, monkeypatch
    ) -> None:
        runner = SimpleNamespace(
            is_hybrid=True,
            cache_config=SimpleNamespace(mamba_cache_mode="align"),
            # Six striped pools: 600 steady bytes per block plus one 100-byte
            # old pool retained during growth.
            hybrid_align_state_bytes_per_block=MagicMock(return_value=600),
            hybrid_align_growth_bytes_per_block=MagicMock(return_value=100),
            draft_scratch_reserve_bytes=MagicMock(return_value=0),
        )
        planner = self._make_planner(
            runner,
            memory_fraction=1.0,
            per_block_bytes=100,
        )
        monkeypatch.setattr(
            WorkerCachePlanner,
            "_metal_limit_bytes",
            lambda self: 10_000,
        )
        monkeypatch.setattr(
            WorkerCachePlanner,
            "get_model_memory_usage",
            lambda self: 1_000,
        )

        plan = planner._paged_attention_plan(
            overhead=1_000,
            require_min_blocks=False,
        )

        assert plan.per_block_bytes == 800
        assert plan.hybrid_gdn_reservation.total_bytes == 0
        assert plan.kv_budget == 8_000
        assert plan.num_blocks == 10

    def test_non_hybrid_oom_error_omits_gdn_reservation(self, monkeypatch) -> None:
        runner = SimpleNamespace(
            is_hybrid=False,
            draft_scratch_reserve_bytes=MagicMock(return_value=0),
        )
        planner = self._make_planner(runner, memory_fraction=0.1)
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

        with pytest.raises(ValueError) as exc_info:
            planner._paged_attention_plan(overhead=100_000_000)

        message = str(exc_info.value)
        assert "hybrid_gdn_state" not in message
        assert "kv_budget_before_hybrid" not in message
        assert "--max-num-seqs" not in message
        assert "kv_budget=-1.10GB" in message

    @pytest.mark.parametrize(
        "is_auto, memory_fraction, gpu_mem_util, expected_fraction",
        [
            pytest.param(True, -1.0, 0.92, 0.92, id="auto_uses_vllm_default"),
            pytest.param(True, -1.0, 0.5, 0.5, id="auto_uses_vllm_flag"),
            pytest.param(False, 0.5, 0.7, 0.5, id="metal_env_wins"),
        ],
    )
    def test_memory_fraction_precedence(
        self,
        is_auto: bool,
        memory_fraction: float,
        gpu_mem_util: float,
        expected_fraction: float,
    ) -> None:
        worker = _make_worker(SimpleNamespace(is_hybrid=False))
        worker.metal_config.memory_fraction = (
            AUTO_MEMORY_FRACTION if is_auto else memory_fraction
        )
        worker.cache_config.gpu_memory_utilization = gpu_mem_util

        fraction = WorkerCachePlanner(worker)._memory_fraction()

        assert fraction == expected_fraction


class TestAlignStateSizing:
    @staticmethod
    def _make_align_runner() -> mr.MetalModelRunner:
        """Qwen-shaped 24-layer GDN plan: 18 state layers striped over 6 SDPA."""
        return make_stub_runner(
            is_hybrid=True,
            kv_cache_dtype=mx.float16,
            hybrid_runtime_plan=build_gdn_hybrid_plan(
                {
                    "full_attention_interval": 4,
                    "linear_num_key_heads": 1,
                    "linear_num_value_heads": 1,
                    "linear_key_head_dim": 1,
                    "linear_value_head_dim": 1,
                    "linear_conv_kernel_dim": 2,
                },
                24,
                (torch.float16, torch.float32),
            ),
        )

    def test_align_state_bytes_per_block_stripes_pools(self) -> None:
        runner = self._make_align_runner()
        # Hand-written: conv (2-1)*3 fp16 values = 6 B plus 1*1*1 fp32 = 4 B
        # per layer, one pool per SDPA layer -> 10 B * 6 pools.
        expected = 60

        assert runner.hybrid_align_state_bytes_per_block() == expected

    def test_align_growth_bytes_retain_one_pool(self) -> None:
        runner = self._make_align_runner()
        # Hand-written: one old pool holds one layer's 10 B.
        expected = 10

        assert runner.hybrid_align_growth_bytes_per_block() == expected


class TestHybridPlanGuard:
    def test_hybrid_sizing_without_plan_rejects(self) -> None:
        # The typed config says hybrid; the stub leaves the plan at None.
        runner = make_stub_runner(is_hybrid=True)

        with pytest.raises(RuntimeError, match="no resolved hybrid_runtime_plan"):
            runner.linear_cache_bytes_per_slot()
