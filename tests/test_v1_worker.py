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
from vllm_metal.attention.caches.placement import KV_CACHE_LAYOUT  # noqa: E402
from vllm_metal.attention.runtime.hybrid import HybridPagedAttentionRuntime
from vllm_metal.config import MetalConfig
from vllm_metal.stt.policy import STT_SCHED_AVAILABLE_BYTES  # noqa: E402
from vllm_metal.utils import CommitProbe  # noqa: E402
from vllm_metal.v1.cache_policy import (  # noqa: E402
    KV_COMMIT_SAMPLE_BYTES,
    KV_COMMIT_SWAP_TOLERANCE_DIVISOR,
    WorkerCachePlanner,
    kv_pool_bytes_after_probe,
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
    @pytest.mark.parametrize("length,sequences", [(512, 1), (4096, 4), (131072, 1)])
    def test_tq_auto_reservation_covers_history_and_stays_fixed(
        self, monkeypatch, length, sequences
    ) -> None:
        from vllm_metal.attention.caches.turboquant import prefill_workspace_bytes

        monkeypatch.setenv("VLLM_METAL_TQ_PREFILL", "1")
        monkeypatch.setenv("VLLM_METAL_TQ_PREFILL_MAX_MIB", "auto")
        monkeypatch.setattr(
            "vllm_metal.v1.cache_policy.get_config",
            lambda: MetalConfig(mlx_device="gpu", turboquant=True),
        )
        runner = make_stub_runner(
            num_kv_heads=2,
            head_dim=128,
            kv_cache_dtype=mx.bfloat16,
            scheduler_config=SimpleNamespace(
                max_num_seqs=sequences, max_num_batched_tokens=sequences * 128
            ),
        )
        # vLLM resolves the model limit before planning even for --max-model-len=-1.
        runner.model_config.original_max_model_len = -1
        runner.model_config.max_model_len = length
        runner.model_config.get_num_attention_heads = lambda _: 8
        allowance = runner.tq_prefill_workspace_bytes
        # All independent histories can be read by this step, even though its
        # new-token budget is far smaller than their combined context length.
        assert allowance >= sequences * length * 2 * 2 * 128 * 2
        assert allowance < prefill_workspace_bytes()

        planner = self._make_planner(runner, gpu_memory_utilization=0.5)
        monkeypatch.setattr(WorkerCachePlanner, "_metal_limit_bytes", lambda _: 10**10)
        monkeypatch.setattr(WorkerCachePlanner, "get_model_memory_usage", lambda _: 0)
        # A later auto-fit shrinks the config, not the already reserved workspace.
        runner.model_config.max_model_len = length // 2
        monkeypatch.setenv("VLLM_METAL_TQ_PREFILL_MAX_MIB", "0")
        assert runner.tq_prefill_workspace_bytes == allowance
        plan = planner._paged_attention_plan(overhead=0)
        assert plan.overhead == allowance
        assert plan.kv_budget == 5 * 10**9 - allowance

    @pytest.mark.parametrize(
        "head_dim,kv_cache_dtype", [(96, mx.bfloat16), (128, mx.float32)]
    )
    def test_tq_cap_zero_when_dtype_or_head_dim_unsupported(
        self, monkeypatch, head_dim, kv_cache_dtype
    ) -> None:
        """Unsupported dtype/head_dim resolves cap=0 instead of an allowance."""
        monkeypatch.setenv("VLLM_METAL_TQ_PREFILL", "1")
        monkeypatch.setenv("VLLM_METAL_TQ_PREFILL_MAX_MIB", "auto")
        monkeypatch.setattr(
            "vllm_metal.v1.cache_policy.get_config",
            lambda: MetalConfig(mlx_device="gpu", turboquant=True),
        )
        runner = make_stub_runner(
            num_kv_heads=2,
            head_dim=head_dim,
            kv_cache_dtype=kv_cache_dtype,
            scheduler_config=SimpleNamespace(
                max_num_seqs=4, max_num_batched_tokens=512
            ),
        )
        assert runner.tq_prefill_workspace_bytes == 0

    def test_tq_speculative_keeps_the_full_auto_allowance(self, monkeypatch) -> None:
        """cap stays None under speculation: the reservation is not bounded."""
        from vllm_metal.attention.caches.turboquant import prefill_workspace_bytes

        monkeypatch.setenv("VLLM_METAL_TQ_PREFILL", "1")
        monkeypatch.setenv("VLLM_METAL_TQ_PREFILL_MAX_MIB", "auto")
        monkeypatch.setattr(
            "vllm_metal.v1.cache_policy.get_config",
            lambda: MetalConfig(mlx_device="gpu", turboquant=True),
        )
        runner = make_stub_runner(
            num_kv_heads=2,
            head_dim=128,
            kv_cache_dtype=mx.bfloat16,
            scheduler_config=SimpleNamespace(
                max_num_seqs=4, max_num_batched_tokens=512
            ),
        )
        runner.vllm_config.speculative_config = object()
        assert runner.tq_prefill_workspace_bytes == prefill_workspace_bytes()

    @pytest.mark.parametrize(
        "turboquant,mode,expected_mib",
        [(True, "1", 64), (True, "0", 0), (False, "1", 0)],
    )
    def test_tq_workspace_is_reserved_inside_memory_fraction(
        self, monkeypatch, turboquant, mode, expected_mib
    ) -> None:
        monkeypatch.setenv("VLLM_METAL_TQ_PREFILL", mode)
        monkeypatch.setenv("VLLM_METAL_TQ_PREFILL_MAX_MIB", "64")
        monkeypatch.setattr(
            "vllm_metal.v1.cache_policy.get_config",
            lambda: MetalConfig(mlx_device="gpu", turboquant=turboquant),
        )
        runner = make_stub_runner(
            num_kv_heads=2,
            head_dim=128,
            kv_cache_dtype=mx.bfloat16,
            scheduler_config=SimpleNamespace(
                max_num_seqs=4, max_num_batched_tokens=2048
            ),
        )
        runner.model_config.get_num_attention_heads = lambda _: 8
        planner = self._make_planner(runner, gpu_memory_utilization=0.5)
        monkeypatch.setattr(
            WorkerCachePlanner, "_metal_limit_bytes", lambda self: 10_000_000_000
        )
        monkeypatch.setattr(
            WorkerCachePlanner, "get_model_memory_usage", lambda self: 2_000_000_000
        )
        plan = planner._paged_attention_plan(overhead=100_000_000)
        assert plan.kv_budget == 2_900_000_000 - expected_mib * 2**20
        assert plan.overhead == 100_000_000 + expected_mib * 2**20

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

    def test_capacity_path_refuses_a_runtime_it_cannot_initialize(
        self, monkeypatch
    ) -> None:
        # Hybrid models always take the layout-budget path; a hybrid runtime on
        # the capacity path is a routing bug and must say so, not raise an
        # AttributeError for the missing ``initialize``.
        hybrid = HybridPagedAttentionRuntime.__new__(HybridPagedAttentionRuntime)
        runner = SimpleNamespace(
            validate_paged_attention_support=lambda: None,
            build_paged_attention_runtime=lambda *, block_size: hybrid,
        )
        planner = WorkerCachePlanner(_make_worker(runner))
        plan = SimpleNamespace(
            block_size=16,
            num_blocks=64,
            per_block_bytes=1,
            format_breakdown=lambda: "stub",
        )
        monkeypatch.setattr(planner, "_paged_attention_plan", lambda **_: plan)
        monkeypatch.setattr(
            planner, "_validate_paged_attention_plan", lambda *a, **k: None
        )

        with pytest.raises(RuntimeError, match="HybridPagedAttentionRuntime"):
            planner.setup_paged_attention(overhead=0)


# The planner's budget before the commit probe: 10 GB * 0.5 - 2 GB weights -
# 0.1 GB overhead. The reserve the probe holds back is max(1 GiB, 10 GB / 16).
_PLANNED_KV_BUDGET = 2_900_000_000
_PROBE_RESERVE = 1 << 30


class TestCommitProbeBudget:
    """The startup probe tells the machine's answer without capping the plan.

    The pool is allocated lazily, so without the probe a plan the machine
    cannot hold stays invisible until a request writes a block -- a swap storm
    or a jetsam kill mid-generation instead of an answer at load time.
    """

    def _plan(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        free_bytes: int,
        swap_growth: int = 0,
        per_block_bytes: int = 1_000_000,
    ):
        worker = _make_worker(SimpleNamespace(is_hybrid=False))
        worker.cache_config.gpu_memory_utilization = 0.5
        worker.get_cache_block_size_bytes = MagicMock(return_value=per_block_bytes)
        planner = WorkerCachePlanner(worker)
        monkeypatch.setattr(
            WorkerCachePlanner, "_metal_limit_bytes", lambda self: 10_000_000_000
        )
        monkeypatch.setattr(
            WorkerCachePlanner, "get_model_memory_usage", lambda self: 2_000_000_000
        )
        monkeypatch.setenv("VLLM_METAL_KV_COMMIT_PROBE", "1")

        def fake_probe(nbytes: int) -> CommitProbe:
            assert nbytes == min(_PLANNED_KV_BUDGET, KV_COMMIT_SAMPLE_BYTES)
            return CommitProbe(
                probed_bytes=nbytes,
                swap_before=1_000,
                swap_after=1_000 + swap_growth,
                available_before=free_bytes,
                available_after=free_bytes,
                seconds=0.01,
            )

        monkeypatch.setattr("vllm_metal.v1.cache_policy.probe_commit", fake_probe)
        return planner._paged_attention_plan(overhead=100_000_000)

    def test_pool_that_fits_free_memory_is_left_alone(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        plan = self._plan(monkeypatch, free_bytes=8 << 30)

        assert plan.kv_budget == _PLANNED_KV_BUDGET
        assert plan.num_blocks == 2_900

    def test_busy_machine_keeps_the_capacity_it_was_given(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Free memory below the plan is reported, not charged to capacity.

        The pool backs blocks as requests use them, so an idle pool never needs
        the blocks the plan allows; shrinking to today's free memory would hand
        back exactly the capacity the lazy allocation exists to keep.
        """
        caplog.set_level("WARNING", logger="vllm_metal.v1.cache_policy")

        plan = self._plan(monkeypatch, free_bytes=2 << 30)

        assert plan.kv_budget == _PLANNED_KV_BUDGET
        assert plan.num_blocks == 2_900
        assert "larger than this machine's free memory" in caplog.text

    def test_a_machine_that_pages_is_charged_to_capacity(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Paging to back the sample is the machine saying it has no headroom."""
        caplog.set_level("WARNING", logger="vllm_metal.v1.cache_policy")

        plan = self._plan(
            monkeypatch,
            free_bytes=2 << 30,
            swap_growth=(KV_COMMIT_SAMPLE_BYTES // KV_COMMIT_SWAP_TOLERANCE_DIVISOR)
            + 1,
        )

        # Free memory less the reserve, rounded down to whole blocks.
        assert plan.kv_budget == 1_073_000_000
        assert plan.num_blocks == 1_073
        # Everything the plan was built from is untouched: vLLM sizes its
        # layout from the bytes it is handed.
        assert plan.per_block_bytes == 1_000_000
        assert plan.fraction == 0.5
        assert "sized down" in caplog.text

    def test_background_paging_is_not_pressure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Other processes paging out a few MB must not cost the pool capacity."""
        plan = self._plan(monkeypatch, free_bytes=2 << 30, swap_growth=8 << 20)

        assert plan.kv_budget == _PLANNED_KV_BUDGET

    def test_probe_can_be_switched_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        worker = _make_worker(SimpleNamespace(is_hybrid=False))
        worker.cache_config.gpu_memory_utilization = 0.5
        worker.get_cache_block_size_bytes = MagicMock(return_value=1_000_000)
        planner = WorkerCachePlanner(worker)
        monkeypatch.setattr(
            WorkerCachePlanner, "_metal_limit_bytes", lambda self: 10_000_000_000
        )
        monkeypatch.setattr(
            WorkerCachePlanner, "get_model_memory_usage", lambda self: 2_000_000_000
        )
        monkeypatch.setenv("VLLM_METAL_KV_COMMIT_PROBE", "0")
        monkeypatch.setattr(
            "vllm_metal.v1.cache_policy.probe_commit",
            lambda nbytes: pytest.fail("probe ran while switched off"),
        )

        plan = planner._paged_attention_plan(overhead=100_000_000)

        assert plan.kv_budget == _PLANNED_KV_BUDGET
        assert plan.num_blocks == 2_900


class TestKvPoolBytesAfterProbe:
    """The probe's arithmetic, where a plan starts costing capacity."""

    _TOLERANCE = (512 << 20) // KV_COMMIT_SWAP_TOLERANCE_DIVISOR

    @staticmethod
    def _probe(available_before: int, probed_bytes: int, swap_growth: int):
        return CommitProbe(
            probed_bytes=probed_bytes,
            swap_before=0,
            swap_after=swap_growth,
            available_before=available_before,
            available_after=available_before,
            seconds=0.0,
        )

    def test_free_memory_short_of_the_plan_is_not_a_reason_to_shrink(self) -> None:
        plan = 6 << 30
        probe = self._probe(2 << 30, 512 << 20, 0)

        assert (
            kv_pool_bytes_after_probe(
                plan,
                probe,
                reserve_bytes=1 << 30,
                swap_tolerance_bytes=self._TOLERANCE,
            )
            == plan
        )

    def test_paging_beyond_the_tolerance_caps_at_free_memory(self) -> None:
        plan = 6 << 30
        free = 3 << 30
        probe = self._probe(free, 512 << 20, self._TOLERANCE + 1)

        assert kv_pool_bytes_after_probe(
            plan,
            probe,
            reserve_bytes=1 << 30,
            swap_tolerance_bytes=self._TOLERANCE,
        ) == free - (1 << 30)

    def test_paging_inside_the_tolerance_is_noise(self) -> None:
        plan = 6 << 30
        probe = self._probe(2 << 30, 512 << 20, self._TOLERANCE)

        assert (
            kv_pool_bytes_after_probe(
                plan,
                probe,
                reserve_bytes=1 << 30,
                swap_tolerance_bytes=self._TOLERANCE,
            )
            == plan
        )
