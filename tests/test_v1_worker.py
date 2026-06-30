# SPDX-License-Identifier: Apache-2.0
"""Tests for v1 MetalWorker STT boundary delegation."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import mlx.core as mx
import pytest

pytest.importorskip("vllm", reason="vllm not installed")

from tests.stub_runner import make_stub_runner  # noqa: E402
from vllm_metal.stt.policy import STT_SCHED_AVAILABLE_BYTES  # noqa: E402
from vllm_metal.v1 import model_runner as mr  # noqa: E402
from vllm_metal.v1.cache_policy import (  # noqa: E402
    ModelCachePolicy,
    WorkerCachePlanner,
)
from vllm_metal.v1.model_adapter import DefaultModelAdapter  # noqa: E402
from vllm_metal.v1.worker import MetalWorker  # noqa: E402


def _make_worker(model_runner: object, *, use_paged_attention: bool) -> MetalWorker:
    worker = MetalWorker.__new__(MetalWorker)
    worker.model_runner = model_runner  # type: ignore[assignment]
    worker.metal_config = SimpleNamespace(use_paged_attention=use_paged_attention)
    worker.cache_config = SimpleNamespace(block_size=16)
    worker.vllm_config = SimpleNamespace(cache_config=worker.cache_config)
    return worker


class TestWorkerRunnerBoundaryDelegation:
    """Worker should honor model runner memory-reporting modes."""

    def test_determine_available_memory_stt_nominal_mode(self) -> None:
        model_runner = SimpleNamespace(
            scheduler_memory_reporting_mode=MagicMock(return_value="stt_nominal"),
        )
        worker = _make_worker(model_runner, use_paged_attention=True)

        available = MetalWorker.determine_available_memory(worker)

        assert available == STT_SCHED_AVAILABLE_BYTES
        model_runner.scheduler_memory_reporting_mode.assert_called_once_with(
            paged_attention_enabled=True
        )

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
        worker = _make_worker(model_runner, use_paged_attention=True)
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

    def test_determine_available_memory_single_sequence_mode(self) -> None:
        """Test MLX path returns one max-length sequence estimate (PR #229)."""
        model_runner = make_stub_runner(
            num_layers=16,
            num_kv_cache_layers=16,
            num_kv_heads=8,
            head_dim=128,
            kv_cache_dtype=mx.float16,
        )
        model_runner.scheduler_memory_reporting_mode = MagicMock(
            return_value="single_sequence_estimate"
        )
        worker = _make_worker(model_runner, use_paged_attention=False)
        worker.model_config = SimpleNamespace(max_model_len=2048)

        try:
            available = MetalWorker.determine_available_memory(worker)

            # Should return one max-length sequence KV cache bytes
            # 2 (K+V) * 16 layers * 2048 tokens * 8 heads * 128 head_dim * 2 bytes
            expected = 2 * 16 * 2048 * 8 * 128 * 2
            assert available == expected
        finally:
            pass


class TestOneSequenceKvBytes:
    """_one_sequence_kv_bytes must account for hybrid linear state and block alignment."""

    def test_non_hybrid_counts_all_layers(self) -> None:
        model_runner = make_stub_runner(
            num_layers=16,
            num_kv_cache_layers=16,
            num_kv_heads=8,
            head_dim=64,
            kv_cache_dtype=mx.float16,
        )
        worker = _make_worker(model_runner, use_paged_attention=False)
        worker.model_config = SimpleNamespace(max_model_len=2048)
        # block_size=16 divides 2048 evenly, so no padding
        worker.vllm_config = SimpleNamespace(
            cache_config=SimpleNamespace(block_size=16)
        )

        # Act
        result = MetalWorker._one_sequence_kv_bytes(worker)

        # Assert — 2 * 16 * 2048 * 8 * 64 * 2
        assert result == 2 * 16 * 2048 * 8 * 64 * 2

    def test_hybrid_adds_linear_state(self) -> None:
        model_runner = make_stub_runner(
            model_args={"full_attention_interval": 2},
            num_sdpa_layers=8,
            num_kv_heads=4,
            head_dim=256,
            kv_cache_dtype=mx.float16,
            linear_conv_kernel_dim=3,
            linear_conv_dim=5,
            linear_num_v_heads=2,
            linear_value_head_dim=7,
            linear_key_head_dim=11,
            num_linear_layers=3,
        )
        worker = _make_worker(model_runner, use_paged_attention=False)
        worker.model_config = SimpleNamespace(max_model_len=2048)
        worker.vllm_config = SimpleNamespace(
            cache_config=SimpleNamespace(block_size=16)
        )

        # Act
        result = MetalWorker._one_sequence_kv_bytes(worker)

        # Assert — SDPA bytes + linear state
        sdpa_bytes = 2 * 8 * 2048 * 4 * 256 * 2
        conv_bytes = (3 - 1) * 5 * mx.float16.size
        recurrent_bytes = 2 * 7 * 11 * mx.float32.size
        linear_bytes = 3 * (conv_bytes + recurrent_bytes)
        assert result == sdpa_bytes + linear_bytes

    def test_linear_cache_bytes_uses_float32_recurrent(self) -> None:
        runner = mr.MetalModelRunner.__new__(mr.MetalModelRunner)
        runner.model_args = {"full_attention_interval": 2}
        runner._model_adapter = DefaultModelAdapter()
        runner._cache_policy = ModelCachePolicy(runner, runner._model_adapter)
        runner.kv_cache_dtype = mx.float16
        runner.linear_conv_kernel_dim = 3
        runner.linear_conv_dim = 5
        runner.linear_num_v_heads = 2
        runner.linear_value_head_dim = 7
        runner.linear_key_head_dim = 11
        runner.num_linear_layers = 3

        conv_bytes = (
            (runner.linear_conv_kernel_dim - 1)
            * runner.linear_conv_dim
            * mx.float16.size
        )
        recurrent_bytes = (
            runner.linear_num_v_heads
            * runner.linear_value_head_dim
            * runner.linear_key_head_dim
            * mx.float32.size
        )
        expected = runner.num_linear_layers * (conv_bytes + recurrent_bytes)

        assert runner.linear_cache_bytes_per_slot() == expected

    def test_block_alignment_rounds_up_token_count(self) -> None:
        """When block_size doesn't divide max_model_len evenly, the token
        count must be rounded up to the next block boundary so that the
        reported bytes match the scheduler's block-aligned accounting.

        This reproduces the KV cache startup failure seen with Mamba-hybrid
        models (e.g. Granite 4.0-H) where the attention block_size is padded
        to 400 to match the mamba page size.
        """
        model_runner = make_stub_runner(
            num_layers=4,
            num_kv_cache_layers=4,
            num_kv_heads=4,
            head_dim=64,
            kv_cache_dtype=mx.float16,
        )
        worker = _make_worker(model_runner, use_paged_attention=False)
        worker.model_config = SimpleNamespace(max_model_len=2048)
        # block_size=400 (Mamba-hybrid): ceil(2048/400)=6, 6*400=2400 tokens
        worker.vllm_config = SimpleNamespace(
            cache_config=SimpleNamespace(block_size=400)
        )

        result = MetalWorker._one_sequence_kv_bytes(worker)

        # Should use 2400 tokens (block-aligned), not 2048
        aligned_tokens = 2400  # ceil(2048/400) * 400
        expected = 2 * 4 * aligned_tokens * 4 * 64 * 2
        assert result == expected
        # Verify this is strictly more than the unaligned calculation
        unaligned = 2 * 4 * 2048 * 4 * 64 * 2
        assert result > unaligned

    def test_mla_uses_latent_only(self) -> None:
        """MLA cache stores one latent vector per token, not K+V.

        head_dim=576 represents kv_lora_rank + qk_rope_head_dim (e.g. GLM-4).
        The 2x K/V factor must NOT be applied — kv_factor=1.
        """
        model_runner = make_stub_runner(
            model_args={"kv_lora_rank": 512},
            num_layers=4,
            num_kv_cache_layers=4,
            num_kv_heads=1,
            head_dim=576,
            kv_cache_dtype=mx.float16,
        )
        worker = _make_worker(model_runner, use_paged_attention=False)
        worker.model_config = SimpleNamespace(max_model_len=2048)
        worker.vllm_config = SimpleNamespace(
            cache_config=SimpleNamespace(block_size=16)
        )

        result = MetalWorker._one_sequence_kv_bytes(worker)

        expected = 1 * 4 * 2048 * 1 * 576 * 2
        assert result == expected

    def test_yoco_uses_unique_cache_layers(self) -> None:
        model_runner = make_stub_runner(
            num_layers=28,
            num_kv_cache_layers=24,
            num_kv_heads=4,
            head_dim=256,
            kv_cache_dtype=mx.float16,
        )
        worker = _make_worker(model_runner, use_paged_attention=False)
        worker.model_config = SimpleNamespace(max_model_len=2048)
        worker.vllm_config = SimpleNamespace(
            cache_config=SimpleNamespace(block_size=16)
        )

        result = MetalWorker._one_sequence_kv_bytes(worker)

        expected = 2 * 24 * 2048 * 4 * 256 * 2
        assert result == expected


class TestPagedAttentionPlanDiagnostics:
    def _make_planner(
        self,
        model_runner: object,
        *,
        memory_fraction: float,
        block_size: int = 16,
        per_block_bytes: int = 1,
    ) -> WorkerCachePlanner:
        worker = _make_worker(model_runner, use_paged_attention=True)
        worker.cache_config.block_size = block_size
        worker.metal_config.is_auto_memory = False
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
            linear_cache_bytes_per_slot=MagicMock(return_value=64_400_000),
        )
        worker = _make_worker(runner, use_paged_attention=True)
        worker.metal_config.is_auto_memory = False
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
        runner.scheduler_memory_reporting_mode.assert_called_once_with(
            paged_attention_enabled=True
        )
        runner.profile_run.assert_called_once_with()
        runner.validate_paged_attention_support.assert_called_once_with()

    def test_hybrid_plan_reserves_bounded_gdn_growth_cushion(self, monkeypatch) -> None:
        runner = SimpleNamespace(
            is_hybrid=True,
            scheduler_config=SimpleNamespace(max_num_seqs=256),
            linear_cache_bytes_per_slot=MagicMock(return_value=64_400_000),
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
            linear_cache_bytes_per_slot=MagicMock(return_value=64_400_000),
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

    def test_non_hybrid_oom_error_omits_gdn_reservation(self, monkeypatch) -> None:
        runner = SimpleNamespace(is_hybrid=False)
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
