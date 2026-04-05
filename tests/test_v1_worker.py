# SPDX-License-Identifier: Apache-2.0
"""Tests for v1 MetalWorker STT boundary delegation."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip("vllm", reason="vllm not installed")

from vllm_metal.stt.policy import STT_SCHED_AVAILABLE_BYTES  # noqa: E402
from vllm_metal.v1.worker import MetalWorker  # noqa: E402


def _make_worker(model_runner: object, *, use_paged_attention: bool) -> MetalWorker:
    worker = MetalWorker.__new__(MetalWorker)
    worker.model_runner = model_runner  # type: ignore[assignment]
    worker.metal_config = SimpleNamespace(use_paged_attention=use_paged_attention)
    return worker


class TestWorkerRunnerBoundaryDelegation:
    """Worker should delegate STT capability decisions to model runner."""

    @pytest.mark.parametrize(
        (
            "use_paged_attention",
            "runner_allows_setup",
            "expect_setup",
            "expect_cap_call",
        ),
        [
            (True, True, True, 1),
            (True, False, False, 1),
            (False, True, False, 0),
        ],
    )
    def test_load_model_delegates_paged_attention_setup_decision(
        self,
        use_paged_attention: bool,
        runner_allows_setup: bool,
        expect_setup: bool,
        expect_cap_call: int,
    ) -> None:
        model_runner = MagicMock()
        model_runner.is_hybrid = False
        model_runner.should_setup_paged_attention.return_value = runner_allows_setup
        worker = _make_worker(model_runner, use_paged_attention=use_paged_attention)
        worker._setup_paged_attention = MagicMock()

        MetalWorker.load_model(worker)

        model_runner.load_model.assert_called_once_with()
        assert model_runner.should_setup_paged_attention.call_count == expect_cap_call
        assert worker._setup_paged_attention.called is expect_setup

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

    def test_determine_available_memory_paged_capacity_mode(self) -> None:
        num_blocks = 8
        block_size_bytes = 16
        model_runner = SimpleNamespace(
            scheduler_memory_reporting_mode=MagicMock(
                return_value="paged_attention_capacity"
            ),
            _paged_attention_backend=SimpleNamespace(num_blocks=lambda: num_blocks),
        )
        worker = _make_worker(model_runner, use_paged_attention=True)
        worker.get_cache_block_size_bytes = MagicMock(return_value=block_size_bytes)

        available = MetalWorker.determine_available_memory(worker)

        assert available == num_blocks * block_size_bytes
        worker.get_cache_block_size_bytes.assert_called_once_with()

    def test_determine_available_memory_single_sequence_mode(self) -> None:
        """Test MLX path returns 80% of remaining Metal memory."""
        import mlx.core as mx

        model_runner = SimpleNamespace(
            scheduler_memory_reporting_mode=MagicMock(
                return_value="single_sequence_estimate"
            ),
        )
        worker = _make_worker(model_runner, use_paged_attention=False)
        worker.model_config = SimpleNamespace(max_model_len=2048)

        # Mock device_info and model memory
        original_device_info = mx.device_info
        mx.device_info = MagicMock(
            return_value={"max_recommended_working_set_size": 16 * 1024**3}
        )
        worker._get_model_memory_usage = MagicMock(return_value=2 * 1024**3)

        try:
            available = MetalWorker.determine_available_memory(worker)

            # Should return 80% of remaining memory
            # (16GB - 2GB) * 0.8 = 11.2GB
            expected = int((16 * 1024**3 - 2 * 1024**3) * 0.8)
            assert available == expected
        finally:
            mx.device_info = original_device_info

    def test_get_supported_tasks_delegates_to_runner_capability(self) -> None:
        model_runner = SimpleNamespace(
            supported_worker_tasks=MagicMock(return_value=("transcription",)),
        )
        worker = _make_worker(model_runner, use_paged_attention=False)

        tasks = MetalWorker.get_supported_tasks(worker)

        assert tasks == ("transcription",)
        model_runner.supported_worker_tasks.assert_called_once_with()


class TestOneSequenceKvBytes:
    """_one_sequence_kv_bytes must account for hybrid linear state."""

    def test_non_hybrid_counts_all_layers(self) -> None:
        # Arrange
        import mlx.core as mx

        model_runner = SimpleNamespace(
            is_hybrid=False,
            num_layers=16,
            num_kv_heads=8,
            head_dim=64,
            kv_cache_dtype=mx.float16,
        )
        worker = _make_worker(model_runner, use_paged_attention=False)
        worker.model_config = SimpleNamespace(max_model_len=2048)

        # Act
        result = MetalWorker._one_sequence_kv_bytes(worker)

        # Assert — 2 * 16 * 2048 * 8 * 64 * 2
        assert result == 2 * 16 * 2048 * 8 * 64 * 2

    def test_hybrid_adds_linear_state(self) -> None:
        # Arrange
        import mlx.core as mx

        linear_bytes = 1_000_000
        model_runner = SimpleNamespace(
            is_hybrid=True,
            num_sdpa_layers=8,
            num_kv_heads=4,
            head_dim=256,
            kv_cache_dtype=mx.float16,
            linear_cache_bytes_per_slot=MagicMock(return_value=linear_bytes),
        )
        worker = _make_worker(model_runner, use_paged_attention=False)
        worker.model_config = SimpleNamespace(max_model_len=2048)

        # Act
        result = MetalWorker._one_sequence_kv_bytes(worker)

        # Assert — SDPA bytes + linear state
        sdpa_bytes = 2 * 8 * 2048 * 4 * 256 * 2
        assert result == sdpa_bytes + linear_bytes
