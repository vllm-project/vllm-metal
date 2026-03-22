# SPDX-License-Identifier: Apache-2.0
"""Tests for v1 MetalWorker STT boundary delegation."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip("vllm", reason="vllm not installed")

from vllm_metal.stt.config import STT_SCHED_AVAILABLE_BYTES  # noqa: E402
from vllm_metal.v1.worker import MetalWorker  # noqa: E402


def _make_worker(model_runner: object, *, use_paged_attention: bool) -> MetalWorker:
    worker = MetalWorker.__new__(MetalWorker)
    worker.model_runner = model_runner  # type: ignore[assignment]
    worker.metal_config = SimpleNamespace(use_paged_attention=use_paged_attention)
    return worker


class TestWorkerRunnerBoundaryDelegation:
    """Worker should delegate STT capability decisions to model runner."""

    @pytest.mark.parametrize(
        ("use_paged_attention", "runner_allows_setup", "expect_setup", "expect_cap_call"),
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
        model_runner = SimpleNamespace(
            scheduler_memory_reporting_mode=MagicMock(
                return_value="paged_attention_capacity"
            ),
            _paged_kv_cache=SimpleNamespace(num_blocks=8),
        )
        worker = _make_worker(model_runner, use_paged_attention=True)
        worker.get_cache_block_size_bytes = MagicMock(return_value=16)

        available = MetalWorker.determine_available_memory(worker)

        assert available == 128
        worker.get_cache_block_size_bytes.assert_called_once_with()

    def test_determine_available_memory_paged_mode_requires_cache(self) -> None:
        model_runner = SimpleNamespace(
            scheduler_memory_reporting_mode=MagicMock(
                return_value="paged_attention_capacity"
            ),
            _paged_kv_cache=None,
        )
        worker = _make_worker(model_runner, use_paged_attention=True)

        with pytest.raises(RuntimeError, match="initialized paged KV cache"):
            MetalWorker.determine_available_memory(worker)

    def test_determine_available_memory_single_sequence_mode(self) -> None:
        model_runner = SimpleNamespace(
            scheduler_memory_reporting_mode=MagicMock(
                return_value="single_sequence_estimate"
            ),
        )
        worker = _make_worker(model_runner, use_paged_attention=False)
        worker._one_sequence_kv_bytes = MagicMock(return_value=4096)
        worker.model_config = SimpleNamespace(max_model_len=2048)

        available = MetalWorker.determine_available_memory(worker)

        assert available == 4096
        worker._one_sequence_kv_bytes.assert_called_once_with()

    def test_get_supported_tasks_delegates_to_runner_capability(self) -> None:
        model_runner = SimpleNamespace(
            supported_worker_tasks=MagicMock(return_value=("transcription",)),
        )
        worker = _make_worker(model_runner, use_paged_attention=False)

        tasks = MetalWorker.get_supported_tasks(worker)

        assert tasks == ("transcription",)
        model_runner.supported_worker_tasks.assert_called_once_with()
