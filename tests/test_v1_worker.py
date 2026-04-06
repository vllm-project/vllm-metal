# SPDX-License-Identifier: Apache-2.0
"""Tests for v1 MetalWorker STT boundary delegation."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import mlx.core as mx
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


class TestOneSequenceKvBytes:
    """_one_sequence_kv_bytes must account for hybrid linear state and block alignment."""

    def test_non_hybrid_counts_all_layers(self) -> None:
        model_runner = SimpleNamespace(
            is_hybrid=False,
            is_mla=False,
            num_layers=16,
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
        linear_bytes = 1_000_000
        model_runner = SimpleNamespace(
            is_hybrid=True,
            is_mla=False,
            num_sdpa_layers=8,
            num_kv_heads=4,
            head_dim=256,
            kv_cache_dtype=mx.float16,
            linear_cache_bytes_per_slot=MagicMock(return_value=linear_bytes),
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
        assert result == sdpa_bytes + linear_bytes

    def test_block_alignment_rounds_up_token_count(self) -> None:
        """When block_size doesn't divide max_model_len evenly, the token
        count must be rounded up to the next block boundary so that the
        reported bytes match the scheduler's block-aligned accounting.

        This reproduces the KV cache startup failure seen with Mamba-hybrid
        models (e.g. Granite 4.0-H) where the attention block_size is padded
        to 400 to match the mamba page size.
        """
        model_runner = SimpleNamespace(
            is_hybrid=False,
            is_mla=False,
            num_layers=4,
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
        model_runner = SimpleNamespace(
            is_hybrid=False,
            is_mla=True,
            num_layers=4,
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
