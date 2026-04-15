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
from vllm_metal.v1.cache_policy import ModelCachePolicy  # noqa: E402
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
    """Worker should delegate STT capability decisions to model runner."""

    def test_load_model_does_not_setup_paged_attention(self) -> None:
        """Paged attention setup moved to determine_available_memory (issue #234)."""
        model_runner = MagicMock()
        worker = _make_worker(model_runner, use_paged_attention=True)
        worker._setup_paged_attention = MagicMock()

        MetalWorker.load_model(worker)

        model_runner.load_model.assert_called_once_with()
        worker._setup_paged_attention.assert_not_called()

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
        measured_overhead = 200 * 1024 * 1024
        model_runner = SimpleNamespace(
            scheduler_memory_reporting_mode=MagicMock(
                return_value="paged_attention_capacity"
            ),
            profile_run=MagicMock(return_value=measured_overhead),
            _paged_attention_backend=None,
        )
        worker = _make_worker(model_runner, use_paged_attention=True)
        worker.get_cache_block_size_bytes = MagicMock(return_value=block_size_bytes)

        def _fake_setup(*, overhead: int) -> None:
            model_runner._paged_attention_backend = SimpleNamespace(
                num_blocks=lambda: num_blocks
            )

        worker._setup_paged_attention = MagicMock(side_effect=_fake_setup)

        available = MetalWorker.determine_available_memory(worker)

        assert available == num_blocks * block_size_bytes
        model_runner.profile_run.assert_called_once_with()
        worker._setup_paged_attention.assert_called_once_with(
            overhead=measured_overhead
        )
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
