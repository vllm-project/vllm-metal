# SPDX-License-Identifier: Apache-2.0
"""Tests for hybrid model dimension extraction, cache specs, and backend allocation."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import mlx.core as mx
import pytest

from vllm_metal.paged_attention_backend.hybrid import HybridPagedAttentionBackend
from vllm_metal.v1.model_runner import MetalModelRunner

QWEN35_4B_ARGS: dict = {
    # Source: Qwen/Qwen3.5-4B config.json → text_config
    "num_hidden_layers": 32,
    "num_attention_heads": 16,
    "num_key_value_heads": 4,
    "head_dim": 256,
    "hidden_size": 2560,
    "full_attention_interval": 4,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 32,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
}

LLAMA_ARGS: dict = {
    # Source: meta-llama/Llama-3.2-1B config.json
    "num_hidden_layers": 16,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 64,
    "hidden_size": 2048,
}


def _make_runner(args: dict) -> MagicMock:
    """Create a mock runner with real property/method wiring."""
    runner = MagicMock(spec=MetalModelRunner)
    runner.model_args = dict(args)
    runner.is_mla = MetalModelRunner.is_mla.fget(runner)
    runner.is_hybrid = MetalModelRunner.is_hybrid.fget(runner)
    runner._is_stt = False
    runner.kv_cache_dtype = mx.float16
    runner.metal_config = MagicMock(block_size=16)
    MetalModelRunner._resolve_model_dims(runner)
    return runner


def _make_hybrid_backend(args: dict = QWEN35_4B_ARGS, **overrides):
    """Create a HybridPagedAttentionBackend from model args."""
    defaults = {
        "num_layers": args["num_hidden_layers"],
        "full_attention_interval": args["full_attention_interval"],
        "max_num_seqs": 8,
        "num_kv_heads": args["num_key_value_heads"],
        "head_dim": args["head_dim"],
        "linear_num_k_heads": args["linear_num_key_heads"],
        "linear_num_v_heads": args["linear_num_value_heads"],
        "linear_key_head_dim": args["linear_key_head_dim"],
        "linear_value_head_dim": args["linear_value_head_dim"],
        "linear_conv_kernel_dim": args["linear_conv_kernel_dim"],
        "linear_conv_dim": (
            args["linear_num_key_heads"] * args["linear_key_head_dim"] * 2
            + args["linear_num_value_heads"] * args["linear_value_head_dim"]
        ),
        "block_size": 16,
        "dtype": mx.float16,
    }
    defaults.update(overrides)
    return HybridPagedAttentionBackend(**defaults)


class TestExtractModelArgs:
    @staticmethod
    def _call_extract(model_args_dict: dict) -> dict:
        runner = MagicMock(spec=MetalModelRunner)
        runner.metal_config = MagicMock(debug=False)
        runner.model = SimpleNamespace(args=SimpleNamespace(**model_args_dict))
        runner._is_vlm = False
        MetalModelRunner._extract_model_args(runner)
        return runner.model_args

    def test_text_config_keys_merged(self) -> None:
        """Keys from text_config are populated into model_args."""
        args = self._call_extract(
            {"model_type": "qwen3_5_text", "text_config": {"new_key": 42}}
        )

        assert args["new_key"] == 42

    def test_top_level_wins_over_text_config(self) -> None:
        """When both top-level and text_config have the same key, top-level wins."""
        args = self._call_extract(
            {"shared_key": "top", "text_config": {"shared_key": "nested"}}
        )

        assert args["shared_key"] == "top"

    def test_no_text_config_unchanged(self) -> None:
        """Models without text_config pass through unchanged."""
        args = self._call_extract({"num_hidden_layers": 24})

        assert args["num_hidden_layers"] == 24


class TestIsHybrid:
    def test_hybrid(self) -> None:
        assert _make_runner(QWEN35_4B_ARGS).is_hybrid is True

    def test_not_hybrid(self) -> None:
        assert _make_runner(LLAMA_ARGS).is_hybrid is False


class TestResolveModelDims:
    def test_hybrid_layer_counts(self) -> None:
        # Arrange
        runner = _make_runner(QWEN35_4B_ARGS)
        n = QWEN35_4B_ARGS["num_hidden_layers"]
        fai = QWEN35_4B_ARGS["full_attention_interval"]
        expected_sdpa = sum(1 for i in range(n) if (i + 1) % fai == 0)

        # Assert
        assert runner.num_layers == n
        assert runner.num_sdpa_layers == expected_sdpa
        assert runner.num_linear_layers == n - expected_sdpa

    def test_hybrid_conv_dim(self) -> None:
        # Arrange
        runner = _make_runner(QWEN35_4B_ARGS)
        a = QWEN35_4B_ARGS
        expected = (
            a["linear_num_key_heads"] * a["linear_key_head_dim"] * 2
            + a["linear_num_value_heads"] * a["linear_value_head_dim"]
        )

        # Assert
        assert runner.linear_conv_dim == expected


class TestCacheBlockSizeBytes:
    def test_hybrid_excludes_linear_layers(self) -> None:
        """Hybrid SDPA block size must be smaller than if all layers were SDPA."""
        # Arrange
        hybrid_runner = _make_runner(QWEN35_4B_ARGS)
        non_hybrid_args = {
            k: v for k, v in QWEN35_4B_ARGS.items() if k != "full_attention_interval"
        }
        non_hybrid_runner = _make_runner(non_hybrid_args)

        # Act
        hybrid_bytes = MetalModelRunner.get_cache_block_size_bytes(hybrid_runner)
        all_layers_bytes = MetalModelRunner.get_cache_block_size_bytes(
            non_hybrid_runner
        )

        # Assert
        assert 0 < hybrid_bytes < all_layers_bytes

    def test_non_hybrid_counts_all_layers(self) -> None:
        # Arrange
        runner = _make_runner(LLAMA_ARGS)
        a = LLAMA_ARGS

        # Act
        result = MetalModelRunner.get_cache_block_size_bytes(runner)

        # Assert
        expected = (
            2
            * a["num_hidden_layers"]
            * 16  # block_size
            * a["num_key_value_heads"]
            * a["head_dim"]
            * runner.kv_cache_dtype.size
        )
        assert result == expected

    def test_linear_cache_bytes_positive(self) -> None:
        # Act
        result = MetalModelRunner.linear_cache_bytes_per_slot(
            _make_runner(QWEN35_4B_ARGS)
        )

        # Assert
        assert result > 0

    def test_linear_cache_bytes_rejects_non_hybrid(self) -> None:
        with pytest.raises(RuntimeError):
            MetalModelRunner.linear_cache_bytes_per_slot(_make_runner(LLAMA_ARGS))


class TestKVCacheSpec:
    def test_hybrid_emits_mixed_specs(self) -> None:
        from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

        # Arrange
        runner = _make_runner(QWEN35_4B_ARGS)

        # Act
        specs = MetalModelRunner.get_kv_cache_spec(runner)

        # Assert
        n = QWEN35_4B_ARGS["num_hidden_layers"]
        assert len(specs) == n
        assert (
            sum(1 for s in specs.values() if isinstance(s, FullAttentionSpec))
            == runner.num_sdpa_layers
        )
        assert (
            sum(1 for s in specs.values() if isinstance(s, MambaSpec))
            == runner.num_linear_layers
        )

    def test_non_hybrid_all_full(self) -> None:
        from vllm.v1.kv_cache_interface import FullAttentionSpec

        # Act
        specs = MetalModelRunner.get_kv_cache_spec(_make_runner(LLAMA_ARGS))

        # Assert
        assert all(isinstance(s, FullAttentionSpec) for s in specs.values())


class TestHybridBackend:
    def test_allocates_separate_caches(self) -> None:
        # Arrange
        backend = _make_hybrid_backend()
        runner = _make_runner(QWEN35_4B_ARGS)

        # Act
        backend.initialize(num_blocks=100)

        # Assert
        assert backend.kv_cache.num_layers == runner.num_sdpa_layers
        assert backend.kv_cache.num_blocks == 100
        assert backend.linear_cache.num_layers == runner.num_linear_layers
        assert backend.linear_cache.num_blocks == 8  # max_num_seqs

    def test_num_blocks_before_init_raises(self) -> None:
        # Arrange
        backend = _make_hybrid_backend()

        # Act / Assert
        with pytest.raises(RuntimeError):
            backend.num_blocks()

    def test_properties_before_init_raise(self) -> None:
        # Arrange
        backend = _make_hybrid_backend()

        # Act / Assert
        with pytest.raises(RuntimeError):
            _ = backend.kv_cache
        with pytest.raises(RuntimeError):
            _ = backend.linear_cache

    def test_kv_budget_subtracts_linear(self) -> None:
        from vllm_metal.v1.worker import MetalWorker

        # Arrange
        budget = MetalWorker._kv_budget_bytes(
            metal_limit=16_000_000_000, model_memory=4_000_000_000, fraction=0.5
        )
        runner = _make_runner(QWEN35_4B_ARGS)
        linear_fixed = MetalModelRunner.linear_cache_bytes_per_slot(runner) * 8
        sdpa_per_block = MetalModelRunner.get_cache_block_size_bytes(runner)

        # Act
        blocks_without = budget // sdpa_per_block
        blocks_with = (budget - linear_fixed) // sdpa_per_block

        # Assert
        assert 0 < blocks_with < blocks_without
