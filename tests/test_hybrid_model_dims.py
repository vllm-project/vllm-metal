# SPDX-License-Identifier: Apache-2.0
"""Tests for hybrid model dimension extraction, cache specs, and backend allocation.

Verifies that Qwen3.5-style hybrid models are correctly detected and that
cache allocation uses the right dimensions.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import mlx.core as mx

from vllm_metal.v1.model_runner import MetalModelRunner

# Qwen3.5-4B model args (after text_config unwrapping).
QWEN35_4B_ARGS = {
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

LLAMA_ARGS = {
    "num_hidden_layers": 24,
    "num_attention_heads": 16,
    "num_key_value_heads": 4,
    "head_dim": 128,
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


# === _extract_model_args: text_config unwrapping ===


class TestExtractModelArgs:
    """Test through the real _extract_model_args method."""

    @staticmethod
    def _call_extract(model_args_dict: dict) -> dict:
        runner = MagicMock(spec=MetalModelRunner)
        runner.metal_config = MagicMock(debug=False)
        runner.model = SimpleNamespace(args=SimpleNamespace(**model_args_dict))
        runner._is_vlm = False
        MetalModelRunner._extract_model_args(runner)
        return runner.model_args

    def test_qwen35_text_config_unwrapped(self) -> None:
        args = self._call_extract(
            {
                "model_type": "qwen3_5_text",
                "text_config": {
                    "num_hidden_layers": 32,
                    "full_attention_interval": 4,
                    "linear_num_value_heads": 32,
                },
            }
        )
        assert args["num_hidden_layers"] == 32
        assert args["full_attention_interval"] == 4

    def test_flat_args_unchanged(self) -> None:
        args = self._call_extract({"num_hidden_layers": 24, "num_attention_heads": 16})
        assert args["num_hidden_layers"] == 24

    def test_top_level_takes_precedence(self) -> None:
        args = self._call_extract(
            {
                "num_hidden_layers": 99,
                "text_config": {"num_hidden_layers": 32},
            }
        )
        assert args["num_hidden_layers"] == 99


# === is_hybrid ===


class TestIsHybrid:
    def test_hybrid_with_interval(self) -> None:
        runner = _make_runner(QWEN35_4B_ARGS)
        assert runner.is_hybrid is True

    def test_not_hybrid_without_interval(self) -> None:
        runner = _make_runner(LLAMA_ARGS)
        assert runner.is_hybrid is False


# === _resolve_model_dims ===


class TestResolveModelDims:
    def test_hybrid_layer_counts(self) -> None:
        runner = _make_runner(QWEN35_4B_ARGS)
        assert runner.num_layers == 32
        assert runner.num_sdpa_layers == 8
        assert runner.num_linear_layers == 24

    def test_hybrid_linear_dims(self) -> None:
        runner = _make_runner(QWEN35_4B_ARGS)
        assert runner.linear_num_v_heads == 32
        assert runner.linear_conv_dim == 8192  # 16*128*2 + 32*128

    def test_non_hybrid_no_linear_attrs(self) -> None:
        runner = _make_runner(LLAMA_ARGS)
        assert runner.num_layers == 24
        with __import__("pytest").raises(AttributeError):
            _ = runner.num_sdpa_layers


# === get_cache_block_size_bytes ===


class TestCacheBlockSizeBytes:
    def test_hybrid_sdpa_only(self) -> None:
        runner = _make_runner(QWEN35_4B_ARGS)
        result = MetalModelRunner.get_cache_block_size_bytes(runner)
        # Pre-computed: 2 * 8 * 16 * 4 * 256 * 2 = 524288
        assert result == 524_288

    def test_non_hybrid_all_layers(self) -> None:
        runner = _make_runner(LLAMA_ARGS)
        result = MetalModelRunner.get_cache_block_size_bytes(runner)
        # Pre-computed: 2 * 24 * 16 * 4 * 128 * 2 = 786432
        assert result == 786_432

    def test_hybrid_linear_cache_bytes(self) -> None:
        runner = _make_runner(QWEN35_4B_ARGS)
        result = MetalModelRunner.linear_cache_bytes_per_slot(runner)
        # Pre-computed: 24 * ((3 * 8192) + (32 * 128 * 128)) * 2 = 26345472
        assert result == 26_345_472


# === get_kv_cache_spec ===


class TestKVCacheSpec:
    def test_hybrid_emits_mixed_specs(self) -> None:
        from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

        runner = _make_runner(QWEN35_4B_ARGS)
        specs = MetalModelRunner.get_kv_cache_spec(runner)

        assert len(specs) == 32
        full_count = sum(1 for s in specs.values() if isinstance(s, FullAttentionSpec))
        mamba_count = sum(1 for s in specs.values() if isinstance(s, MambaSpec))
        assert full_count == 8
        assert mamba_count == 24

    def test_hybrid_layer_names(self) -> None:
        runner = _make_runner(QWEN35_4B_ARGS)
        specs = MetalModelRunner.get_kv_cache_spec(runner)
        for i in range(32):
            if (i + 1) % 4 == 0:
                assert f"layers.{i}.self_attn" in specs
            else:
                assert f"layers.{i}.linear_attn" in specs

    def test_non_hybrid_all_full(self) -> None:
        from vllm.v1.kv_cache_interface import FullAttentionSpec

        runner = _make_runner(LLAMA_ARGS)
        specs = MetalModelRunner.get_kv_cache_spec(runner)
        assert len(specs) == 24
        assert all(isinstance(s, FullAttentionSpec) for s in specs.values())


# === HybridPagedAttentionBackend ===


class TestHybridBackend:
    def test_allocates_separate_caches(self) -> None:
        from vllm_metal.paged_attention_backend.hybrid import (
            HybridPagedAttentionBackend,
        )

        backend = HybridPagedAttentionBackend(
            num_layers=32,
            full_attention_interval=4,
            max_num_seqs=8,
            num_kv_heads=4,
            head_dim=256,
            linear_num_k_heads=16,
            linear_num_v_heads=32,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_conv_kernel_dim=4,
            linear_conv_dim=8192,
            block_size=16,
            dtype=mx.float16,
        )
        backend.initialize(num_blocks=100)

        assert backend.kv_cache.num_layers == 8
        assert backend.kv_cache.num_blocks == 100
        assert backend.linear_cache.num_layers == 24
        assert backend.linear_cache.num_blocks == 8

    def test_num_blocks_before_init_raises(self) -> None:
        import pytest

        from vllm_metal.paged_attention_backend.hybrid import (
            HybridPagedAttentionBackend,
        )

        backend = HybridPagedAttentionBackend(
            num_layers=32,
            full_attention_interval=4,
            max_num_seqs=8,
            num_kv_heads=4,
            head_dim=256,
            linear_num_k_heads=16,
            linear_num_v_heads=32,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_conv_kernel_dim=4,
            linear_conv_dim=8192,
            block_size=16,
            dtype=mx.float16,
        )
        with pytest.raises(RuntimeError):
            backend.num_blocks()

    def test_kv_budget_subtracts_linear(self) -> None:
        """Worker._kv_budget_bytes minus linear fixed cost yields fewer SDPA blocks."""
        from vllm_metal.v1.worker import MetalWorker

        budget = MetalWorker._kv_budget_bytes(
            metal_limit=16_000_000_000,
            model_memory=4_000_000_000,
            fraction=0.5,
        )
        runner = _make_runner(QWEN35_4B_ARGS)
        linear_fixed = (
            MetalModelRunner.linear_cache_bytes_per_slot(runner) * 8
        )  # max_num_seqs=8
        sdpa_per_block = MetalModelRunner.get_cache_block_size_bytes(runner)

        blocks_without_linear = budget // sdpa_per_block
        blocks_with_linear = (budget - linear_fixed) // sdpa_per_block

        assert blocks_with_linear > 0
        assert blocks_with_linear < blocks_without_linear
