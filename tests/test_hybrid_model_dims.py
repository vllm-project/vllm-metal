# SPDX-License-Identifier: Apache-2.0
"""Tests for hybrid model dimension extraction, cache specs, and backend allocation."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import mlx.core as mx
import pytest

from vllm_metal.paged_attention_backend.hybrid import HybridPagedAttentionBackend
from vllm_metal.v1.model_runner import MetalModelRunner

# Source: Qwen/Qwen3.5-4B config.json → text_config
QWEN35_4B_ARGS: dict = {
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

# Source: meta-llama/Llama-3.2-1B config.json
LLAMA_ARGS: dict = {
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


def _make_hybrid_backend(**overrides):
    """Create a HybridPagedAttentionBackend with Qwen3.5-4B defaults."""
    defaults = {
        "num_layers": 32,
        "full_attention_interval": 4,
        "max_num_seqs": 8,
        "num_kv_heads": 4,
        "head_dim": 256,
        "linear_num_v_heads": 32,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "linear_conv_dim": 8192,
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


class TestResolveModelDims:
    def test_hybrid_layer_counts(self) -> None:
        # Arrange
        runner = _make_runner(QWEN35_4B_ARGS)

        # Assert — fixed ground truth for Qwen3.5-4B (32 layers, interval=4)
        assert runner.num_layers == 32
        assert runner.num_sdpa_layers == 8
        assert runner.num_linear_layers == 24

    def test_hybrid_conv_dim(self) -> None:
        # Arrange
        runner = _make_runner(QWEN35_4B_ARGS)

        # Assert — fixed ground truth: 16*128*2 + 32*128 = 8192
        assert runner.linear_conv_dim == 8192


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
        all_bytes = MetalModelRunner.get_cache_block_size_bytes(non_hybrid_runner)

        # Assert
        assert 0 < hybrid_bytes < all_bytes

    def test_linear_cache_bytes_exact(self) -> None:
        """Exact byte count for Qwen3.5-4B linear cache per slot."""
        # Arrange
        runner = _make_runner(QWEN35_4B_ARGS)

        # Act
        result = MetalModelRunner.linear_cache_bytes_per_slot(runner)

        # Assert — 24 layers * ((3*8192) + (32*128*128)) * 2 bytes = 26345472
        assert result == 26_345_472

    def test_linear_cache_bytes_rejects_non_hybrid(self) -> None:
        with pytest.raises(RuntimeError):
            MetalModelRunner.linear_cache_bytes_per_slot(_make_runner(LLAMA_ARGS))


class TestKVCacheSpec:
    def test_hybrid_layer_spec_types(self) -> None:
        """Specific layers get the right spec type."""
        from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

        # Arrange
        runner = _make_runner(QWEN35_4B_ARGS)

        # Act
        specs = MetalModelRunner.get_kv_cache_spec(runner)

        # Assert — check concrete layer keys and types
        assert isinstance(specs["layers.0.linear_attn"], MambaSpec)
        assert isinstance(specs["layers.1.linear_attn"], MambaSpec)
        assert isinstance(specs["layers.2.linear_attn"], MambaSpec)
        assert isinstance(specs["layers.3.self_attn"], FullAttentionSpec)
        assert isinstance(specs["layers.7.self_attn"], FullAttentionSpec)
        assert isinstance(specs["layers.31.self_attn"], FullAttentionSpec)
        assert len(specs) == 32

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

        # Act
        backend.initialize(num_blocks=100)

        # Assert — Qwen3.5-4B: 8 SDPA, 24 linear
        assert backend.kv_cache.num_layers == 8
        assert backend.kv_cache.num_blocks == 100
        assert backend.linear_cache.num_layers == 24
        assert backend.linear_cache.num_blocks == 8  # max_num_seqs

    def test_num_blocks_before_init_raises(self) -> None:
        with pytest.raises(RuntimeError):
            _make_hybrid_backend().num_blocks()

    def test_properties_before_init_raise(self) -> None:
        backend = _make_hybrid_backend()

        with pytest.raises(RuntimeError):
            _ = backend.kv_cache
        with pytest.raises(RuntimeError):
            _ = backend.linear_cache
