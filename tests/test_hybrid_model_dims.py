# SPDX-License-Identifier: Apache-2.0
"""Tests for hybrid model dimension extraction and cache spec generation.

Verifies that Qwen3.5-style models with ``text_config`` nested dimensions
are correctly unwrapped, and that ``get_kv_cache_spec`` emits the right
spec types per layer.
"""

from __future__ import annotations

from unittest.mock import MagicMock

# === _extract_model_args: text_config unwrapping ===


class TestExtractModelArgs:
    """Verify the text_config unwrapping logic added to _extract_model_args.

    Tests the unwrapping directly on a dict rather than mocking the full
    _extract_model_args method, since the interesting logic is just the
    text_config flattening at the end.
    """

    @staticmethod
    def _unwrap(model_args: dict) -> dict:
        """Replicate the text_config unwrapping from _extract_model_args."""
        if (
            "text_config" in model_args
            and isinstance(model_args["text_config"], dict)
            and "num_hidden_layers" not in model_args
        ):
            for k, v in model_args["text_config"].items():
                model_args.setdefault(k, v)
        return model_args

    def test_qwen35_text_config_unwrapped(self) -> None:
        args = self._unwrap(
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
        assert args["linear_num_value_heads"] == 32

    def test_flat_args_unchanged(self) -> None:
        args = self._unwrap(
            {
                "num_hidden_layers": 24,
                "num_attention_heads": 16,
            }
        )
        assert args["num_hidden_layers"] == 24
        assert "text_config" not in args

    def test_top_level_takes_precedence(self) -> None:
        args = self._unwrap(
            {
                "model_type": "custom",
                "num_hidden_layers": 99,
                "text_config": {"num_hidden_layers": 32},
            }
        )
        # num_hidden_layers exists at top level → text_config not flattened
        assert args["num_hidden_layers"] == 99


# === is_hybrid ===


class TestIsHybrid:
    @staticmethod
    def _runner_with_model_args(args: dict) -> MagicMock:
        from vllm_metal.v1.model_runner import MetalModelRunner

        runner = MagicMock(spec=MetalModelRunner)
        runner.model_args = args
        runner.is_hybrid = MetalModelRunner.is_hybrid.fget(runner)
        return runner

    def test_hybrid_with_interval(self) -> None:
        runner = self._runner_with_model_args({"full_attention_interval": 4})
        assert runner.is_hybrid is True

    def test_not_hybrid_without_interval(self) -> None:
        runner = self._runner_with_model_args({})
        assert runner.is_hybrid is False

    def test_not_hybrid_with_zero(self) -> None:
        runner = self._runner_with_model_args({"full_attention_interval": 0})
        assert runner.is_hybrid is False


# === _resolve_model_dims: hybrid dimensions ===


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


class TestResolveModelDimsHybrid:
    @staticmethod
    def _make_runner(args: dict) -> MagicMock:
        from vllm_metal.v1.model_runner import MetalModelRunner

        runner = MagicMock(spec=MetalModelRunner)
        runner.model_args = dict(args)
        # Wire up real properties
        runner.is_mla = MetalModelRunner.is_mla.fget(runner)
        runner.is_hybrid = MetalModelRunner.is_hybrid.fget(runner)
        MetalModelRunner._resolve_model_dims(runner)
        return runner

    def test_layer_counts(self) -> None:
        runner = self._make_runner(QWEN35_4B_ARGS)
        assert runner.num_layers == 32
        assert runner.num_sdpa_layers == 8
        assert runner.num_linear_layers == 24

    def test_sdpa_dims(self) -> None:
        runner = self._make_runner(QWEN35_4B_ARGS)
        assert runner.num_kv_heads == 4
        assert runner.head_dim == 256

    def test_linear_dims(self) -> None:
        runner = self._make_runner(QWEN35_4B_ARGS)
        assert runner.linear_num_v_heads == 32
        assert runner.linear_key_head_dim == 128
        assert runner.linear_value_head_dim == 128
        assert runner.linear_conv_kernel_dim == 4

    def test_non_hybrid_has_no_linear_dims(self) -> None:
        args = {
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "hidden_size": 2048,
        }
        runner = self._make_runner(args)
        assert runner.num_layers == 24
        assert not hasattr(runner, "num_sdpa_layers")


# === get_cache_block_size_bytes ===


class TestCacheBlockSizeBytes:
    @staticmethod
    def _make_runner(args: dict) -> MagicMock:
        import mlx.core as mx

        from vllm_metal.v1.model_runner import MetalModelRunner

        runner = MagicMock(spec=MetalModelRunner)
        runner.model_args = dict(args)
        runner.is_mla = MetalModelRunner.is_mla.fget(runner)
        runner.is_hybrid = MetalModelRunner.is_hybrid.fget(runner)
        runner._is_stt = False
        runner.kv_cache_dtype = mx.float16
        runner.metal_config = MagicMock(block_size=16)
        MetalModelRunner._resolve_model_dims(runner)
        return runner

    def test_hybrid_uses_sdpa_layers_only(self) -> None:
        from vllm_metal.v1.model_runner import MetalModelRunner

        runner = self._make_runner(QWEN35_4B_ARGS)
        result = MetalModelRunner.get_cache_block_size_bytes(runner)
        # 2 * 8 sdpa_layers * 16 block_size * 4 kv_heads * 256 head_dim * 2 bytes
        expected = 2 * 8 * 16 * 4 * 256 * 2
        assert result == expected

    def test_non_hybrid_uses_all_layers(self) -> None:
        from vllm_metal.v1.model_runner import MetalModelRunner

        args = {
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "hidden_size": 2048,
        }
        runner = self._make_runner(args)
        result = MetalModelRunner.get_cache_block_size_bytes(runner)
        expected = 2 * 24 * 16 * 4 * 128 * 2
        assert result == expected


# === get_kv_cache_spec ===


class TestKVCacheSpec:
    @staticmethod
    def _make_runner(args: dict) -> MagicMock:
        import mlx.core as mx

        from vllm_metal.v1.model_runner import MetalModelRunner

        runner = MagicMock(spec=MetalModelRunner)
        runner.model_args = dict(args)
        runner.is_mla = MetalModelRunner.is_mla.fget(runner)
        runner.is_hybrid = MetalModelRunner.is_hybrid.fget(runner)
        runner._is_stt = False
        runner.kv_cache_dtype = mx.float16
        runner.metal_config = MagicMock(block_size=16)
        MetalModelRunner._resolve_model_dims(runner)
        return runner

    def test_hybrid_emits_mixed_specs(self) -> None:
        from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

        from vllm_metal.v1.model_runner import MetalModelRunner

        runner = self._make_runner(QWEN35_4B_ARGS)
        specs = MetalModelRunner.get_kv_cache_spec(runner)

        # 32 layers total
        assert len(specs) == 32

        # Count types
        full_count = sum(1 for s in specs.values() if isinstance(s, FullAttentionSpec))
        mamba_count = sum(1 for s in specs.values() if isinstance(s, MambaSpec))
        assert full_count == 8  # SDPA layers
        assert mamba_count == 24  # GDN layers

    def test_hybrid_sdpa_layer_names(self) -> None:
        from vllm_metal.v1.model_runner import MetalModelRunner

        runner = self._make_runner(QWEN35_4B_ARGS)
        specs = MetalModelRunner.get_kv_cache_spec(runner)

        # SDPA at (i+1) % 4 == 0 → indices 3, 7, 11, ..., 31
        for i in range(32):
            if (i + 1) % 4 == 0:
                assert f"layers.{i}.self_attn" in specs
            else:
                assert f"layers.{i}.linear_attn" in specs

    def test_non_hybrid_all_full_attention(self) -> None:
        from vllm.v1.kv_cache_interface import FullAttentionSpec

        from vllm_metal.v1.model_runner import MetalModelRunner

        args = {
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "hidden_size": 2048,
        }
        runner = self._make_runner(args)
        specs = MetalModelRunner.get_kv_cache_spec(runner)
        assert len(specs) == 24
        assert all(isinstance(s, FullAttentionSpec) for s in specs.values())
