# SPDX-License-Identifier: Apache-2.0
"""Tests for runtime compatibility patches."""

from __future__ import annotations

import builtins
import importlib.util
import os
import sys
from types import ModuleType

import numpy as np
import pytest

import vllm_metal.compat as compat


def _install_fake_qwen35_modules(monkeypatch, *, include_moe: bool):
    mlx_pkg = ModuleType("mlx")
    mlx_core = ModuleType("mlx.core")
    mlx_core.bfloat16 = np.float32
    mlx_core.from_fp8 = lambda weight, dtype=None: np.asarray(weight, dtype=np.float32)
    mlx_core.pad = lambda weight, pad_width: np.pad(weight, pad_width)
    mlx_pkg.core = mlx_core
    monkeypatch.setitem(sys.modules, "mlx", mlx_pkg)
    monkeypatch.setitem(sys.modules, "mlx.core", mlx_core)

    mlx_lm_pkg = ModuleType("mlx_lm")
    mlx_lm_models = ModuleType("mlx_lm.models")
    mlx_lm_pkg.models = mlx_lm_models
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm_pkg)
    monkeypatch.setitem(sys.modules, "mlx_lm.models", mlx_lm_models)

    dense_module = ModuleType("mlx_lm.models.qwen3_5")

    class DenseModel:
        def sanitize(self, weights):
            return dict(weights)

    dense_module.Model = DenseModel
    monkeypatch.setitem(sys.modules, "mlx_lm.models.qwen3_5", dense_module)
    mlx_lm_models.qwen3_5 = dense_module

    moe_module = None
    if include_moe:
        moe_module = ModuleType("mlx_lm.models.qwen3_5_moe")

        class MoeModel:
            def sanitize(self, weights):
                return dict(weights)

        moe_module.Model = MoeModel
        monkeypatch.setitem(sys.modules, "mlx_lm.models.qwen3_5_moe", moe_module)
        mlx_lm_models.qwen3_5_moe = moe_module

    def _fake_find_spec(name: str):
        if name == "mlx_lm.models.qwen3_5":
            return object()
        if name == "mlx_lm.models.qwen3_5_moe":
            return object() if include_moe else None
        return None

    monkeypatch.setattr(importlib.util, "find_spec", _fake_find_spec)
    return dense_module, moe_module


class TestQwen35Fp8CompatPatch:
    def test_logs_when_mlx_core_is_unavailable(self, monkeypatch) -> None:
        original_import = builtins.__import__
        warnings = []

        def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "mlx.core":
                raise ImportError("missing mlx.core")
            return original_import(name, globals, locals, fromlist, level)

        def _record_warning(message, *args, **_kwargs):
            warnings.append(message % args)

        monkeypatch.setattr(builtins, "__import__", _fake_import)
        monkeypatch.setattr(compat.logger, "warning", _record_warning)

        compat._patch_mlx_lm_qwen35_fp8_sanitize()

        assert any("mlx.core is unavailable" in warning for warning in warnings)

    def test_patches_dense_qwen35_even_when_moe_module_is_missing(
        self, monkeypatch
    ) -> None:
        dense_module, _ = _install_fake_qwen35_modules(monkeypatch, include_moe=False)

        compat._patch_mlx_lm_qwen35_fp8_sanitize()

        sanitized = dense_module.Model().sanitize(
            {
                "language_model.layers.0.linear.weight": np.ones((128, 128)),
                "language_model.layers.0.linear.weight_scale_inv": np.ones((1, 1)),
            }
        )

        assert "language_model.layers.0.linear.weight_scale_inv" not in sanitized
        assert sanitized["language_model.layers.0.linear.weight"].shape == (128, 128)

    def test_dequant_applies_scale_values_by_fp8_block(self, monkeypatch) -> None:
        _install_fake_qwen35_modules(monkeypatch, include_moe=False)

        weight = np.arange(129 * 130, dtype=np.float32).reshape(129, 130)
        scale_inv = np.array(
            [[2.0, 3.0], [5.0, 7.0]],
            dtype=np.float32,
        )
        dequantized = compat._dequantize_qwen35_fp8_weight(
            weight,
            scale_inv,
            sys.modules["mlx.core"],
        )

        expected = weight.copy()
        expected[:128, :128] *= 2.0
        expected[:128, 128:] *= 3.0
        expected[128:, :128] *= 5.0
        expected[128:, 128:] *= 7.0
        assert dequantized.shape == (129, 130)
        np.testing.assert_allclose(dequantized, expected)

    def test_dequant_real_mlx_fp8_values_when_enabled(self) -> None:
        if os.environ.get("VLLM_METAL_RUN_REAL_MLX_FP8_TESTS") != "1":
            pytest.skip("VLLM_METAL_RUN_REAL_MLX_FP8_TESTS=1 not set")

        import mlx.core as mx

        fp8_dtype = getattr(mx, "float8_e4m3fn", None)
        if fp8_dtype is None:
            pytest.skip("mlx.core has no float8_e4m3fn dtype")

        weight = mx.array([[1.0, -2.0], [0.5, 4.0]], dtype=mx.float32).astype(fp8_dtype)
        scale_inv = mx.array([[2.0]], dtype=mx.float32)

        dequantized = compat._dequantize_qwen35_fp8_weight(weight, scale_inv, mx)
        mx.eval(dequantized)

        np.testing.assert_allclose(
            np.array(dequantized, dtype=np.float32),
            np.array([[2.0, -4.0], [1.0, 8.0]], dtype=np.float32),
        )

    def test_rejects_unexpected_fp8_block_scale_shape(self, monkeypatch) -> None:
        dense_module, _ = _install_fake_qwen35_modules(monkeypatch, include_moe=False)

        compat._patch_mlx_lm_qwen35_fp8_sanitize()

        with pytest.raises(ValueError, match="128x128 FP8 blocks"):
            dense_module.Model().sanitize(
                {
                    "language_model.layers.0.linear.weight": np.ones((128, 128)),
                    "language_model.layers.0.linear.weight_scale_inv": np.ones((2, 1)),
                }
            )

    def test_patches_higher_rank_weights_for_moe(self, monkeypatch) -> None:
        _, moe_module = _install_fake_qwen35_modules(monkeypatch, include_moe=True)
        gate_up_proj_prefix = "language_model.layers.0.mlp.experts.gate_up_proj"

        compat._patch_mlx_lm_qwen35_fp8_sanitize()

        sanitized = moe_module.Model().sanitize(
            {
                f"{gate_up_proj_prefix}.weight": np.ones((2, 256, 128)),
                f"{gate_up_proj_prefix}.weight_scale_inv": np.ones((2, 2, 1)),
                f"{gate_up_proj_prefix}.activation_scale": np.ones((2, 2, 1)),
            }
        )

        assert f"{gate_up_proj_prefix}.weight_scale_inv" not in sanitized
        assert f"{gate_up_proj_prefix}.activation_scale" not in sanitized
        assert sanitized[f"{gate_up_proj_prefix}.weight"].shape == (2, 256, 128)


def _install_fake_gemma4_text_module(
    monkeypatch,
    *,
    num_hidden_layers: int,
    num_kv_shared_layers: int,
):
    mlx_lm_pkg = ModuleType("mlx_lm")
    mlx_lm_models = ModuleType("mlx_lm.models")
    mlx_lm_pkg.models = mlx_lm_models
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm_pkg)
    monkeypatch.setitem(sys.modules, "mlx_lm.models", mlx_lm_models)

    module = ModuleType("mlx_lm.models.gemma4_text")

    class FakeArgs:
        def __init__(self) -> None:
            self.num_hidden_layers = num_hidden_layers
            self.num_kv_shared_layers = num_kv_shared_layers

    class FakeModel:
        def __init__(self) -> None:
            self.args = FakeArgs()

        def sanitize(self, weights):
            return dict(weights)

    module.Model = FakeModel
    monkeypatch.setitem(sys.modules, "mlx_lm.models.gemma4_text", module)
    mlx_lm_models.gemma4_text = module

    def _fake_find_spec(name: str):
        if name == "mlx_lm.models.gemma4_text":
            return object()
        return None

    monkeypatch.setattr(importlib.util, "find_spec", _fake_find_spec)
    return module


class TestGemma4KvSharedCompatPatch:
    def test_drop_helper_removes_54_phantom_keys_for_e4b_layout(self) -> None:
        # E4B: 42 layers total, last 18 are KV-shared.
        weights = {}
        for i in range(42):
            for suffix in (
                "k_proj",
                "v_proj",
                "k_norm",
                "q_proj",
                "q_norm",
                "o_proj",
            ):
                weights[
                    f"language_model.model.layers.{i}.self_attn.{suffix}.weight"
                ] = f"T{i}_{suffix}"

        out = compat._drop_gemma4_kv_shared_phantom_weights(
            weights, num_hidden_layers=42, num_kv_shared_layers=18
        )

        assert len(weights) - len(out) == 54
        for i in range(24):
            for suffix in (
                "k_proj",
                "v_proj",
                "k_norm",
                "q_proj",
                "q_norm",
                "o_proj",
            ):
                assert (
                    f"language_model.model.layers.{i}.self_attn.{suffix}.weight" in out
                )
        for i in range(24, 42):
            for suffix in ("k_proj", "v_proj", "k_norm"):
                assert (
                    f"language_model.model.layers.{i}.self_attn.{suffix}.weight"
                    not in out
                )
            for suffix in ("q_proj", "q_norm", "o_proj"):
                assert (
                    f"language_model.model.layers.{i}.self_attn.{suffix}.weight" in out
                )

    def test_drop_helper_is_noop_without_sharing(self) -> None:
        weights = {
            "language_model.model.layers.0.self_attn.k_proj.weight": "T",
            "language_model.model.layers.41.self_attn.k_proj.weight": "T",
        }
        out = compat._drop_gemma4_kv_shared_phantom_weights(
            weights, num_hidden_layers=42, num_kv_shared_layers=0
        )
        assert out == weights

    def test_drop_helper_ignores_unrelated_or_malformed_keys(self) -> None:
        weights = {
            "language_model.model.layers.30.self_attn.q_proj.weight": "keep",
            "language_model.model.weird.self_attn.k_proj.weight": "keep",
            "language_model.model.layers.5.self_attn.k_proj.weight": "keep",
            "language_model.model.layers.30.self_attn.k_proj.weight": "drop",
        }
        out = compat._drop_gemma4_kv_shared_phantom_weights(
            weights, num_hidden_layers=42, num_kv_shared_layers=18
        )
        assert "language_model.model.layers.30.self_attn.k_proj.weight" not in out
        assert "language_model.model.layers.30.self_attn.q_proj.weight" in out
        assert "language_model.model.weird.self_attn.k_proj.weight" in out
        assert "language_model.model.layers.5.self_attn.k_proj.weight" in out

    def test_patch_wraps_gemma4_text_sanitize_and_drops_phantom_keys(
        self, monkeypatch
    ) -> None:
        module = _install_fake_gemma4_text_module(
            monkeypatch, num_hidden_layers=42, num_kv_shared_layers=18
        )

        compat._patch_mlx_lm_gemma4_kv_shared_sanitize()

        weights = {
            "language_model.model.layers.0.self_attn.k_proj.weight": "T",
            "language_model.model.layers.30.self_attn.k_proj.weight": "phantom",
            "language_model.model.layers.30.self_attn.q_proj.weight": "real",
        }
        sanitized = module.Model().sanitize(weights)

        assert "language_model.model.layers.30.self_attn.k_proj.weight" not in sanitized
        assert "language_model.model.layers.0.self_attn.k_proj.weight" in sanitized
        assert "language_model.model.layers.30.self_attn.q_proj.weight" in sanitized

    def test_patch_is_idempotent(self, monkeypatch) -> None:
        module = _install_fake_gemma4_text_module(
            monkeypatch, num_hidden_layers=42, num_kv_shared_layers=18
        )

        compat._patch_mlx_lm_gemma4_kv_shared_sanitize()
        once = module.Model.sanitize
        compat._patch_mlx_lm_gemma4_kv_shared_sanitize()
        twice = module.Model.sanitize

        assert once is twice
        assert getattr(twice, "_vllm_metal_gemma4_kv_shared_patch", False) is True


class TestWrapModelSanitize:
    def test_returns_false_when_class_has_no_sanitize(self) -> None:
        class ModelWithoutSanitize:
            pass

        applied = compat._wrap_model_sanitize(
            ModelWithoutSanitize,
            "_vllm_metal_test_patch",
            lambda _self, weights: weights,
        )

        assert applied is False
        assert not hasattr(ModelWithoutSanitize, "sanitize")
