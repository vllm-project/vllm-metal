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


def _write_bytelevel_tokenizer_json(path) -> None:
    from tokenizers import Tokenizer
    from tokenizers.decoders import ByteLevel
    from tokenizers.models import WordLevel

    tokenizer = Tokenizer(
        WordLevel(
            {
                "\u0120Hello": 0,
                "\u010a": 1,
                "<unk>": 2,
            },
            unk_token="<unk>",
        )
    )
    tokenizer.decoder = ByteLevel(add_prefix_space=False, use_regex=False)
    tokenizer.save(str(path / "tokenizer.json"))


class _WrongDecoderBackend:
    decoder = 'Sequence(decoders=[Replace(pattern=String("\\u2581"), content=" ")])'


class _ByteLevelBackend:
    decoder = "ByteLevel(add_prefix_space=False, trim_offsets=False, use_regex=False)"


class _FakeTokenizer:
    def __init__(self, path, backend) -> None:
        self.backend_tokenizer = backend
        self.init_kwargs = {
            "clean_up_tokenization_spaces": False,
            "model_max_length": 128,
        }
        self.name_or_path = str(path)
        self.bos_token = None
        self.eos_token = "<unk>"
        self.unk_token = "<unk>"
        self.pad_token = None
        self.sep_token = None
        self.cls_token = None
        self.mask_token = None
        self.additional_special_tokens = []
        self.chat_template = None
        self.model_max_length = 128
        self.clean_up_tokenization_spaces = False


class TestByteLevelTokenizerCompatPatch:
    def test_rebuilds_when_tokenizer_json_is_bytelevel_but_loaded_decoder_is_not(
        self, tmp_path
    ) -> None:
        _write_bytelevel_tokenizer_json(tmp_path)
        broken = _FakeTokenizer(tmp_path, _WrongDecoderBackend())

        fixed = compat._maybe_rebuild_bytelevel_tokenizer(broken, tmp_path, {})

        decoded = fixed.decode([0, 1])
        assert fixed is not broken
        assert "\u0120" not in decoded
        assert "\u010a" not in decoded
        assert "Hello" in decoded
        assert "\n" in decoded
        assert compat._loaded_tokenizer_decoder_uses_bytelevel(fixed)

    def test_keeps_loaded_tokenizer_when_decoder_is_already_bytelevel(
        self, tmp_path
    ) -> None:
        _write_bytelevel_tokenizer_json(tmp_path)
        tokenizer = _FakeTokenizer(tmp_path, _ByteLevelBackend())

        fixed = compat._maybe_rebuild_bytelevel_tokenizer(tokenizer, tmp_path, {})

        assert fixed is tokenizer

    def test_cached_lookup_forwards_tokenizer_location_kwargs(
        self, monkeypatch, tmp_path
    ) -> None:
        tokenizer_json = tmp_path / "tokenizer.json"
        tokenizer_json.write_text("{}", encoding="utf-8")
        captured_kwargs = {}

        def _fake_cached_file(path_or_repo_id, filename, **kwargs):
            captured_kwargs.update(kwargs)
            assert path_or_repo_id == "org/repo"
            assert filename == "tokenizer.json"
            return str(tokenizer_json)

        monkeypatch.setattr("transformers.utils.cached_file", _fake_cached_file)

        path = compat._cached_tokenizer_json_path(
            "org/repo",
            {
                "cache_dir": "/tmp/hf-cache",
                "force_download": True,
                "proxies": {"https": "proxy"},
                "revision": "refs/pr/1",
                "local_files_only": True,
                "subfolder": "tokenizer",
                "repo_type": "model",
                "user_agent": {"vllm-metal": "test"},
                "use_auth_token": "secret",
                "_commit_hash": "abc123",
            },
        )

        assert path == tokenizer_json
        assert captured_kwargs["cache_dir"] == "/tmp/hf-cache"
        assert captured_kwargs["force_download"] is True
        assert captured_kwargs["proxies"] == {"https": "proxy"}
        assert captured_kwargs["revision"] == "refs/pr/1"
        assert captured_kwargs["local_files_only"] is True
        assert captured_kwargs["subfolder"] == "tokenizer"
        assert captured_kwargs["repo_type"] == "model"
        assert captured_kwargs["user_agent"] == {"vllm-metal": "test"}
        assert captured_kwargs["token"] == "secret"
        assert captured_kwargs["_commit_hash"] == "abc123"


def _install_fake_qwen35_modules(monkeypatch, *, include_moe: bool):
    mlx_pkg = ModuleType("mlx")
    mlx_core = ModuleType("mlx.core")
    mlx_core.bfloat16 = np.float32
    mlx_core.from_fp8 = lambda weight, dtype=None: np.asarray(weight, dtype=np.float32)
    mlx_core.pad = lambda weight, pad_width: np.pad(weight, pad_width)
    mlx_core.stack = lambda arrays, axis=0: np.stack(arrays, axis=axis)
    mlx_core.concatenate = lambda arrays, axis=0: np.concatenate(arrays, axis=axis)
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

    def test_per_expert_moe_tensors_stack_to_combined(self, monkeypatch) -> None:
        # Qwen/Qwen3.6-35B-A3B-FP8 ships expert MLPs per-expert. The MoE
        # sanitize wrapper must stack them along axis 0 and concatenate
        # gate+up along the intermediate-dim axis, producing the combined
        # form upstream sanitize already handles.
        _, moe_module = _install_fake_qwen35_modules(monkeypatch, include_moe=True)
        prefix = "model.language_model.layers.0.mlp.experts"

        compat._patch_mlx_lm_qwen35_fp8_sanitize()

        per_expert = {
            f"{prefix}.0.gate_proj.weight": np.full((6, 4), 1.0),
            f"{prefix}.0.up_proj.weight": np.full((6, 4), 2.0),
            f"{prefix}.0.down_proj.weight": np.full((4, 6), 3.0),
            f"{prefix}.1.gate_proj.weight": np.full((6, 4), 4.0),
            f"{prefix}.1.up_proj.weight": np.full((6, 4), 5.0),
            f"{prefix}.1.down_proj.weight": np.full((4, 6), 6.0),
        }
        sanitized = moe_module.Model().sanitize(per_expert)

        gate_up_key = f"{prefix}.gate_up_proj"
        down_key = f"{prefix}.down_proj"
        assert gate_up_key in sanitized
        assert down_key in sanitized
        # gate_up: (num_experts, 2*intermediate, hidden); down: (num_experts, hidden, intermediate)
        assert sanitized[gate_up_key].shape == (2, 12, 4)
        assert sanitized[down_key].shape == (2, 4, 6)
        # Per-expert keys must not leak through after stacking.
        assert all(".experts.0." not in k for k in sanitized)
        assert all(".experts.1." not in k for k in sanitized)
        # Stacking preserves per-expert content along axis 0; gate occupies
        # the first half of axis -2, up occupies the second half.
        np.testing.assert_array_equal(
            sanitized[gate_up_key][0, :6, :], np.full((6, 4), 1.0)
        )
        np.testing.assert_array_equal(
            sanitized[gate_up_key][0, 6:, :], np.full((6, 4), 2.0)
        )
        np.testing.assert_array_equal(
            sanitized[down_key][1, :, :], np.full((4, 6), 6.0)
        )

    def test_pre_stacked_moe_is_noop_for_per_expert_helper(self, monkeypatch) -> None:
        # Pre-stacked checkpoints (mlx-community redistributions, Qwen3.6 bf16
        # master) ship `experts.gate_up_proj` / `experts.down_proj` already
        # combined. The per-expert helper must short-circuit and pass them
        # through unchanged, leaving the combined-format branch in upstream
        # sanitize free to do its split.
        _, moe_module = _install_fake_qwen35_modules(monkeypatch, include_moe=True)
        prefix = "model.language_model.layers.0.mlp.experts"

        compat._patch_mlx_lm_qwen35_fp8_sanitize()

        gate_up = np.arange(2 * 12 * 4, dtype=np.float32).reshape(2, 12, 4)
        down = np.arange(2 * 4 * 6, dtype=np.float32).reshape(2, 4, 6)
        weights = {
            f"{prefix}.gate_up_proj": gate_up,
            f"{prefix}.down_proj": down,
        }
        sanitized = moe_module.Model().sanitize(weights)

        # Helper is a no-op: combined keys present unchanged, no per-expert
        # keys appear.
        np.testing.assert_array_equal(sanitized[f"{prefix}.gate_up_proj"], gate_up)
        np.testing.assert_array_equal(sanitized[f"{prefix}.down_proj"], down)
        assert not any(f"{prefix}.0." in k for k in sanitized)

    def test_non_contiguous_per_expert_indices_raise(self, monkeypatch) -> None:
        # Defensive: a malformed checkpoint shipping experts {0, 1, 3} (skipping
        # 2) would silently drop expert 3 if the stacker walked indices in
        # order. Helper must raise loudly so the user diagnoses the missing
        # tensor instead of getting subtly wrong output.
        _, moe_module = _install_fake_qwen35_modules(monkeypatch, include_moe=True)
        prefix = "model.language_model.layers.0.mlp.experts"

        compat._patch_mlx_lm_qwen35_fp8_sanitize()

        gapped = {
            f"{prefix}.0.gate_proj.weight": np.zeros((6, 4)),
            f"{prefix}.0.up_proj.weight": np.zeros((6, 4)),
            f"{prefix}.0.down_proj.weight": np.zeros((4, 6)),
            f"{prefix}.1.gate_proj.weight": np.zeros((6, 4)),
            f"{prefix}.1.up_proj.weight": np.zeros((6, 4)),
            f"{prefix}.1.down_proj.weight": np.zeros((4, 6)),
            f"{prefix}.3.gate_proj.weight": np.zeros((6, 4)),
            f"{prefix}.3.up_proj.weight": np.zeros((6, 4)),
            f"{prefix}.3.down_proj.weight": np.zeros((4, 6)),
        }

        with pytest.raises(ValueError, match="non-contiguous"):
            moe_module.Model().sanitize(gapped)

    def test_missing_projection_family_raises(self, monkeypatch) -> None:
        # Defensive: a malformed checkpoint missing one entire projection
        # family (e.g., no down_proj at all) must surface as a clear
        # ValueError naming the missing family, rather than a raw KeyError
        # leaking from the walk step. The same path also covers the case
        # where only some experts have a given projection (mismatched index
        # sets across families).
        _, moe_module = _install_fake_qwen35_modules(monkeypatch, include_moe=True)
        prefix = "model.language_model.layers.0.mlp.experts"

        compat._patch_mlx_lm_qwen35_fp8_sanitize()

        # 1) Entire down_proj family absent.
        no_down = {
            f"{prefix}.0.gate_proj.weight": np.zeros((6, 4)),
            f"{prefix}.0.up_proj.weight": np.zeros((6, 4)),
            f"{prefix}.1.gate_proj.weight": np.zeros((6, 4)),
            f"{prefix}.1.up_proj.weight": np.zeros((6, 4)),
        }
        with pytest.raises(ValueError, match="missing projection families"):
            moe_module.Model().sanitize(no_down)

        # 2) down_proj missing for one expert (mismatched index sets).
        partial_down = {
            f"{prefix}.0.gate_proj.weight": np.zeros((6, 4)),
            f"{prefix}.0.up_proj.weight": np.zeros((6, 4)),
            f"{prefix}.0.down_proj.weight": np.zeros((4, 6)),
            f"{prefix}.1.gate_proj.weight": np.zeros((6, 4)),
            f"{prefix}.1.up_proj.weight": np.zeros((6, 4)),
            # missing f"{prefix}.1.down_proj.weight"
        }
        with pytest.raises(ValueError, match="mismatched down_proj"):
            moe_module.Model().sanitize(partial_down)

    def test_per_expert_helper_does_not_run_on_dense_qwen35(self, monkeypatch) -> None:
        # The dense qwen3_5 patch wraps sanitize with FP8 dequant only — the
        # per-expert stacking helper must NOT run on dense Qwen3.5/3.6
        # checkpoints (no expert tensors exist in dense models, so even an
        # accidental call would be a no-op, but the patch architecture
        # makes the MoE-only nature explicit).
        dense_module, _ = _install_fake_qwen35_modules(monkeypatch, include_moe=True)

        compat._patch_mlx_lm_qwen35_fp8_sanitize()

        # Dense weights with FP8 quant; no expert tensors anywhere.
        sanitized = dense_module.Model().sanitize(
            {
                "model.language_model.layers.0.self_attn.q_proj.weight": np.ones(
                    (128, 128)
                ),
                "model.language_model.layers.0.self_attn.q_proj.weight_scale_inv": np.ones(
                    (1, 1)
                ),
            }
        )
        assert (
            "model.language_model.layers.0.self_attn.q_proj.weight_scale_inv"
            not in sanitized
        )
        assert sanitized[
            "model.language_model.layers.0.self_attn.q_proj.weight"
        ].shape == (128, 128)


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
