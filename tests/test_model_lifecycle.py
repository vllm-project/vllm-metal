# SPDX-License-Identifier: Apache-2.0
"""Tests for model lifecycle behavior."""

from __future__ import annotations

import contextlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import mlx.core as mx
import mlx.nn as nn
import pytest
import torch
from mlx_lm.models.nemotron_h import Model as NemotronHModel
from mlx_lm.models.nemotron_h import ModelArgs as NemotronHModelArgs
from vllm.model_executor.models import ModelRegistry

import vllm_metal.envs as envs
from tests.stub_runner import NEMOTRON_H_TINY_ARGS, make_stub_runner
from vllm_metal.attention.impls.mla import MLA_DEFAULT_QK_ROPE_HEAD_DIM
from vllm_metal.config import reset_config
from vllm_metal.distributed.pipeline import PipelineGroup
from vllm_metal.multimodal.gemma4 import Gemma4MultimodalAdapter, Gemma4VisionSidecar
from vllm_metal.multimodal.qwen3_vl import Qwen3VLMultimodalAdapter
from vllm_metal.v1 import model_adapter as model_adapter_module
from vllm_metal.v1 import model_lifecycle
from vllm_metal.v1.gemma4_mtp import Gemma4MTPAssistantLoader
from vllm_metal.v1.mm import EncoderCache
from vllm_metal.v1.model_adapter import DefaultModelAdapter
from vllm_metal.v1.model_lifecycle import GenerationLoadRequest, ModelLifecycle

_TEXT_MODEL_ARGS = {
    "vocab_size": 32000,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "hidden_size": 4096,
}


@pytest.fixture(autouse=True)
def _reset_env(monkeypatch: pytest.MonkeyPatch):
    for var in envs.environment_variables:
        monkeypatch.delenv(var, raising=False)
    reset_config()
    yield
    reset_config()


def _runner_model_config(**overrides: object) -> object:
    values = {
        "model": "stub-model",
        "hf_config": None,
        "multimodal_config": None,
        "is_multimodal_model": False,
        "trust_remote_code": False,
        "dtype": torch.float16,
        "quantization": None,
        "model_weights": "",
        "hf_token": None,
        "revision": None,
        "tokenizer": None,
        "tokenizer_revision": None,
        "is_hybrid": False,
        "architecture": "Qwen3NextForCausalLM",
        "model_impl": "vllm",
        "registry": ModelRegistry,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


_GDN_HYBRID_ARGS = {
    "model_type": "qwen3_5",
    "num_hidden_layers": 8,
    "num_attention_heads": 16,
    "num_key_value_heads": 4,
    "hidden_size": 1024,
    "full_attention_interval": 4,
    "linear_num_key_heads": 2,
    "linear_num_value_heads": 4,
    "linear_key_head_dim": 32,
    "linear_value_head_dim": 16,
    "linear_conv_kernel_dim": 3,
}

_JAMBA_ARGS = {
    "model_type": "jamba",
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "hidden_size": 4096,
}

_NEMOTRON_H_ARGS = {
    "model_type": "nemotron_h",
    "num_hidden_layers": 52,
    "num_attention_heads": 32,
    "num_key_value_heads": 2,
    "hidden_size": 2688,
    "head_dim": 128,
    "hybrid_override_pattern": list(
        "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"
    ),
    "mamba_num_heads": 64,
    "mamba_head_dim": 64,
    "ssm_state_size": 128,
    "n_groups": 8,
    "conv_kernel": 4,
}


def _text_config(**overrides: object) -> SimpleNamespace:
    return SimpleNamespace(**(_TEXT_MODEL_ARGS | overrides))


class _Qwen35LanguageModelStub:
    """Mirrors mlx_vlm 0.4.x ``LanguageModel.__call__`` so signature sniffing works.

    Also exposes ``self.model.embed_tokens`` so ``from_loaded_model`` can
    resolve the bottom-level embedding callable at load time.
    """

    def __init__(self) -> None:
        self.model = SimpleNamespace(embed_tokens=lambda input_ids: input_ids)

    def __call__(
        self,
        inputs: object,
        inputs_embeds: object | None = None,
        cache: object | None = None,
        position_ids: object | None = None,
        mask: object | None = None,
    ) -> None:
        return None


def _qwen35_vlm_model(
    *,
    vision_tower: object | None = None,
    language_model: object | None = None,
    spatial_merge_size: int = 2,
) -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(
            text_config=_text_config(),
            vision_config=SimpleNamespace(spatial_merge_size=spatial_merge_size),
        ),
        vision_tower=object() if vision_tower is None else vision_tower,
        language_model=(
            _Qwen35LanguageModelStub() if language_model is None else language_model
        ),
    )


def _stub_generation_model(
    monkeypatch: pytest.MonkeyPatch,
    *,
    config: object,
    tokenizer: object | None = None,
    is_vlm: bool = False,
    model: object | None = None,
) -> tuple[object, object]:
    fake_model = model or SimpleNamespace(config=config)
    fake_tokenizer = object() if tokenizer is None else tokenizer

    def _load_generation_model(
        self: ModelLifecycle,
        model_name: str,
        actual_is_vlm: bool,
        **_: object,
    ) -> tuple[object, object]:
        assert model_name == "stub-model"
        assert actual_is_vlm is is_vlm
        return fake_model, fake_tokenizer

    monkeypatch.setattr(
        ModelLifecycle,
        "_load_generation_model",
        _load_generation_model,
    )
    return fake_model, fake_tokenizer


def _make_lifecycle(
    *,
    model_args: dict[str, object] | None = None,
    model_config: object | None = None,
) -> tuple[ModelLifecycle, object]:
    runner = make_stub_runner(
        model_args=model_args,
        model_config=model_config or _runner_model_config(),
    )
    lifecycle = ModelLifecycle(runner, runner._model_adapter)
    return lifecycle, runner


class TestModelLifecycle:
    @pytest.mark.parametrize("revision", [None, "release-tag", "a" * 40])
    @pytest.mark.parametrize("backend", ["text", "vlm", "awq"])
    def test_generation_load_preserves_revision(
        self, monkeypatch: pytest.MonkeyPatch, revision: str | None, backend: str
    ) -> None:
        model, tokenizer = object(), object()
        loader = Mock(return_value=(model, tokenizer))
        awq_loader = SimpleNamespace(load=loader) if backend == "awq" else None
        detect_awq = Mock(return_value=awq_loader)
        monkeypatch.setattr(model_lifecycle.AWQQuantLoader, "for_model", detect_awq)
        monkeypatch.setattr(model_lifecycle, "mlx_lm_load", loader)
        monkeypatch.setattr(model_lifecycle, "mlx_vlm_load", loader)
        resolve_path = Mock(return_value="org/model")
        monkeypatch.setattr(model_lifecycle, "get_model_download_path", resolve_path)
        lifecycle, runner = _make_lifecycle(
            model_config=_runner_model_config(
                model="org/model",
                revision=revision,
                is_multimodal_model=backend == "vlm",
            )
        )
        request = GenerationLoadRequest.from_runner(
            runner, SimpleNamespace(should_force_text_backbone=lambda _: False)
        )

        result = lifecycle._load_generation_model(
            request.model_name,
            request.is_vlm,
            model_config=request.model_config,
            target_dtype=request.target_dtype,
        )

        assert result == (model, tokenizer)
        resolve_path.assert_called_once_with("org/model", revision=revision)
        loader.assert_called_once()
        assert loader.call_args.args == ("org/model",)
        assert loader.call_args.kwargs["revision"] == revision
        if backend != "vlm":
            detect_awq.assert_called_once_with("org/model", revision=revision)

    def test_private_mlx_lm_compatible_model_path_adapts_indexed_custom_shards(
        self, tmp_path: Path
    ) -> None:
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
            (model_dir / name).write_text("{}", encoding="utf-8")

        for name in ("layers-0.safetensors", "outside.safetensors", "mtp.safetensors"):
            (model_dir / name).write_text("", encoding="utf-8")

        (model_dir / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "weight_map": {
                        "a": "outside.safetensors",
                        "b": "layers-0.safetensors",
                        "c": "mtp.safetensors",
                    }
                }
            ),
            encoding="utf-8",
        )

        with model_lifecycle._mlx_lm_compatible_model_path(str(model_dir)) as compat:
            compat_path = Path(compat)

            assert compat_path != model_dir
            assert (compat_path / "config.json").is_symlink()
            assert (compat_path / "tokenizer.json").is_symlink()
            compat_shards = sorted(
                p.name for p in compat_path.glob("model*.safetensors")
            )
            assert compat_shards == [
                "model-00001-of-00003.safetensors",
                "model-00002-of-00003.safetensors",
                "model-00003-of-00003.safetensors",
            ]

    def test_private_mlx_lm_compatible_model_path_keeps_standard_model_shards(
        self, tmp_path: Path
    ) -> None:
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "model.safetensors").write_text("", encoding="utf-8")

        with model_lifecycle._mlx_lm_compatible_model_path(str(model_dir)) as compat:
            assert Path(compat) == model_dir

    def test_load_uses_adapter_override_for_text_only_multimodal_model(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _stub_generation_model(monkeypatch, config=_text_config())
        lifecycle, runner = _make_lifecycle(
            model_config=_runner_model_config(
                hf_config=SimpleNamespace(model_type="gemma4"),
                is_multimodal_model=True,
            )
        )

        lifecycle.load()

        assert runner._is_vlm is False

    def test_model_load_request_resolves_effective_vlm_once(self) -> None:
        hf_config = SimpleNamespace(model_type="custom")
        calls: list[object] = []

        class _Adapter:
            def should_force_text_backbone(self, config: object) -> bool:
                calls.append(config)
                return True

        runner = make_stub_runner(
            model_config=_runner_model_config(
                hf_config=hf_config,
                is_multimodal_model=True,
                trust_remote_code=True,
            ),
        )

        request = GenerationLoadRequest.from_runner(runner, _Adapter())

        assert request.model_name == "stub-model"
        assert request.hf_config is hf_config
        assert request.is_vlm is False
        assert request.target_dtype is not None
        assert request.tokenizer_config == {"trust_remote_code": True}
        assert calls == [hf_config]

    def test_effective_multimodal_gguf_is_rejected(self) -> None:
        runner = make_stub_runner(
            model_config=_runner_model_config(
                hf_config=SimpleNamespace(model_type="custom_vlm"),
                is_multimodal_model=True,
                quantization="gguf",
                model_weights="stub-model.gguf",
            ),
        )

        with pytest.raises(NotImplementedError, match="Multimodal GGUF"):
            GenerationLoadRequest.from_runner(
                runner,
                SimpleNamespace(should_force_text_backbone=lambda _: False),
            )

    def test_model_load_request_marks_pipeline_stage_lazy(self) -> None:
        # A pipeline-parallel stage (pp.size > 1) loads weights lazily so it can
        # prune its non-owned layers before the first eval; a single-stage load
        # stays eager. lazy_weights is the typed contract the mlx_lm loader reads.
        class _Adapter:
            def should_force_text_backbone(self, config: object) -> bool:
                return False

        class _FakeGroup:
            def __init__(self, size: int) -> None:
                self._size = size

            def rank(self) -> int:
                return 0

            def size(self) -> int:
                return self._size

        def _lazy_for(pp: PipelineGroup | None) -> bool:
            runner = make_stub_runner(
                model_config=_runner_model_config(),
                pp=pp,
            )
            return GenerationLoadRequest.from_runner(runner, _Adapter()).lazy_weights

        assert _lazy_for(None) is False
        assert _lazy_for(PipelineGroup(_FakeGroup(1))) is False
        assert _lazy_for(PipelineGroup(_FakeGroup(2))) is True

    def test_load_mlx_lm_text_model_forwards_lazy_flag(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The pp lazy_weights flag must reach mlx_lm's loader so a stage can prune
        # its non-owned layers before the first eval; a single-stage load is eager.
        captured: dict[str, object] = {}

        def _fake_load(
            path: str, *, tokenizer_config: object, lazy: bool, revision: str | None
        ) -> tuple[object, object]:
            captured["lazy"] = lazy
            return object(), object()

        monkeypatch.setattr(model_lifecycle, "mlx_lm_load", _fake_load)
        monkeypatch.setattr(
            model_lifecycle,
            "_mlx_lm_compatible_model_path",
            lambda name: contextlib.nullcontext(name),
        )
        lifecycle, _ = _make_lifecycle()

        lifecycle._load_mlx_lm_text_model("stub", {}, lazy=True)
        assert captured["lazy"] is True
        lifecycle._load_mlx_lm_text_model("stub", {}, lazy=False)
        assert captured["lazy"] is False

    def test_load_uses_adapter_override_for_qwen35_fp8_conditional_generation(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _stub_generation_model(monkeypatch, config=_text_config())
        lifecycle, runner = _make_lifecycle(
            model_config=_runner_model_config(
                hf_config=SimpleNamespace(
                    model_type="qwen3_5",
                    architectures=["Qwen3_5ForConditionalGeneration"],
                    quantization_config={"quant_method": "fp8"},
                ),
                is_multimodal_model=True,
            )
        )

        lifecycle.load()

        assert runner._is_vlm is False
        assert runner._multimodal_adapter is None

    def test_load_uses_adapter_override_for_qwen36_fp8_conditional_generation(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _stub_generation_model(monkeypatch, config=_text_config())
        lifecycle, runner = _make_lifecycle(
            model_config=_runner_model_config(
                hf_config=SimpleNamespace(
                    model_type="qwen3_6",
                    architectures=["Qwen3_6ForConditionalGeneration"],
                    quantization_config={"quant_method": "fp8"},
                ),
                is_multimodal_model=True,
            )
        )

        lifecycle.load()

        assert runner._is_vlm is False

    def test_load_keeps_qwen35_mlx_quant_dense_wrapper_as_vlm(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # The auto-mode override used to route MLX-quantized wrappers to the
        # text backbone; the pinned mlx-vlm floor serves them natively, so auto
        # mode now keeps the multimodal path.
        vision_tower = object()
        language_model = _Qwen35LanguageModelStub()
        fake_model = _qwen35_vlm_model(
            vision_tower=vision_tower,
            language_model=language_model,
        )
        _stub_generation_model(
            monkeypatch,
            config=fake_model.config,
            is_vlm=True,
            model=fake_model,
        )
        lifecycle, runner = _make_lifecycle(
            model_config=_runner_model_config(
                hf_config=SimpleNamespace(
                    model_type="qwen3_5",
                    architectures=["Qwen3_5ForConditionalGeneration"],
                    quantization={"group_size": 64, "bits": 4, "mode": "affine"},
                    vision_config=SimpleNamespace(spatial_merge_size=2),
                    text_config=SimpleNamespace(model_type="qwen3_5_text"),
                ),
                is_multimodal_model=True,
            )
        )

        lifecycle.load()

        assert runner._is_vlm is True
        assert isinstance(runner._multimodal_adapter, Qwen3VLMultimodalAdapter)

    def test_load_multimodal_native_mode_keeps_qwen35_fp8_as_vlm(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("VLLM_METAL_MULTIMODAL_MODE", "multimodal-native")
        reset_config()
        fake_model = _qwen35_vlm_model()
        _stub_generation_model(
            monkeypatch,
            config=fake_model.config,
            is_vlm=True,
            model=fake_model,
        )
        lifecycle, runner = _make_lifecycle(
            model_config=_runner_model_config(
                hf_config=SimpleNamespace(
                    model_type="qwen3_5",
                    architectures=["Qwen3_5ForConditionalGeneration"],
                    quantization_config={"quant_method": "fp8"},
                ),
                is_multimodal_model=True,
            )
        )

        lifecycle.load()

        assert runner._is_vlm is True

    def test_load_multimodal_native_qwen35_builds_model_adapter(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("VLLM_METAL_MULTIMODAL_MODE", "multimodal-native")
        reset_config()
        vision_tower = object()
        language_model = _Qwen35LanguageModelStub()
        fake_model = _qwen35_vlm_model(
            vision_tower=vision_tower,
            language_model=language_model,
        )
        _stub_generation_model(
            monkeypatch,
            config=fake_model.config,
            is_vlm=True,
            model=fake_model,
        )
        lifecycle, runner = _make_lifecycle(
            model_config=_runner_model_config(
                hf_config=SimpleNamespace(
                    model_type="qwen3_5",
                    architectures=["Qwen3_5ForConditionalGeneration"],
                    quantization_config={"quant_method": "fp8"},
                ),
                is_multimodal_model=True,
            )
        )

        lifecycle.load()

        assert runner._is_vlm is True
        assert isinstance(runner._multimodal_adapter, Qwen3VLMultimodalAdapter)
        assert runner._multimodal_adapter.text_model() is language_model
        assert isinstance(runner.encoder_cache, EncoderCache)

    def test_load_generic_vlm_leaves_model_adapter_unset(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _stub_generation_model(
            monkeypatch,
            config=SimpleNamespace(text_config=_text_config()),
            is_vlm=True,
        )
        lifecycle, runner = _make_lifecycle(
            model_config=_runner_model_config(
                hf_config=SimpleNamespace(model_type="phi3_v", architectures=[]),
                is_multimodal_model=True,
            )
        )

        lifecycle.load()

        assert runner._is_vlm is True
        assert runner._multimodal_adapter is None
        assert runner.encoder_cache is None

    def test_load_forced_text_backbone_uses_mlx_lm_loader(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        text_model = SimpleNamespace(config=_text_config())
        text_tokenizer = object()

        class _StubAWQLoader:
            @classmethod
            def for_model(cls, _model_name: str, *, revision: str | None) -> None:
                return None

        def _load_text(
            _model_name: str,
            *,
            tokenizer_config: object,
            lazy: bool,
            revision: str | None,
        ) -> tuple[object, object]:
            return text_model, text_tokenizer

        monkeypatch.setattr(model_lifecycle, "AWQQuantLoader", _StubAWQLoader)
        monkeypatch.setattr(model_lifecycle, "mlx_lm_load", _load_text)

        lifecycle, runner = _make_lifecycle(
            model_config=_runner_model_config(
                hf_config=SimpleNamespace(model_type="gemma4"),
                is_multimodal_model=True,
            )
        )
        lifecycle.load()

        assert runner.model is text_model
        assert runner.tokenizer is text_tokenizer
        assert runner._is_vlm is False

    def test_load_vlm_pipeline_parallel_uses_mlx_vlm_lazy_loading(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        vlm_model = _qwen35_vlm_model()
        vlm_tokenizer = object()
        vlm_lazy: list[bool] = []

        def _load_vlm(
            _model_name: str, *, lazy: bool, revision: str | None
        ) -> tuple[object, object]:
            vlm_lazy.append(lazy)
            return vlm_model, vlm_tokenizer

        monkeypatch.setattr(model_lifecycle, "mlx_vlm_load", _load_vlm)
        monkeypatch.setenv("VLLM_METAL_MULTIMODAL_MODE", "multimodal-native")
        reset_config()

        lifecycle, runner = _make_lifecycle(
            model_config=_runner_model_config(
                hf_config=SimpleNamespace(
                    model_type="qwen3_5",
                    architectures=["Qwen3_5ForConditionalGeneration"],
                    quantization_config={"quant_method": "fp8"},
                ),
                is_multimodal_model=True,
            )
        )
        runner.pp = PipelineGroup(SimpleNamespace(rank=lambda: 0, size=lambda: 2))
        lifecycle.load()

        assert runner.model is vlm_model
        assert runner.tokenizer is vlm_tokenizer
        assert runner._is_vlm is True
        assert vlm_lazy == [True]

    @pytest.mark.slow
    def test_load_auto_mode_real_qwen_fp8_checkpoint(
        self,
    ) -> None:
        model_path = os.environ.get("VLLM_METAL_QWEN_FP8_COMPAT_MODEL_PATH")
        if not model_path:
            pytest.skip("VLLM_METAL_QWEN_FP8_COMPAT_MODEL_PATH not set")
        if not Path(model_path).exists():
            pytest.skip(f"Model path does not exist: {model_path}")

        from transformers import AutoConfig

        from vllm_metal.compat import _patch_mlx_lm_qwen35_fp8_sanitize

        Gemma4MTPAssistantLoader.clear_cache()
        _patch_mlx_lm_qwen35_fp8_sanitize()

        hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=False)
        lifecycle, runner = _make_lifecycle(
            model_config=_runner_model_config(
                model=model_path,
                hf_config=hf_config,
                is_multimodal_model=True,
            )
        )
        try:
            lifecycle.load()

            assert runner._is_vlm is False
            assert runner.model is not None
            assert int(runner.model_args["vocab_size"]) > 0
        finally:
            Gemma4MTPAssistantLoader.clear_cache()

    def test_load_extracts_text_model_config_from_loaded_model(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake_tokenizer = object()
        fake_model, _ = _stub_generation_model(
            monkeypatch,
            config=_text_config(),
            tokenizer=fake_tokenizer,
        )
        lifecycle, runner = _make_lifecycle()

        lifecycle.load()

        assert runner.model is fake_model
        assert runner.tokenizer is fake_tokenizer
        assert runner.model_args["vocab_size"] == 32000
        assert runner.hidden_size == 4096
        assert runner.kv_cache_dtype is not None

    def test_load_wires_gemma4_mtp_assistant_after_target_dims(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = object()
        calls: list[dict[str, object]] = []
        call_time_dims: list[tuple[int, int]] = []

        class _StubGemma4MTPAssistantLoader:
            def load_if_needed(self, **kwargs: object) -> object:
                call_time_dims.append((runner.hidden_size, runner.head_dim))
                calls.append(kwargs)
                return runtime

        _stub_generation_model(monkeypatch, config=_text_config())
        monkeypatch.setattr(
            model_lifecycle,
            "Gemma4MTPAssistantLoader",
            _StubGemma4MTPAssistantLoader,
        )
        speculative_config = SimpleNamespace(
            method="mtp",
            draft_model_config=SimpleNamespace(
                model="assistant",
                revision=None,
                hf_config=SimpleNamespace(model_type="gemma4_mtp"),
            ),
        )
        lifecycle, runner = _make_lifecycle(
            model_config=_runner_model_config(),
        )
        runner.vllm_config.speculative_config = speculative_config

        lifecycle.load()

        assert runner._gemma4_mtp_assistant is runtime
        assert len(calls) == 1
        call = calls[0]
        assert call["speculative_config"] is speculative_config
        assert call["target_model_args"] == runner.model_args
        assert call_time_dims == [(4096, 128)]
        assert runner.hidden_size == 4096

    def test_load_clears_stale_gemma4_mtp_assistant_before_reload(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class _FailingGemma4MTPAssistantLoader:
            def load_if_needed(self, **kwargs: object) -> object:
                raise RuntimeError("assistant load failed")

        _stub_generation_model(monkeypatch, config=_text_config())
        monkeypatch.setattr(
            model_lifecycle,
            "Gemma4MTPAssistantLoader",
            _FailingGemma4MTPAssistantLoader,
        )
        speculative_config = SimpleNamespace(
            method="mtp",
            draft_model_config=SimpleNamespace(
                model="assistant",
                revision=None,
                hf_config=SimpleNamespace(model_type="gemma4_mtp"),
            ),
        )
        lifecycle, runner = _make_lifecycle(
            model_config=_runner_model_config(),
        )
        runner.vllm_config.speculative_config = speculative_config
        runner._gemma4_mtp_assistant = object()

        with pytest.raises(RuntimeError, match="assistant load failed"):
            lifecycle.load()

        assert runner._gemma4_mtp_assistant is None

    def test_gemma4_mtp_assistant_loader_clear_cache_resets_cached_load(
        self,
    ) -> None:
        load_model_calls = 0

        def _load_model(
            *args: object, **kwargs: object
        ) -> tuple[object, dict[str, object]]:
            nonlocal load_model_calls
            load_model_calls += 1
            return object(), {
                "model_type": "gemma4_assistant",
                "architectures": ["Gemma4AssistantForCausalLM"],
                "backbone_hidden_size": 4096,
                "text_config": {
                    "model_type": "gemma4_text",
                    "vocab_size": 32000,
                    "hidden_size": 256,
                    "num_hidden_layers": 1,
                    "layer_types": ["full_attention"],
                },
            }

        spec_config = SimpleNamespace(
            method="mtp",
            revision=None,
            draft_model_config=SimpleNamespace(
                model="/assistant",
                revision=None,
                hf_config=SimpleNamespace(model_type="gemma4_mtp"),
            ),
        )
        target_args = {
            "model_type": "gemma4_text",
            "vocab_size": 32000,
            "hidden_size": 4096,
            "num_hidden_layers": 1,
            "num_kv_shared_layers": 0,
            "layer_types": ["full_attention"],
        }
        loader = Gemma4MTPAssistantLoader(
            load_model_fn=_load_model,
            download_fn=lambda model_name, revision: Path(model_name),
        )

        first = loader.load_if_needed(
            speculative_config=spec_config,
            target_model_args=target_args,
        )
        second = loader.load_if_needed(
            speculative_config=spec_config,
            target_model_args=target_args,
        )
        assert first is second
        assert load_model_calls == 1

        Gemma4MTPAssistantLoader.clear_cache()

        third = loader.load_if_needed(
            speculative_config=spec_config,
            target_model_args=target_args,
        )
        assert third is not first
        assert load_model_calls == 2

    def test_load_merges_nested_text_config_for_non_vlm_model(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _stub_generation_model(
            monkeypatch,
            config=SimpleNamespace(
                vocab_size=_TEXT_MODEL_ARGS["vocab_size"],
                text_config=_text_config(),
            ),
        )
        lifecycle, runner = _make_lifecycle()

        lifecycle.load()

        assert runner._is_vlm is False
        assert runner.model_args["hidden_size"] == 4096
        assert runner.num_layers == 32
        assert runner.head_dim == 128

    def test_load_merges_nested_text_config_from_mlx_lm_args(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """mlx-lm Gemma4 exposes .args with dims nested inside text_config.

        Pin that model arg extraction flattens text_config onto the top level
        on the .args path as well, so every dim key sits at the top level for
        models whose mlx-lm ModelArgs only declares
        ``{model_type, text_config, vocab_size}`` at the top level.
        """
        args = SimpleNamespace(
            model_type="gemma4",
            vocab_size=_TEXT_MODEL_ARGS["vocab_size"],
            text_config=dict(_TEXT_MODEL_ARGS),
        )
        fake_model = SimpleNamespace(args=args)
        _stub_generation_model(monkeypatch, config=args, model=fake_model)
        lifecycle, runner = _make_lifecycle()

        lifecycle.load()

        assert runner.model is fake_model
        assert runner.num_layers == _TEXT_MODEL_ARGS["num_hidden_layers"]
        assert runner.num_kv_heads == _TEXT_MODEL_ARGS["num_key_value_heads"]
        assert runner.hidden_size == _TEXT_MODEL_ARGS["hidden_size"]
        assert runner.head_dim == (
            _TEXT_MODEL_ARGS["hidden_size"] // _TEXT_MODEL_ARGS["num_attention_heads"]
        )
        assert runner.model_args["model_type"] == "gemma4"
        assert runner.model_args["vocab_size"] == _TEXT_MODEL_ARGS["vocab_size"]

    def test_load_reads_omitted_text_keys_off_the_built_language_model(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A re-saved Qwen3.5 config omits full_attention_interval; mlx-lm
        resolves it to 4 on the built text model, and routing must see that."""
        text_config = dict(_TEXT_MODEL_ARGS) | {
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "linear_key_head_dim": 32,
            "linear_value_head_dim": 16,
            "linear_conv_kernel_dim": 3,
        }
        args = SimpleNamespace(model_type="qwen3_5", text_config=text_config)
        fake_model = SimpleNamespace(
            args=args,
            language_model=SimpleNamespace(
                args=SimpleNamespace(**text_config, full_attention_interval=4)
            ),
        )
        _stub_generation_model(monkeypatch, config=None, model=fake_model)
        lifecycle, runner = _make_lifecycle(
            model_config=_runner_model_config(is_hybrid=True)
        )

        lifecycle.load()

        assert runner.model_args["full_attention_interval"] == 4
        assert runner.hybrid_runtime_plan.family.label == "gdn"
        assert runner.hybrid_runtime_plan.layers.num_attention == (
            _TEXT_MODEL_ARGS["num_hidden_layers"] // 4
        )

    def test_load_routes_the_mlx_vlm_text_model_type_to_the_gdn_family(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """mlx-vlm flattens text_config, whose model_type is the ``_text`` name."""
        text_config = _text_config(
            model_type="qwen3_5_text",
            full_attention_interval=4,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=32,
            linear_value_head_dim=16,
            linear_conv_kernel_dim=3,
        )
        _stub_generation_model(
            monkeypatch, config=SimpleNamespace(text_config=text_config), is_vlm=True
        )
        lifecycle, runner = _make_lifecycle(
            model_config=_runner_model_config(is_hybrid=True, is_multimodal_model=True)
        )

        lifecycle.load()

        assert runner.model_args["model_type"] == "qwen3_5_text"
        assert runner.hybrid_runtime_plan.family.label == "gdn"

    def test_load_stt_model_loads_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_model = SimpleNamespace(
            create_runtime_adapter=lambda model_name: (object(), model_name)
        )
        calls: list[str] = []

        def _load_model(model_name: str) -> object:
            calls.append(model_name)
            return fake_model

        monkeypatch.setitem(
            sys.modules,
            "vllm_metal.stt.loader",
            SimpleNamespace(load_model=_load_model),
        )

        assert model_lifecycle.load_stt_model("stub-model") is fake_model
        assert calls == ["stub-model"]

    @pytest.mark.parametrize(
        "is_awq", [True, False], ids=["awq-checkpoint", "non-awq-checkpoint"]
    )
    def test_load_dispatches_by_awq_detection(
        self,
        monkeypatch: pytest.MonkeyPatch,
        is_awq: bool,
    ) -> None:
        """Pin both sides of the lifecycle dispatch contract introduced
        by the owner-shape refactor.

        When ``AWQQuantLoader.for_model`` reports an AWQ checkpoint,
        ``ModelLifecycle._load_generation_model`` delegates the actual
        load to ``AWQQuantLoader.load`` and never falls back to the
        generic ``mlx_lm.load``. When it returns ``None`` (non-AWQ),
        lifecycle uses the generic ``mlx_lm.load`` and never passes
        ``model_config`` (which is reserved for the AWQ owner's
        normalized quant config kwargs). AWQ *detection* — not the
        model-name string or any other heuristic — is what gates the
        dispatch.
        """
        fake_model = SimpleNamespace(config=_text_config())
        fake_tokenizer = object()
        awq_load_calls: list[dict[str, object]] = []
        mlx_lm_load_calls: list[dict[str, object]] = []

        class _StubAWQLoader:
            @classmethod
            def for_model(
                cls, _model_name: str, *, revision: str | None
            ) -> _StubAWQLoader | None:
                return cls() if is_awq else None

            def load(
                self,
                model_path: str,
                *,
                target_dtype: object,
                tokenizer_config: dict[str, object] | None,
                revision: str | None,
            ) -> tuple[object, object]:
                awq_load_calls.append(
                    {
                        "model_path": model_path,
                        "target_dtype": target_dtype,
                        "tokenizer_config": (
                            dict(tokenizer_config) if tokenizer_config else None
                        ),
                    }
                )
                return fake_model, fake_tokenizer

        def _fake_mlx_lm_load(*args: object, **kwargs: object) -> tuple[object, object]:
            mlx_lm_load_calls.append({"args": args, "kwargs": kwargs})
            return fake_model, fake_tokenizer

        monkeypatch.setattr(model_lifecycle, "AWQQuantLoader", _StubAWQLoader)
        monkeypatch.setattr(model_lifecycle, "mlx_lm_load", _fake_mlx_lm_load)

        lifecycle, runner = _make_lifecycle()
        lifecycle.load()

        assert runner.model is fake_model
        assert runner.tokenizer is fake_tokenizer

        if is_awq:
            assert len(awq_load_calls) == 1, (
                f"expected exactly one AWQQuantLoader.load() call, "
                f"got {len(awq_load_calls)}"
            )
            assert mlx_lm_load_calls == [], (
                "generic mlx_lm.load must NOT be called when AWQQuantLoader "
                "owns the load path"
            )
            call = awq_load_calls[0]
            assert call["model_path"] == "stub-model"
            assert call["target_dtype"] is not None, (
                "lifecycle must derive target_dtype from "
                "runner.model_config.dtype and thread it to the loader"
            )
            assert call["tokenizer_config"] == {"trust_remote_code": False}
        else:
            assert awq_load_calls == [], (
                "AWQQuantLoader.load must NOT be called for a non-AWQ checkpoint"
            )
            assert len(mlx_lm_load_calls) == 1
            # The generic path must NOT pass ``model_config`` (which is
            # reserved for the AWQ owner's normalized quant config kwargs).
            kwargs = mlx_lm_load_calls[0]["kwargs"]
            assert isinstance(kwargs, dict)
            assert "model_config" not in kwargs

    def test_load_routes_gguf_to_owner_on_quantization_detection(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """``quantization == "gguf"`` (set by the GGUF engine integration for a
        .gguf) delegates the whole load to ``GGUFModelLoader`` (lazily imported)
        and never calls generic ``mlx_lm.load`` — detection-routed like the AWQ
        branch, no env flag. The lifecycle threads the ``.gguf`` path from
        ``model_config.model_weights``, the ``--tokenizer`` dir, derived dtype,
        and tokenizer config to the owner.
        """
        fake_model = SimpleNamespace(config=_text_config())
        fake_tokenizer = object()
        loader_calls: list[dict[str, object]] = []
        mlx_lm_load_calls: list[dict[str, object]] = []

        class _StubGGUFLoader:
            def __init__(
                self,
                gguf_path: str,
                *,
                config_dir: str,
                tokenizer_dir: str,
                target_dtype: object,
                tokenizer_config: dict[str, object] | None = None,
            ) -> None:
                loader_calls.append(
                    {
                        "gguf_path": gguf_path,
                        "config_dir": config_dir,
                        "tokenizer_dir": tokenizer_dir,
                        "target_dtype": target_dtype,
                        "tokenizer_config": (
                            dict(tokenizer_config) if tokenizer_config else None
                        ),
                    }
                )

            def load(self) -> tuple[object, object]:
                return fake_model, fake_tokenizer

        def _fake_mlx_lm_load(*args: object, **kwargs: object) -> tuple[object, object]:
            mlx_lm_load_calls.append({"args": args, "kwargs": kwargs})
            return fake_model, fake_tokenizer

        # The owner is imported lazily inside the branch; inject the stub module
        # so the `from vllm_metal.gguf.loader import GGUFModelLoader` resolves to it.
        monkeypatch.setitem(
            sys.modules,
            "vllm_metal.gguf.loader",
            SimpleNamespace(GGUFModelLoader=_StubGGUFLoader),
        )
        monkeypatch.setattr(model_lifecycle, "mlx_lm_load", _fake_mlx_lm_load)
        config_dir = tmp_path / "config"
        tokenizer_dir = tmp_path / "tokenizer"
        config_dir.mkdir()
        tokenizer_dir.mkdir()

        lifecycle, runner = _make_lifecycle(
            model_config=_runner_model_config(
                model=str(config_dir),
                quantization="gguf",
                tokenizer=str(tokenizer_dir),
                model_weights="stub-model.gguf",
            )
        )
        lifecycle.load()

        assert runner.model is fake_model
        assert runner.tokenizer is fake_tokenizer
        assert mlx_lm_load_calls == [], (
            "generic mlx_lm.load must NOT be called when the GGUF owner owns "
            "the load path"
        )
        assert len(loader_calls) == 1
        call = loader_calls[0]
        assert call["gguf_path"] == "stub-model.gguf"
        assert call["config_dir"] == str(config_dir)
        assert call["tokenizer_dir"] == str(tokenizer_dir)
        assert call["target_dtype"] is not None, (
            "lifecycle must derive target_dtype from runner.model_config.dtype "
            "and thread it to the owner"
        )
        assert call["tokenizer_config"] == {"trust_remote_code": False}

    def test_non_gguf_model_does_not_route_or_import_gguf(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A non-GGUF model (``quantization != "gguf"``) is never routed to the
        GGUF owner, and the optional ``gguf`` package is never imported.

        Tripwire: block the ``gguf`` package itself (``sys.modules[...] = None``
        raises on import). A clean generic load proves a default install with no
        gguf extra is unaffected by the lazy owner import.
        """
        fake_model = SimpleNamespace(config=_text_config())
        fake_tokenizer = object()

        class _StubAWQLoader:
            @classmethod
            def for_model(cls, _model_name: str, *, revision: str | None) -> None:
                return None

        monkeypatch.setitem(sys.modules, "gguf", None)
        monkeypatch.setattr(model_lifecycle, "AWQQuantLoader", _StubAWQLoader)
        monkeypatch.setattr(
            model_lifecycle,
            "mlx_lm_load",
            lambda *_args, **_kwargs: (fake_model, fake_tokenizer),
        )
        lifecycle, runner = _make_lifecycle(
            model_config=_runner_model_config()  # no quantization -> not gguf
        )

        lifecycle.load()  # must not raise: the gguf owner import is never reached

        assert runner.model is fake_model
        assert runner._is_vlm is False
        assert runner.model_args["vocab_size"] == 32000

    def test_gguf_load_threads_config_and_tokenizer_sources(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """GGUF loads keep the config source separate from the tokenizer source."""
        loader_sources: list[tuple[str, str]] = []

        class _StubGGUFLoader:
            def __init__(
                self,
                gguf_path: str,
                *,
                config_dir: str,
                tokenizer_dir: str,
                target_dtype: object,
                tokenizer_config: dict[str, object] | None = None,
            ) -> None:
                loader_sources.append((config_dir, tokenizer_dir))
                self._tokenizer_dir = tokenizer_dir

            def load(self) -> tuple[object, object]:
                return (
                    SimpleNamespace(config=_text_config()),
                    f"tokenizer::{self._tokenizer_dir}",
                )

        monkeypatch.setitem(
            sys.modules,
            "vllm_metal.gguf.loader",
            SimpleNamespace(GGUFModelLoader=_StubGGUFLoader),
        )

        def _load_tokenizer_for(tokenizer_dir: str) -> object:
            lifecycle, runner = _make_lifecycle(
                model_config=_runner_model_config(
                    model="/config-dir",
                    quantization="gguf",
                    tokenizer=tokenizer_dir,
                    model_weights="stub-model.gguf",
                )
            )
            lifecycle.load()
            return runner.tokenizer

        first = _load_tokenizer_for("/dir-a")
        second = _load_tokenizer_for("/dir-b")

        assert loader_sources == [("/config-dir", "/dir-a"), ("/config-dir", "/dir-b")]
        assert first == "tokenizer::/dir-a"
        assert second == "tokenizer::/dir-b"


class TestResolveModelDims:
    def _resolve(self, args: dict[str, object], *, is_hybrid: bool = False) -> object:
        lifecycle, runner = _make_lifecycle(
            model_args=args, model_config=_runner_model_config(is_hybrid=is_hybrid)
        )
        lifecycle.resolve_model_dims()
        return runner

    def test_hybrid_model_installs_its_family_plan(self) -> None:
        lifecycle, runner = _make_lifecycle(
            model_args=_GDN_HYBRID_ARGS,
            model_config=_runner_model_config(is_hybrid=True, dtype=torch.bfloat16),
        )
        runner.cache_config.mamba_ssm_cache_dtype = "auto"
        lifecycle.resolve_model_dims()

        assert runner.hybrid_runtime_plan.family.label == "gdn"
        assert runner.hybrid_runtime_plan.layers.attention_indices == (3, 7)
        assert runner.hybrid_runtime_plan.state_dtypes == (
            torch.bfloat16,
            torch.bfloat16,
        )

    def test_hybrid_state_dtypes_come_from_the_configured_registry(self) -> None:
        state_dtypes = (torch.float32, torch.float32)
        model_cls = SimpleNamespace(
            get_mamba_state_dtype_from_config=lambda _: state_dtypes
        )
        registry = SimpleNamespace(
            resolve_model_cls=lambda *_args, **_kwargs: (model_cls, "override")
        )
        lifecycle, runner = _make_lifecycle(
            model_args=_GDN_HYBRID_ARGS,
            model_config=_runner_model_config(is_hybrid=True, registry=registry),
        )

        lifecycle.resolve_model_dims()

        assert runner.hybrid_runtime_plan.state_dtypes == state_dtypes

    @pytest.mark.parametrize(
        ("model_dtype", "conv_dtype", "ssm_dtype", "supported"),
        [
            (torch.bfloat16, "auto", "auto", True),
            (torch.bfloat16, "auto", "float32", True),
            (torch.float16, "auto", "auto", True),
            (torch.float32, "auto", "auto", True),
            (torch.bfloat16, "auto", "float16", False),
            (torch.bfloat16, "float32", "bfloat16", False),
            (torch.float32, "float16", "float16", False),
        ],
    )
    def test_gdn_dtype_support_is_checked_at_initialization(
        self, model_dtype, conv_dtype, ssm_dtype, supported
    ) -> None:
        lifecycle, runner = _make_lifecycle(
            model_args=_GDN_HYBRID_ARGS,
            model_config=_runner_model_config(is_hybrid=True, dtype=model_dtype),
        )
        runner.cache_config.mamba_cache_dtype = conv_dtype
        runner.cache_config.mamba_ssm_cache_dtype = ssm_dtype

        if supported:
            lifecycle.resolve_model_dims()
            assert runner.hybrid_runtime_plan is not None
        else:
            with pytest.raises(ValueError, match="--mamba-ssm-cache-dtype float32"):
                lifecycle.resolve_model_dims()
            assert runner.hybrid_runtime_plan is None

    def test_routing_follows_the_typed_field_not_the_args(self) -> None:
        runner = self._resolve(_GDN_HYBRID_ARGS, is_hybrid=False)

        assert runner.hybrid_runtime_plan is None

    def test_hybrid_model_without_a_family_rejects_before_any_plan(self) -> None:
        lifecycle, runner = _make_lifecycle(
            model_args=_JAMBA_ARGS,
            model_config=_runner_model_config(is_hybrid=True),
        )

        with pytest.raises(NotImplementedError, match="model_type='jamba'"):
            lifecycle.resolve_model_dims()
        assert runner.hybrid_runtime_plan is None

    def test_nemotron_pattern_resolves_from_layers_block_type(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The published checkpoint config carries layers_block_type only;
        # mlx-lm resolves hybrid_override_pattern on the built args.
        raw_args = {
            k: v
            for k, v in NEMOTRON_H_TINY_ARGS.items()
            if k != "hybrid_override_pattern"
        }
        raw_args["layers_block_type"] = ["mamba", "mlp", "attention", "mamba"]
        raw_args["num_hidden_layers"] = 4
        model = NemotronHModel(NemotronHModelArgs(**raw_args))
        _stub_generation_model(monkeypatch, config=None, model=model)
        lifecycle, runner = _make_lifecycle(
            model_config=_runner_model_config(is_hybrid=True)
        )

        lifecycle.load()

        assert runner.hybrid_runtime_plan.family.label == "nemotron_h"
        assert runner.hybrid_runtime_plan.layers.layer_roles == (
            "state",
            "stateless",
            "attention",
            "state",
        )

    def test_nemotron_model_installs_its_family_plan(self) -> None:
        runner = self._resolve(_NEMOTRON_H_ARGS, is_hybrid=True)

        assert runner.hybrid_runtime_plan.family.label == "nemotron_h"
        assert runner.hybrid_runtime_plan.layers.attention_indices == (
            5,
            12,
            19,
            26,
            33,
            42,
        )
        assert runner.hybrid_runtime_plan.layers.num_state == 23
        assert runner.head_dim == 128

    def test_nemotron_head_dim_resolves_like_mlx_lm_when_omitted(self) -> None:
        args = {k: v for k, v in _NEMOTRON_H_ARGS.items() if k != "head_dim"}

        runner = self._resolve(args, is_hybrid=True)

        assert runner.head_dim == 2688 // 32

    def test_standard_attention(self) -> None:
        runner = self._resolve(
            {
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "hidden_size": 4096,
            }
        )

        assert runner.num_layers == 32
        assert runner.num_kv_heads == 8
        assert runner.head_dim == 128

    @pytest.mark.parametrize(
        ("args", "expected_head_dim"),
        [
            (
                {
                    "num_hidden_layers": 47,
                    "num_attention_heads": 20,
                    "num_key_value_heads": 20,
                    "hidden_size": 2048,
                    "kv_lora_rank": 512,
                    "qk_rope_head_dim": 64,
                },
                512 + 64,
            ),
            (
                {
                    "num_hidden_layers": 28,
                    "num_attention_heads": 16,
                    "hidden_size": 2048,
                    "kv_lora_rank": 256,
                },
                256 + MLA_DEFAULT_QK_ROPE_HEAD_DIM,
            ),
        ],
    )
    def test_mla_sets_expected_head_dim(
        self,
        args: dict[str, object],
        expected_head_dim: int,
    ) -> None:
        runner = self._resolve(args)

        assert runner.num_kv_heads == 1
        assert runner.head_dim == expected_head_dim
        assert runner.mla_latent_dim == expected_head_dim

    def test_missing_dims_raise(self) -> None:
        lifecycle, _ = _make_lifecycle(model_args={"num_hidden_layers": 32})

        with pytest.raises(ValueError, match="Cannot resolve model dimensions"):
            lifecycle.resolve_model_dims()

    # Gemma4-style interleaved sliding/full attention -> non-None per-layer KV.
    _NON_UNIFORM_ARGS: dict[str, object] = {
        "num_hidden_layers": 2,
        "num_attention_heads": 8,
        "num_key_value_heads": 8,
        "hidden_size": 256,
        "layer_types": ["sliding_attention", "full_attention"],
        "sliding_window": 128,
    }

    def test_non_uniform_per_layer_kv_rejects_pipeline_parallel(self) -> None:
        """PP is rejected for non-uniform per-layer KV models: the split only
        rebinds num_layers, so a stage would index the full-length per-layer
        lists by LOCAL layer and read the wrong global layer."""
        lifecycle, runner = _make_lifecycle(model_args=dict(self._NON_UNIFORM_ARGS))
        runner.pp = SimpleNamespace(size=2)  # worker sets the PP group before load

        with pytest.raises(NotImplementedError, match="non-uniform per-layer KV"):
            lifecycle.resolve_model_dims()

    def test_non_uniform_per_layer_kv_allowed_without_pipeline_parallel(self) -> None:
        """The same model loads on the single-stage path (pp=1, stub default)."""
        lifecycle, runner = _make_lifecycle(model_args=dict(self._NON_UNIFORM_ARGS))

        lifecycle.resolve_model_dims()  # must not raise

        assert runner.sliding_window_per_layer == [128, -1]

    def test_uniform_model_leaves_per_layer_shapes_none(self) -> None:
        runner = self._resolve(
            {
                "num_hidden_layers": 4,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "hidden_size": 2048,
            }
        )

        assert runner.kv_heads_per_layer is None
        assert runner.head_dim_per_layer is None

    def test_gemma4_31b_sets_heterogeneous_per_layer_shapes(self) -> None:
        runner = self._resolve(
            {
                "num_hidden_layers": 4,
                "num_attention_heads": 32,
                "num_key_value_heads": 16,
                "head_dim": 256,
                "hidden_size": 5376,
                "layer_types": [
                    "sliding_attention",
                    "full_attention",
                    "sliding_attention",
                    "full_attention",
                ],
                "global_head_dim": 512,
                "num_global_key_value_heads": 4,
            }
        )

        # Cache allocation uses the max head_dim; per-layer lists carry
        # the true sliding vs full shapes.
        assert runner.head_dim == 512
        assert runner.num_kv_heads == 16
        assert runner.kv_heads_per_layer == [16, 4, 16, 4]
        assert runner.head_dim_per_layer == [256, 512, 256, 512]

    def test_gemma4_e2b_sets_heterogeneous_per_layer_shapes_without_global_kv(
        self,
    ) -> None:
        """E2B-style configs omit ``num_global_key_value_heads`` entirely."""
        runner = self._resolve(
            {
                "num_hidden_layers": 4,
                "num_attention_heads": 8,
                "num_key_value_heads": 1,
                "head_dim": 256,
                "hidden_size": 2048,
                "layer_types": [
                    "sliding_attention",
                    "full_attention",
                    "sliding_attention",
                    "full_attention",
                ],
                "global_head_dim": 512,
            }
        )

        assert runner.head_dim == 512
        assert runner.kv_heads_per_layer == [1, 1, 1, 1]
        assert runner.head_dim_per_layer == [256, 512, 256, 512]


class _Gemma4Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(16, 8)
        self.embed_scale = 8**0.5


class _Gemma4TextModel:
    """Shape of the mlx_lm ``gemma4.Model`` wrapper."""

    def __init__(self) -> None:
        # hidden_size/num_attention_heads/num_hidden_layers/num_key_value_heads
        # are the minimum resolve_model_dims() needs to compute a head_dim:
        # with only hidden_size present, num_layers/num_kv_heads/head_dim all
        # come back None and _install_runner_attention_dims raises before
        # load() reaches any of this test's own assertions.
        self.args = {
            "model_type": "gemma4",
            "vocab_size": 16,
            "text_config": {
                "hidden_size": 8,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
            },
        }
        self.language_model = SimpleNamespace(model=_Gemma4Backbone())

    def __call__(self, inputs, cache=None, input_embeddings=None):
        return SimpleNamespace(logits=mx.zeros((1, inputs.shape[1], 16)))


def _fake_sidecar() -> Gemma4VisionSidecar:
    return Gemma4VisionSidecar(
        vision_tower=object(),
        embed_vision=object(),
        pixel_dtype=mx.bfloat16,
        num_parameters=1,
        num_bytes=2,
    )


def _gemma4_runner_config(
    *,
    multimodal_config: object | None = None,
    is_multimodal_model: bool = True,
) -> object:
    return _runner_model_config(
        hf_config=SimpleNamespace(
            model_type="gemma4",
            architectures=["Gemma4ForConditionalGeneration"],
            text_config=SimpleNamespace(
                model_type="gemma4_text",
                hidden_size=8,
                use_bidirectional_attention="vision",
            ),
        ),
        is_multimodal_model=is_multimodal_model,
        multimodal_config=multimodal_config,
    )


_CACHED_SNAPSHOT = Path("/hf-cache/models--org--gemma-4/snapshots/0123abcd")


class TestTextSidecarLifecycle:
    def _force_mode(self, monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
        monkeypatch.setattr(
            DefaultModelAdapter,
            "multimodal_backbone_mode",
            lambda self, model_config, speculative_config=None: mode,
        )
        # "stub-model" is no local directory, so it resolves like a repo id:
        # to the cached snapshot mode selection accepted.
        monkeypatch.setattr(
            model_adapter_module,
            "_resolve_cached_snapshot",
            lambda model_config: _CACHED_SNAPSHOT,
        )

    def test_text_sidecar_loads_mlx_lm_backbone_and_sidecar(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._force_mode(monkeypatch, "text_sidecar")
        text_model = _Gemma4TextModel()
        _stub_generation_model(monkeypatch, config=None, is_vlm=False, model=text_model)
        loaded_paths: list[Path] = []
        sidecar = _fake_sidecar()

        def _load(path: Path, **_: object) -> Gemma4VisionSidecar:
            loaded_paths.append(Path(path))
            return sidecar

        monkeypatch.setattr(
            model_lifecycle.Gemma4VisionSidecar, "load", staticmethod(_load)
        )
        lifecycle, runner = _make_lifecycle(model_config=_gemma4_runner_config())

        lifecycle.load()

        # The text model loaded from "stub-model" (_stub_generation_model
        # asserts it); mlx-vlm's load_model reads a path, so the sidecar gets
        # the resolved snapshot.
        assert loaded_paths == [_CACHED_SNAPSHOT]
        assert runner._is_vlm is True
        assert isinstance(runner._multimodal_adapter, Gemma4MultimodalAdapter)
        assert runner._multimodal_adapter.text_model() is runner.model
        assert runner._forward_model is runner.model
        assert runner.encoder_cache is not None
        assert runner._multimodal_adapter.bidirectional_layer_kinds == frozenset(
            {"sliding"}
        )

    def test_text_only_mode_keeps_today_s_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._force_mode(monkeypatch, "text_only")
        text_model = _Gemma4TextModel()
        _stub_generation_model(monkeypatch, config=None, is_vlm=False, model=text_model)
        lifecycle, runner = _make_lifecycle(model_config=_gemma4_runner_config())

        lifecycle.load()

        assert runner._is_vlm is False
        assert runner._multimodal_adapter is None
        assert runner.encoder_cache is None

    def test_drafter_with_text_sidecar_is_fatal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._force_mode(monkeypatch, "text_sidecar")
        _stub_generation_model(
            monkeypatch, config=None, is_vlm=False, model=_Gemma4TextModel()
        )
        lifecycle, runner = _make_lifecycle(model_config=_gemma4_runner_config())
        runner.vllm_config.speculative_config = object()

        with pytest.raises(RuntimeError, match="speculative decoding"):
            lifecycle.load()

    def test_sidecar_failure_is_fatal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._force_mode(monkeypatch, "text_sidecar")
        _stub_generation_model(
            monkeypatch, config=None, is_vlm=False, model=_Gemma4TextModel()
        )

        def _boom(path: Path, **_: object) -> Gemma4VisionSidecar:
            raise ValueError("Missing parameters: vision_tower.encoder")

        monkeypatch.setattr(
            model_lifecycle.Gemma4VisionSidecar, "load", staticmethod(_boom)
        )
        lifecycle, _ = _make_lifecycle(model_config=_gemma4_runner_config())

        with pytest.raises(ValueError, match="Missing parameters"):
            lifecycle.load()

    def test_from_runner_without_mode_method_falls_back_to_predicate(self) -> None:
        runner = make_stub_runner(model_config=_gemma4_runner_config())
        request = GenerationLoadRequest.from_runner(
            runner, SimpleNamespace(should_force_text_backbone=lambda _: True)
        )
        assert request.backbone_mode == "text_only"
        assert request.is_vlm is False

    def test_drift_to_text_only_with_multimodal_config_kept_is_fatal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The API process kept `multimodal_config` (it decided text_sidecar
        # or native earlier), but this process's own mode resolution now
        # says text_only -- the checkpoint directory or
        # VLLM_METAL_MULTIMODAL_MODE changed between the two processes.
        self._force_mode(monkeypatch, "text_only")
        runner = make_stub_runner(
            model_config=_gemma4_runner_config(multimodal_config=SimpleNamespace())
        )

        with pytest.raises(RuntimeError, match="drifted"):
            GenerationLoadRequest.from_runner(runner, runner._model_adapter)

    def test_load_request_carries_the_resolved_sidecar_checkpoint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._force_mode(monkeypatch, "text_sidecar")
        runner = make_stub_runner(model_config=_gemma4_runner_config())

        request = GenerationLoadRequest.from_runner(runner, runner._model_adapter)

        assert request.model_name == "stub-model"
        assert request.sidecar_checkpoint == _CACHED_SNAPSHOT

    def test_unresolvable_sidecar_checkpoint_is_fatal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Mode selection accepted the checkpoint, but by the time this
        # process loads, the snapshot no longer resolves (cache cleared).
        self._force_mode(monkeypatch, "text_sidecar")
        monkeypatch.setattr(
            model_adapter_module, "_resolve_cached_snapshot", lambda model_config: None
        )
        runner = make_stub_runner(model_config=_gemma4_runner_config())

        with pytest.raises(RuntimeError, match="no longer resolves"):
            GenerationLoadRequest.from_runner(runner, runner._model_adapter)

    def test_other_modes_carry_no_sidecar_checkpoint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._force_mode(monkeypatch, "native")
        runner = make_stub_runner(model_config=_gemma4_runner_config())

        request = GenerationLoadRequest.from_runner(runner, runner._model_adapter)

        assert request.sidecar_checkpoint is None

    def test_sidecar_not_loaded_when_not_flagged_multimodal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # backbone_mode can resolve to text_sidecar while is_multimodal_model
        # is False (e.g. a stub/adapter disagreement); is_vlm then stays
        # False, and the sidecar must not be loaded for a request nothing
        # will ever route images to.
        self._force_mode(monkeypatch, "text_sidecar")
        text_model = _Gemma4TextModel()
        _stub_generation_model(monkeypatch, config=None, is_vlm=False, model=text_model)
        sidecar_loads: list[Path] = []

        def _load(path: Path, **_: object) -> Gemma4VisionSidecar:
            sidecar_loads.append(Path(path))
            return _fake_sidecar()

        monkeypatch.setattr(
            model_lifecycle.Gemma4VisionSidecar, "load", staticmethod(_load)
        )
        lifecycle, runner = _make_lifecycle(
            model_config=_gemma4_runner_config(is_multimodal_model=False)
        )

        lifecycle.load()

        assert sidecar_loads == []
        assert runner._is_vlm is False
        assert runner._multimodal_adapter is None
        assert runner.encoder_cache is None

    def test_sidecar_is_not_reachable_from_runner_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Spec invariant (5.1/10.1): sidecar submodules never hang off
        runner.model, or patch_model/CompiledMLPBlocks/LoRA would walk them."""
        self._force_mode(monkeypatch, "text_sidecar")
        text_model = _Gemma4TextModel()
        _stub_generation_model(monkeypatch, config=None, is_vlm=False, model=text_model)
        sidecar = _fake_sidecar()
        monkeypatch.setattr(
            model_lifecycle.Gemma4VisionSidecar,
            "load",
            staticmethod(lambda path, **_: sidecar),
        )
        lifecycle, runner = _make_lifecycle(model_config=_gemma4_runner_config())

        lifecycle.load()

        visited: set[int] = set()

        def _reaches_sidecar_submodule(obj: object) -> bool:
            if id(obj) in visited:
                return False
            visited.add(id(obj))
            if obj is sidecar.vision_tower or obj is sidecar.embed_vision:
                return True
            if isinstance(obj, dict):
                children = obj.values()
            elif hasattr(obj, "__dict__"):
                children = vars(obj).values()
            else:
                return False
            return any(_reaches_sidecar_submodule(child) for child in children)

        assert _reaches_sidecar_submodule(runner.model) is False

    def test_turboquant_with_text_sidecar_is_fatal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._force_mode(monkeypatch, "text_sidecar")
        _stub_generation_model(
            monkeypatch, config=None, is_vlm=False, model=_Gemma4TextModel()
        )
        monkeypatch.setattr(
            model_lifecycle, "get_config", lambda: SimpleNamespace(turboquant=True)
        )
        lifecycle, _ = _make_lifecycle(model_config=_gemma4_runner_config())

        with pytest.raises(RuntimeError, match="unquantized KV cache"):
            lifecycle.load()

    def test_softcap_with_text_sidecar_is_fatal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._force_mode(monkeypatch, "text_sidecar")
        _stub_generation_model(
            monkeypatch, config=None, is_vlm=False, model=_Gemma4TextModel()
        )
        model_config = _gemma4_runner_config()
        model_config.hf_config.text_config.attn_logit_softcapping = 50.0
        lifecycle, _ = _make_lifecycle(model_config=model_config)

        with pytest.raises(RuntimeError, match="softcap"):
            lifecycle.load()

    def test_attention_sinks_with_text_sidecar_are_fatal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._force_mode(monkeypatch, "text_sidecar")
        text_model = _Gemma4TextModel()
        text_model.language_model.model.layers = [
            SimpleNamespace(self_attn=SimpleNamespace(sinks=mx.zeros((2,))))
        ]
        _stub_generation_model(monkeypatch, config=None, is_vlm=False, model=text_model)
        monkeypatch.setattr(
            model_lifecycle.Gemma4VisionSidecar,
            "load",
            staticmethod(lambda p, **_: _fake_sidecar()),
        )
        lifecycle, _ = _make_lifecycle(model_config=_gemma4_runner_config())

        with pytest.raises(RuntimeError, match="attention sinks"):
            lifecycle.load()
