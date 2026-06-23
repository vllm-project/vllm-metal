# SPDX-License-Identifier: Apache-2.0
"""Tests for model lifecycle behavior."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import vllm_metal.envs as envs
from tests.stub_runner import make_stub_runner
from vllm_metal.attention.impls.mla import MLA_DEFAULT_QK_ROPE_HEAD_DIM
from vllm_metal.config import reset_config
from vllm_metal.multimodal.qwen3_vl import Qwen3VLMultimodalAdapter
from vllm_metal.v1 import model_lifecycle
from vllm_metal.v1.gemma4_mtp import Gemma4MTPAssistantLoader
from vllm_metal.v1.mm import EncoderCache
from vllm_metal.v1.model_lifecycle import ModelLifecycle

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


class _BaseSlotTextConfig:
    __slots__ = ("vocab_size", "num_hidden_layers", "num_attention_heads")

    def __init__(
        self,
        *,
        vocab_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
    ) -> None:
        self.vocab_size = vocab_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads


class _SlotTextConfig(_BaseSlotTextConfig):
    __slots__ = ("num_key_value_heads", "hidden_size")

    def __init__(
        self,
        *,
        vocab_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        hidden_size: int,
    ) -> None:
        super().__init__(
            vocab_size=vocab_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
        )
        self.num_key_value_heads = num_key_value_heads
        self.hidden_size = hidden_size


def _runner_model_config(**overrides: object) -> object:
    values = {
        "model": "stub-model",
        "hf_config": None,
        "is_multimodal_model": False,
        "trust_remote_code": False,
        "dtype": torch.float16,
        "quantization": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


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


def _cache_generation_model(
    monkeypatch: pytest.MonkeyPatch,
    *,
    config: object,
    tokenizer: object | None = None,
    is_vlm: bool = False,
    model: object | None = None,
) -> tuple[object, object]:
    fake_model = model or SimpleNamespace(config=config)
    fake_tokenizer = object() if tokenizer is None else tokenizer
    cache_key = model_lifecycle._generation_cache_key("stub-model", is_vlm=is_vlm)
    monkeypatch.setattr(
        model_lifecycle,
        "_MODEL_CACHE",
        {cache_key: (fake_model, fake_tokenizer)},
    )
    return fake_model, fake_tokenizer


def _make_lifecycle(
    *,
    model_args: dict[str, object] | None = None,
    model_config: object | None = None,
) -> tuple[ModelLifecycle, object]:
    runner = make_stub_runner(
        model_args=model_args,
        metal_config=SimpleNamespace(debug=False),
        model_config=model_config or _runner_model_config(),
    )
    lifecycle = ModelLifecycle(runner, runner._model_adapter)
    return lifecycle, runner


class TestModelLifecycle:
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
        _cache_generation_model(monkeypatch, config=_text_config())
        lifecycle, runner = _make_lifecycle(
            model_config=_runner_model_config(
                hf_config=SimpleNamespace(model_type="gemma4"),
                is_multimodal_model=True,
            )
        )

        lifecycle.load()

        assert runner._is_vlm is False

    def test_load_uses_adapter_override_for_qwen35_fp8_conditional_generation(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _cache_generation_model(monkeypatch, config=_text_config())
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
        _cache_generation_model(monkeypatch, config=_text_config())
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

    def test_load_multimodal_native_mode_keeps_qwen35_fp8_as_vlm(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("VLLM_METAL_MULTIMODAL_MODE", "multimodal-native")
        reset_config()
        fake_model = _qwen35_vlm_model()
        _cache_generation_model(
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
        _cache_generation_model(
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
        _cache_generation_model(
            monkeypatch,
            config=SimpleNamespace(text_config=_text_config()),
            is_vlm=True,
        )
        lifecycle, runner = _make_lifecycle(
            model_config=_runner_model_config(
                hf_config=SimpleNamespace(model_type="phi3_v"),
                is_multimodal_model=True,
            )
        )

        lifecycle.load()

        assert runner._is_vlm is True
        assert runner._multimodal_adapter is None
        assert runner.encoder_cache is None

    def test_generation_cache_separates_text_and_vlm_variants(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        text_model = SimpleNamespace(config=_text_config())
        text_tokenizer = object()
        vlm_model = _qwen35_vlm_model()
        vlm_tokenizer = object()
        monkeypatch.setattr(
            model_lifecycle,
            "_MODEL_CACHE",
            {
                model_lifecycle._generation_cache_key("stub-model", is_vlm=False): (
                    text_model,
                    text_tokenizer,
                ),
                model_lifecycle._generation_cache_key("stub-model", is_vlm=True): (
                    vlm_model,
                    vlm_tokenizer,
                ),
            },
        )

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
        lifecycle.load()

        assert runner.model is vlm_model
        assert runner.tokenizer is vlm_tokenizer
        assert runner._is_vlm is True

    def test_load_text_only_compat_mode_keeps_generic_vlm_native(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("VLLM_METAL_MULTIMODAL_MODE", "text-only-compat")
        reset_config()
        _cache_generation_model(
            monkeypatch,
            config=SimpleNamespace(text_config=_text_config()),
            is_vlm=True,
        )
        lifecycle, runner = _make_lifecycle(
            model_config=_runner_model_config(
                hf_config=SimpleNamespace(model_type="phi3_v"),
                is_multimodal_model=True,
            )
        )

        lifecycle.load()

        assert runner._is_vlm is True

    @pytest.mark.slow
    def test_load_text_only_compat_real_qwen_fp8_checkpoint(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        model_path = os.environ.get("VLLM_METAL_QWEN_FP8_COMPAT_MODEL_PATH")
        if not model_path:
            pytest.skip("VLLM_METAL_QWEN_FP8_COMPAT_MODEL_PATH not set")
        if not Path(model_path).exists():
            pytest.skip(f"Model path does not exist: {model_path}")

        from transformers import AutoConfig

        from vllm_metal.compat import _patch_mlx_lm_qwen35_fp8_sanitize

        monkeypatch.setenv("VLLM_METAL_MULTIMODAL_MODE", "text-only-compat")
        reset_config()
        model_lifecycle.reset_model_cache()
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
            model_lifecycle.reset_model_cache()

    def test_load_extracts_text_model_config_from_cached_model(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake_tokenizer = object()
        fake_model, _ = _cache_generation_model(
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

        class _StubGemma4MTPAssistantLoader:
            def load_if_needed(self, **kwargs: object) -> object:
                calls.append(kwargs)
                return runtime

        _cache_generation_model(monkeypatch, config=_text_config())
        monkeypatch.setattr(
            model_lifecycle,
            "Gemma4MTPAssistantLoader",
            _StubGemma4MTPAssistantLoader,
        )
        speculative_config = SimpleNamespace(
            method="mtp",
            draft_model_config=SimpleNamespace(
                model="assistant",
                hf_config=SimpleNamespace(model_type="gemma4_mtp"),
            ),
        )
        lifecycle, runner = _make_lifecycle(
            model_config=_runner_model_config(),
        )
        runner.vllm_config = SimpleNamespace(speculative_config=speculative_config)

        lifecycle.load()

        assert runner._gemma4_mtp_assistant is runtime
        assert len(calls) == 1
        call = calls[0]
        assert call["speculative_config"] is speculative_config
        assert call["target_model_args"] == runner.model_args
        assert runner.hidden_size == 4096

    def test_load_clears_stale_gemma4_mtp_assistant_before_reload(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class _FailingGemma4MTPAssistantLoader:
            def load_if_needed(self, **kwargs: object) -> object:
                raise RuntimeError("assistant load failed")

        _cache_generation_model(monkeypatch, config=_text_config())
        monkeypatch.setattr(
            model_lifecycle,
            "Gemma4MTPAssistantLoader",
            _FailingGemma4MTPAssistantLoader,
        )
        speculative_config = SimpleNamespace(
            method="mtp",
            draft_model_config=SimpleNamespace(
                model="assistant",
                hf_config=SimpleNamespace(model_type="gemma4_mtp"),
            ),
        )
        lifecycle, runner = _make_lifecycle(
            model_config=_runner_model_config(),
        )
        runner.vllm_config = SimpleNamespace(speculative_config=speculative_config)
        runner._gemma4_mtp_assistant = object()

        with pytest.raises(RuntimeError, match="assistant load failed"):
            lifecycle.load()

        assert runner._gemma4_mtp_assistant is None

    def test_reset_model_cache_clears_gemma4_mtp_assistant_cache(self) -> None:
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
                hf_config=SimpleNamespace(model_type="gemma4_mtp"),
            ),
        )
        target_args = {
            "model_type": "gemma4_text",
            "vocab_size": 32000,
            "hidden_size": 4096,
            "num_kv_shared_layers": 0,
            "layer_types": ["full_attention"],
        }
        loader = Gemma4MTPAssistantLoader(
            load_model_fn=_load_model,
            download_fn=lambda model_name, revision: Path(model_name),
        )

        first = loader.load_if_needed(
            speculative_config=spec_config,
            target_hf_config=None,
            target_model_args=target_args,
        )
        second = loader.load_if_needed(
            speculative_config=spec_config,
            target_hf_config=None,
            target_model_args=target_args,
        )
        assert first is second
        assert load_model_calls == 1

        model_lifecycle.reset_model_cache()

        third = loader.load_if_needed(
            speculative_config=spec_config,
            target_hf_config=None,
            target_model_args=target_args,
        )
        assert third is not first
        assert load_model_calls == 2

    def test_load_merges_nested_text_config_for_non_vlm_model(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _cache_generation_model(
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
        monkeypatch.setattr(
            model_lifecycle,
            "_MODEL_CACHE",
            {
                model_lifecycle._generation_cache_key("stub-model", is_vlm=False): (
                    fake_model,
                    object(),
                )
            },
        )
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

    def test_load_extracts_vlm_text_config_with_inherited_slots(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _cache_generation_model(
            monkeypatch,
            config=SimpleNamespace(
                text_config=_SlotTextConfig(
                    **_TEXT_MODEL_ARGS,
                )
            ),
            is_vlm=True,
        )
        lifecycle, runner = _make_lifecycle(
            model_config=_runner_model_config(
                is_multimodal_model=True,
            )
        )

        lifecycle.load()

        assert runner._is_vlm is True
        assert runner.model_args["vocab_size"] == 32000
        assert runner.model_args["hidden_size"] == 4096
        assert runner.num_layers == 32
        assert runner.head_dim == 128

    def test_load_stt_model_reuses_cached_model(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake_model = SimpleNamespace(
            create_runtime_adapter=lambda model_name: (object(), model_name)
        )
        monkeypatch.setattr(
            model_lifecycle,
            "_MODEL_CACHE",
            {model_lifecycle._stt_cache_key("stub-model"): (fake_model, None)},
        )

        assert model_lifecycle.load_stt_model("stub-model") is fake_model

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
            def for_model(cls, _model_name: str) -> _StubAWQLoader | None:
                return cls() if is_awq else None

            @staticmethod
            def cache_key(model_name: str, *, target_dtype: object) -> tuple[str, str]:
                return (model_name, f"mlx_lm-awq:{target_dtype}")

            def load(
                self,
                model_path: str,
                *,
                target_dtype: object,
                tokenizer_config: dict[str, object] | None,
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
        monkeypatch.setattr(model_lifecycle, "_MODEL_CACHE", {})
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
            assert "model_config" not in mlx_lm_load_calls[0]["kwargs"]

    def test_load_routes_gguf_to_owner_on_quantization_detection(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``quantization == "gguf"`` (set by vLLM core for a .gguf) delegates the
        whole load to ``GGUFModelLoader`` (lazily imported) and never calls generic
        ``mlx_lm.load`` — detection-routed like the AWQ branch, no env flag. The
        lifecycle threads the resolved model path, ``--tokenizer`` dir, derived
        dtype, and tokenizer config to the owner.
        """
        fake_model = SimpleNamespace(config=_text_config())
        fake_tokenizer = object()
        for_model_calls: list[dict[str, object]] = []
        mlx_lm_load_calls: list[dict[str, object]] = []

        class _StubGGUFLoader:
            @classmethod
            def for_model(
                cls,
                model_name: str,
                *,
                config_dir: str,
                target_dtype: object,
                tokenizer_config: dict[str, object] | None = None,
            ) -> _StubGGUFLoader:
                for_model_calls.append(
                    {
                        "model_name": model_name,
                        "config_dir": config_dir,
                        "target_dtype": target_dtype,
                        "tokenizer_config": (
                            dict(tokenizer_config) if tokenizer_config else None
                        ),
                    }
                )
                return cls()

            @staticmethod
            def cache_key(
                model_name: str, config_dir: str, *, target_dtype: object
            ) -> tuple[str, str]:
                return (model_name, f"gguf:{config_dir}:{target_dtype}")

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
        monkeypatch.setattr(model_lifecycle, "_MODEL_CACHE", {})
        monkeypatch.setattr(model_lifecycle, "mlx_lm_load", _fake_mlx_lm_load)

        lifecycle, runner = _make_lifecycle(
            model_config=_runner_model_config(
                quantization="gguf", tokenizer="tokenizer-dir"
            )
        )
        lifecycle.load()

        assert runner.model is fake_model
        assert runner.tokenizer is fake_tokenizer
        assert mlx_lm_load_calls == [], (
            "generic mlx_lm.load must NOT be called when the GGUF owner owns "
            "the load path"
        )
        assert len(for_model_calls) == 1
        call = for_model_calls[0]
        assert call["model_name"] == "stub-model"
        assert call["config_dir"] == "tokenizer-dir"
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
        raises on import). A clean load through the generic cached path proves a
        default install with no gguf extra is unaffected — the detection guard
        short-circuits before the lazy owner import that would pull in ``gguf``.
        """
        monkeypatch.setitem(sys.modules, "gguf", None)
        _cache_generation_model(monkeypatch, config=_text_config())
        lifecycle, runner = _make_lifecycle(
            model_config=_runner_model_config()  # no quantization -> not gguf
        )

        lifecycle.load()  # must not raise: the gguf owner import is never reached

        assert runner._is_vlm is False
        assert runner.model_args["vocab_size"] == 32000

    def test_gguf_cache_key_separates_tokenizer_dirs(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Two loads of the same .gguf with different ``--tokenizer`` dirs must
        not collide in the process cache. A .gguf carries weights only, so the
        model and tokenizer are built from the config dir; returning the first's
        result for the second (and bypassing the config-dir fail-fast once the
        cache is warm) would be a silent wrong load.
        """
        for_model_dirs: list[str] = []

        class _StubGGUFLoader:
            def __init__(self, config_dir: str) -> None:
                self._config_dir = config_dir

            @classmethod
            def for_model(
                cls,
                model_name: str,
                *,
                config_dir: str,
                target_dtype: object,
                tokenizer_config: dict[str, object] | None = None,
            ) -> _StubGGUFLoader:
                for_model_dirs.append(config_dir)
                return cls(config_dir)

            @staticmethod
            def cache_key(
                model_name: str, config_dir: str, *, target_dtype: object
            ) -> tuple[str, str]:
                return (model_name, f"gguf:{config_dir}:{target_dtype}")

            def load(self) -> tuple[object, object]:
                return (
                    SimpleNamespace(config=_text_config()),
                    f"tokenizer::{self._config_dir}",
                )

        monkeypatch.setitem(
            sys.modules,
            "vllm_metal.gguf.loader",
            SimpleNamespace(GGUFModelLoader=_StubGGUFLoader),
        )
        monkeypatch.setattr(model_lifecycle, "_MODEL_CACHE", {})

        def _load_tokenizer_for(tokenizer_dir: str) -> object:
            lifecycle, runner = _make_lifecycle(
                model_config=_runner_model_config(
                    quantization="gguf", tokenizer=tokenizer_dir
                )
            )
            lifecycle.load()
            return runner.tokenizer

        first = _load_tokenizer_for("/dir-a")
        second = _load_tokenizer_for("/dir-b")

        assert for_model_dirs == ["/dir-a", "/dir-b"], (
            "for_model must run for each distinct --tokenizer dir, not be skipped "
            "by a colliding cache key"
        )
        assert first == "tokenizer::/dir-a"
        assert second == "tokenizer::/dir-b"


class TestResolveModelDims:
    def _resolve(self, args: dict[str, object]) -> object:
        lifecycle, runner = _make_lifecycle(model_args=args)
        lifecycle.resolve_model_dims()
        return runner

    def test_standard_mha(self) -> None:
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
