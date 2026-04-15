# SPDX-License-Identifier: Apache-2.0
"""Tests for model lifecycle behavior."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tests.stub_runner import make_stub_runner
from vllm_metal.paged_attention_backend.mla import MLA_DEFAULT_QK_ROPE_HEAD_DIM
from vllm_metal.v1 import model_lifecycle
from vllm_metal.v1.model_lifecycle import ModelLifecycle

_TEXT_MODEL_ARGS = {
    "vocab_size": 32000,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "hidden_size": 4096,
}


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
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _text_config(**overrides: object) -> SimpleNamespace:
    return SimpleNamespace(**(_TEXT_MODEL_ARGS | overrides))


def _cache_generation_model(
    monkeypatch: pytest.MonkeyPatch,
    *,
    config: object,
    tokenizer: object | None = None,
) -> tuple[object, object]:
    fake_model = SimpleNamespace(config=config)
    fake_tokenizer = object() if tokenizer is None else tokenizer
    monkeypatch.setattr(
        model_lifecycle,
        "_MODEL_CACHE",
        {"stub-model": (fake_model, fake_tokenizer)},
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
            {"stub-model": (fake_model, object())},
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

    def test_load_reuses_cached_stt_model(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        adapter = object()
        fake_model = SimpleNamespace(
            create_runtime_adapter=lambda model_name: (adapter, model_name)
        )
        monkeypatch.setattr(
            model_lifecycle,
            "_MODEL_CACHE",
            {"stub-model": (fake_model, None)},
        )
        monkeypatch.setattr(model_lifecycle, "is_stt_model", lambda _model_name: True)
        lifecycle, runner = _make_lifecycle()

        lifecycle.load()

        assert runner.model is fake_model
        assert runner.tokenizer is None
        assert runner.model_args == {}
        assert runner.kv_cache_dtype is None
        assert runner._is_vlm is False
        assert runner._is_stt is True
        assert runner._stt_runtime_adapter == (adapter, "stub-model")


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
