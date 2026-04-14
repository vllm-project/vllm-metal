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


def _make_lifecycle(
    *,
    model: object,
    model_args: dict[str, object] | None = None,
    model_config: object | None = None,
    is_vlm: bool = False,
) -> tuple[ModelLifecycle, object]:
    runner = make_stub_runner(
        model=model,
        model_args=model_args,
        _is_vlm=is_vlm,
        metal_config=SimpleNamespace(debug=False),
        model_config=model_config
        or SimpleNamespace(
            model="stub-model",
            hf_config=None,
            is_multimodal_model=False,
            trust_remote_code=False,
            dtype=torch.float16,
        ),
    )
    lifecycle = ModelLifecycle(runner, runner._model_adapter)
    return lifecycle, runner


class TestModelLifecycle:
    def test_load_uses_adapter_override_for_text_only_multimodal_model(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake_model = SimpleNamespace(
            config=SimpleNamespace(
                vocab_size=32000,
                num_hidden_layers=32,
                num_attention_heads=32,
                num_key_value_heads=8,
                hidden_size=4096,
            )
        )
        monkeypatch.setattr(
            model_lifecycle,
            "_MODEL_CACHE",
            {"stub-model": (fake_model, object())},
        )
        lifecycle, runner = _make_lifecycle(
            model=SimpleNamespace(),
            model_config=SimpleNamespace(
                model="stub-model",
                hf_config=SimpleNamespace(model_type="gemma4"),
                is_multimodal_model=True,
                trust_remote_code=False,
                dtype=torch.float16,
            ),
        )

        lifecycle.load()

        assert runner._is_vlm is False

    def test_load_extracts_text_model_config_from_cached_model(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake_model = SimpleNamespace(
            config=SimpleNamespace(
                vocab_size=32000,
                num_hidden_layers=32,
                num_attention_heads=32,
                num_key_value_heads=8,
                hidden_size=4096,
            )
        )
        fake_tokenizer = object()
        monkeypatch.setattr(
            model_lifecycle,
            "_MODEL_CACHE",
            {"stub-model": (fake_model, fake_tokenizer)},
        )
        lifecycle, runner = _make_lifecycle(model=SimpleNamespace())

        lifecycle.load()

        assert runner.model is fake_model
        assert runner.tokenizer is fake_tokenizer
        assert runner.model_args["vocab_size"] == 32000
        assert runner.hidden_size == 4096
        assert runner.kv_cache_dtype is not None

    def test_load_extracts_vlm_text_config_with_inherited_slots(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake_model = SimpleNamespace(
            config=SimpleNamespace(
                text_config=_SlotTextConfig(
                    vocab_size=32000,
                    num_hidden_layers=32,
                    num_attention_heads=32,
                    num_key_value_heads=8,
                    hidden_size=4096,
                )
            )
        )
        monkeypatch.setattr(
            model_lifecycle,
            "_MODEL_CACHE",
            {"stub-model": (fake_model, object())},
        )
        lifecycle, runner = _make_lifecycle(
            model=SimpleNamespace(),
            is_vlm=True,
            model_config=SimpleNamespace(
                model="stub-model",
                hf_config=None,
                is_multimodal_model=True,
                trust_remote_code=False,
                dtype=torch.float16,
            ),
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
        lifecycle, runner = _make_lifecycle(model=object())

        lifecycle.load()

        assert runner.model is fake_model
        assert runner.tokenizer is None
        assert runner.model_args == {}
        assert runner.kv_cache_dtype is None
        assert runner._is_vlm is False
        assert runner._is_stt is True
        assert runner._stt_runtime_adapter == (adapter, "stub-model")


class TestResolveModelDims:
    def _make_runner(self, args: dict[str, object]) -> tuple[ModelLifecycle, object]:
        return _make_lifecycle(model=object(), model_args=args)

    def test_standard_mha(self) -> None:
        lifecycle, runner = self._make_runner(
            {
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "hidden_size": 4096,
            }
        )

        lifecycle.resolve_model_dims()

        assert runner.num_layers == 32
        assert runner.num_kv_heads == 8
        assert runner.head_dim == 128

    def test_mla_overrides_kv_heads_and_head_dim(self) -> None:
        lifecycle, runner = self._make_runner(
            {
                "num_hidden_layers": 47,
                "num_attention_heads": 20,
                "num_key_value_heads": 20,
                "hidden_size": 2048,
                "kv_lora_rank": 512,
                "qk_rope_head_dim": 64,
            }
        )

        lifecycle.resolve_model_dims()

        assert runner.num_kv_heads == 1
        assert runner.head_dim == 576
        assert runner.mla_latent_dim == 576

    def test_mla_default_rope_head_dim(self) -> None:
        lifecycle, runner = self._make_runner(
            {
                "num_hidden_layers": 28,
                "num_attention_heads": 16,
                "hidden_size": 2048,
                "kv_lora_rank": 256,
            }
        )

        lifecycle.resolve_model_dims()

        assert runner.head_dim == 256 + MLA_DEFAULT_QK_ROPE_HEAD_DIM
        assert runner.mla_latent_dim == 256 + MLA_DEFAULT_QK_ROPE_HEAD_DIM

    def test_missing_dims_raise(self) -> None:
        lifecycle, _ = self._make_runner({"num_hidden_layers": 32})

        with pytest.raises(ValueError, match="Cannot resolve model dimensions"):
            lifecycle.resolve_model_dims()
