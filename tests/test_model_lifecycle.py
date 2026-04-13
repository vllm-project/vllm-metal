# SPDX-License-Identifier: Apache-2.0
"""Tests for model lifecycle ownership and config normalization."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tests.stub_runner import make_stub_runner
from vllm_metal.paged_attention_backend.mla import MLA_DEFAULT_QK_ROPE_HEAD_DIM
from vllm_metal.v1 import model_lifecycle
from vllm_metal.v1.model_lifecycle import ModelLifecycle


class _SlotConfig:
    __slots__ = ("vocab_size", "hidden_size")

    def __init__(self, *, vocab_size: int, hidden_size: int) -> None:
        self.vocab_size = vocab_size
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
    def test_is_vlm_model_defaults_false_when_flag_missing(self) -> None:
        lifecycle, _ = _make_lifecycle(
            model=SimpleNamespace(),
            model_config=SimpleNamespace(
                model="stub-model",
                hf_config=None,
                trust_remote_code=False,
                dtype=torch.float16,
            ),
        )

        assert lifecycle.is_vlm_model() is False

    def test_text_model_args_falls_back_to_config(self) -> None:
        lifecycle, _ = _make_lifecycle(
            model=SimpleNamespace(
                config=SimpleNamespace(vocab_size=32000, hidden_size=4096)
            )
        )

        model_args = lifecycle._text_model_args()

        assert model_args["vocab_size"] == 32000
        assert model_args["hidden_size"] == 4096

    def test_vlm_model_args_accepts_slot_backed_text_config(self) -> None:
        lifecycle, _ = _make_lifecycle(
            model=SimpleNamespace(
                config=SimpleNamespace(
                    text_config=_SlotConfig(vocab_size=32000, hidden_size=4096)
                )
            ),
            is_vlm=True,
        )

        model_args = lifecycle._vlm_model_args()

        assert model_args["vocab_size"] == 32000
        assert model_args["hidden_size"] == 4096

    def test_merge_text_config_accepts_namespace(self) -> None:
        lifecycle, _ = _make_lifecycle(model=SimpleNamespace())

        merged = lifecycle._merge_text_config(
            {
                "hidden_size": 1024,
                "text_config": SimpleNamespace(
                    hidden_size=4096,
                    vocab_size=32000,
                ),
            }
        )

        assert merged["hidden_size"] == 1024
        assert merged["vocab_size"] == 32000

    def test_load_stt_reuses_cached_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        adapter = object()
        fake_model = SimpleNamespace(
            create_runtime_adapter=lambda model_name: (adapter, model_name)
        )
        monkeypatch.setattr(
            model_lifecycle,
            "_MODEL_CACHE",
            {"openai/whisper-tiny": (fake_model, None)},
        )

        lifecycle, runner = _make_lifecycle(model=object())

        lifecycle.load_stt("openai/whisper-tiny")

        assert runner.model is fake_model
        assert runner.tokenizer is None
        assert runner.model_args == {}
        assert runner.kv_cache_dtype is None
        assert runner._is_vlm is False
        assert runner._is_stt is True
        assert runner._stt_runtime_adapter == (adapter, "openai/whisper-tiny")


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
