# SPDX-License-Identifier: Apache-2.0
"""Tests for VLM forward-model dispatch."""

from __future__ import annotations

from unittest.mock import MagicMock

from tests.stub_runner import make_stub_runner


class TestForwardModelProperty:
    def test_text_model_returns_self(self) -> None:
        model = MagicMock()
        runner = make_stub_runner(model=model, _is_vlm=False)
        assert runner._forward_model is model

    def test_vlm_with_language_model_returns_language_model(self) -> None:
        language_model = MagicMock()
        vlm = MagicMock()
        vlm.language_model = language_model
        runner = make_stub_runner(model=vlm, _is_vlm=True)
        assert runner._forward_model is language_model

    def test_vlm_without_language_model_returns_model(self) -> None:
        model = MagicMock(spec=[])
        runner = make_stub_runner(model=model, _is_vlm=True)
        assert runner._forward_model is model

    def test_vlm_flag_false_bypasses_language_model(self) -> None:
        language_model = MagicMock()
        model = MagicMock()
        model.language_model = language_model
        runner = make_stub_runner(model=model, _is_vlm=False)
        assert runner._forward_model is model


class TestValidatePagedAttentionSupport:
    def test_delegates_to_model_adapter(self) -> None:
        calls: list[tuple[dict[str, int], int]] = []

        class RecordingAdapter:
            def require_uniform_kv_heads(
                self, args: dict[str, int], num_kv_heads: int
            ) -> None:
                calls.append((args, num_kv_heads))

        model_args = {"num_global_key_value_heads": 8}
        runner = make_stub_runner(
            model_args=model_args,
            num_kv_heads=8,
            _model_adapter=RecordingAdapter(),
        )

        runner.validate_paged_attention_support()

        assert calls == [(model_args, 8)]
