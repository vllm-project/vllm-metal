# SPDX-License-Identifier: Apache-2.0
"""Tests for native MTP head registry behavior."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from vllm_metal.v1.mtp_heads.registry import (
    NativeMTPBuildContext,
    NativeMTPHeadRegistry,
)


def _mtp_config(*, model_type: str) -> SimpleNamespace:
    return SimpleNamespace(
        method="mtp",
        draft_model_config=SimpleNamespace(
            hf_config=SimpleNamespace(model_type=model_type),
        ),
    )


class _FakeProposer:
    def needs_target_hidden_states(
        self, decode_segments: Any, *, has_final_prefill: bool
    ) -> bool:
        return False

    def propose(self, ctx: Any) -> None:
        return None


class _FakeHead:
    def __init__(self, model_type: str = "dummy_mtp") -> None:
        self.model_type = model_type

    def build_proposer(self, context: NativeMTPBuildContext) -> _FakeProposer:
        return _FakeProposer()


@pytest.mark.parametrize(
    "config",
    [
        pytest.param(None, id="no_config"),
        pytest.param(
            SimpleNamespace(method="ngram", draft_model_config=None),
            id="non_mtp_method",
        ),
        pytest.param(
            SimpleNamespace(
                method="draft_model",
                draft_model_config=SimpleNamespace(
                    hf_config=SimpleNamespace(model_type="llama"),
                ),
            ),
            id="draft_model_method",
        ),
        pytest.param(_mtp_config(model_type="eagle"), id="unregistered_model_type"),
        pytest.param(
            SimpleNamespace(method="mtp", draft_model_config=None),
            id="missing_draft_model_config",
        ),
        pytest.param(
            SimpleNamespace(
                method="mtp",
                draft_model_config=SimpleNamespace(hf_config=None),
            ),
            id="missing_hf_config",
        ),
    ],
)
def test_find_returns_none_for_unregistered_or_non_mtp_config(config: Any) -> None:
    assert NativeMTPHeadRegistry.find(config) is None


def test_returns_registered_head_for_matching_model_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    head = _FakeHead()
    monkeypatch.setattr(NativeMTPHeadRegistry, "_heads", {})

    NativeMTPHeadRegistry.register(head)

    assert NativeMTPHeadRegistry.find(_mtp_config(model_type="dummy_mtp")) is head
    assert NativeMTPHeadRegistry.registered_types() == ["dummy_mtp"]


def test_unsupported_message_names_model_type_and_supported_path() -> None:
    message = NativeMTPHeadRegistry.unsupported_message(_mtp_config(model_type="eagle"))

    assert "'eagle'" in message
    assert "Gemma4 MTP" in message


def test_unsupported_message_names_registered_heads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(NativeMTPHeadRegistry, "_heads", {})
    head = _FakeHead("glm4_moe_lite_mtp")
    NativeMTPHeadRegistry.register(head)

    message = NativeMTPHeadRegistry.unsupported_message(_mtp_config(model_type="eagle"))

    assert "glm4_moe_lite_mtp" in message
