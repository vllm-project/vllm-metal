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


@pytest.fixture(autouse=True)
def _empty_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(NativeMTPHeadRegistry, "_heads", {})


def test_find_returns_none_for_unregistered_model_type() -> None:
    config = _mtp_config(model_type="eagle")

    head = NativeMTPHeadRegistry.find(config)

    assert head is None


def test_returns_registered_head_for_matching_model_type() -> None:
    head = _FakeHead()
    config = _mtp_config(model_type="dummy_mtp")

    NativeMTPHeadRegistry.register(head)
    found = NativeMTPHeadRegistry.find(config)

    assert found is head
    assert NativeMTPHeadRegistry.registered_types() == ["dummy_mtp"]


def test_unsupported_message_names_model_type_and_supported_path() -> None:
    config = _mtp_config(model_type="eagle")

    message = NativeMTPHeadRegistry.unsupported_message(config)

    assert "'eagle'" in message
    assert "Gemma4 MTP" in message


def test_unsupported_message_names_registered_heads() -> None:
    head = _FakeHead("glm4_moe_lite_mtp")
    config = _mtp_config(model_type="eagle")

    NativeMTPHeadRegistry.register(head)
    message = NativeMTPHeadRegistry.unsupported_message(config)

    assert "'eagle'" in message
    assert "glm4_moe_lite_mtp" in message
