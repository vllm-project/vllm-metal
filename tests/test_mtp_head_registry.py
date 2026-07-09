# SPDX-License-Identifier: Apache-2.0
"""Tests for the native MTP head registry lookup and unsupported-head message."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

# Imported via the package re-exports on purpose: doubles as the back-compat
# check for existing import sites.
from vllm_metal.v1.mtp_heads import (
    NativeMTPBuildContext,
    find_native_mtp_head,
    registered_mtp_head_types,
    registry,
    unsupported_mtp_message,
)


def _mtp_config(*, model_type: str) -> SimpleNamespace:
    return SimpleNamespace(
        method="mtp",
        draft_model_config=SimpleNamespace(
            hf_config=SimpleNamespace(model_type=model_type),
        ),
    )


class _FakeProposer:
    """Minimal ``MetalProposer`` double."""

    def needs_target_hidden_states(
        self, decode_segments: Any, *, has_final_prefill: bool
    ) -> bool:
        return False

    def propose(self, ctx: Any) -> None:
        return None


class _FakeHead:
    """Minimal :class:`NativeMTPHead` for registry-lookup tests."""

    model_type = "dummy_mtp"

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
def test_find_native_mtp_head_returns_none(config: Any) -> None:
    assert find_native_mtp_head(config) is None


def test_returns_registered_head_for_matching_model_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    head = _FakeHead()
    monkeypatch.setattr(registry, "MTP_HEAD_REGISTRY", {"dummy_mtp": head})

    assert find_native_mtp_head(_mtp_config(model_type="dummy_mtp")) is head


def test_registered_mtp_head_types_reads_registry_at_call_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A from-import of the dict would freeze a stale reference; the helper must
    # re-read the module attribute so a rebound registry is reflected.
    monkeypatch.setattr(registry, "MTP_HEAD_REGISTRY", {"dummy_mtp": _FakeHead()})

    assert registered_mtp_head_types() == ["dummy_mtp"]


def test_unsupported_mtp_message_names_model_type_and_supported_path() -> None:
    message = unsupported_mtp_message(_mtp_config(model_type="eagle"))

    assert "'eagle'" in message
    assert "Gemma4 MTP" in message


def test_unsupported_mtp_message_names_registered_heads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        registry, "MTP_HEAD_REGISTRY", {"glm4_moe_lite_mtp": _FakeHead()}
    )

    message = unsupported_mtp_message(_mtp_config(model_type="eagle"))

    assert "glm4_moe_lite_mtp" in message
