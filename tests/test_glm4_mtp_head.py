# SPDX-License-Identifier: Apache-2.0
"""Tests for the GLM-4.7-Flash native MTP head and its registration."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_metal.v1.mtp_heads.glm4 import Glm4MoeLiteMTPHead
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


def _context(
    *,
    num_speculative_tokens: int = 1,
    enable_prefix_caching: bool = False,
    target_config: dict | None = None,
    dtype: object = None,
) -> NativeMTPBuildContext:
    return NativeMTPBuildContext(
        speculative_config=SimpleNamespace(
            num_speculative_tokens=num_speculative_tokens
        ),
        controller=object(),
        vllm_config=SimpleNamespace(
            cache_config=SimpleNamespace(enable_prefix_caching=enable_prefix_caching),
        ),
        target_config=target_config if target_config is not None else {},
        dtype=dtype,
    )


def test_glm_head_is_registered_under_its_model_type() -> None:
    head = NativeMTPHeadRegistry.find(_mtp_config(model_type="glm4_moe_lite_mtp"))

    assert isinstance(head, Glm4MoeLiteMTPHead)
    assert "glm4_moe_lite_mtp" in NativeMTPHeadRegistry.registered_types()


def test_unsupported_message_points_at_glm_head() -> None:
    message = NativeMTPHeadRegistry.unsupported_message(_mtp_config(model_type="eagle"))

    assert "glm4_moe_lite_mtp" in message


def test_build_proposer_rejects_prefix_caching() -> None:
    with pytest.raises(NotImplementedError, match="no-enable-prefix-caching"):
        Glm4MoeLiteMTPHead().build_proposer(_context(enable_prefix_caching=True))


def test_build_proposer_rejects_multi_token_spec() -> None:
    with pytest.raises(NotImplementedError, match="at most 1 speculative"):
        Glm4MoeLiteMTPHead().build_proposer(_context(num_speculative_tokens=2))


def test_build_proposer_delegates_to_native_mtp_proposer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm_metal.v1.mtp_proposer as mtp_proposer

    captured: dict[str, object] = {}
    sentinel = object()

    def _fake_build(**kwargs: object) -> object:
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(mtp_proposer.NativeMTPProposer, "build", _fake_build)
    context = _context(target_config={"hidden_size": 2048}, dtype="DTYPE")

    result = Glm4MoeLiteMTPHead().build_proposer(context)

    assert result is sentinel
    assert captured["speculative_config"] is context.speculative_config
    assert captured["controller"] is context.controller
    assert captured["model_type"] == "glm4_moe_lite_mtp"
    assert captured["target_config"] == {"hidden_size": 2048}
    assert captured["dtype"] == "DTYPE"
    from vllm_metal.v1.mtp_heads.glm4_moe_lite_mtp import Glm4MoeLiteMTPHeadLoader

    assert isinstance(captured["loader"], Glm4MoeLiteMTPHeadLoader)
