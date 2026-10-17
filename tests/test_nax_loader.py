# SPDX-License-Identifier: Apache-2.0
"""Launch-policy tests for the optional NAX prefill shader library."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import vllm_metal.envs as envs
from vllm_metal.metal import _try_init_nax_library


def _ops(*, supported: bool = True, load_error: Exception | None = None):
    return SimpleNamespace(
        nax_supported=Mock(return_value=supported),
        init_nax_library_path=Mock(side_effect=load_error),
    )


def test_disable_nax_env_is_a_negative_override(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_METAL_DISABLE_NAX", raising=False)
    assert envs.VLLM_METAL_DISABLE_NAX is False

    monkeypatch.setenv("VLLM_METAL_DISABLE_NAX", "1")
    assert envs.VLLM_METAL_DISABLE_NAX is True


@pytest.mark.parametrize(
    ("disabled", "supported", "has_library", "expected"),
    [
        (True, True, True, False),
        (False, False, True, False),
        (False, True, True, True),
    ],
)
def test_prebuilt_nax_policy(
    tmp_path: Path,
    disabled: bool,
    supported: bool,
    has_library: bool,
    expected: bool,
) -> None:
    ops = _ops(supported=supported)
    lib = tmp_path / "paged_attention_nax_kern.metallib"
    if has_library:
        lib.write_bytes(b"lib")

    loaded = _try_init_nax_library(
        ops,  # type: ignore[arg-type]
        disabled=disabled,
        prebuilt_path=lib,
    )
    assert loaded is expected
    assert ops.nax_supported.called is (not disabled)
    assert ops.init_nax_library_path.called is expected


def test_missing_prebuilt_nax_warns_and_keeps_fallback(tmp_path: Path, caplog) -> None:
    ops = _ops(supported=True)
    missing = tmp_path / "paged_attention_nax_kern.metallib"

    assert not _try_init_nax_library(
        ops,  # type: ignore[arg-type]
        disabled=False,
        prebuilt_path=missing,
    )
    assert f"prebuilt library is missing at {missing}" in caplog.text
    assert "using the non-NAX fallback" in caplog.text


def test_optional_load_failure_warns_and_keeps_fallback(
    tmp_path: Path,
    caplog,
) -> None:
    ops = _ops(load_error=RuntimeError("bad optional library"))
    lib = tmp_path / "paged_attention_nax_kern.metallib"
    lib.write_bytes(b"lib")

    assert not _try_init_nax_library(
        ops,  # type: ignore[arg-type]
        disabled=False,
        prebuilt_path=lib,
    )
    assert "using the non-NAX fallback" in caplog.text
