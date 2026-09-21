# SPDX-License-Identifier: Apache-2.0
"""Tests for shared Metal utilities."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import mlx.core as mx
import pytest

from vllm_metal.utils import get_model_download_path, set_wired_limit


@pytest.mark.parametrize("revision", [None, "release-tag", "a" * 40])
def test_modelscope_download_preserves_revision(monkeypatch, revision):
    monkeypatch.setattr("vllm.envs.VLLM_USE_MODELSCOPE", True)
    monkeypatch.setenv("VLLM_METAL_MODELSCOPE_CACHE", "/model-cache")
    download = Mock(return_value="/model-cache/snapshot")
    monkeypatch.setitem(
        sys.modules,
        "modelscope.hub.snapshot_download",
        SimpleNamespace(snapshot_download=download),
    )

    assert (
        get_model_download_path("org/model", revision=revision)
        == "/model-cache/snapshot"
    )
    kwargs = {"revision": revision} if revision is not None else {}
    download.assert_called_once_with("org/model", cache_dir="/model-cache", **kwargs)


def test_local_model_path_is_preserved(tmp_path):
    assert get_model_download_path(str(tmp_path), revision="release-tag") == str(
        tmp_path
    )


def test_set_wired_limit_uses_pinned_mlx_api(monkeypatch) -> None:
    calls: list[int] = []
    monkeypatch.delenv("VLLM_METAL_DISABLE_WIRED_LIMIT", raising=False)

    monkeypatch.setattr(
        mx.metal,
        "device_info",
        lambda: {"max_recommended_working_set_size": 123},
    )
    monkeypatch.setattr(mx, "set_wired_limit", calls.append)

    set_wired_limit()

    assert calls == [123]


def test_disabled_wired_limit_does_not_inspect_or_change_device(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_METAL_DISABLE_WIRED_LIMIT", "1")
    device_info = Mock(side_effect=AssertionError("must not inspect device"))
    set_limit = Mock(side_effect=AssertionError("must preserve existing limit"))
    monkeypatch.setattr(mx.metal, "device_info", device_info)
    monkeypatch.setattr(mx, "set_wired_limit", set_limit)

    set_wired_limit()

    device_info.assert_not_called()
    set_limit.assert_not_called()
