# SPDX-License-Identifier: Apache-2.0
"""Tests for shared Metal utilities."""

import importlib.metadata
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import mlx.core as mx
import psutil
import pytest

from tools.attention_bench_utils import package_versions
from vllm_metal.utils import get_model_download_path, probe_commit, set_wired_limit


def test_benchmark_versions_allow_missing_distributions(monkeypatch):
    def version(name):
        if name == "optional":
            raise importlib.metadata.PackageNotFoundError(name)
        return "1.0"

    monkeypatch.setattr(importlib.metadata, "version", version)
    assert package_versions("installed", "optional", "another") == {
        "installed": "1.0",
        "optional": None,
        "another": "1.0",
    }


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

    monkeypatch.setattr(
        mx.metal,
        "device_info",
        lambda: {"max_recommended_working_set_size": 123},
    )
    monkeypatch.setattr(mx, "set_wired_limit", calls.append)

    set_wired_limit()

    assert calls == [123]


def test_probe_commit_reports_what_the_machine_spent() -> None:
    """The probe measures free memory and swap around forcing pages resident."""

    probe = probe_commit(16 << 20)

    assert probe.probed_bytes == 16 << 20
    assert probe.available_before > 0
    assert probe.available_after > 0
    assert probe.swap_growth >= 0
    assert probe.seconds > 0


def test_probe_commit_does_not_keep_the_memory_it_touched() -> None:
    """The check must not hand the lazy pool the footprint it exists to avoid.

    A probe that left its pages behind would make the pool resident at startup
    again, one sample at a time, which is the cost the lazy allocation removes.
    The margins are wide on purpose: the failure is a whole sample sticking
    around, not noise.
    """

    rss_before = psutil.Process().memory_info().rss
    probe_commit(128 << 20)
    rss_after = psutil.Process().memory_info().rss

    assert rss_after - rss_before < 32 << 20


def test_probe_commit_reports_a_machine_that_cannot_map(monkeypatch) -> None:
    """A failed mapping is the answer, not something to paper over."""

    def refuse(*args, **kwargs):
        raise OSError("cannot map the sample")

    monkeypatch.setattr("mmap.mmap", refuse)

    with pytest.raises(OSError):
        probe_commit(1 << 20)
