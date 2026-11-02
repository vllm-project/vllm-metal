# SPDX-License-Identifier: Apache-2.0
"""Tests for shared Metal utilities."""

import importlib.metadata
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import mlx.core as mx
import psutil
import pytest

from tools.attention_bench_utils import attention_tolerances, package_versions
from vllm_metal.utils import (
    CommitProbe,
    _paging_counters,
    _parse_compressed_bytes,
    _parse_swap_out_bytes,
    get_model_download_path,
    probe_commit,
    resolve_mlx_lm_model_path,
    set_wired_limit,
)


def test_attention_tolerances_reject_unsupported_dtype():
    with pytest.raises(ValueError, match="expected float16, bfloat16 or float32"):
        attention_tolerances(mx.int32)


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


@pytest.mark.parametrize("modelscope_flag", [None, "false", "TrUe"])
def test_mlx_lm_snapshot_preserves_provider_revision_and_filters(
    monkeypatch, modelscope_flag
):
    from mlx_lm.utils import DEFAULT_ALLOW_PATTERNS

    if modelscope_flag is None:
        monkeypatch.delenv("MLXLM_USE_MODELSCOPE", raising=False)
    else:
        monkeypatch.setenv("MLXLM_USE_MODELSCOPE", modelscope_flag)
    hf_download = Mock(return_value="/hf/snapshot")
    ms_download = Mock(return_value="/modelscope/snapshot")
    monkeypatch.setattr("huggingface_hub.snapshot_download", hf_download)
    monkeypatch.setitem(
        sys.modules, "modelscope", SimpleNamespace(snapshot_download=ms_download)
    )
    expected, unused = (
        (ms_download, hf_download)
        if modelscope_flag == "TrUe"
        else (hf_download, ms_download)
    )

    assert resolve_mlx_lm_model_path("org/model", revision="release-tag") == Path(
        expected.return_value
    )
    expected.assert_called_once_with(
        "org/model", revision="release-tag", allow_patterns=DEFAULT_ALLOW_PATTERNS
    )
    unused.assert_not_called()


def test_mlx_lm_local_path_skips_downloads(tmp_path, monkeypatch):
    monkeypatch.setenv("MLXLM_USE_MODELSCOPE", "true")
    download = Mock(side_effect=AssertionError("local model must not download"))
    monkeypatch.setattr("huggingface_hub.snapshot_download", download)
    monkeypatch.setitem(
        sys.modules, "modelscope", SimpleNamespace(snapshot_download=download)
    )

    assert resolve_mlx_lm_model_path(str(tmp_path), revision="release-tag") == tmp_path
    download.assert_not_called()


def test_set_wired_limit_uses_pinned_mlx_api(monkeypatch) -> None:
    calls: list[int] = []

    monkeypatch.setattr(
        mx,
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
    assert probe.swap_out_bytes >= 0
    assert probe.seconds > 0


def test_probe_commit_does_not_keep_the_memory_it_touched() -> None:
    """The check must not hand the lazy pool the footprint it exists to avoid.

    A probe that left its pages behind would make the pool resident at startup
    again, one sample at a time, which is the cost the lazy allocation removes.
    The bound is relative to the sample, so the failure it catches is "a whole
    sample stuck around" rather than any particular number of bytes.
    """

    sample = 32 << 20
    rss_before = psutil.Process().memory_info().rss
    probe = probe_commit(sample)
    rss_after = psutil.Process().memory_info().rss

    assert probe.probed_bytes == sample
    assert probe.seconds > 0
    assert rss_after - rss_before < sample // 2


def test_probe_commit_reports_a_machine_that_cannot_map(monkeypatch) -> None:
    """A failed mapping is the answer, not something to paper over."""

    def refuse(*args, **kwargs):
        raise OSError("cannot map the sample")

    monkeypatch.setattr("mmap.mmap", refuse)

    with pytest.raises(OSError):
        probe_commit(1 << 20)


_VM_STAT = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                     6663.
Compressions:                               45258515577.
Pageins:                                    16116825.
Pageouts:                                     216236.
Swapins:                                    12239475.
Swapouts:                                   15761162.
"""


def test_swap_out_counter_reads_the_swap_file_not_pageouts() -> None:
    """Only ``Swapouts`` tracks the swap file.

    psutil's macOS ``swap_memory().sout`` is the ``Pageouts`` counter, which
    counts page-outs generally: it moves for file writeback and can stay put
    while the kernel swaps. The probe needs the counter that only moves when
    the swap file does.
    """

    assert _parse_swap_out_bytes(_VM_STAT) == 15761162 * 16384


@pytest.mark.parametrize(
    "output",
    [
        "Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPageouts: 1.\n",
        "Mach Virtual Memory Statistics:\nSwapouts: 1.\n",
    ],
    ids=["no_swapouts", "no_page_size"],
)
def test_swap_out_counter_needs_both_the_page_size_and_the_swapouts_line(
    output: str,
) -> None:
    with pytest.raises(ValueError, match="missing"):
        _parse_swap_out_bytes(output)


def test_paging_counters_read_with_a_timeout(monkeypatch):
    seen = {}

    def run(*args, **kwargs):
        seen.update(kwargs)
        return subprocess.CompletedProcess(args, 0, _VM_STAT, "")

    monkeypatch.setattr(subprocess, "run", run)

    assert _paging_counters() == (15761162 * 16384, 45258515577 * 16384)
    assert seen["timeout"] > 0


def test_compressed_counter_reads_the_compressor_line() -> None:
    assert _parse_compressed_bytes(_VM_STAT) == 45258515577 * 16384


def test_compressed_counter_needs_its_line() -> None:
    with pytest.raises(ValueError, match="Compressions"):
        _parse_compressed_bytes(_VM_STAT.replace("Compressions", "Nothing"))


def test_probe_reports_what_the_compressor_absorbed() -> None:
    # Measured on an 8 GB machine under pressure: a 128 MiB touch was met
    # almost entirely by the compressor while the swap file barely moved.
    probe = CommitProbe(
        probed_bytes=128 << 20,
        swap_out_before=0,
        swap_out_after=1 << 20,
        compressed_before=0,
        compressed_after=126 << 20,
        available_before=1 << 30,
        available_after=1 << 30,
        seconds=0.03,
    )

    assert probe.compressed_bytes == 126 << 20
    assert probe.displaced_bytes == 127 << 20
    assert "compressed +126 MiB, swap out +1 MiB" in probe.describe()


@pytest.mark.parametrize(
    "failure",
    [
        FileNotFoundError("no vm_stat"),
        subprocess.CalledProcessError(1, "vm_stat"),
        subprocess.TimeoutExpired("vm_stat", 5.0),
    ],
    ids=["missing", "failed", "hung"],
)
def test_paging_counters_fail_loud_and_name_the_opt_out(monkeypatch, failure):
    # The probe is a default-on safety gate: an unreadable counter ends
    # startup instead of passing an unverified plan.
    def run(*args, **kwargs):
        raise failure

    monkeypatch.setattr(subprocess, "run", run)

    with pytest.raises(RuntimeError, match="VLLM_METAL_KV_COMMIT_PROBE=0") as excinfo:
        _paging_counters()
    assert excinfo.value.__cause__ is failure


def test_paging_counters_fail_loud_on_unexpected_output(monkeypatch):
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, "???", "")
    )

    with pytest.raises(RuntimeError, match="VLLM_METAL_KV_COMMIT_PROBE=0"):
        _paging_counters()
