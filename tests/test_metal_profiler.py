# SPDX-License-Identifier: Apache-2.0
"""Unit tests for MetalProfilerWrapper."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import mlx.core.metal  # noqa: F401 — submodule must be loaded for monkeypatch
import pytest
from vllm.config import ProfilerConfig

from vllm_metal.profiler import MetalProfilerWrapper


@pytest.mark.parametrize(
    ("delay", "max_iters"),
    [(1, 0), (0, 1), (1, 1)],
    ids=["delay=1", "max=1", "both"],
)
def test_rejects_unsupported_scheduling_fields(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    delay: int,
    max_iters: int,
) -> None:
    monkeypatch.setenv("MTL_CAPTURE_ENABLED", "1")
    cfg = ProfilerConfig(
        profiler="torch",
        torch_profiler_dir=str(tmp_path),
        delay_iterations=delay,
        max_iterations=max_iters,
    )

    with pytest.raises(ValueError, match="WorkerProfiler.step"):
        MetalProfilerWrapper(cfg, trace_name="run")


def test_raises_when_capture_env_var_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("MTL_CAPTURE_ENABLED", raising=False)
    cfg = ProfilerConfig(
        profiler="torch",
        torch_profiler_dir=str(tmp_path),
    )

    with pytest.raises(RuntimeError, match="MTL_CAPTURE_ENABLED"):
        MetalProfilerWrapper(cfg, trace_name="run")


def test_raises_when_trace_dir_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MTL_CAPTURE_ENABLED", "1")
    # profiler=None bypasses the upstream validator's own dir check, so the
    # wrapper's defensive check is what fires.
    cfg = ProfilerConfig()

    with pytest.raises(ValueError, match="torch_profiler_dir"):
        MetalProfilerWrapper(cfg, trace_name="run")


@pytest.mark.parametrize("new_wrapper", [False, True])
def test_captures_use_distinct_trace_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    new_wrapper: bool,
) -> None:
    monkeypatch.setenv("MTL_CAPTURE_ENABLED", "1")
    captures: list[Path] = []

    def start_capture(path: str) -> None:
        trace_path = Path(path)
        # Like Metal, refuse to overwrite an earlier capture.
        trace_path.mkdir()
        captures.append(trace_path)

    monkeypatch.setattr("mlx.core.metal.start_capture", start_capture)
    monkeypatch.setattr("mlx.core.metal.stop_capture", MagicMock())

    cfg = ProfilerConfig(
        profiler="torch",
        torch_profiler_dir=str(tmp_path),
    )
    wrapper = MetalProfilerWrapper(cfg, trace_name="run42")
    wrapper.start()
    assert wrapper.is_running
    wrapper.stop()

    if new_wrapper:
        wrapper = MetalProfilerWrapper(cfg, trace_name="run42")
    wrapper.start()
    assert wrapper.is_running
    wrapper.stop()

    assert len(captures) == 2
    for path in captures:
        assert path.parent == Path(cfg.torch_profiler_dir)
        assert path.name.startswith("run42_")
        assert path.suffix == ".gputrace"


def test_stop_calls_mlx_stop_capture(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("MTL_CAPTURE_ENABLED", "1")
    monkeypatch.setattr("mlx.core.metal.start_capture", MagicMock())
    mock_stop = MagicMock()
    monkeypatch.setattr("mlx.core.metal.stop_capture", mock_stop)

    cfg = ProfilerConfig(
        profiler="torch",
        torch_profiler_dir=str(tmp_path),
    )
    wrapper = MetalProfilerWrapper(cfg, trace_name="run")
    wrapper.start()
    wrapper.stop()

    mock_stop.assert_called_once_with()
