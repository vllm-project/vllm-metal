# SPDX-License-Identifier: Apache-2.0
"""Unit tests for MetalProfilerWrapper."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock

import mlx.core.metal  # noqa: F401 — submodule must be loaded for monkeypatch
import pytest
from vllm.config import ProfilerConfig, VllmConfig
from vllm.distributed.utils import get_worker_rank_suffix

from vllm_metal.profiler import MetalProfilerWrapper
from vllm_metal.v1.worker import MetalWorker


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


@pytest.fixture
def profiled_worker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Iterator[tuple[MetalWorker, list[Path]]]:
    monkeypatch.setenv("MTL_CAPTURE_ENABLED", "1")
    captures: list[Path] = []

    def start_capture(path: str) -> None:
        trace_path = Path(path)
        trace_path.mkdir(parents=True)
        captures.append(trace_path)

    monkeypatch.setattr("mlx.core.metal.start_capture", start_capture)
    monkeypatch.setattr("mlx.core.metal.stop_capture", MagicMock())
    worker = MetalWorker(
        vllm_config=VllmConfig(
            profiler_config=ProfilerConfig(
                profiler="torch", torch_profiler_dir=str(tmp_path)
            )
        ),
        local_rank=0,
        rank=0,
        distributed_init_method=f"file://{tmp_path}/unused-rendezvous",
        is_driver_worker=True,
    )
    try:
        yield worker, captures
    finally:
        worker.shutdown()


@pytest.mark.parametrize("next_prefix", ["decode", None])
def test_worker_honors_each_capture_prefix(
    profiled_worker: tuple[MetalWorker, list[Path]], next_prefix: str | None
) -> None:
    worker, captures = profiled_worker
    worker.profile(is_start=True, profile_prefix="prefill")
    worker.profile(is_start=False)

    worker.profile(is_start=True, profile_prefix=next_prefix)
    worker.profile(is_start=False)

    suffix: str = get_worker_rank_suffix(global_rank=worker.rank)
    expected_prefix = f"{next_prefix}_{suffix}_" if next_prefix else f"{suffix}_"
    assert len(captures) == 2
    assert captures[0].name.startswith(f"prefill_{suffix}_")
    assert captures[1].name.startswith(expected_prefix)


def test_worker_recovers_with_a_valid_prefix_after_capture_failure(
    profiled_worker: tuple[MetalWorker, list[Path]], tmp_path: Path
) -> None:
    worker, captures = profiled_worker
    (tmp_path / "blocked").write_text("not a directory")
    worker.profile(is_start=True, profile_prefix="blocked/capture")
    assert worker._metal_profiler is not None
    assert not worker._metal_profiler.is_running
    worker.profile(is_start=False)

    worker.profile(is_start=True, profile_prefix="recovered")

    assert worker._metal_profiler is not None
    assert worker._metal_profiler.is_running
    worker.profile(is_start=False)
    assert len(captures) == 1
    assert captures[0].name.startswith("recovered_")


def test_worker_keeps_an_active_capture_until_stop(
    profiled_worker: tuple[MetalWorker, list[Path]],
) -> None:
    worker, captures = profiled_worker
    worker.profile(is_start=False)
    worker.profile(is_start=True, profile_prefix="prefill")
    worker.profile(is_start=True, profile_prefix="ignored_while_running")
    assert worker._metal_profiler is not None
    assert worker._metal_profiler.is_running
    assert len(captures) == 1
    worker.profile(is_start=False)
    worker.profile(is_start=False)
    assert len(captures) == 1
