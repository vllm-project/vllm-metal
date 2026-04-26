# SPDX-License-Identifier: Apache-2.0
"""Metal frame-capture wrapper for vLLM's WorkerProfiler abstraction.

Subclasses ``vllm.profiler.wrapper.WorkerProfiler`` so that the entire
upstream profiling surface — ``LLM.start_profile`` / ``LLM.stop_profile``,
the ``/start_profile`` and ``/stop_profile`` HTTP endpoints, the engine's
``collective_rpc("profile", ...)`` plumbing, and the ``delay_iterations`` /
``max_iterations`` state machine — routes through unchanged.

Only the two abstract methods need bodies: ``_start`` calls
``mlx.metal.start_capture`` and ``_stop`` calls ``mlx.metal.stop_capture``.
The output is a ``.gputrace`` bundle that opens directly in Xcode (the same
artifact Xcode's "Capture GPU Frame" produces).

Apple gates frame capture behind ``MTL_CAPTURE_ENABLED=1`` in the process
environment.  We check that up front and raise with an actionable message;
the alternative is a generic "Capturing is not supported" from MLX.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import override

import mlx.core as mx
from vllm.config import ProfilerConfig
from vllm.logger import init_logger
from vllm.profiler.wrapper import WorkerProfiler

logger = init_logger(__name__)


class MetalProfilerWrapper(WorkerProfiler):
    """Metal frame-capture flavor of vLLM's WorkerProfiler.

    Trace output: ``<profiler_config.torch_profiler_dir>/<trace_name>.gputrace``
    """

    def __init__(self, profiler_config: ProfilerConfig, trace_name: str) -> None:
        super().__init__(profiler_config)

        if os.environ.get("MTL_CAPTURE_ENABLED") != "1":
            raise RuntimeError(
                "Metal frame capture requires MTL_CAPTURE_ENABLED=1 in the "
                "process environment. Restart the engine with that variable "
                "set, then retry."
            )

        trace_dir = profiler_config.torch_profiler_dir
        if not trace_dir:
            raise ValueError(
                "MetalProfilerWrapper requires profiler_config.torch_profiler_dir "
                "to be set (e.g. --profiler-config.torch_profiler_dir=/tmp/trace)."
            )

        Path(trace_dir).mkdir(parents=True, exist_ok=True)
        self._trace_path = str(Path(trace_dir) / f"{trace_name}.gputrace")

        logger.info_once(
            "Metal frame capture enabled. Trace will be saved to %s",
            self._trace_path,
            scope="local",
        )

    @override
    def _start(self) -> None:
        mx.metal.start_capture(self._trace_path)

    @override
    def _stop(self) -> None:
        mx.metal.stop_capture()
