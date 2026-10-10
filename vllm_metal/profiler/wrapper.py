# SPDX-License-Identifier: Apache-2.0
"""Metal frame-capture wrapper for vLLM's WorkerProfiler abstraction.

Subclasses ``vllm.profiler.wrapper.WorkerProfiler`` so that the manual
start/stop surface — ``LLM.start_profile`` / ``LLM.stop_profile``, the
``/start_profile`` and ``/stop_profile`` HTTP endpoints, and the engine's
``collective_rpc("profile", ...)`` plumbing — routes through unchanged.

Uses the selected backend's capture API: MLX start/stop calls or PyTorch's
``torch.mps.profiler.metal_capture`` context manager. The output is a
``.gputrace`` bundle that opens directly in Xcode.

The ``delay_iterations`` / ``max_iterations`` scheduling fields are
**rejected** at construction time: they are advanced by
``WorkerProfiler.step()``, which the metal worker never calls.

Apple gates frame capture behind ``MTL_CAPTURE_ENABLED=1`` in the process
environment. We check that up front and raise with an actionable message.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import override
from uuid import uuid4

from vllm.config import ProfilerConfig
from vllm.logger import init_logger
from vllm.profiler.wrapper import WorkerProfiler

from vllm_metal import envs

logger = init_logger(__name__)


@contextmanager
def _mlx_capture(path: str) -> Iterator[None]:
    import mlx.core as mx

    mx.metal.start_capture(path)
    try:
        yield
    finally:
        mx.metal.stop_capture()


@contextmanager
def _mps_capture(path: str) -> Iterator[None]:
    from torch.mps.profiler import metal_capture

    # PyTorch 2.13 writes <counter>-<name>.gputrace in the worker's cwd.
    # Give it a unique basename, then honor our configured output directory.
    name = f"vllm_{uuid4().hex}"
    with metal_capture(name):
        yield
    (capture,) = Path.cwd().glob(f"*-{name}.gputrace")
    shutil.move(str(capture), path)


class MetalProfilerWrapper(WorkerProfiler):
    """Metal frame-capture flavor of vLLM's WorkerProfiler.

    Trace output: ``<torch_profiler_dir>/<trace_name>_<capture_id>.gputrace``
    """

    def __init__(self, profiler_config: ProfilerConfig, trace_name: str) -> None:
        # delay_iterations / max_iterations are advanced by
        # WorkerProfiler.step(), which the metal worker never calls. Reject
        # up front rather than letting them silently no-op.
        if profiler_config.delay_iterations > 0 or profiler_config.max_iterations > 0:
            raise ValueError(
                "Metal frame capture does not support delay_iterations or "
                "max_iterations — the metal worker does not pump "
                "WorkerProfiler.step(), so those scheduling knobs are inert. "
                "Use start_profile / stop_profile to bracket the work "
                "manually, and bound capture via short prompts and small "
                "max_tokens."
            )

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
        self._trace_prefix = Path(trace_dir) / f"{trace_name}_"
        self._capture = ExitStack()

    @override
    def _start(self) -> None:
        # Metal refuses to capture to an existing bundle, including one left
        # by an earlier start/stop cycle or a previous worker instance.
        trace_path = f"{self._trace_prefix}{uuid4().hex}.gputrace"
        capture = _mps_capture if envs.VLLM_METAL_BACKEND == "mps" else _mlx_capture
        # Adapt the native capture context to vLLM's separate start/stop RPCs.
        self._capture.enter_context(capture(trace_path))
        logger.info(
            "Metal frame capture started. Trace will be saved to %s", trace_path
        )

    @override
    def _stop(self) -> None:
        self._capture.close()
