# SPDX-License-Identifier: Apache-2.0
"""MPS Model Runner for vLLM v1 API on Apple Silicon.

Performance optimizations for high throughput streaming:
- Deferred synchronization: Only sync when absolutely necessary
- Rust-accelerated tensor conversion: Fast Python list building
- Unified memory exploitation: Minimize copies in MPS unified memory model
- Optimized decode path: Streamlined single-sequence decode
"""

import os
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)

# Try to import Rust extensions for accelerated tensor operations
try:
    from vllm_metal_rust import tensor_1d_to_nested_list, tensor_to_nested_list
    RUST_AVAILABLE = True
    logger.info("Rust extensions loaded for accelerated tensor operations")
except ImportError:
    RUST_AVAILABLE = False
    logger.warning("Rust extensions not available, using Python fallback")

# Profiling counters - enabled via VLLM_MPS_PROFILE=1
_profile_enabled = os.environ.get("VLLM_MPS_PROFILE", "0") == "1"
_profile_counts: dict[str, int] = {}
_profile_times: dict[str, float] = {}
_profile_start_time: float = 0.0


def _profile_start(name: str) -> float:
    """Start profiling a section."""
    if not _profile_enabled:
        return 0.0
    torch.mps.synchronize()
    return time.perf_counter()


def _profile_end(name: str, start: float) -> None:
    """End profiling a section."""
    if not _profile_enabled or start == 0.0:
        return
    torch.mps.synchronize()
    elapsed = time.perf_counter() - start
    _profile_counts[name] = _profile_counts.get(name, 0) + 1
    _profile_times[name] = _profile_times.get(name, 0.0) + elapsed


def print_mps_profile() -> None:
    """Print profiling summary."""
    if not _profile_times:
        print("No profiling data collected. Set VLLM_MPS_PROFILE=1 to enable.")
        return
    total = sum(_profile_times.values())
    print("\n=== MPS Model Runner Profile ===")
    for name, time_s in sorted(_profile_times.items(), key=lambda x: -x[1]):
        count = _profile_counts.get(name, 1)
        avg_ms = (time_s / count) * 1000 if count > 0 else 0
        pct = (time_s / total) * 100 if total > 0 else 0
        print(f"  {name}: {count} calls, {avg_ms:.2f}ms avg, {pct:.1f}%")
    print(f"  Total: {total * 1000:.1f}ms")


class MPSModelRunner(GPUModelRunner):
    """Model runner for Apple MPS (Metal Performance Shaders) backend.

    This inherits from GPUModelRunner but adapts it for MPS devices,
    similar to how CPUModelRunner adapts it for CPU.
    """

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        with _torch_cuda_wrapper():
            super().__init__(vllm_config, device)

        assert device.type == "mps", f"Expected MPS device, got {device}"
        assert self.speculative_config is None, (
            "Speculative decoding is not supported on MPS."
        )

        # MPS doesn't support CUDA graphs
        self.use_cuda_graph = False
        self.cascade_attn_enabled = False

        self._postprocess_tensors()

    def _postprocess_tensors(self) -> None:
        """Replace CUDA-specific tensors with MPS-compatible ones.

        MPS uses unified memory, so we can use the same tensors for both
        CPU and GPU operations in many cases.
        """

        def replace_tensor(obj: Any, cpu_attr_name: str, device_attr_name: str) -> None:
            cpu_tensor = getattr(obj, cpu_attr_name, None)
            device_tensor = getattr(obj, device_attr_name, None)
            if cpu_tensor is not None and device_tensor is not None:
                assert isinstance(cpu_tensor, torch.Tensor)
                assert isinstance(device_tensor, torch.Tensor)
                # Move CPU tensor to MPS device
                setattr(obj, device_attr_name, cpu_tensor.to(self.device))

        # Handle CpuGpuBuffer objects - MPS can use unified memory
        for v in vars(self).values():
            if isinstance(v, CpuGpuBuffer):
                # For MPS, move the cpu buffer to MPS device
                v.gpu = v.cpu.to(self.device)

        # Handle input batch tensors
        for k, v in vars(self.input_batch).items():
            if k.endswith("_cpu_tensor") and isinstance(v, torch.Tensor):
                replace_tensor(self.input_batch, k, k[:-11])

        # Handle block tables
        for block_table in self.input_batch.block_table.block_tables:
            for v in vars(block_table).values():
                if isinstance(v, CpuGpuBuffer):
                    v.gpu = v.cpu.to(self.device)

    def get_model(self) -> nn.Module:
        return self.model

    def warming_up_model(self) -> None:
        """Warm up the model for MPS compilation."""
        logger.info("Warming up model for MPS...")

        # Run a dummy forward pass to compile any lazy operations
        with _set_mps_compilation_settings(self.vllm_config):
            self._dummy_run(
                min(
                    max(16, self.max_num_reqs),
                    self.scheduler_config.max_num_batched_tokens,
                )
            )

        # Synchronize MPS to ensure compilation is complete
        if torch.backends.mps.is_available():
            torch.mps.synchronize()

        logger.info("MPS warming up done.")

    def _init_device_properties(self) -> None:
        """Initialize MPS device properties.

        MPS doesn't have compute capability like CUDA, so this is mostly a no-op.
        """
        pass

    def _sync_device(self) -> None:
        """Synchronize MPS device."""
        if torch.backends.mps.is_available():
            torch.mps.synchronize()

    def get_dp_padding(self, num_tokens: int) -> tuple[int, torch.Tensor | None]:
        """Get data parallel padding.

        MPS doesn't support distributed training, so no padding is needed.
        """
        return 0, None

    def capture_model(self) -> int:
        """Capture model for graph execution.

        MPS doesn't support CUDA graphs, so this returns 0.
        """
        logger.debug("MPS does not support CUDA graph capture, skipping.")
        return 0

    @contextmanager
    def synchronize_input_prep(self):
        """Synchronize input preparation for MPS.

        The parent class uses CUDA events for async synchronization, but MPS
        doesn't support the same stream semantics. However, we DON'T sync here
        because synced_forward() already syncs before reading tensors.
        Removing this sync improves GPU utilization by allowing async operations.
        """
        # Don't sync here - synced_forward() handles synchronization before
        # the model reads input tensors. This allows CPU work to overlap with
        # previous GPU work, improving utilization.
        try:
            yield
        finally:
            pass

    def load_model(self, eep_scale_up: bool = False) -> None:
        """Load model onto MPS device.

        MPS uses unified memory, so no special wrapping is needed.
        The model will run on the GPU and data transfers are essentially
        zero-copy since CPU and GPU share the same physical memory.
        """
        # Call parent's load_model which handles all the complexity
        super().load_model(eep_scale_up)

        # Verify model is on MPS device
        try:
            first_param = next(iter(self.model.parameters()))
            if first_param.device.type != "mps":
                logger.warning(
                    f"Model is NOT on MPS! Device: {first_param.device}. "
                    "This may cause low GPU utilization."
                )
            else:
                logger.info(f"Model loaded on device: {first_param.device}")
        except StopIteration:
            logger.warning("Model has no parameters!")

    def _sample(self, logits, spec_decode_metadata):
        """Sample tokens from logits.

        Don't sync here - defer to _to_list() which is called when we actually
        need to read the results. This allows more GPU work to happen in parallel.
        """
        logger.debug("MPSModelRunner._sample: calling parent _sample")
        result = super()._sample(logits, spec_decode_metadata)
        # Don't sync here - defer synchronization to when we actually need
        # to read the results (in _to_list). This improves GPU utilization.
        return result

    def _to_list(self, sampled_token_ids: torch.Tensor) -> list[list[int]]:
        """Convert sampled token IDs tensor to Python list with MPS sync.

        MPS requires explicit synchronization before reading GPU tensor values
        because our placeholder CUDA events don't actually synchronize.

        Optimizations:
        - Uses Rust extension for fast list building when available
        - Minimizes Python object allocations
        - Single sync point before tensor read
        """
        start = _profile_start("_to_list")

        # Sync MPS to ensure sampling is complete before reading tensor values
        # This is the critical sync point - we've deferred it as long as possible
        sync_start = _profile_start("_to_list.sync")
        torch.mps.synchronize()
        _profile_end("_to_list.sync", sync_start)

        conv_start = _profile_start("_to_list.convert")

        # Move to CPU and get numpy array (unified memory makes this fast)
        cpu_tensor = sampled_token_ids.cpu()
        arr = cpu_tensor.numpy()

        if RUST_AVAILABLE:
            # Use Rust extension for fast conversion
            # This avoids Python object allocation overhead
            if sampled_token_ids.dim() == 1:
                result = tensor_1d_to_nested_list(arr.astype(np.int64))
            else:
                result = tensor_to_nested_list(arr.astype(np.int64))
        else:
            # Python fallback
            if sampled_token_ids.dim() == 1:
                result = [[int(x)] for x in arr]
            else:
                result = arr.tolist()

        _profile_end("_to_list.convert", conv_start)
        _profile_end("_to_list", start)
        return result

    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors:
        """Execute model with profiling for MPS.

        This wraps the parent execute_model to add detailed timing information.
        """
        if not _profile_enabled:
            return super().execute_model(scheduler_output, intermediate_tensors)

        # Profile the full execution
        total_start = _profile_start("execute_model.total")

        # The parent execute_model does everything, so just time it
        result = super().execute_model(scheduler_output, intermediate_tensors)

        _profile_end("execute_model.total", total_start)

        # Print summary every 100 calls
        count = _profile_counts.get("execute_model.total", 0)
        if count > 0 and count % 100 == 0:
            print_mps_profile()

        return result


@contextmanager
def _torch_cuda_wrapper():
    """Context manager to mock CUDA operations for MPS.

    This allows us to reuse GPU-focused code paths that reference
    CUDA-specific constructs like Events and Streams.
    """

    class _EventPlaceholder:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def record(self, stream=None):
            pass

        def synchronize(self):
            pass

        def wait(self, stream=None):
            pass

        def query(self):
            return True

        def elapsed_time(self, other):
            return 0.0

    class _StreamPlaceholder:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def synchronize(self):
            pass

        def wait_event(self, event):
            pass

        def wait_stream(self, stream):
            pass

        def record_event(self, event=None):
            return _EventPlaceholder()

        def query(self):
            return True

    # Save originals
    cuda_event = torch.cuda.Event
    cuda_stream = torch.cuda.Stream
    try:
        # Patch torch.cuda.Event and torch.cuda.Stream
        torch.cuda.Event = _EventPlaceholder
        torch.cuda.Stream = _StreamPlaceholder
        yield
    finally:
        torch.cuda.Event = cuda_event
        torch.cuda.Stream = cuda_stream


@contextmanager
def _set_mps_compilation_settings(config: VllmConfig):
    """Set up compilation settings for MPS.

    MPS uses different compilation paths than CUDA.
    """
    import torch._inductor.config as torch_inductor_config

    inductor_config = config.compilation_config.inductor_compile_config
    freezing_value = torch_inductor_config.freezing
    try:
        if inductor_config.get("max_autotune", False):
            torch_inductor_config.freezing = True
        yield
    finally:
        torch_inductor_config.freezing = freezing_value
