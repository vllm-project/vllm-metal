# SPDX-License-Identifier: Apache-2.0
"""Metal Worker for vLLM v1 engine."""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING, Any

import mlx.core as mx
import torch
from vllm.config import VllmConfig  # noqa: TC002
from vllm.distributed import (
    ensure_model_parallel_initialized,
    init_distributed_environment,
)
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest  # noqa: TC002
from vllm.model_executor import set_random_seed
from vllm.tasks import SupportedTask  # noqa: TC002
from vllm.v1.core.sched.output import SchedulerOutput  # noqa: TC002
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec  # noqa: TC002
from vllm.v1.outputs import ModelRunnerOutput  # noqa: TC002
from vllm.v1.worker.worker_base import WorkerBase

from vllm_metal.config import get_config
from vllm_metal.platform import MetalPlatform
from vllm_metal.utils import set_wired_limit

if TYPE_CHECKING:
    from vllm_metal.v1.model_runner import MetalModelRunner

logger = init_logger(__name__)


def init_worker_distributed_environment(
    vllm_config: VllmConfig,
    rank: int,
    distributed_init_method: str,
    local_rank: int,
) -> None:
    """Initialize distributed environment for Metal worker."""
    parallel_config = vllm_config.parallel_config

    init_distributed_environment(
        parallel_config.world_size,
        rank,
        distributed_init_method,
        local_rank,
        backend="gloo",  # Use gloo for CPU-based distributed
    )

    ensure_model_parallel_initialized(
        parallel_config.tensor_parallel_size,
        parallel_config.pipeline_parallel_size,
    )


class MetalWorker(WorkerBase):
    """Worker implementation for Apple Silicon Metal/MLX.

    This worker handles model loading and inference on Apple Silicon
    using MLX as the primary compute backend.
    """

    # Override model_runner type from base class
    model_runner: MetalModelRunner  # type: ignore[assignment]

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,  # noqa: FBT001, FBT002
        **kwargs: Any,  # noqa: ARG002, ANN401
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            is_driver_worker=is_driver_worker,
        )
        self.metal_config = get_config()

        # Disable custom all reduce (not supported on Metal)
        self.parallel_config.disable_custom_all_reduce = True

    def init_device(self) -> None:
        """Initialize the Metal device and distributed environment."""
        import psutil  # noqa: PLC0415

        # Set up MLX device
        if self.metal_config.use_mlx:
            device_type = (
                mx.DeviceType.gpu
                if self.metal_config.mlx_device == "gpu"
                else mx.DeviceType.cpu
            )
            mx.set_default_device(mx.Device(device_type))
            logger.info("MLX device set to: %s", mx.default_device())
            set_wired_limit()

            # Log system memory info for debugging OOM issues
            total_memory = psutil.virtual_memory().total
            available_memory = psutil.virtual_memory().available
            device_info = mx.metal.device_info()
            max_buffer_size = device_info.get("max_buffer_length", 0)
            if isinstance(max_buffer_size, str):
                max_buffer_size = (
                    int(max_buffer_size) if max_buffer_size.isdigit() else 0
                )
            elif isinstance(max_buffer_size, float):
                max_buffer_size = int(max_buffer_size)

            logger.info(
                "[Memory] System: total=%.2fGB, available=%.2fGB, max_buffer=%.2fGB",
                total_memory / (1024**3),
                available_memory / (1024**3),
                max_buffer_size / (1024**3) if max_buffer_size > 0 else 0,
            )
            # Note: MLX memory limit is set after model load in _set_auto_memory_limit()

        # Use MetalPlatform.get_torch_device() to properly support MPS when available.
        # This ensures consistency with the platform's device selection logic and
        # allows using MPS for PyTorch operations (like vLLM's sampler) when supported,
        # while falling back to CPU if MPS is not available.
        self.device = MetalPlatform.get_torch_device(0)
        logger.info("PyTorch device set to: %s", self.device)

        # Initialize distributed environment only if actually needed (multi-GPU setup)
        # For single-device Metal usage, we can skip distributed initialization
        # to avoid network timeout issues during benchmarking
        if self.parallel_config.world_size > 1:
            logger.info(
                "Multi-device setup detected (world_size=%d), initializing distributed environment",
                self.parallel_config.world_size,
            )
            init_worker_distributed_environment(
                self.vllm_config,
                self.rank,
                self.distributed_init_method,
                self.local_rank,
            )
        else:
            # For single-device Metal, just ensure model parallel is initialized appropriately
            # but avoid network-based distributed setup which can cause timeouts
            logger.info("Single-device Metal setup detected, skipping distributed init")

            # Explicitly initialize model parallel for single device to avoid issues
            # but avoid network-based initialization that can cause timeouts
            from vllm.distributed.parallel_state import (  # noqa: PLC0415
                ensure_model_parallel_initialized,
                init_distributed_environment,
            )

            # Initialize with local world size of 1 to avoid network calls
            try:
                init_distributed_environment(
                    world_size=1,
                    rank=0,
                    local_rank=self.local_rank,
                    distributed_init_method=None,  # No network init method
                    backend="nccl" if torch.cuda.is_available() else "gloo",
                )
                ensure_model_parallel_initialized(1, 1)
            except Exception as e:
                logger.warning(
                    "Failed to initialize distributed environment for single device: %s. "
                    "This may cause issues in some scenarios.",
                    e,
                )

        # Set random seed
        set_random_seed(self.model_config.seed)

        # Import here to avoid circular imports
        from vllm_metal.v1.model_runner import MetalModelRunner  # noqa: PLC0415

        # Create model runner
        self.model_runner = MetalModelRunner(
            vllm_config=self.vllm_config,
            device=self.device,
        )

    def load_model(self) -> None:
        """Load the model onto the Metal device."""
        self.model_runner.load_model()

        # Always calculate memory budget after model loads
        # This sets both MLX memory limit and _kv_cache_budget
        self._set_auto_memory_limit()

        # Clear cache after model loading to free up memory
        try:
            mx.metal.clear_cache()
        except (RuntimeError, OSError):
            # Log the exception instead of silently ignoring it
            logger.debug("Failed to clear MLX cache", exc_info=True)

    def _get_model_memory_usage(self) -> int:
        """Get current model memory usage from MLX.

        Returns:
            Memory usage in bytes

        """
        # Force evaluation of any pending computations
        try:
            mx.eval(mx.array([0]))
        except RuntimeError as e:
            if "Attempting to allocate" in str(
                e,
            ) and "greater than the maximum allowed buffer size" in str(e):
                # Even tiny arrays can fail in extreme cases - clear cache and try again
                mx.metal.clear_cache()
                mx.eval(mx.array([0]))
            else:
                raise

        # Get active memory usage - try new API first, then deprecated
        if hasattr(mx, "get_active_memory"):
            return mx.get_active_memory()
        if hasattr(mx, "metal") and hasattr(mx.metal, "get_active_memory"):
            return mx.metal.get_active_memory()

        # Fallback: estimate from model config if available
        if hasattr(self, "model_runner") and self.model_runner is not None:
            model_config = self.model_config
            hidden_size = getattr(model_config, "hidden_size", 4096)
            num_layers = getattr(model_config, "num_hidden_layers", 32)
            # Rough parameter count estimate
            params = hidden_size * hidden_size * 4 * num_layers
            return params * 2

        return 0

    def _set_auto_memory_limit(self) -> None:
        """Set MLX memory limit and calculate KV cache budget.

        This is the unified memory calculation - called after model loads.
        Sets both the MLX memory limit and stores _kv_cache_budget for
        determine_available_memory() to return.
        """
        import psutil  # noqa: PLC0415

        # Get model memory after loading
        model_memory = self._get_model_memory_usage()

        # Get constraints
        available_memory = psutil.virtual_memory().available
        device_info = mx.metal.device_info()
        max_buffer = device_info.get("max_buffer_length", 0)
        if isinstance(max_buffer, str):
            max_buffer = int(max_buffer) if max_buffer.isdigit() else 0
        elif isinstance(max_buffer, float):
            max_buffer = int(max_buffer)

        # Calculate max allowed (respect both system RAM and Metal limits)
        if max_buffer > 0:
            max_allowed = min(available_memory, max_buffer)
        else:
            max_allowed = available_memory

        # TODO: overhead should be profiled; using 0.2 * model_memory for now
        overhead = int(model_memory * 0.2)

        # KV cache budget = what's left after model and overhead
        kv_cache_budget = max_allowed - overhead
        kv_cache_budget = max(0, kv_cache_budget)  # safety floor
        kv_cache_budget = int(0.6 * kv_cache_budget)  # for debugging

        # Store for determine_available_memory() to use
        self._kv_cache_budget = kv_cache_budget

        # MLX limit = everything we plan to use
        mlx_limit = model_memory + kv_cache_budget + overhead

        logger.info(
            "[Memory] Budget: available=%.2fGB, max_buffer=%.2fGB, "
            "model=%.2fGB, overhead=%.2fGB, kv_cache=%.2fGB",
            available_memory / (1024**3),
            max_buffer / (1024**3) if max_buffer > 0 else 0,
            model_memory / (1024**3),
            overhead / (1024**3),
            kv_cache_budget / (1024**3),
        )

        # Set MLX memory limit
        if hasattr(mx, "set_memory_limit"):
            mx.set_memory_limit(mlx_limit)
            logger.info(
                "[Memory] Set MLX limit: %.2fGB",
                mlx_limit / (1024**3),
            )
        else:
            logger.warning(
                "mx.set_memory_limit not available, memory may grow unbounded",
            )

    def determine_available_memory(self) -> int:
        """Return available memory for KV cache.

        Returns the budget calculated by _set_auto_memory_limit() after model load.

        Returns:
            Available memory in bytes

        """
        if hasattr(self, "_kv_cache_budget"):
            logger.info(
                "[Memory] Returning KV cache budget: %.2fGB",
                self._kv_cache_budget / (1024**3),
            )
            return self._kv_cache_budget

        # Fallback: if called before model load (shouldn't happen)
        import psutil  # noqa: PLC0415

        logger.warning("[Memory] determine_available_memory called before model load")
        return int(psutil.virtual_memory().available * 0.5)

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """Get KV cache specification.

        Returns:
            Dictionary mapping layer names to KV cache specs

        """
        return self.model_runner.get_kv_cache_spec()

    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks: int) -> None:
        """Initialize the KV cache.

        Args:
            num_gpu_blocks: Number of GPU cache blocks
            num_cpu_blocks: Number of CPU cache blocks (unused on Metal)

        """
        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        """Initialize from KV cache configuration.

        Args:
            kv_cache_config: KV cache configuration for this worker

        """
        self.model_runner.initialize_kv_cache(kv_cache_config)

    def compile_or_warm_up_model(self) -> None:
        """Warm up the model for inference."""
        # Reset seed for reproducibility
        set_random_seed(self.model_config.seed)
        self.model_runner.warm_up()

    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
    ) -> ModelRunnerOutput | None:
        """Execute model inference.

        Args:
            scheduler_output: Scheduler output with batch information

        Returns:
            Model runner output with generated tokens

        """
        return self.model_runner.execute_model(scheduler_output)

    def get_model(self) -> object:
        """Get the underlying model.

        Returns:
            The loaded model

        """
        return self.model_runner.model

    def get_cache_block_size_bytes(self) -> int:
        """Get the size of a single cache block in bytes.

        Returns:
            Block size in bytes

        """
        return self.model_runner.get_cache_block_size_bytes()

    def get_cache_usage(self) -> tuple[int, int, float]:
        """Get KV cache usage statistics.

        Returns:
            Tuple of (used_blocks, total_blocks, usage_ratio)

        """
        return self.model_runner.get_cache_usage()

    def add_lora(self, lora_request: LoRARequest) -> bool:  # noqa: ARG002
        """Add a LoRA adapter.

        Args:
            lora_request: LoRA request

        Returns:
            False (LoRA not supported on Metal yet)

        """
        logger.warning("LoRA is not supported on Metal platform")
        return False

    def remove_lora(self, lora_id: int) -> bool:  # noqa: ARG002
        """Remove a LoRA adapter.

        Args:
            lora_id: LoRA adapter ID

        Returns:
            False (LoRA not supported on Metal yet)

        """
        return False

    def pin_lora(self, lora_id: int) -> bool:  # noqa: ARG002
        """Pin a LoRA adapter.

        Args:
            lora_id: LoRA adapter ID

        Returns:
            False (LoRA not supported on Metal yet)

        """
        return False

    def list_loras(self) -> set[int]:
        """List loaded LoRA adapters.

        Returns:
            Empty set (LoRA not supported)

        """
        return set()

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        """Get supported tasks for this worker.

        Returns:
            Tuple of supported task types

        """
        return ("generate",)

    def sleep(self, level: int = 1) -> None:  # noqa: ARG002
        """Enter sleep mode (not supported on Metal).

        Args:
            level: Sleep level

        """
        logger.warning("Sleep mode is not supported on Metal, ignoring")

    def wake_up(self, tags: list[str] | None = None) -> None:  # noqa: ARG002
        """Wake up from sleep mode (not supported on Metal).

        Args:
            tags: Wake up tags

        """
        logger.warning("Sleep mode is not supported on Metal, ignoring")

    def check_health(self) -> None:
        """Check worker health."""
        # Metal worker is healthy if MLX is available
        try:
            mx.eval(mx.array([1.0]))
        except RuntimeError as e:
            if "Attempting to allocate" in str(
                e,
            ) and "greater than the maximum allowed buffer size" in str(e):
                # Even tiny arrays can fail in extreme cases - clear cache and try again
                mx.metal.clear_cache()
                mx.eval(mx.array([1.0]))
            else:
                error_msg = f"Metal worker health check failed: {e}"
                raise RuntimeError(error_msg) from e
        except Exception as e:
            error_msg = f"Metal worker health check failed: {e}"
            raise RuntimeError(error_msg) from e

    def shutdown(self) -> None:
        """Shutdown the worker and cleanup resources."""
        if hasattr(self, "model_runner") and self.model_runner is not None:
            del self.model_runner
            self.model_runner = None

        gc.collect()
        logger.info("Metal worker shutdown complete")
