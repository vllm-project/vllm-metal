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
from vllm_metal.platform import MetalPlatform, set_wired_limit

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

            # Additionally, set memory limits to prevent large allocations
            try:
                device_info = mx.metal.device_info()
                max_buffer_size = device_info.get("max_buffer_size", 0)
                if isinstance(max_buffer_size, str):
                    max_buffer_size = (
                        int(max_buffer_size) if max_buffer_size.isdigit() else 0
                    )
                elif isinstance(max_buffer_size, float):
                    max_buffer_size = int(max_buffer_size)

                if max_buffer_size > 0 and hasattr(mx, "set_memory_limit"):
                    # Set memory limit to 60% of max buffer size to leave more headroom
                    # This is more conservative to prevent the allocation error seen in CI
                    memory_limit = int(max_buffer_size * 0.60)
                    mx.set_memory_limit(memory_limit)
                    logger.info(
                        "Set MLX memory limit to %.1f GB (60%% of max buffer size)",
                        memory_limit / (1024**3),
                    )
            except (RuntimeError, ValueError, OSError) as e:
                logger.warning("Failed to set memory limit: %s", e)

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

        # In auto mode, set MLX memory limit to prevent unbounded growth
        if self.metal_config.is_auto_memory:
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
        """Set MLX memory limit based on auto calculation.

        This prevents MLX from using unbounded memory in auto mode.
        """
        import psutil  # noqa: PLC0415

        # Get model memory after loading
        model_memory = self._get_model_memory_usage()
        total_memory = psutil.virtual_memory().total

        # Calculate KV cache memory for concurrent sequences
        block_size_bytes = self.get_cache_block_size_bytes()
        block_size_tokens = self.metal_config.block_size
        max_model_len = self.model_config.max_model_len

        # Blocks needed per sequence at max length
        blocks_per_seq = (max_model_len + block_size_tokens - 1) // block_size_tokens

        # Get max concurrent sequences
        max_num_seqs = getattr(self.vllm_config.scheduler_config, "max_num_seqs", 256)

        # Estimate cache memory for concurrent sequences (avg 25% of max length initially)
        avg_blocks_per_seq = max(1, blocks_per_seq // 4)
        target_blocks = int(max_num_seqs * avg_blocks_per_seq * 1.1)
        target_cache_memory = target_blocks * block_size_bytes

        # Use 50% of total memory for cache (more conservative to prevent allocation errors)
        # This allows more parallel requests to run while staying within buffer limits
        memory_cap = int((total_memory - model_memory) * 0.5)
        kv_cache_memory = min(target_cache_memory, max(0, memory_cap))

        # Ensure minimum for at least one full sequence
        min_cache_memory = int(blocks_per_seq * block_size_bytes * 1.1)
        kv_cache_memory = max(kv_cache_memory, min_cache_memory)

        # Total memory limit: model + KV cache + 5% overhead (reduced from 15%)
        # This provides more headroom to prevent allocation errors
        memory_limit = int((model_memory + kv_cache_memory) * 1.05)

        # Set MLX memory limit
        if hasattr(mx, "set_memory_limit"):
            # Check if the calculated memory limit exceeds the device's max buffer size
            device_info = mx.metal.device_info()
            max_buffer_size = device_info.get("max_buffer_size", 0)
            if isinstance(max_buffer_size, str):
                max_buffer_size = (
                    int(max_buffer_size) if max_buffer_size.isdigit() else 0
                )
            elif isinstance(max_buffer_size, float):
                max_buffer_size = int(max_buffer_size)

            # If max buffer size is known and our limit is higher, use a safer limit
            if max_buffer_size > 0 and memory_limit > max_buffer_size * 0.9:
                # Use 60% of max buffer size to be more conservative and prevent allocation errors
                # The original 75% was still too aggressive for some CI environments
                memory_limit = int(max_buffer_size * 0.60)

            mx.set_memory_limit(memory_limit)
            logger.info(
                "Auto mode: set MLX memory limit to %.2fGB "
                "(model=%.2fGB, kv_cache=%.2fGB, "
                "max_num_seqs=%d)",
                memory_limit / 1e9,
                model_memory / 1e9,
                kv_cache_memory / 1e9,
                max_num_seqs,
            )
        else:
            logger.warning(
                "mx.set_memory_limit not available, memory may grow unbounded",
            )

    def determine_available_memory(self) -> int:
        """Determine available memory for KV cache.

        Returns:
            Available memory in bytes

        """
        import psutil  # noqa: PLC0415

        total_memory = psutil.virtual_memory().total

        # Handle auto memory mode
        if self.metal_config.is_auto_memory:
            # Get actual model memory usage
            model_memory = self._get_model_memory_usage()

            # Get block size for cache calculation
            block_size_bytes = self.get_cache_block_size_bytes()
            block_size_tokens = self.metal_config.block_size

            # Calculate blocks needed per sequence at max_model_len
            max_model_len = self.model_config.max_model_len
            blocks_per_seq = (
                max_model_len + block_size_tokens - 1
            ) // block_size_tokens

            # Get max concurrent sequences from scheduler config
            # This determines how many sequences can run simultaneously
            max_num_seqs = getattr(
                self.vllm_config.scheduler_config,
                "max_num_seqs",
                256,
            )

            # Scale for concurrency: allocate blocks for concurrent sequences
            # Use a reasonable estimate - not all sequences will be at max_model_len
            # Assume average sequence uses ~25% of max_model_len worth of blocks initially
            avg_blocks_per_seq = max(1, blocks_per_seq // 4)
            target_blocks = max_num_seqs * avg_blocks_per_seq

            # Add 10% safety buffer
            target_blocks = int(target_blocks * 1.1)

            # Calculate memory needed
            target_cache_memory = target_blocks * block_size_bytes

            # Calculate available memory for cache (45% of remaining memory after model)
            # This is more conservative to prevent allocation errors seen in CI
            # The original 65% was still too aggressive for some CI environments
            available_for_cache = int((total_memory - model_memory) * 0.45)
            cache_memory = min(target_cache_memory, available_for_cache)

            # Ensure minimum for at least one full sequence
            min_cache_memory = int(blocks_per_seq * block_size_bytes * 1.1)
            cache_memory = max(cache_memory, min_cache_memory)

            logger.info(
                "Auto memory mode: model=%.2fGB, "
                "max_model_len=%d, max_num_seqs=%d, "
                "blocks_per_seq=%d, target_blocks=%d, "
                "cache_memory=%.2fGB",
                model_memory / 1e9,
                max_model_len,
                max_num_seqs,
                blocks_per_seq,
                target_blocks,
                cache_memory / 1e9,
            )

            available = cache_memory
        else:
            # Use configured fraction of system memory (45% instead of 65% for more safety)
            # The original 65% was still too aggressive for some CI environments
            available = int(total_memory * self.metal_config.memory_fraction * 0.45)

        logger.info("Metal available memory for KV cache: %.2f GB", available / 1e9)
        return available

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
