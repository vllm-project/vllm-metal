# SPDX-License-Identifier: Apache-2.0
"""Metal Worker for vLLM v1 engine."""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import mlx.core as mx
from vllm.config import VllmConfig
from vllm.distributed import (
    ensure_model_parallel_initialized,
    init_distributed_environment,
)
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor import set_random_seed
from vllm.tasks import SupportedTask
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.worker_base import WorkerBase

from vllm_metal.config import (
    AUTO_MEMORY_MIN_BLOCKS_BUFFER_FACTOR,
    AUTO_MEMORY_OVERHEAD_FACTOR,
    get_config,
)
from vllm_metal.platform import MetalPlatform
from vllm_metal.utils import set_wired_limit

if TYPE_CHECKING:
    from vllm_metal.v1.model_runner import MetalModelRunner

logger = init_logger(__name__)


@dataclass(frozen=True)
class _AutoMemoryEstimate:
    total_memory: int
    model_memory: int
    kv_cache_memory: int
    total_needed: int
    needed_fraction: float
    max_model_len: int
    block_size_tokens: int
    min_blocks: int
    overhead_factor: float


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
        is_driver_worker: bool = False,
        **kwargs: Any,
    ):
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
            logger.info(f"MLX device set to: {mx.default_device()}")
            set_wired_limit()

        # Use MetalPlatform.get_torch_device() to properly support MPS when available.
        # This ensures consistency with the platform's device selection logic and
        # allows using MPS for PyTorch operations (like vLLM's sampler) when supported,
        # while falling back to CPU if MPS is not available.
        self.device = MetalPlatform.get_torch_device(0)
        logger.info(f"PyTorch device set to: {self.device}")

        # Initialize distributed environment
        init_worker_distributed_environment(
            self.vllm_config,
            self.rank,
            self.distributed_init_method,
            self.local_rank,
        )

        # Set random seed
        set_random_seed(self.model_config.seed)

        # Import here to avoid circular imports
        from vllm_metal.v1.model_runner import MetalModelRunner

        # Create model runner
        self.model_runner = MetalModelRunner(
            vllm_config=self.vllm_config,
            device=self.device,
        )

    def load_model(self) -> None:
        """Load the model onto the Metal device."""
        self.model_runner.load_model()

        # Patch model for paged attention if enabled
        if self.metal_config.use_paged_attention:
            self._setup_paged_attention()

        # In auto mode, set MLX memory limit to prevent unbounded growth
        if self.metal_config.is_auto_memory:
            self._set_auto_memory_limit()

    def _setup_paged_attention(self) -> None:
        """Create MPSPagedKVCache and patch model attention for HF Metal kernel."""
        import torch

        from vllm_metal.metal_kernel_backend.cache import MPSPagedKVCache
        from vllm_metal.metal_kernel_backend.paged_attention import (
            patch_model_attention_metal_kernel,
        )

        runner = self.model_runner
        block_size = self.metal_config.block_size

        # Extract model dimensions
        num_layers = (
            runner.model_args.get("num_hidden_layers")
            or runner.model_args.get("n_layers")
            or 32
        )
        num_attention_heads = runner.model_args.get("num_attention_heads") or 32
        num_kv_heads = (
            runner.model_args.get("num_key_value_heads")
            or runner.model_args.get("n_kv_heads")
            or num_attention_heads
        )
        hidden_size = runner.model_args.get("hidden_size") or 4096
        head_dim = runner.model_args.get("head_dim") or (
            hidden_size // num_attention_heads
        )

        # Allocate blocks
        max_model_len = self.model_config.max_model_len
        num_blocks = (max_model_len + block_size - 1) // block_size
        num_blocks = int(num_blocks * 4)  # buffer for batched decode

        mps_kv_cache = MPSPagedKVCache(
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            num_blocks=num_blocks,
            block_size=block_size,
            dtype=torch.float16,
        )

        n_patched = patch_model_attention_metal_kernel(
            runner.model, mps_kv_cache, block_size
        )
        logger.info(
            "Metal kernel paged attention enabled: %d layers patched, "
            "%d blocks allocated (block_size=%d, kv_heads=%d, head_dim=%d)",
            n_patched,
            num_blocks,
            block_size,
            num_kv_heads,
            head_dim,
        )

        # Store on model runner for use by paged prefill/decode
        runner._paged_kv_cache = mps_kv_cache
        runner._paged_block_size = block_size

    def _get_model_memory_usage(self) -> int:
        """Get current model memory usage from MLX.

        Returns:
            Memory usage in bytes
        """
        # Force evaluation of any pending computations
        mx.eval(mx.array([0]))

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
        estimate = self._estimate_auto_memory()
        if estimate.total_needed > estimate.total_memory:
            raise ValueError(self._format_auto_memory_infeasible_error(estimate))

        # Set MLX memory limit
        if hasattr(mx, "set_memory_limit"):
            mx.set_memory_limit(estimate.total_needed)
            logger.info(
                f"Auto mode: set MLX memory limit to {estimate.total_needed / 1e9:.2f}GB "
                f"(model={estimate.model_memory / 1e9:.2f}GB, kv_cache={estimate.kv_cache_memory / 1e9:.2f}GB)"
            )
        else:
            logger.warning(
                "mx.set_memory_limit not available, memory may grow unbounded"
            )

    def _estimate_auto_memory(self) -> _AutoMemoryEstimate:
        import psutil

        total_memory = psutil.virtual_memory().total
        model_memory = self._get_model_memory_usage()

        block_size_bytes = self.get_cache_block_size_bytes()
        if block_size_bytes <= 0:
            msg = f"Computed KV cache block size is invalid ({block_size_bytes} bytes)."
            raise ValueError(msg)

        block_size_tokens = self.metal_config.block_size
        max_model_len = self.model_config.max_model_len

        min_blocks = (max_model_len + block_size_tokens - 1) // block_size_tokens
        min_blocks = int(min_blocks * AUTO_MEMORY_MIN_BLOCKS_BUFFER_FACTOR)
        kv_cache_memory = min_blocks * block_size_bytes

        overhead_factor = AUTO_MEMORY_OVERHEAD_FACTOR
        total_needed = int((model_memory + kv_cache_memory) * overhead_factor)
        needed_fraction = total_needed / total_memory

        return _AutoMemoryEstimate(
            total_memory=total_memory,
            model_memory=model_memory,
            kv_cache_memory=kv_cache_memory,
            total_needed=total_needed,
            needed_fraction=needed_fraction,
            max_model_len=max_model_len,
            block_size_tokens=block_size_tokens,
            min_blocks=min_blocks,
            overhead_factor=overhead_factor,
        )

    def _format_auto_memory_infeasible_error(
        self, estimate: _AutoMemoryEstimate
    ) -> str:
        return (
            "Auto memory mode (VLLM_METAL_MEMORY_FRACTION=auto) requires more "
            "memory than is available. "
            f"total={estimate.total_memory / 1e9:.2f}GB, "
            f"model={estimate.model_memory / 1e9:.2f}GB, "
            f"min_kv_cache={estimate.kv_cache_memory / 1e9:.2f}GB, "
            f"overhead_factor={estimate.overhead_factor:.1f} "
            f"(max_model_len={estimate.max_model_len}, "
            f"block_size={estimate.block_size_tokens}, "
            f"min_blocks={estimate.min_blocks}, "
            f"needed_fraction={estimate.needed_fraction:.3f}). "
            "Mitigations: reduce max_model_len, reduce VLLM_METAL_BLOCK_SIZE, "
            "or use a smaller model."
        )

    def determine_available_memory(self) -> int:
        """Determine available memory for KV cache.

        When paged attention is enabled, reports the actual MPS paged cache
        capacity so the vLLM scheduler allocates the right number of blocks.

        Returns:
            Available memory in bytes
        """
        import psutil

        # When paged attention is on, the real KV storage is the MPS paged
        # cache (already allocated).  Report its capacity so the scheduler
        # can make use of all the blocks we actually have.
        if self.metal_config.use_paged_attention:
            runner = self.model_runner
            if (
                hasattr(runner, "_paged_kv_cache")
                and runner._paged_kv_cache is not None
            ):
                paged_cache = runner._paged_kv_cache
                block_size_bytes = self.get_cache_block_size_bytes()
                available = paged_cache.num_blocks * block_size_bytes
                logger.info(
                    "Paged attention: reporting MPS cache capacity "
                    "(%d blocks × %d bytes = %.2f GB)",
                    paged_cache.num_blocks,
                    block_size_bytes,
                    available / 1e9,
                )
                return available

        # Handle auto memory mode
        if self.metal_config.is_auto_memory:
            estimate = self._estimate_auto_memory()
            if estimate.total_needed > estimate.total_memory:
                raise ValueError(self._format_auto_memory_infeasible_error(estimate))

            logger.info(
                f"Auto memory mode: model={estimate.model_memory / 1e9:.2f}GB, "
                f"max_model_len={estimate.max_model_len}, min_blocks={estimate.min_blocks}, "
                f"min_kv_cache={estimate.kv_cache_memory / 1e9:.2f}GB, "
                f"total_needed={estimate.total_needed / 1e9:.2f}GB, "
                f"needed_fraction={estimate.needed_fraction:.3f}"
            )

            # Return just the cache portion for KV cache allocation
            available = estimate.kv_cache_memory
        else:
            total_memory = psutil.virtual_memory().total
            # Use configured fraction of system memory
            available = int(total_memory * self.metal_config.memory_fraction * 0.5)

        logger.info(f"Metal available memory for KV cache: {available / 1e9:.2f} GB")
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
        self, scheduler_output: SchedulerOutput
    ) -> ModelRunnerOutput | None:
        """Execute model inference.

        Args:
            scheduler_output: Scheduler output with batch information

        Returns:
            Model runner output with generated tokens
        """
        return self.model_runner.execute_model(scheduler_output)

    def get_model(self) -> Any:
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

    def add_lora(self, lora_request: LoRARequest) -> bool:
        """Add a LoRA adapter.

        Args:
            lora_request: LoRA request

        Returns:
            False (LoRA not supported on Metal yet)
        """
        logger.warning("LoRA is not supported on Metal platform")
        return False

    def remove_lora(self, lora_id: int) -> bool:
        """Remove a LoRA adapter.

        Args:
            lora_id: LoRA adapter ID

        Returns:
            False (LoRA not supported on Metal yet)
        """
        return False

    def pin_lora(self, lora_id: int) -> bool:
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

    def sleep(self, level: int = 1) -> None:
        """Enter sleep mode (not supported on Metal).

        Args:
            level: Sleep level
        """
        logger.warning("Sleep mode is not supported on Metal, ignoring")

    def wake_up(self, tags: list[str] | None = None) -> None:
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
        except Exception as e:
            raise RuntimeError(f"Metal worker health check failed: {e}") from e

    def shutdown(self) -> None:
        """Shutdown the worker and cleanup resources."""
        if hasattr(self, "model_runner") and self.model_runner is not None:
            del self.model_runner
            self.model_runner = None

        gc.collect()
        logger.info("Metal worker shutdown complete")
