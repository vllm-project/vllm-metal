# SPDX-License-Identifier: Apache-2.0
"""Metal V2 Worker implementation for vLLM."""

import gc
from contextlib import AbstractContextManager, nullcontext

import torch
from vllm.config import VllmConfig
from vllm.distributed import (
    ensure_model_parallel_initialized,
    init_distributed_environment,
    set_custom_all_reduce,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.batch_invariant import init_batch_invariance
from vllm.model_executor.utils import set_random_seed
from vllm.platforms import current_platform
from vllm.v1.utils import report_usage_stats
from vllm.v1.worker.gpu_worker import Worker

from vllm_metal.utils import (
    check_metal_availability,
    get_metal_device_info,
    get_metal_memory_info,
    metal_empty_cache,
)
from vllm_metal.v2.model_runner import MetalModelRunner

logger = init_logger(__name__)


class MetalWorker(Worker):
    """V2 Worker implementation for Apple Metal/MLX backend.

    This worker extends the base GPU Worker to provide Metal-specific
    device initialization and model execution using MLX as the primary
    compute backend.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ):
        # Validate Metal availability before initialization
        available, error = check_metal_availability()
        if not available:
            raise RuntimeError(f"Metal not available: {error}")

        super().__init__(
            vllm_config=vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            is_driver_worker=is_driver_worker,
        )

        # Disable custom all reduce - not supported on Metal
        self.parallel_config.disable_custom_all_reduce = True

        # Disable profiler for Metal (CUDA-specific)
        self.profiler = None

        logger.info(
            f"Initialized MetalWorker: local_rank={local_rank}, "
            f"rank={rank}, is_driver={is_driver_worker}"
        )

    def init_device(self):
        """Initialize the Metal/MPS device and MLX."""
        import mlx.core as mx

        # Set the device to MPS for PyTorch compatibility
        self.device = torch.device("mps")
        current_platform.set_device(self.device)

        # Set MLX to use GPU
        mx.set_default_device(mx.gpu)

        # Log device info
        info = get_metal_device_info()
        logger.info(
            f"Metal device initialized: {info['name']}, "
            f"MLX available: {info.get('mlx_available', False)}, "
            f"Total memory: {info['total_memory'] / 1e9:.1f}GB"
        )

        # Initialize batch invariance
        init_batch_invariance()

        # Disable custom all reduce for Metal
        set_custom_all_reduce(False)

        # Initialize distributed environment for single-process Metal
        # Use gloo backend since nccl is CUDA-only
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method=self.distributed_init_method
            or "tcp://127.0.0.1:12345",
            local_rank=0,
            backend="gloo",
        )

        # Initialize model parallel groups (single GPU, so all sizes are 1)
        ensure_model_parallel_initialized(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
        )

        # Set random seed
        set_random_seed(self.model_config.seed)
        mx.random.seed(self.model_config.seed)

        # Clear cache before measuring memory
        gc.collect()
        metal_empty_cache()

        # Get memory info for Metal
        allocated, total = get_metal_memory_info()
        free = total - allocated

        # Store memory snapshot info (simplified for Metal)
        class MetalMemorySnapshot:
            def __init__(self, free_mem, total_mem):
                self.free_memory = free_mem
                self.total_memory = total_mem

        self.init_snapshot = MetalMemorySnapshot(free, total)
        self.requested_memory = int(total * self.cache_config.gpu_memory_utilization)

        # Create the model runner
        self.model_runner = MetalModelRunner(self.vllm_config, self.device)

        if self.rank == 0:
            # If usage stat is enabled, collect relevant info.
            report_usage_stats(self.vllm_config)

    def compile_or_warm_up_model(self) -> None:
        """Compile or warm up the model.

        Metal uses MLX for compute, no graph compilation needed.
        """
        logger.info("Metal warmup: using MLX compute backend")

    def sleep(self, level: int = 1) -> None:
        """Sleep mode - clear Metal cache."""
        metal_empty_cache()
        logger.info("Metal worker entered sleep mode")

    def wake_up(self, tags: list[str] | None = None) -> None:
        """Wake up from sleep mode."""
        logger.info("Metal worker woke up")

    def determine_available_memory(self) -> int:
        """Determine available memory for KV cache.

        Returns:
            Available memory in bytes for KV cache.
        """
        info = get_metal_device_info()
        total_memory = info.get("total_memory", 0)

        # Apply memory utilization factor
        gpu_memory_utilization = self.cache_config.gpu_memory_utilization
        available = int(total_memory * gpu_memory_utilization)

        logger.info(
            f"Metal memory: {total_memory / 1e9:.1f}GB total, "
            f"{available / 1e9:.1f}GB available for cache "
            f"({gpu_memory_utilization * 100:.0f}% utilization)"
        )

        return available

    def _maybe_get_memory_pool_context(self, tag: str) -> AbstractContextManager:
        """Metal doesn't use CUDA memory pools, return no-op context."""
        return nullcontext()

    def load_model(self) -> None:
        """Load the model onto the Metal device."""
        logger.info("Loading model on Metal device with MLX backend...")
        self.model_runner.load_model()
        logger.info("Model loaded successfully on Metal device")
