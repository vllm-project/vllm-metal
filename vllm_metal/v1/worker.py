# SPDX-License-Identifier: Apache-2.0
"""Metal Worker for vLLM v1 engine."""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING, Any

import mlx.core as mx
from vllm.config import VllmConfig
from vllm.distributed import (
    ensure_model_parallel_initialized,
    init_distributed_environment,
)
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.tasks import SupportedTask
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.worker_base import WorkerBase

from vllm_metal.config import (
    PAGED_ATTENTION_DEFAULT_MEMORY_FRACTION,
    PAGED_ATTENTION_MIN_BLOCKS,
    PAGED_ATTENTION_OVERHEAD_BYTES,
    get_config,
)
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

    def _setup_paged_attention(self) -> None:
        """Create MPSPagedKVCache and patch model attention for HF Metal kernel.

        Computes num_blocks from available system RAM, model weight size, and
        a configurable memory fraction, rather than blindly scaling from
        max_model_len.
        """
        import psutil

        from vllm_metal.metal_kernel_backend.cache import MPSPagedKVCache
        from vllm_metal.metal_kernel_backend.paged_attention import (
            patch_model_attention_metal_kernel,
        )

        runner = self.model_runner
        block_size = self.metal_config.block_size

        # --- Determine memory fraction ---
        if self.metal_config.is_auto_memory:
            fraction = PAGED_ATTENTION_DEFAULT_MEMORY_FRACTION
            logger.info(
                "Paged attention: VLLM_METAL_MEMORY_FRACTION=auto, "
                "defaulting to %.2f for paged path",
                fraction,
            )
        else:
            fraction = self.metal_config.memory_fraction

        # --- Gather memory numbers ---
        total_ram = psutil.virtual_memory().total
        model_memory = self._get_model_memory_usage()
        per_block_bytes = self.get_cache_block_size_bytes()

        # --- Compute KV budget ---
        usable_ram = int(total_ram * fraction)
        available_ram = psutil.virtual_memory().available

        if usable_ram > available_ram:
            raise ValueError(
                "Paged attention: requested memory exceeds available RAM. "
                f"total_ram={total_ram / 1e9:.2f}GB, "
                f"fraction={fraction}, "
                f"usable_ram={usable_ram / 1e9:.2f}GB, "
                f"available_ram={available_ram / 1e9:.2f}GB. "
                "The OS and other processes are using "
                f"{(total_ram - available_ram) / 1e9:.2f}GB. "
                "Mitigations: lower VLLM_METAL_MEMORY_FRACTION "
                f"(try {available_ram / total_ram:.2f} or less), "
                "close other applications, or add more RAM."
            )

        kv_budget = usable_ram - model_memory - PAGED_ATTENTION_OVERHEAD_BYTES

        if kv_budget <= 0:
            raise ValueError(
                "Paged attention: not enough memory for KV cache. "
                f"total_ram={total_ram / 1e9:.2f}GB, "
                f"fraction={fraction}, "
                f"usable_ram={usable_ram / 1e9:.2f}GB, "
                f"model_memory={model_memory / 1e9:.2f}GB, "
                f"overhead={PAGED_ATTENTION_OVERHEAD_BYTES / 1e9:.2f}GB, "
                f"kv_budget={kv_budget / 1e9:.2f}GB. "
                "Mitigations: increase VLLM_METAL_MEMORY_FRACTION, "
                "use a smaller model, or add more RAM."
            )

        num_blocks = kv_budget // per_block_bytes

        if num_blocks < PAGED_ATTENTION_MIN_BLOCKS:
            raise ValueError(
                "Paged attention: computed num_blocks too low "
                f"({num_blocks} < minimum {PAGED_ATTENTION_MIN_BLOCKS}). "
                f"total_ram={total_ram / 1e9:.2f}GB, "
                f"fraction={fraction}, "
                f"usable_ram={usable_ram / 1e9:.2f}GB, "
                f"model_memory={model_memory / 1e9:.2f}GB, "
                f"overhead={PAGED_ATTENTION_OVERHEAD_BYTES / 1e9:.2f}GB, "
                f"kv_budget={kv_budget / 1e9:.2f}GB, "
                f"per_block_bytes={per_block_bytes}. "
                "Mitigations: increase VLLM_METAL_MEMORY_FRACTION, "
                "use a smaller model, or add more RAM."
            )

        max_tokens_cached = num_blocks * block_size

        logger.info(
            "Paged attention memory breakdown: "
            "total_ram=%.2fGB, fraction=%.2f, usable_ram=%.2fGB, "
            "model_memory=%.2fGB, overhead=%.2fGB, "
            "kv_budget=%.2fGB, per_block_bytes=%d, "
            "num_blocks=%d, max_tokens_cached=%d",
            total_ram / 1e9,
            fraction,
            usable_ram / 1e9,
            model_memory / 1e9,
            PAGED_ATTENTION_OVERHEAD_BYTES / 1e9,
            kv_budget / 1e9,
            per_block_bytes,
            num_blocks,
            max_tokens_cached,
        )

        # --- Create cache and patch model ---
        if runner.kv_cache_dtype is None:
            raise RuntimeError("KV cache dtype not initialized; runner.load_model()")
        mps_kv_cache = MPSPagedKVCache(
            num_layers=runner.num_layers,
            num_kv_heads=runner.num_kv_heads,
            head_dim=runner.head_dim,
            num_blocks=num_blocks,
            block_size=block_size,
            dtype=runner.kv_cache_dtype,
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
            runner.num_kv_heads,
            runner.head_dim,
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

    def _one_sequence_kv_bytes(self) -> int:
        """Bytes for one max-length sequence of KV cache (K + V)."""
        runner = self.model_runner
        dtype_size = (
            runner.kv_cache_dtype.itemsize if runner.kv_cache_dtype is not None else 2
        )
        return (
            2  # K and V
            * runner.num_layers
            * self.model_config.max_model_len
            * runner.num_kv_heads
            * runner.head_dim
            * dtype_size
        )

    def determine_available_memory(self) -> int:
        """Determine available memory for KV cache.

        Paged attention: reports the actual MPS paged cache capacity.
        MLX path (default): reports one max-length sequence of KV cache
        so the scheduler budgets for one concurrent sequence.

        Returns:
            Available memory in bytes
        """
        # --- Paged attention: report real MPS cache capacity ---
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

        # --- MLX path: one max-length sequence for admission control ---
        available = self._one_sequence_kv_bytes()
        logger.info(
            "MLX path: reporting %.2fGB for scheduler admission control "
            "(one max-length sequence, max_model_len=%d)",
            available / 1e9,
            self.model_config.max_model_len,
        )
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

    def sample_tokens(
        self, grammar_output: GrammarOutput | None
    ) -> ModelRunnerOutput | None:
        """Return sampled tokens for the previously executed batch."""
        return self.model_runner.sample_tokens(grammar_output)

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
