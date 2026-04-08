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
from vllm_metal.paged_attention_backend.mha import MHAPagedAttentionBackend
from vllm_metal.paged_attention_backend.mla import MLAPagedAttentionBackend
from vllm_metal.platform import MetalPlatform
from vllm_metal.stt.policy import STT_SCHED_AVAILABLE_BYTES
from vllm_metal.utils import set_wired_limit

if TYPE_CHECKING:
    from vllm_metal.v1.model_runner import (
        MetalModelRunner,
        SchedulerMemoryReportingMode,
    )

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

        # Boundary ownership:
        # - Worker owns resource setup.
        # - Runner owns STT/runtime capability decisions.
        # Note: For hybrid models (Qwen3.5), we don't auto-enable paged attention
        # because MLX's make_prompt_cache() handles hybrid KV cache natively.
        # Paged attention for hybrid models requires splitting KV cache across
        # multiple buffers to avoid Metal's max buffer size limit (~9.5GB).
        if (
            self.metal_config.use_paged_attention
            and self.model_runner.should_setup_paged_attention()
        ):
            self._setup_paged_attention()

    @staticmethod
    def _kv_budget_bytes(
        metal_limit: int,
        model_memory: int,
        fraction: float,
        overhead: int = PAGED_ATTENTION_OVERHEAD_BYTES,
    ) -> int:
        """KV cache budget = fraction of Metal limit minus model and overhead.

        All three quantities live in the same domain: Metal-managed memory.
        psutil.available is intentionally excluded — it reflects OS page-cache
        state and is blind to MLX wired buffers holding model weights.
        """
        return int(metal_limit * fraction) - model_memory - overhead

    def _setup_paged_attention(self) -> None:
        """Allocate paged KV cache and patch model attention layers.

        Computes num_blocks from Metal memory headroom, model weight size, and
        a configurable memory fraction, rather than blindly scaling from
        max_model_len.
        """
        runner = self.model_runner
        # Use cache_config.block_size (not metal_config) because vLLM's
        # hybrid alignment may have adjusted it to match mamba page size.
        block_size = self.vllm_config.cache_config.block_size

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

        # --- Gather Metal memory numbers ---
        # KV cache lives in Metal-managed (wired) memory. psutil.available
        # reflects OS page-cache state and excludes MLX wired buffers, making
        # it appear nearly zero when a large model is loaded. Use
        # max_recommended_working_set_size — the OS-reported Metal headroom —
        # as the budget ceiling instead.
        device_info = mx.device_info()
        metal_limit = int(device_info.get("max_recommended_working_set_size", 0))
        if metal_limit <= 0:
            raise RuntimeError(
                "Paged attention: mx.device_info() did not return "
                "max_recommended_working_set_size. "
                "Ensure MLX is up to date and running on Apple Silicon. "
                f"Reported device_info keys: {list(device_info.keys())}"
            )
        model_memory = self._get_model_memory_usage()
        per_block_bytes = self.get_cache_block_size_bytes()

        # --- Compute KV budget ---
        usable_metal = int(metal_limit * fraction)
        kv_budget = self._kv_budget_bytes(metal_limit, model_memory, fraction)

        # For hybrid models, subtract the fixed linear state cost first.
        if runner.is_hybrid:
            kv_budget -= runner.linear_cache_bytes_per_slot() * (
                runner.scheduler_config.max_num_seqs
            )

        if kv_budget <= 0:
            raise ValueError(
                "Paged attention: not enough Metal memory for KV cache. "
                f"metal_limit={metal_limit / 1e9:.2f}GB, "
                f"fraction={fraction}, "
                f"usable_metal={usable_metal / 1e9:.2f}GB, "
                f"model_memory={model_memory / 1e9:.2f}GB, "
                f"overhead={PAGED_ATTENTION_OVERHEAD_BYTES / 1e9:.2f}GB, "
                f"kv_budget={kv_budget / 1e9:.2f}GB. "
                "Mitigations: increase VLLM_METAL_MEMORY_FRACTION, "
                "use a smaller or more quantized model."
            )

        num_blocks = kv_budget // per_block_bytes

        if num_blocks < PAGED_ATTENTION_MIN_BLOCKS:
            raise ValueError(
                "Paged attention: computed num_blocks too low "
                f"({num_blocks} < minimum {PAGED_ATTENTION_MIN_BLOCKS}). "
                f"metal_limit={metal_limit / 1e9:.2f}GB, "
                f"fraction={fraction}, "
                f"usable_metal={usable_metal / 1e9:.2f}GB, "
                f"model_memory={model_memory / 1e9:.2f}GB, "
                f"overhead={PAGED_ATTENTION_OVERHEAD_BYTES / 1e9:.2f}GB, "
                f"kv_budget={kv_budget / 1e9:.2f}GB, "
                f"per_block_bytes={per_block_bytes}. "
                "Mitigations: increase VLLM_METAL_MEMORY_FRACTION, "
                "use a smaller or more quantized model."
            )

        max_tokens_cached = num_blocks * block_size

        logger.info(
            "Paged attention memory breakdown: "
            "metal_limit=%.2fGB, fraction=%.2f, usable_metal=%.2fGB, "
            "model_memory=%.2fGB, overhead=%.2fGB, "
            "kv_budget=%.2fGB, per_block_bytes=%d, "
            "num_blocks=%d, max_tokens_cached=%d",
            metal_limit / 1e9,
            fraction,
            usable_metal / 1e9,
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

        backend = self._make_backend(runner, block_size)
        backend.initialize(num_blocks)
        n_patched = backend.patch_model(runner.model)
        logger.info(
            "Paged attention enabled: %d layers patched, "
            "%d blocks allocated (block_size=%d, mla=%s)",
            n_patched,
            num_blocks,
            block_size,
            runner.is_mla,
        )

        runner._paged_attention_backend = backend
        runner._paged_block_size = block_size

    @staticmethod
    def _make_backend(runner: MetalModelRunner, block_size: int) -> Any:
        """Create the right paged attention backend for the model type."""
        if runner.is_hybrid:
            from vllm_metal.paged_attention_backend.hybrid import (
                HybridPagedAttentionBackend,
            )

            return HybridPagedAttentionBackend(
                num_layers=runner.num_layers,
                full_attention_interval=runner.full_attention_interval,
                max_num_seqs=runner.scheduler_config.max_num_seqs,
                num_kv_heads=runner.num_kv_heads,
                head_dim=runner.head_dim,
                linear_num_v_heads=runner.linear_num_v_heads,
                linear_key_head_dim=runner.linear_key_head_dim,
                linear_value_head_dim=runner.linear_value_head_dim,
                linear_conv_kernel_dim=runner.linear_conv_kernel_dim,
                linear_conv_dim=runner.linear_conv_dim,
                block_size=block_size,
                dtype=runner.kv_cache_dtype,
            )
        if runner.is_mla:
            return MLAPagedAttentionBackend(
                num_layers=runner.num_layers,
                latent_dim=runner.mla_latent_dim,
                block_size=block_size,
                dtype=runner.kv_cache_dtype,
            )
        return MHAPagedAttentionBackend(
            num_layers=runner.num_layers,
            num_kv_heads=runner.num_kv_heads,
            head_dim=runner.head_dim,
            block_size=block_size,
            dtype=runner.kv_cache_dtype,
        )

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
        """Bytes for one max-length sequence of cache state.

        Uses block-aligned token count so the estimate matches the upstream
        ``_check_enough_kv_cache_memory`` calculation, which rounds
        ``max_model_len`` up to the nearest ``block_size`` boundary via
        ``cdiv(max_model_len, block_size) * page_size_bytes``.
        """
        runner = self.model_runner
        if runner.kv_cache_dtype is None:
            raise RuntimeError("KV cache dtype not initialized; runner.load_model()")
        dtype_size = runner.kv_cache_dtype.size

        num_kv_layers = (
            runner.num_sdpa_layers if runner.is_hybrid else runner.num_layers
        )

        # Round token count up to block boundary to match the scheduler's
        # block-aligned memory accounting.
        block_size = self.vllm_config.cache_config.block_size
        max_tokens = -(-self.model_config.max_model_len // block_size) * block_size

        kv_factor = 1 if runner.is_mla else 2
        sdpa_kv_bytes = (
            kv_factor
            * num_kv_layers
            * max_tokens
            * runner.num_kv_heads
            * runner.head_dim
            * dtype_size
        )
        if runner.is_hybrid:
            return sdpa_kv_bytes + runner.linear_cache_bytes_per_slot()
        return sdpa_kv_bytes

    def determine_available_memory(self) -> int:
        """Determine available memory for KV cache.

        Paged attention: reports the actual MPS paged cache capacity.
        MLX path (default): reports one max-length sequence of KV cache
        so the scheduler budgets for one concurrent sequence.

        Returns:
            Available memory in bytes
        """
        mode: SchedulerMemoryReportingMode = (
            self.model_runner.scheduler_memory_reporting_mode(
                paged_attention_enabled=self.metal_config.use_paged_attention
            )
        )

        if mode == "stt_nominal":
            # STT models don't use vLLM's KV cache. Return a nominal value so
            # scheduler minimum-memory checks pass.
            logger.info("STT model: reporting nominal memory for scheduler")
            return STT_SCHED_AVAILABLE_BYTES

        if mode == "paged_attention_capacity":
            # Runner only reports this mode when paged cache is initialized.
            backend = self.model_runner._paged_attention_backend
            assert backend is not None
            num_blocks = backend.num_blocks()
            block_size_bytes = self.get_cache_block_size_bytes()
            available = num_blocks * block_size_bytes
            logger.info(
                "Paged attention: reporting MPS cache capacity "
                "(%d blocks × %d bytes = %.2f GB)",
                num_blocks,
                block_size_bytes,
                available / 1e9,
            )
            return available

        # Default MLX path: report one max-length sequence for admission control.
        # This matches the design from PR #229, which ensures the scheduler
        # can admit at least one sequence without over-committing memory.
        # MLX's make_prompt_cache() dynamically allocates KV cache per request,
        # so we only need to report enough for one sequence.
        available = self._one_sequence_kv_bytes()
        logger.info(
            "MLX path: reporting %.2f GB for scheduler admission control "
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
        return self.model_runner.supported_worker_tasks()

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
