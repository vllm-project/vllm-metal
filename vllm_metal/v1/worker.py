# SPDX-License-Identifier: Apache-2.0
"""Metal Worker for vLLM v1 engine."""

from __future__ import annotations

import gc
import time
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
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
from vllm.v1.worker.worker_base import CompilationTimes, WorkerBase

from vllm_metal.config import get_config
from vllm_metal.distributed import PipelineGroup
from vllm_metal.platform import MetalPlatform
from vllm_metal.utils import set_wired_limit
from vllm_metal.v1.cache_policy import WorkerCachePlanner

if TYPE_CHECKING:
    from vllm_metal.profiler.wrapper import MetalProfilerWrapper
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

    # Override model_runner type from base class. Typed as MetalModelRunner
    # because its worker-facing method set is a superset of STTModelRunner's;
    # STT models are assigned an STTModelRunner at runtime (see init_device).
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

        # Apply TurboQuant config from --additional-config (needed because worker
        # runs in a separate process and doesn't inherit the config singleton
        # state set in MetalPlatform.check_and_update_config).
        add = vllm_config.additional_config
        if isinstance(add, dict) and add.get("turboquant"):
            self.metal_config.turboquant = True
            self.metal_config.k_quant = add.get("k_quant", "q8_0")
            self.metal_config.v_quant = add.get("v_quant", "q3_0")
            self.metal_config._validate_turboquant()

        # Disable custom all reduce (not supported on Metal)
        self.parallel_config.disable_custom_all_reduce = True

        # Metal frame-capture profiler — created lazily on the first
        # profile(is_start=True) call so we pay zero cost when unused.
        self._metal_profiler: MetalProfilerWrapper | None = None

        # Pipeline-parallel group over the mx.distributed data plane. Created in
        # init_device only when pipeline_parallel_size > 1; stays None on the
        # default single-stage path (no group object, no behavior change).
        self.pp: PipelineGroup | None = None

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

        # Initialize distributed environment (vLLM gloo control plane).
        init_worker_distributed_environment(
            self.vllm_config,
            self.rank,
            self.distributed_init_method,
            self.local_rank,
        )

        # Pipeline parallelism: bring up the mx.distributed ring (the activation
        # data plane, separate from gloo) and wrap it in a PipelineGroup. With
        # TP forced to 1, the global rank IS the pipeline stage index, so we use
        # self.rank directly. Gated on pipeline_parallel_size > 1 so the default
        # single-stage path creates no group and is byte-for-byte unchanged.
        pp_size = self.parallel_config.pipeline_parallel_size
        if pp_size > 1:
            # PP+STT is rejected at config time (platform.check_and_update_config,
            # with the other PP guards), so only generation models reach here.
            # Discover every stage's node IP over the gloo control plane (already
            # initialized above) so the MLX ring spans real hosts — the same path
            # works single- or multi-node. peer_ips[r] is the IP of global rank r;
            # with TP forced to 1 the global rank is the pipeline-stage index.
            import torch.distributed as dist
            from vllm.utils.network_utils import get_ip

            peer_ips: list[str] = [""] * self.parallel_config.world_size
            dist.all_gather_object(peer_ips, get_ip())
            self.pp = PipelineGroup.bootstrap_ring(self.rank, peer_ips)
            logger.info(
                "Pipeline stage %d/%d (is_first=%s, is_last=%s)",
                self.pp.rank,
                self.pp.size,
                self.pp.is_first,
                self.pp.is_last,
            )

        # Set random seed
        set_random_seed(self.model_config.seed)

        # STT checkpoints run a one-shot transcribe that shares none of the
        # generation runner's machinery, so they use a dedicated runner. Detect
        # here using the same predicate as the generation load path and pick the
        # runner before construction. Imports are local to avoid circular imports.
        from vllm_metal.stt.detection import is_stt_model
        from vllm_metal.utils import get_model_download_path

        if is_stt_model(get_model_download_path(self.model_config.model)):
            from vllm_metal.v1.stt_model_runner import STTModelRunner

            self.model_runner = STTModelRunner(
                vllm_config=self.vllm_config,
                device=self.device,
            )
        else:
            from vllm_metal.v1.model_runner import MetalModelRunner

            self.model_runner = MetalModelRunner(
                vllm_config=self.vllm_config,
                device=self.device,
            )
            # Hand the pipeline group to the runner so its forward path can pipe
            # activations stage-to-stage. None on the default single-stage path.
            self.model_runner.pp = self.pp

    def load_model(self) -> None:
        """Load the model onto the Metal device."""
        self.model_runner.load_model()
        # Phase 0 = load-then-slice: every stage loaded full weights above; now
        # drop the layers (and, off the last stage, the final norm) this stage
        # does not own. No-op when self.pp is None (single-stage path).
        if self.pp is not None:
            self.model_runner.apply_pipeline_split(self.pp)

    def _one_sequence_kv_bytes(self) -> int:
        """Bytes for one max-length sequence of cache state.

        Uses block-aligned token count so the estimate matches the upstream
        ``_check_enough_kv_cache_memory`` calculation, which rounds
        ``max_model_len`` up to the nearest ``block_size`` boundary via
        ``cdiv(max_model_len, block_size) * page_size_bytes``.
        """
        block_size = self.vllm_config.cache_config.block_size
        return self.model_runner.estimate_one_sequence_kv_bytes(
            max_model_len=self.model_config.max_model_len,
            block_size=block_size,
        )

    def determine_available_memory(self) -> int:
        """Determine available memory for KV cache.

        Paged attention: reports the actual MPS paged cache capacity.
        MLX path: reports one max-length sequence of KV cache
        so the scheduler budgets for one concurrent sequence.

        Returns:
            Available memory in bytes
        """
        return WorkerCachePlanner(self).determine_available_memory()

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

    def compile_or_warm_up_model(self) -> CompilationTimes:
        """Warm up the model for inference."""
        # Reset seed for reproducibility
        set_random_seed(self.model_config.seed)
        start = time.perf_counter()
        self.model_runner.warm_up()
        return CompilationTimes(language_model=time.perf_counter() - start, encoder=0.0)

    def reset_mm_cache(self) -> None:
        """Reset profiling-time multimodal cache state."""
        self.model_runner.reset_mm_cache()

    def reset_encoder_cache(self) -> None:
        """Clear cached multimodal encoder outputs."""
        self.model_runner.reset_encoder_cache()

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

    def take_draft_token_ids(self) -> DraftTokenIds | None:
        """Return draft tokens generated by the model runner."""
        return self.model_runner.take_draft_token_ids()

    def get_model(self) -> Any:
        """Get the underlying model.

        Returns:
            The loaded model
        """
        return self.model_runner.model

    def update_max_model_len(self, max_model_len: int) -> None:
        """Update max_model_len after engine auto-fits context to GPU memory."""
        self.model_config.max_model_len = max_model_len

    def get_cache_block_size_bytes(self) -> int:
        """Get the size of a single cache block in bytes.

        Returns:
            Block size in bytes
        """
        return self.model_runner.get_cache_block_size_bytes()

    def add_lora(self, lora_request: LoRARequest) -> bool:
        """Add a LoRA/QLoRA adapter."""
        return self.model_runner.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        """Remove a LoRA adapter."""
        return self.model_runner.remove_lora(lora_id)

    def pin_lora(self, lora_id: int) -> bool:
        """Pin a LoRA adapter."""
        return self.model_runner.pin_lora(lora_id)

    def list_loras(self) -> set[int]:
        """List loaded LoRA adapter IDs."""
        return self.model_runner.list_loras()

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

    def profile(self, is_start: bool = True, profile_prefix: str | None = None) -> None:
        """Start or stop a Metal frame capture.

        Routed via the engine's ``collective_rpc("profile", ...)``, which is
        triggered by ``LLM.start_profile`` / ``LLM.stop_profile`` and by the
        ``POST /start_profile`` / ``POST /stop_profile`` HTTP endpoints. We
        get all of those for free — only this method needs to exist.
        """
        profiler_config = self.vllm_config.profiler_config
        if profiler_config is None:
            raise RuntimeError(
                "Profiling is not enabled. Pass --profiler-config to enable; "
                "e.g. --profiler-config.profiler=torch "
                "--profiler-config.torch_profiler_dir=/tmp/metal-trace"
            )

        if is_start:
            if self._metal_profiler is None:
                from vllm.distributed.utils import get_worker_rank_suffix

                from vllm_metal.profiler.wrapper import MetalProfilerWrapper

                rank_suffix = get_worker_rank_suffix(global_rank=self.rank)
                trace_name = (
                    f"{profile_prefix}_{rank_suffix}" if profile_prefix else rank_suffix
                )
                self._metal_profiler = MetalProfilerWrapper(profiler_config, trace_name)
            self._metal_profiler.start()
        else:
            if self._metal_profiler is None:
                logger.warning("Profiler was not started; nothing to stop.")
                return
            self._metal_profiler.stop()

    def shutdown(self) -> None:
        """Shutdown the worker and cleanup resources."""
        if self._metal_profiler is not None:
            self._metal_profiler.shutdown()
            self._metal_profiler = None

        if hasattr(self, "model_runner") and self.model_runner is not None:
            del self.model_runner
            self.model_runner = None

        gc.collect()
        logger.info("Metal worker shutdown complete")
