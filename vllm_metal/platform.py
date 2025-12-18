# SPDX-License-Identifier: Apache-2.0
"""Metal Platform implementation for vLLM."""

from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch

from vllm_metal._compat import (
    CompilationMode,
    CUDAGraphMode,
    Platform,
    PlatformEnum,
    init_logger,
)
from vllm_metal.envs import VLLM_METAL_MEMORY_FRACTION
from vllm_metal.utils import (
    check_metal_availability,
    get_apple_chip_name,
    get_metal_device_info,
    get_metal_memory_info,
    metal_empty_cache,
    metal_synchronize,
)

if TYPE_CHECKING:
    from vllm_metal._compat import VllmConfig

logger = init_logger(__name__)


class MetalPlatform(Platform):
    """Platform implementation for Apple Metal backend using MLX.

    This platform provides high-performance LLM inference on Apple Silicon
    by using MLX as the primary compute backend. It integrates with vLLM's
    plugin system to provide a seamless experience.

    Key features:
    - MLX for GPU operations (attention, normalization, activations)
    - PyTorch for model loading and tensor interface compatibility
    - Unified memory architecture (no CPU/GPU copies needed)
    - V2 model runner with Triton kernel replacements
    """

    # Out-of-tree platform enum (OOT is only available in vLLM 0.12+)
    _enum = PlatformEnum.OOT if PlatformEnum is not None else None
    device_name: str = "mps"
    device_type: str = "mps"
    dispatch_key: str = "MPS"

    # Supported quantization methods
    supported_quantization = ["awq", "gptq", "compressed-tensors"]

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        """Get the name of the Metal device."""
        return get_apple_chip_name()

    @classmethod
    def get_device_uuid(cls, device_id: int = 0) -> str:
        """Get a unique identifier for the Metal device."""
        chip_name = get_apple_chip_name().replace(" ", "_")
        return f"metal:{chip_name}:{device_id}"

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        """Get total memory available to the device in bytes.

        On Apple Silicon, GPU uses unified memory shared with CPU.
        """
        info = get_metal_device_info()
        total_mem = info.get("total_memory", 0)
        return int(total_mem * VLLM_METAL_MEMORY_FRACTION)

    @classmethod
    def get_device_capability(cls, device_id: int = 0):
        """Get device capability.

        Returns None as Metal doesn't have compute capability like CUDA.
        """
        return None

    @classmethod
    def is_async_output_supported(cls, enforce_eager: bool) -> bool:
        """Check if async output is supported."""
        return not enforce_eager

    @classmethod
    def inference_mode(cls):
        """Return the appropriate inference mode context manager."""
        return torch.inference_mode()

    @classmethod
    def seed_everything(cls, seed: int) -> None:
        """Seed all random number generators."""
        import mlx.core as mx

        torch.manual_seed(seed)
        mx.random.seed(seed)

        try:
            torch.mps.manual_seed(seed)
        except Exception:
            pass

    @classmethod
    def set_device(cls, device) -> None:
        """Set the current device.

        Metal only has one device, so this is mostly a no-op.
        MLX handles device placement automatically.
        """
        pass

    @classmethod
    def get_current_memory_usage(cls, device=None) -> int:
        """Get current memory usage in bytes."""
        allocated, _ = get_metal_memory_info()
        return allocated

    @classmethod
    def empty_cache(cls) -> None:
        """Empty the Metal memory cache."""
        metal_empty_cache()

    @classmethod
    def synchronize(cls) -> None:
        """Synchronize Metal operations."""
        metal_synchronize()

    @classmethod
    def mem_get_info(cls) -> tuple[int, int]:
        """Get memory info (free, total).

        Note: Metal uses unified memory, so 'free' is estimated.
        """
        allocated, total = get_metal_memory_info()
        free = total - allocated
        return free, total

    @classmethod
    def check_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        """Check and update vLLM configuration for Metal backend.

        This method validates the configuration and sets Metal-specific
        defaults for optimal performance.
        """
        # Validate platform availability
        available, error = check_metal_availability()
        if not available:
            raise RuntimeError(f"Metal backend not available: {error}")

        # Get config objects (may be None in early calls)
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        parallel_config = vllm_config.parallel_config
        compilation_config = vllm_config.compilation_config

        # Set the worker class for Metal platform
        if parallel_config is not None:
            if parallel_config.worker_cls == "auto":
                parallel_config.worker_cls = "vllm_metal.v2.worker.MetalWorker"
                logger.info("Metal backend: Using MetalWorker V2")

            # Metal doesn't support tensor parallelism
            if parallel_config.tensor_parallel_size > 1:
                raise ValueError(
                    "Metal backend does not support tensor parallelism. "
                    "Please set tensor_parallel_size=1"
                )

            # Metal doesn't support pipeline parallelism
            if parallel_config.pipeline_parallel_size > 1:
                raise ValueError(
                    "Metal backend does not support pipeline parallelism. "
                    "Please set pipeline_parallel_size=1"
                )

        # Set default block size if not specified
        if cache_config is not None:
            if cache_config.block_size is None:
                cache_config.block_size = 16
                logger.info("Metal backend: Using block_size=16 for KV cache")

        # Metal always runs in eager mode (no CUDA graph support)
        if model_config is not None:
            model_config.enforce_eager = True

        # Disable CUDA graphs and torch.compile
        if compilation_config is not None and CompilationMode is not None:
            compilation_config.cudagraph_mode = CUDAGraphMode.NONE
            compilation_config.cudagraph_capture_sizes = []
            compilation_config.compile_sizes = []
            compilation_config.level = 0
            compilation_config.mode = CompilationMode.NONE
            logger.info(
                "Metal backend: Disabled CUDA graphs and compilation "
                "(not supported on Metal)"
            )

        # Log initialization info
        if parallel_config is not None:
            logger.info(
                f"Metal backend initialized: device={cls.get_device_name()}, "
                f"memory={cls.get_device_total_memory() / 1e9:.1f}GB"
            )

    @classmethod
    def verify_quantization(cls, quant: str) -> None:
        """Verify that the quantization method is supported."""
        if quant and quant not in cls.supported_quantization:
            raise ValueError(
                f"Quantization method '{quant}' not supported on Metal. "
                f"Supported: {cls.supported_quantization}"
            )

    @classmethod
    def verify_model_arch(cls, model_arch: str) -> None:
        """Verify that the model architecture is supported."""
        unsupported = {"mamba", "rwkv", "xlnet"}
        if model_arch.lower() in unsupported:
            logger.warning(
                f"Model architecture '{model_arch}' may not be fully "
                "supported on Metal backend"
            )

    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend,
        head_size: int,
        dtype,
        kv_cache_dtype,
        block_size: int,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        attn_type: str | None = None,
    ) -> str:
        """Get the attention backend class path for Metal.

        Returns our custom Metal attention backend that uses MLX's
        scaled_dot_product_attention.
        """
        if use_mla:
            raise NotImplementedError("MLA is not supported on Metal.")
        if use_sparse:
            raise NotImplementedError("Sparse Attention is not supported on Metal.")

        return "vllm_metal.attention.backend.MetalAttentionBackend"

    @classmethod
    def is_pin_memory_available(cls) -> bool:
        """Check if pin memory is available.

        Metal uses unified memory, so pinned memory isn't applicable.
        """
        return False

    @classmethod
    def check_if_supports_dtype(cls, dtype: torch.dtype) -> bool:
        """Check if the dtype is supported on Metal."""
        supported = {
            torch.float32,
            torch.float16,
            torch.bfloat16,
            torch.int32,
            torch.int64,
            torch.int16,
            torch.int8,
            torch.uint8,
            torch.bool,
        }
        return dtype in supported

    @classmethod
    def supports_fp8(cls) -> bool:
        """Check if FP8 is supported."""
        return False

    @classmethod
    def supports_mx(cls) -> bool:
        """Check if MX formats are supported."""
        return False

    @classmethod
    def get_punica_wrapper(cls):
        """Get the Punica wrapper for LoRA.

        Returns None as Punica is CUDA-specific.
        """
        return None

    @classmethod
    def can_update_inplace(cls) -> bool:
        """Check if in-place updates are supported."""
        return True

    @classmethod
    def support_hybrid_kv_cache(cls) -> bool:
        """Check if hybrid KV cache is supported."""
        return False

    @classmethod
    def support_static_graph_mode(cls) -> bool:
        """Check if static graph mode is supported."""
        return False

    @classmethod
    @contextmanager
    def device_scope(cls, device_id: int = 0):
        """Context manager for device scope."""
        yield

    @classmethod
    def get_device_communicator_cls(cls):
        """Get the device communicator class.

        Returns None as Metal doesn't support distributed.
        """
        return None

    @classmethod
    def stateless_init_device_torch_dist_pg(
        cls,
        backend: str,
        timeout,
    ):
        """Initialize torch distributed process group.

        Metal doesn't support distributed training.
        """
        raise NotImplementedError(
            "Metal backend does not support distributed operations"
        )

    @classmethod
    def import_kernels(cls) -> None:
        """Import Metal-specific kernels (MLX operations)."""
        from vllm_metal import ops

        ops.register_metal_ops()
        logger.debug("Metal/MLX kernels imported")
