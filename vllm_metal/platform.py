# SPDX-License-Identifier: Apache-2.0
"""Metal Platform implementation for vLLM."""

import logging
import platform as py_platform
from typing import TYPE_CHECKING

import psutil
import torch
from vllm.platforms.interface import DeviceCapability, Platform, PlatformEnum
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from vllm_metal.config import get_config

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.attention.selector import AttentionSelectorConfig

logger = logging.getLogger(__name__)


class MetalPlatform(Platform):
    """Platform implementation for Apple Silicon Metal/MLX.

    This class provides vLLM with information about the Metal platform
    capabilities and handles device management.
    """

    _enum: PlatformEnum = PlatformEnum.OOT  # Out-of-tree platform
    device_name: str = "cpu"  # PyTorch device name (use CPU for compatibility)
    device_type: str = "cpu"  # PyTorch device type (use CPU for compatibility)
    dispatch_key: str = "CPU"  # PyTorch dispatch key

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        """Get the name of the Metal device.

        Args:
            device_id: Device index (ignored for Metal, single GPU)

        Returns:
            Device name string
        """
        try:
            import mlx.core as mx

            device = mx.default_device()
            return f"Apple Silicon ({device})"
        except ImportError:
            return "Apple Silicon (MLX not available)"

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        """Get total memory available for the device.

        On Apple Silicon, this returns the fraction of unified memory
        configured for use by the plugin.

        Args:
            device_id: Device index (ignored for Metal)

        Returns:
            Total memory in bytes
        """
        config = get_config()
        total_memory = psutil.virtual_memory().total
        # In auto mode, report full memory - actual allocation is dynamic
        if config.is_auto_memory:
            return total_memory
        return int(total_memory * config.memory_fraction)

    @classmethod
    def get_device_available_memory(cls, device_id: int = 0) -> int:
        """Get available memory for the device.

        Args:
            device_id: Device index (ignored for Metal)

        Returns:
            Available memory in bytes
        """
        config = get_config()
        available = psutil.virtual_memory().available
        # In auto mode, report full available memory - actual allocation is dynamic
        if config.is_auto_memory:
            return available
        return int(available * config.memory_fraction)

    @classmethod
    def is_available(cls) -> bool:
        """Check if Metal platform is available.

        Returns:
            True if running on Apple Silicon with MLX support
        """
        # Check architecture
        if py_platform.machine() != "arm64":
            return False

        # Check OS
        if py_platform.system() != "Darwin":
            return False

        # Check MLX availability without mutating global device state
        try:
            import mlx.core as mx

            return bool(mx.metal.is_available())
        except (ImportError, AttributeError, RuntimeError):
            return False

    @classmethod
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability:
        """Get device compute capability.

        Returns a fake capability for compatibility with CUDA-centric code.

        Args:
            device_id: Device index (ignored)

        Returns:
            DeviceCapability with (major, minor) version
        """
        # Return a reasonable value for compatibility
        return DeviceCapability(major=8, minor=0)

    @classmethod
    def get_device_count(cls) -> int:
        """Get number of available devices.

        Apple Silicon has unified memory, so we expose a single device.

        Returns:
            Always 1 for Metal
        """
        return 1

    @classmethod
    def set_device(cls, device_id: int) -> None:
        """Set the current device.

        Args:
            device_id: Device index (must be 0 for Metal)
        """
        if device_id != 0:
            msg = f"Metal only supports device 0, got {device_id}"
            raise ValueError(msg)

        config = get_config()
        if config.use_mlx:
            import mlx.core as mx

            device_type = (
                mx.DeviceType.gpu if config.mlx_device == "gpu" else mx.DeviceType.cpu
            )
            mx.set_default_device(mx.Device(device_type))

    @classmethod
    def current_device(cls) -> int:
        """Get the current device index.

        Returns:
            Always 0 for Metal
        """
        return 0

    @classmethod
    def synchronize(cls, device_id: int = 0) -> None:
        """Synchronize the device.

        Args:
            device_id: Device index (ignored)
        """
        import mlx.core as mx

        # Prefer an explicit MLX barrier when available; otherwise force evaluation.
        # `mx.eval([])` is a no-op, so we evaluate a tiny scalar as a safe fallback.
        try:
            mx.synchronize()
        except (AttributeError, TypeError):
            mx.eval(mx.array(0, dtype=mx.int32))

        if torch.backends.mps.is_available():
            torch.mps.synchronize()

    @classmethod
    def get_torch_device(cls, device_id: int = 0) -> torch.device:
        """Get the corresponding PyTorch device.

        Args:
            device_id: Device index (ignored)

        Returns:
            PyTorch device (MPS or CPU)
        """
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @classmethod
    def check_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        """Check and update vLLM configuration for Metal compatibility.

        Args:
            vllm_config: vLLM configuration object
        """
        config = get_config()
        parallel_config = vllm_config.parallel_config
        cache_config = vllm_config.cache_config
        model_config = vllm_config.model_config

        # Apply TurboQuant config from --additional-config
        # Example: --additional-config '{"turboquant": true, "k_quant": "q4_0"}'
        add = getattr(vllm_config, "additional_config", None) or {}
        if add.get("turboquant"):
            config.turboquant = True
            config.k_quant = add.get("k_quant", "q8_0")
            config.v_quant = add.get("v_quant", "q3_0")
            config._validate_turboquant()
            logger.info(
                f"TurboQuant enabled via --additional-config: "
                f"k_quant={config.k_quant}, v_quant={config.v_quant}"
            )

        if config.debug:
            logger.info(f"Metal config: {config}")

        # Set worker class for Metal
        if parallel_config.worker_cls == "auto":
            parallel_config.worker_cls = "vllm_metal.v1.worker.MetalWorker"

        # Set executor backend (use uniproc for single device)
        if parallel_config.distributed_executor_backend in ("auto", None):
            parallel_config.distributed_executor_backend = "uni"

        # Disable features not supported on Metal
        parallel_config.disable_custom_all_reduce = True

        scheduler_config = vllm_config.scheduler_config
        if getattr(scheduler_config, "enable_chunked_prefill", False):
            if config.use_paged_attention:
                # The paged path uses a unified varlen Metal kernel that
                # handles mixed prefill + decode in a single forward pass,
                # so chunked prefill works correctly.
                logger.info(
                    "Metal: chunked prefill enabled (paged attention), "
                    "max_num_batched_tokens=%d",
                    scheduler_config.max_num_batched_tokens,
                )
            else:
                # The non-paged MLX path does not honor chunked-prefill
                # scheduler boundaries.  Disable so the scheduler only
                # requests full prefills.
                scheduler_config.enable_chunked_prefill = False

                # Without chunked prefill, the scheduler must fit the
                # entire prompt in a single step.  Ensure
                # max_num_batched_tokens (and max_num_scheduled_tokens)
                # are at least max_model_len; otherwise the scheduler
                # silently refuses to schedule any prompt that exceeds
                # the budget.
                if model_config is not None:
                    model_max = model_config.max_model_len
                    if scheduler_config.max_num_batched_tokens < model_max:
                        scheduler_config.max_num_batched_tokens = model_max
                    if (
                        scheduler_config.max_num_scheduled_tokens is not None
                        and scheduler_config.max_num_scheduled_tokens < model_max
                    ):
                        scheduler_config.max_num_scheduled_tokens = model_max

                logger.info(
                    "Metal: disabled chunked prefill (non-paged path), "
                    "max_num_batched_tokens=%d",
                    scheduler_config.max_num_batched_tokens,
                )

        # Configure cache — ensure block_size is at least the Metal kernel
        # minimum.  With chunked prefill enabled, upstream may default to
        # block_size=1 for fine-grained scheduling, but our Metal paged
        # attention kernel requires multiples of 8.
        if (
            cache_config.block_size is None
            or cache_config.block_size < config.block_size
        ):
            cache_config.block_size = config.block_size

        # Disable cascade attention (not supported), then let the adapter
        # apply any model-specific normalisations (e.g. clearing
        # ``multimodal_config`` for model types served on the text-only
        # backbone — see ``DefaultModelAdapter.normalize_model_config``).
        if model_config is not None:
            model_config.disable_cascade_attn = True
            from vllm_metal.v1.model_adapter import DefaultModelAdapter

            DefaultModelAdapter().normalize_model_config(model_config)

        # STT model detection — set tokenizer fallback if not already configured.
        # Lazy imports to avoid circular import: platform.py is loaded during
        # vllm.config init, and stt.detection imports from vllm.config.
        from vllm_metal.stt.detection import is_stt_model
        from vllm_metal.stt.policy import apply_stt_scheduler_policy
        from vllm_metal.utils import get_model_download_path

        resolved_model = (
            get_model_download_path(model_config.model)
            if model_config is not None
            else None
        )
        if resolved_model is not None and is_stt_model(resolved_model):
            was_async_scheduling = bool(scheduler_config.async_scheduling)
            apply_stt_scheduler_policy(model_config, scheduler_config)
            if was_async_scheduling and not scheduler_config.async_scheduling:
                logger.info("STT: disabled async_scheduling")
            logger.info("STT model detected")

        # Log memory configuration
        total_mem = cls.get_device_total_memory()
        available_mem = cls.get_device_available_memory()
        logger.info(
            f"Metal memory: {total_mem / 1e9:.1f}GB total, "
            f"{available_mem / 1e9:.1f}GB available"
        )

    @classmethod
    def support_hybrid_kv_cache(cls) -> bool:
        """Metal supports hybrid KV cache for models like Qwen3.5 (SDPA + GDN)."""
        return True

    @classmethod
    def update_block_size_for_backend(
        cls,
        vllm_config,
    ) -> None:
        """Update block_size to unify page sizes for hybrid models.

        Hybrid models (e.g., Qwen3.5) have two types of layers:
        - SDPA layers: page_size scales with block_size
        - Mamba/linear layers: page_size is fixed

        vLLM requires all layer page sizes to be divisible. This method adjusts
        block_size and sets mamba_page_size_padded to satisfy vLLM's validation.

        Note:
            This is a "logical" fix for vLLM's scheduler validation only.
            The Metal plugin manages KV cache internally via MLX's make_prompt_cache(),
            independent of vLLM's block_size and page_size calculations.
            These parameters are used only to pass vLLM's initialization checks.

        Steps:
        1. Compute attention page size per token (MLAAttentionSpec or FullAttentionSpec)
        2. Get Mamba page size from model class
        3. Calculate block_size so SDPA page_size >= Mamba page_size
        4. Sync mamba_block_size if using align mode
        5. Pad mamba_page_size to match SDPA page_size exactly

        Args:
            vllm_config: vLLM configuration (modified in-place for vLLM validation)

        Raises:
            ValueError: If hybrid model is used with paged attention on Metal,
                        or if computed mamba_page_size is zero
            Exception: Model class resolution or mamba state query failures
        """
        from vllm.model_executor.models import ModelRegistry
        from vllm.utils.math_utils import cdiv
        from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE
        from vllm.v1.kv_cache_interface import (
            FullAttentionSpec,
            MambaSpec,
            MLAAttentionSpec,
        )

        cache_config = vllm_config.cache_config
        model_config = vllm_config.model_config

        if not model_config:
            return

        # Skip non-hybrid models
        is_hybrid = getattr(model_config, "is_hybrid", False)
        if not is_hybrid:
            return

        # For hybrid models with paged attention, log a warning explaining the
        # block-size translation mechanism.
        #
        # Background:
        # - vLLM requires block_size=160 (or larger) for hybrid models to satisfy
        #   page size divisibility validation between SDPA and Mamba layers.
        # - Metal paged attention kernels only support block_size in {8, 16, 32}.
        #
        # Solution (PR #235):
        # - vLLM sees a large block_size (e.g., 160) for its scheduler validation.
        # - The Metal kernel uses a translated block_size (e.g., 32) that it supports.
        # - Each vLLM block is split into ratio = cache_block_size / kernel_block_size
        #   kernel blocks. For example, one vLLM block of 160 tokens becomes 5 kernel
        #   blocks of 32 tokens each.
        # - The KV cache is reshaped (zero-copy) to match: [num_blocks, 160, ...] →
        #   [num_blocks*5, 32, ...]. The physical memory layout is unchanged.
        # - Block tables are expanded so the kernel reads the correct blocks.
        #
        # This is a logical transformation only — the computation is identical, just
        # the kernel sees more, smaller blocks.
        from vllm_metal.config import get_config

        metal_config = get_config()
        if metal_config.use_paged_attention:
            logger.warning(
                "Hybrid model (e.g., Qwen3.5) with paged attention enabled. "
                "Using block-size translation (PR #235) to convert vLLM's large "
                "block_size to a Metal kernel-compatible size.\n"
                "  Mechanism: Each vLLM block is split into multiple kernel blocks.\n"
                "  Example: vLLM block_size=160 → kernel block_size=32 (ratio=5).\n"
                "  The KV cache is reshaped (zero-copy) and block tables are expanded.\n"
                "  This is a logical transformation — physical memory is unchanged.\n"
                "  Note: The default MLX path (without paged attention) is recommended "
                "for hybrid models as it has no translation overhead."
            )

        # Step 1: Compute attention page size per token
        # Handle cache_dtype conversion
        if cache_config.cache_dtype == "auto":
            kv_cache_dtype = model_config.dtype
        else:
            kv_cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]

        # Use MLAAttentionSpec for MLA models, FullAttentionSpec otherwise
        spec_class = (
            MLAAttentionSpec
            if getattr(model_config, "use_mla", False)
            else FullAttentionSpec
        )
        attn_page_size_1_token = spec_class(
            block_size=1,
            num_kv_heads=model_config.get_num_kv_heads(vllm_config.parallel_config),
            head_size=model_config.get_head_size(),
            dtype=kv_cache_dtype,
        ).page_size_bytes

        # Step 2: Get Mamba page size (fixed, independent of block_size)
        try:
            model_cls, _ = ModelRegistry.resolve_model_cls(
                model_config.architecture,
                model_config=model_config,
            )
            mamba_state_shape = model_cls.get_mamba_state_shape_from_config(vllm_config)
            mamba_state_dtype = model_cls.get_mamba_state_dtype_from_config(vllm_config)

            mamba_page_size = MambaSpec(
                shapes=mamba_state_shape,
                dtypes=mamba_state_dtype,
                block_size=-1,
            ).page_size_bytes
        except Exception as e:
            # For hybrid models, re-raise exception instead of silently returning
            logger.error(
                "Failed to get mamba state for hybrid model %s: %s",
                model_config.architecture,
                e,
            )
            raise

        if mamba_page_size == 0:
            raise ValueError(
                f"Computed mamba_page_size is zero for hybrid model "
                f"{model_config.architecture}"
            )

        # Step 3: Calculate block_size so SDPA page_size >= Mamba page_size
        # Use the same formula as vLLM's CPU platform for consistency
        #
        # Note: kernel_block_alignment_size=32 is chosen for Metal GPU performance.
        # Common Metal threadgroup sizes are multiples of 32 (e.g., 32, 64, 128, 256).
        # However, this value has no actual impact on MLX execution because:
        # - MLX manages its own KV cache via make_prompt_cache()
        # - This block_size is only used to satisfy vLLM's validation logic
        # - The actual Metal kernel uses MLX's native memory layout
        #
        # Using 32 provides a reasonable balance between:
        # - GPU performance (aligned to Metal threadgroup preferences)
        # - Memory efficiency (not excessively large)
        # - Compatibility with vLLM's page size unification requirements
        kernel_block_alignment_size = 32  # Metal GPU kernel alignment
        attn_block_size = kernel_block_alignment_size * cdiv(
            mamba_page_size,
            kernel_block_alignment_size * attn_page_size_1_token,
        )

        if cache_config.block_size < attn_block_size:
            cache_config.block_size = attn_block_size
            logger.info(
                "Setting attention block size to %d tokens "
                "to ensure that attention page size is >= mamba page size.",
                attn_block_size,
            )

        # Step 4: Sync mamba_block_size if using align mode
        if cache_config.mamba_cache_mode == "align":
            cache_config.mamba_block_size = cache_config.block_size

        # Step 5: Pad Mamba page size to exactly match SDPA page size
        attn_page_size = cache_config.block_size * attn_page_size_1_token
        if attn_page_size > mamba_page_size:
            cache_config.mamba_page_size_padded = attn_page_size
            mamba_padding_pct = (
                100 * (attn_page_size - mamba_page_size) / mamba_page_size
            )
            logger.info(
                "Padding mamba page size by %.2f%% to ensure "
                "that mamba page size and attention page size are "
                "exactly equal.",
                mamba_padding_pct,
            )

    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend: "AttentionBackendEnum",
        attn_selector_config: "AttentionSelectorConfig",
    ) -> str:
        """Get the attention backend class for Metal."""
        if selected_backend and selected_backend != AttentionBackendEnum.CPU_ATTN:
            logger.info(f"Cannot use {selected_backend} backend on Metal/MLX.")
        if attn_selector_config.use_mla:
            # MLA attention is handled by the vllm-metal model runner (MLAPagedAttentionWrapper),
            # not by vLLM's attention backend selector. Continue to return CPU_ATTN below.
            logger.info(
                "MLA model detected; attention handled by vllm-metal model runner"
            )
        if attn_selector_config.use_sparse:
            raise NotImplementedError("Sparse Attention is not supported on Metal/MLX.")
        return AttentionBackendEnum.CPU_ATTN.get_path()

    @classmethod
    def verify_quantization(cls, quant: str) -> None:
        """Verify that quantization method is supported.

        Args:
            quant: Quantization method name

        Raises:
            ValueError: If quantization is not supported
        """
        # Allow all quantization methods to pass through - actual support
        # depends on the model implementation. This avoids blocking models
        # that use quantization formats we may be able to handle.
        pass

    @classmethod
    def is_pin_memory_available(cls) -> bool:
        """Check if pin_memory is available for Metal platform.

        Returns:
            False - pin_memory is not needed/supported on Metal/MLX

        Note:
            Although MLX uses unified memory (which theoretically could benefit
            from pin_memory), we disable it because:
            1. PyTorch's pin_memory is primarily designed for CUDA
            2. In our architecture, PyTorch tensors are on CPU for MLX interop
            3. pin_memory on CPU can cause issues or errors
            4. Unified memory already provides fast CPU-GPU transfers without pinning
        """
        return False
