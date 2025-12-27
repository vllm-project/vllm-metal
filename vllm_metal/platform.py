# SPDX-License-Identifier: Apache-2.0
"""Metal Platform implementation for vLLM."""

import logging
import platform as py_platform
from typing import TYPE_CHECKING, Any

import psutil
import torch
from vllm.platforms.interface import Platform, PlatformEnum

from vllm_metal.config import get_config

if TYPE_CHECKING:
    from vllm.config import VllmConfig

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

        # Check MLX availability
        try:
            import mlx.core as mx

            # Try to use the GPU
            config = get_config()
            if config.use_mlx:
                mx.set_default_device(mx.Device(mx.DeviceType.gpu))
            return True
        except (ImportError, RuntimeError):
            return False

    @classmethod
    def get_device_capability(cls, device_id: int = 0) -> tuple[int, int]:
        """Get device compute capability.

        Returns a fake capability for compatibility with CUDA-centric code.

        Args:
            device_id: Device index (ignored)

        Returns:
            Tuple of (major, minor) version
        """
        # Return a reasonable value for compatibility
        return (8, 0)

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

        mx.eval([])

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

        # Configure cache
        if cache_config.block_size is None:
            cache_config.block_size = config.block_size

        # Disable cascade attention (not supported)
        if model_config is not None:
            model_config.disable_cascade_attn = True

        # Log memory configuration
        total_mem = cls.get_device_total_memory()
        available_mem = cls.get_device_available_memory()
        logger.info(
            f"Metal memory: {total_mem / 1e9:.1f}GB total, "
            f"{available_mem / 1e9:.1f}GB available"
        )

    @classmethod
    def get_attn_backend_cls(cls) -> Any:
        """Get the attention backend class for Metal.

        Returns:
            Attention backend class
        """
        # Return None to use default attention
        return None

    @classmethod
    def verify_quantization(cls, quant: str) -> None:
        """Verify that quantization method is supported.

        Args:
            quant: Quantization method name

        Raises:
            ValueError: If quantization is not supported
        """
        supported = {"none", None, "fp16", "bfloat16"}
        if quant not in supported:
            msg = f"Metal does not support quantization: {quant}"
            raise ValueError(msg)

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
