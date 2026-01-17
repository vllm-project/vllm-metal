# SPDX-License-Identifier: Apache-2.0
"""Metal Worker implementation for vLLM."""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vllm_metal.config import (
    AUTO_MEMORY_MIN_BLOCKS_BUFFER_FACTOR,
    AUTO_MEMORY_OVERHEAD_FACTOR,
    get_config,
)
from vllm_metal.platform import MetalPlatform

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = logging.getLogger(__name__)


@dataclass
class MetalWorkerConfig:
    """Configuration for Metal worker."""

    model_name: str
    max_model_len: int
    block_size: int
    dtype: str = "float16"
    trust_remote_code: bool = True


class MetalWorker:
    """Worker for Metal/MLX-based inference.

    The worker handles model loading, execution, and memory management
    on Apple Silicon hardware.
    """

    def __init__(
        self,
        vllm_config: "VllmConfig | None" = None,
        **kwargs: Any,
    ):
        """Initialize Metal worker.

        Args:
            vllm_config: Optional vLLM configuration
            **kwargs: Additional configuration options
        """
        self.config = get_config()
        self.vllm_config = vllm_config
        self.model_runner: Any = None
        self.is_initialized = False

        # Set up the device
        MetalPlatform.set_device(0)

        if self.config.debug:
            logger.info(f"MetalWorker initialized with config: {self.config}")

    def init_device(self) -> None:
        """Initialize the Metal device."""
        import mlx.core as mx

        if self.config.use_mlx:
            device_type = (
                mx.DeviceType.gpu
                if self.config.mlx_device == "gpu"
                else mx.DeviceType.cpu
            )
            mx.set_default_device(mx.Device(device_type))

        logger.info(f"Metal device initialized: {mx.default_device()}")

    def load_model(self) -> None:
        """Load the model for inference."""
        from vllm_metal.model_runner import MetalModelRunner

        if self.vllm_config is None:
            msg = "vllm_config is required to load model"
            raise ValueError(msg)

        self.model_runner = MetalModelRunner(self.vllm_config)
        self.model_runner.load_model()
        self.is_initialized = True

        logger.info("Model loaded successfully")

    def _get_model_memory_usage(self) -> int:
        """Get current model memory usage from MLX.

        Returns:
            Memory usage in bytes
        """
        import mlx.core as mx

        # Force evaluation of any pending computations
        mx.eval(mx.array([0]))

        # Get active memory usage - try new API first, then deprecated
        if hasattr(mx, "get_active_memory"):
            return mx.get_active_memory()
        if hasattr(mx, "metal") and hasattr(mx.metal, "get_active_memory"):
            return mx.metal.get_active_memory()

        # Fallback: estimate from model config
        if self.model_runner is not None and self.model_runner.model_config is not None:
            # Rough estimate: assume 2 bytes per parameter (float16)
            # This is a very rough estimate
            config = self.model_runner.model_config
            hidden_size = config.get("hidden_size", 4096)
            num_layers = config.get("num_hidden_layers", 32)
            # Rough parameter count estimate
            params = hidden_size * hidden_size * 4 * num_layers  # Very rough
            return params * 2

        return 0

    def determine_num_available_blocks(self) -> tuple[int, int]:
        """Determine the number of available KV cache blocks.

        Returns:
            Tuple of (num_gpu_blocks, num_cpu_blocks)
        """
        import psutil

        total_memory = psutil.virtual_memory().total

        # Calculate block memory size
        if self.model_runner is not None and self.model_runner.model_config is not None:
            model_config = self.model_runner.model_config
            num_layers = model_config.get("num_hidden_layers", 32)
            num_kv_heads = model_config.get(
                "num_key_value_heads",
                model_config.get("num_attention_heads", 32),
            )
            head_dim = model_config.get("hidden_size", 4096) // model_config.get(
                "num_attention_heads", 32
            )

            # Each block stores key and value for all layers
            # Block memory = 2 * num_layers * block_size * num_kv_heads * head_dim * dtype_size
            dtype_size = 2  # float16
            block_memory = (
                2
                * num_layers
                * self.config.block_size
                * num_kv_heads
                * head_dim
                * dtype_size
            )
        else:
            # Default estimate: ~4KB per block
            block_memory = 4096

        if block_memory <= 0:
            msg = (
                "Computed KV cache block size is invalid "
                f"({block_memory} bytes). Check model config and VLLM_METAL_BLOCK_SIZE."
            )
            raise ValueError(msg)

        # Handle auto memory mode
        if self.config.is_auto_memory:
            # Get actual model memory usage
            model_memory = self._get_model_memory_usage()

            # Calculate minimum blocks needed to handle at least one request
            # at max_model_len (vLLM requires this minimum)
            max_model_len = 2048  # Default fallback
            if self.vllm_config is not None:
                max_model_len = self.vllm_config.model_config.max_model_len

            min_blocks = (
                max_model_len + self.config.block_size - 1
            ) // self.config.block_size
            # Add a small buffer for safety (e.g., 10% more blocks)
            min_blocks = int(min_blocks * AUTO_MEMORY_MIN_BLOCKS_BUFFER_FACTOR)

            min_cache_memory = min_blocks * block_memory

            # Add 20% overhead buffer for MLX operations
            minimal_needed = int(
                (model_memory + min_cache_memory) * AUTO_MEMORY_OVERHEAD_FACTOR
            )

            # Calculate how much of total unified memory the minimum requirement
            # consumes (useful for user diagnostics in logs/errors).
            needed_fraction = minimal_needed / total_memory

            if minimal_needed > total_memory:
                msg = (
                    "Auto memory mode (VLLM_METAL_MEMORY_FRACTION=auto) requires more "
                    "memory than is available. "
                    f"total={total_memory / 1e9:.2f}GB, "
                    f"model={model_memory / 1e9:.2f}GB, "
                    f"min_kv_cache={min_cache_memory / 1e9:.2f}GB, "
                    f"overhead_factor={AUTO_MEMORY_OVERHEAD_FACTOR:.1f} "
                    f"(max_model_len={max_model_len}, "
                    f"block_size={self.config.block_size}, "
                    f"min_blocks={min_blocks}, "
                    f"needed_fraction={needed_fraction:.3f}). "
                    "Mitigations: reduce max_model_len, reduce VLLM_METAL_BLOCK_SIZE, "
                    "or use a smaller model."
                )
                raise ValueError(msg)

            logger.info(
                f"Auto memory mode: model={model_memory / 1e9:.2f}GB, "
                f"max_model_len={max_model_len}, min_blocks={min_blocks}, "
                f"min_kv_cache={min_cache_memory / 1e9:.2f}GB, "
                f"total_needed={minimal_needed / 1e9:.2f}GB, "
                f"needed_fraction={needed_fraction:.3f}"
            )

            available_memory = minimal_needed
            # In auto mode, use most of the allocated memory for cache
            # since we already calculated minimal needed
            cache_memory = min_cache_memory
        else:
            available_memory = int(total_memory * self.config.memory_fraction)
            # Reserve some memory for model weights and overhead
            cache_memory = available_memory * 0.5  # Use 50% for cache

        num_blocks = int(cache_memory / block_memory)

        # Ensure at least 1 block
        num_blocks = max(1, num_blocks)

        # Metal has unified memory, so all blocks are "GPU" blocks
        return (num_blocks, 0)

    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks: int = 0) -> None:
        """Initialize the KV cache.

        Args:
            num_gpu_blocks: Number of GPU cache blocks
            num_cpu_blocks: Number of CPU cache blocks (unused on Metal)
        """
        if self.model_runner is not None:
            self.model_runner.initialize_cache(num_gpu_blocks)
            logger.info(f"KV cache initialized with {num_gpu_blocks} blocks")

    def execute_model(
        self,
        seq_group_metadata_list: list[Any] | None = None,
        **kwargs: Any,
    ) -> list[Any]:
        """Execute model inference.

        Args:
            seq_group_metadata_list: Sequence group metadata
            **kwargs: Additional arguments

        Returns:
            List of model outputs
        """
        if not self.is_initialized:
            msg = "Worker not initialized. Call load_model() first."
            raise RuntimeError(msg)

        if self.model_runner is None:
            msg = "Model runner not initialized"
            raise RuntimeError(msg)

        return self.model_runner.execute_model(
            seq_group_metadata_list=seq_group_metadata_list,
            **kwargs,
        )

    def get_cache_block_size_bytes(self) -> int:
        """Get the size of a cache block in bytes.

        Returns:
            Block size in bytes
        """
        if self.model_runner is not None and self.model_runner.model_config is not None:
            model_config = self.model_runner.model_config
            num_layers = model_config.get("num_hidden_layers", 32)
            num_kv_heads = model_config.get(
                "num_key_value_heads",
                model_config.get("num_attention_heads", 32),
            )
            head_dim = model_config.get("hidden_size", 4096) // model_config.get(
                "num_attention_heads", 32
            )

            dtype_size = 2  # float16
            return (
                2
                * num_layers
                * self.config.block_size
                * num_kv_heads
                * head_dim
                * dtype_size
            )

        return 4096  # Default estimate

    def __del__(self) -> None:
        """Cleanup worker resources."""
        self.model_runner = None
