# SPDX-License-Identifier: Apache-2.0
"""Metal V2 Model Runner - extends GPU model runner for Metal/MLX backend."""

from contextlib import contextmanager

import torch
from vllm.config import VllmConfig
from vllm.logger import init_logger

logger = init_logger(__name__)

# ============================================================================
# Module-level patching for Metal (must happen before importing GPUModelRunner)
# ============================================================================


def _patched_bincount_metal(
    prefill_token_ids: torch.Tensor,
    prefill_len: int,
    prompt_len: int,
    prompt_bin_mask: torch.Tensor,
    output_bin_counts: torch.Tensor,
) -> None:
    """PyTorch-based bincount replacement for Metal (no Triton)."""
    prompt_bin_mask.zero_()
    output_bin_counts.zero_()

    # Get the tokens in the range [prompt_len, prefill_len)
    if prefill_len > prompt_len:
        tokens = prefill_token_ids[prompt_len:prefill_len]
        tokens_cpu = tokens.cpu().to(torch.int64)
        vocab_size = output_bin_counts.shape[0]
        counts = torch.bincount(tokens_cpu, minlength=vocab_size)
        min_len = min(len(counts), vocab_size)
        output_bin_counts[:min_len] = counts[:min_len].to(output_bin_counts.device)

    # Set prompt_bin_mask for tokens in [0, prompt_len)
    if prompt_len > 0:
        prompt_tokens = prefill_token_ids[:prompt_len]
        prompt_tokens_cpu = prompt_tokens.cpu().to(torch.int64)
        vocab_size = prompt_bin_mask.shape[0]
        for token in prompt_tokens_cpu:
            if 0 <= token < vocab_size:
                prompt_bin_mask[token] = 1


# Patch bincount BEFORE any vLLM modules that use it are imported
try:
    import vllm.v1.worker.gpu.sample.penalties as penalties_module

    penalties_module.bincount = _patched_bincount_metal
    logger.debug("Patched penalties_module.bincount for Metal")
except ImportError:
    pass

# Patch input_batch functions BEFORE importing GPUModelRunner
try:
    import vllm.v1.worker.gpu.input_batch as input_batch_module

    from vllm_metal.v2.input_batch import (
        combine_sampled_and_draft_tokens,
        post_update,
        prepare_pos_seq_lens,
        prepare_prefill_inputs,
    )

    input_batch_module.prepare_prefill_inputs = prepare_prefill_inputs
    input_batch_module.prepare_pos_seq_lens = prepare_pos_seq_lens
    input_batch_module.combine_sampled_and_draft_tokens = (
        combine_sampled_and_draft_tokens
    )
    input_batch_module.post_update = post_update
    logger.debug("Patched input_batch module functions for Metal")
except ImportError as e:
    logger.warning(f"Failed to patch input_batch module: {e}")

# Patch penalties module - uses Triton kernels
try:
    import vllm.v1.worker.gpu.sample.penalties as penalties_module

    from vllm_metal.v2.penalties import apply_penalties_and_temperature

    penalties_module.apply_penalties_and_temperature = apply_penalties_and_temperature
    logger.debug("Patched penalties_module.apply_penalties_and_temperature for Metal")
except ImportError as e:
    logger.warning(f"Failed to patch penalties module: {e}")

# Patch gumbel module - uses Triton kernel
try:
    import vllm.v1.worker.gpu.sample.gumbel as gumbel_module

    from vllm_metal.v2.gumbel import gumbel_sample

    gumbel_module.gumbel_sample = gumbel_sample
    logger.debug("Patched gumbel_module.gumbel_sample for Metal")
except ImportError as e:
    logger.warning(f"Failed to patch gumbel module: {e}")

# Patch async_utils module - uses CUDA streams
try:
    import vllm.v1.worker.gpu.async_utils as async_utils_module

    from vllm_metal.v2.async_utils import MetalAsyncOutput

    async_utils_module.AsyncOutput = MetalAsyncOutput
    logger.debug("Patched async_utils_module.AsyncOutput for Metal")
except ImportError as e:
    logger.warning(f"Failed to patch async_utils module: {e}")

# Now import the rest of vLLM modules (they will get our patched functions)
from vllm.model_executor.model_loader import get_model  # noqa: E402
from vllm.v1.kv_cache_interface import KVCacheConfig  # noqa: E402
from vllm.v1.utils import CpuGpuBuffer  # noqa: E402
from vllm.v1.worker.gpu.attn_utils import (  # noqa: E402
    init_attn_backend,
    init_kv_cache,
)
from vllm.v1.worker.gpu.model_runner import GPUModelRunner  # noqa: E402

# Use our Metal-compatible BlockTables
from vllm_metal.v2.block_table import MetalBlockTables as BlockTables  # noqa: E402

# Patch states module's bincount reference
try:
    import vllm.v1.worker.gpu.states as states_module

    states_module.bincount = _patched_bincount_metal
    logger.debug("Patched states_module.bincount for Metal")
except (ImportError, AttributeError):
    pass


@contextmanager
def _torch_cuda_wrapper():
    """Context manager to handle CUDA references during init.

    Some vLLM code paths reference torch.cuda even on non-CUDA platforms.
    This wrapper provides a no-op for those references.
    """
    yield


class MetalModelRunner(GPUModelRunner):
    """Metal/MLX model runner that extends the GPU model runner.

    This class inherits all the complex input batch management, attention
    metadata building, and model execution from GPUModelRunner. It only
    overrides Metal-specific functionality like:
    - Disabling CUDA-specific features (pinned memory, CUDA graphs)
    - Using MPS/MLX synchronization instead of CUDA
    - Metal-specific device handling
    """

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        # Use the CUDA wrapper to prevent CUDA stream/event creation
        with _torch_cuda_wrapper():
            super().__init__(vllm_config, device)

        # Override CUDA-specific settings
        self.pin_memory = False  # Metal uses unified memory
        self.use_cuda_graph = False
        self.cascade_attn_enabled = False

        # Replace GPU tensors with MPS equivalents
        self._postprocess_tensors()

        # Log initialization
        logger.info(
            f"MetalModelRunner V2 initialized: "
            f"hidden_size={self.model_config.get_hidden_size()}, "
            f"num_heads={self.model_config.get_num_attention_heads(self.parallel_config)}, "
            f"num_kv_heads={self.model_config.get_num_kv_heads(self.parallel_config)}, "
            f"head_dim={self.model_config.get_head_size()}, "
            f"block_size={self.cache_config.block_size}"
        )

    def _postprocess_tensors(self) -> None:
        """Replace GPU tensors with device tensors for Metal."""
        # For Metal, we don't need separate CPU and GPU buffers
        # since MPS/MLX uses unified memory
        for v in vars(self).values():
            if isinstance(v, CpuGpuBuffer):
                v.gpu = v.cpu

    def _sync_device(self) -> None:
        """Synchronize the MPS/MLX device instead of CUDA."""
        import mlx.core as mx

        mx.eval([])  # Force MLX evaluation
        torch.mps.synchronize()

    def load_model(self, *args, **kwargs) -> None:
        """Load the model to the MPS device."""
        logger.info("Loading model with MLX acceleration...")

        # Load model using standard vLLM loader
        self.model = get_model(
            vllm_config=self.vllm_config,
        )

        # Move model to MPS device
        self.model = self.model.to(self.device)

        logger.info("Model loaded successfully")
