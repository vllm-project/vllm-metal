# SPDX-License-Identifier: Apache-2.0
"""MLX compute backend for vLLM Metal.

This module provides MLX-based implementations of core LLM operations
that are significantly faster than PyTorch MPS on Apple Silicon.
"""

from vllm_metal.mlx.attention import (
    mlx_scaled_dot_product_attention,
    mlx_paged_attention,
)
from vllm_metal.mlx.cache import (
    mlx_reshape_and_cache,
    mlx_copy_blocks,
    mlx_swap_blocks,
)
from vllm_metal.mlx.norm import (
    mlx_rms_norm,
    mlx_layer_norm,
    mlx_fused_add_rms_norm,
)
from vllm_metal.mlx.rope import (
    mlx_rotary_embedding,
    mlx_apply_rope,
)
from vllm_metal.mlx.tensor_bridge import (
    to_mlx,
    to_torch,
    TensorBridge,
)
from vllm_metal.mlx.device import (
    get_mlx_device,
    mlx_synchronize,
    mlx_clear_cache,
)

__all__ = [
    # Attention
    "mlx_scaled_dot_product_attention",
    "mlx_paged_attention",
    # Cache
    "mlx_reshape_and_cache",
    "mlx_copy_blocks",
    "mlx_swap_blocks",
    # Normalization
    "mlx_rms_norm",
    "mlx_layer_norm",
    "mlx_fused_add_rms_norm",
    # RoPE
    "mlx_rotary_embedding",
    "mlx_apply_rope",
    # Tensor bridge
    "to_mlx",
    "to_torch",
    "TensorBridge",
    # Device
    "get_mlx_device",
    "mlx_synchronize",
    "mlx_clear_cache",
]
