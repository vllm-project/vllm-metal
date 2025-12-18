# SPDX-License-Identifier: Apache-2.0
"""Metal operations for vLLM.

This module provides MLX-based implementations of core LLM operations
that are optimized for Apple Silicon.
"""

from vllm_metal.ops.activation import (
    gelu_and_mul,
    gelu_tanh_and_mul,
    silu_and_mul,
)
from vllm_metal.ops.attention import (
    paged_attention_v1,
    paged_attention_v2,
)
from vllm_metal.ops.cache import (
    copy_blocks,
    reshape_and_cache,
    swap_blocks,
)
from vllm_metal.ops.layernorm import (
    fused_add_rms_norm,
    rms_norm,
)
from vllm_metal.ops.rotary import (
    rotary_embedding,
)
from vllm_metal.ops.sampling import (
    sampling_from_probs,
)

_registered = False


def register_metal_ops() -> None:
    """Register Metal-specific operations with vLLM.

    This function is called by the plugin system to register
    MLX-based operation implementations.
    """
    global _registered
    if _registered:
        return

    # Operations are already implemented using MLX
    # Registration is mainly for tracking purposes
    from vllm_metal._compat import init_logger

    logger = init_logger(__name__)
    logger.info("Metal/MLX operations registered")

    _registered = True


__all__ = [
    # Attention
    "paged_attention_v1",
    "paged_attention_v2",
    # Cache
    "copy_blocks",
    "reshape_and_cache",
    "swap_blocks",
    # Activation
    "gelu_and_mul",
    "gelu_tanh_and_mul",
    "silu_and_mul",
    # Normalization
    "rms_norm",
    "fused_add_rms_norm",
    # Rotary
    "rotary_embedding",
    # Sampling
    "sampling_from_probs",
    # Registration
    "register_metal_ops",
]
