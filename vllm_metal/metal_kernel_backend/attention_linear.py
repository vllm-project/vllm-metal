# SPDX-License-Identifier: Apache-2.0
"""Linear attention (Gated DeltaNet) on Metal — NOT YET IMPLEMENTED.

Targets models like Qwen/Qwen3.5-0.8B (mlx_lm module: ``qwen3_next``) that use hybrid architectures
with a mix of SDPA and linear attention layers.  In mlx_lm, the linear
attention module is ``Qwen3NextGatedDeltaNet`` and lives on
``layer.linear_attn`` (as opposed to ``layer.self_attn`` for SDPA layers).

Detection heuristic: the module has ``in_proj_qkvz`` and ``conv1d``
(combined Q/K/V/Z projection + 1-D convolution) — attributes that no
standard SDPA module has.

All operations use MLX arrays end-to-end — no PyTorch MPS bridge.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from vllm_metal.metal_kernel_backend.cache import MetalPagedKVCache
from vllm_metal.paged_attention_common import PagedAttentionContext


def is_linear_attention(module: nn.Module) -> bool:
    """Return True if *module* is a linear attention layer (e.g. GatedDeltaNet)."""
    return hasattr(module, "in_proj_qkvz") and hasattr(module, "conv1d")


def linear_attention_forward(
    inner: nn.Module,
    x: mx.array,
    ctx: PagedAttentionContext,
    kv_cache: MetalPagedKVCache,
    layer_idx: int,
) -> mx.array:
    """Linear attention forward pass — not yet implemented.

    This is a placeholder for future implementation.  Contributing guide:

    1. Linear attention (GatedDeltaNet) uses a recurrent state instead of
       a standard KV cache.  The cache layout will differ from SDPA — you
       will likely need a separate cache class or a per-layer cache spec.

    2. There is no softmax — the kernel computes gated delta updates:
       ``gated_delta_update(q, k, v, a, b, A_log, dt_bias, state, ...)``.

    3. The model uses ``conv1d`` over concatenated Q/K/V before the
       attention computation.  This stateful convolution needs its own
       cache slot (``MambaCache`` in mlx_lm).

    4. Qwen3.5 is a hybrid model: SDPA layers (every ``full_attention_interval``-th)
       coexist with linear layers.  The patching mechanism in
       ``paged_attention.py`` needs to handle both ``self_attn`` and
       ``linear_attn`` attributes on the same model.
    """
    raise NotImplementedError(
        f"Linear attention (GatedDeltaNet) is not yet implemented for Metal paged "
        f"attention. Module: {type(inner).__name__}, layer: {layer_idx}. "
        f"See attention_linear.py for contributing guide."
    )
