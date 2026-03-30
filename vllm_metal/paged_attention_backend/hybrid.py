# SPDX-License-Identifier: Apache-2.0
"""Hybrid model helpers for paged attention backend.

Provides spec construction for GDN linear attention layers in hybrid
models (Qwen3.5).  The full hybrid backend will be added in Stage C
when the linear attention kernel is implemented.
"""

from __future__ import annotations

import torch
from vllm.v1.kv_cache_interface import MambaSpec


def _build_linear_layer_spec(
    *,
    conv_kernel_dim: int,
    conv_dim: int,
    num_v_heads: int,
    value_head_dim: int,
    key_head_dim: int,
    torch_dtype: torch.dtype,
) -> MambaSpec:
    """Build a MambaSpec for one GDN linear attention layer."""
    return MambaSpec(
        shapes=(
            (conv_kernel_dim - 1, conv_dim),
            (num_v_heads, value_head_dim, key_head_dim),
        ),
        dtypes=(torch_dtype, torch_dtype),
        block_size=1,
    )
