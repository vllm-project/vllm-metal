# SPDX-License-Identifier: Apache-2.0
"""GDN linear-attention state family (Qwen3.5 / Qwen3.6, Qwen3-Next)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

from vllm_metal.attention.impls.linear import (
    GDNPagedAttentionWrapper,
    is_linear_attention,
)
from vllm_metal.attention.runtime.hybrid_plan import (
    ATTENTION_LAYER,
    STATE_LAYER,
    HybridLayerPlan,
    HybridRuntimePlan,
    LayerKind,
    RecurrentStateGeometry,
    StateFamilySpec,
)

_INTERVAL_KEY = "full_attention_interval"
# Scheduler-side mamba caching strategies the GDN state path implements.
_SUPPORTED_CACHE_MODES = ("none", "align")


def build_gdn_hybrid_plan(
    model_args: Mapping[str, Any], num_layers: int
) -> HybridRuntimePlan:
    """Resolve GDN layer topology and recurrent geometry from model args."""
    interval = _require_positive_int(model_args, _INTERVAL_KEY)
    if num_layers <= 0:
        raise ValueError(
            f"GDN hybrid requires a positive layer count, got {num_layers}."
        )
    kinds: tuple[LayerKind, ...] = tuple(
        ATTENTION_LAYER if (i + 1) % interval == 0 else STATE_LAYER
        for i in range(num_layers)
    )
    return HybridRuntimePlan(
        layers=HybridLayerPlan.from_kinds(kinds),
        family=_GDN_FAMILY,
        geometry=_gdn_geometry(model_args),
    )


_GDN_FAMILY = StateFamilySpec(
    label="gdn",
    wrapper_cls=GDNPagedAttentionWrapper,
    is_state_module=is_linear_attention,
    mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
    # Match the fp32 recurrent pool used for kernel accumulation.
    recurrent_dtype=torch.float32,
    supported_cache_modes=_SUPPORTED_CACHE_MODES,
)


def _gdn_geometry(model_args: Mapping[str, Any]) -> RecurrentStateGeometry:
    num_k_heads = _require_positive_int(model_args, "linear_num_key_heads")
    num_v_heads = _require_positive_int(model_args, "linear_num_value_heads")
    key_head_dim = _require_positive_int(model_args, "linear_key_head_dim")
    value_head_dim = _require_positive_int(model_args, "linear_value_head_dim")
    return RecurrentStateGeometry(
        conv_kernel_dim=_require_positive_int(model_args, "linear_conv_kernel_dim"),
        # GDN packs q/k at key_dim and v at value_dim into one conv stream.
        conv_dim=num_k_heads * key_head_dim * 2 + num_v_heads * value_head_dim,
        num_v_heads=num_v_heads,
        value_head_dim=value_head_dim,
        key_head_dim=key_head_dim,
    )


def _require_positive_int(model_args: Mapping[str, Any], key: str) -> int:
    value = model_args.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(
            f"GDN hybrid model args are missing a usable {key!r}: got {value!r}."
        )
    return int(value)
