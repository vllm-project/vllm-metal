# SPDX-License-Identifier: Apache-2.0
"""GDN linear-attention state family (Qwen3.5 / Qwen3.6, Qwen3-Next)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
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
    LayerRole,
    RecurrentStateGeometry,
    StateFamilySpec,
)

_GDN_FAMILY = StateFamilySpec(
    label="gdn",
    wrapper_cls=GDNPagedAttentionWrapper,
    is_state_module=is_linear_attention,
    mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
    # Match the fp32 recurrent pool used for kernel accumulation.
    recurrent_dtype=torch.float32,
    # Scheduler-side mamba caching strategies the GDN state path implements.
    supported_cache_modes=("none", "align"),
)


def _read_positive_int(model_args: Mapping[str, Any], key: str) -> int:
    """Read one GDN dimension from model args, rejecting anything unusable."""
    value = model_args.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(
            f"GDN hybrid model args are missing a usable {key!r}: got {value!r}."
        )
    return value


@dataclass(frozen=True, slots=True)
class GDNHybridConfig:
    """GDN dims parsed once from the third-party model args boundary."""

    full_attention_interval: int
    num_key_heads: int
    num_value_heads: int
    key_head_dim: int
    value_head_dim: int
    conv_kernel_dim: int

    @classmethod
    def from_model_args(cls, model_args: Mapping[str, Any]) -> GDNHybridConfig:
        return cls(
            full_attention_interval=_read_positive_int(
                model_args, "full_attention_interval"
            ),
            num_key_heads=_read_positive_int(model_args, "linear_num_key_heads"),
            num_value_heads=_read_positive_int(model_args, "linear_num_value_heads"),
            key_head_dim=_read_positive_int(model_args, "linear_key_head_dim"),
            value_head_dim=_read_positive_int(model_args, "linear_value_head_dim"),
            conv_kernel_dim=_read_positive_int(model_args, "linear_conv_kernel_dim"),
        )


def build_gdn_hybrid_plan(
    model_args: Mapping[str, Any], num_layers: int
) -> HybridRuntimePlan:
    """Resolve GDN layer topology and recurrent geometry from model args."""
    gdn_config = GDNHybridConfig.from_model_args(model_args)
    if num_layers <= 0:
        raise ValueError(
            f"GDN hybrid requires a positive layer count, got {num_layers}."
        )
    layer_roles: tuple[LayerRole, ...] = tuple(
        ATTENTION_LAYER
        if (i + 1) % gdn_config.full_attention_interval == 0
        else STATE_LAYER
        for i in range(num_layers)
    )
    return HybridRuntimePlan(
        layers=HybridLayerPlan(layer_roles=layer_roles),
        family=_GDN_FAMILY,
        geometry=RecurrentStateGeometry(
            conv_kernel_dim=gdn_config.conv_kernel_dim,
            # GDN packs q/k at key_dim and v at value_dim into one conv stream.
            conv_dim=gdn_config.num_key_heads * gdn_config.key_head_dim * 2
            + gdn_config.num_value_heads * gdn_config.value_head_dim,
            num_v_heads=gdn_config.num_value_heads,
            value_head_dim=gdn_config.value_head_dim,
            key_head_dim=gdn_config.key_head_dim,
        ),
    )
