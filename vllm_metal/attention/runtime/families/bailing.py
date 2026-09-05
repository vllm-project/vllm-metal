# SPDX-License-Identifier: Apache-2.0
"""Bailing V3 hybrid MLA/KDA layer topology and recurrent state family."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

from vllm_metal.attention.impls.kda import KDAPagedAttentionWrapper, is_bailing_kda
from vllm_metal.attention.runtime.hybrid_plan import (
    ATTENTION_LAYER,
    STATE_LAYER,
    HybridLayerPlan,
    HybridRuntimePlan,
    RecurrentStateGeometry,
    StateFamilySpec,
)

_KDA_FAMILY = StateFamilySpec(
    label="kda",
    wrapper_cls=KDAPagedAttentionWrapper,
    is_state_module=is_bailing_kda,
    # KDA uses the same scheduler state layout as GDN, not its compute kernel.
    mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
    recurrent_dtype=torch.float32,
    supported_cache_modes=("none", "align"),
)


def build_bailing_hybrid_plan(
    model_args: Mapping[str, Any], num_layers: int
) -> HybridRuntimePlan:
    """Resolve the supported Bailing V3 MLA/KDA layout from model args."""
    for name in (
        "layer_group_size",
        "num_attention_heads",
        "head_dim",
        "short_conv_kernel_size",
    ):
        value = model_args.get(name)
        if type(value) is not int or value <= 0:
            raise ValueError(f"Bailing V3 requires a positive integer {name}")
    for name in ("no_kda_lora", "kda_safe_gate"):
        if model_args.get(name) is not True:
            raise NotImplementedError(f"Bailing V3 requires {name}=true")

    group_size = model_args["layer_group_size"]
    grouped_layers = num_layers // group_size * group_size
    num_heads = model_args["num_attention_heads"]
    head_dim = model_args["head_dim"]
    return HybridRuntimePlan(
        layers=HybridLayerPlan(
            layer_roles=tuple(
                ATTENTION_LAYER
                if (i + 1) % group_size == 0 or i >= grouped_layers
                else STATE_LAYER
                for i in range(num_layers)
            )
        ),
        family=_KDA_FAMILY,
        geometry=RecurrentStateGeometry(
            conv_kernel_dim=model_args["short_conv_kernel_size"],
            conv_dim=3 * num_heads * head_dim,
            num_v_heads=num_heads,
            value_head_dim=head_dim,
            key_head_dim=head_dim,
        ),
    )
