# SPDX-License-Identifier: Apache-2.0
"""Granite hybrid topology and Mamba-2 state geometry."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

from vllm_metal.attention.impls.mamba2 import Mamba2PagedStateWrapper, is_mamba2_mixer
from vllm_metal.attention.runtime.hybrid_plan import (
    ATTENTION_LAYER,
    STATE_LAYER,
    HybridLayerPlan,
    HybridRuntimePlan,
    RecurrentStateGeometry,
    StateFamilySpec,
)

GRANITE_MODEL_TYPES = frozenset({"granitemoehybrid"})

GRANITE_FAMILY = StateFamilySpec(
    label="granite",
    wrapper_cls=Mamba2PagedStateWrapper,
    is_state_module=is_mamba2_mixer,
    mamba_type=MambaAttentionBackendEnum.MAMBA2,
    supported_cache_modes=("none",),
    layer_name="mamba",
)


def build_granite_hybrid_plan(
    model_args: Mapping[str, Any],
    num_layers: int,
    state_dtypes: tuple[torch.dtype, ...],
) -> HybridRuntimePlan:
    """Resolve Granite's explicit layer types and mlx-lm Mamba-2 dimensions."""
    num_heads = model_args["mamba_n_heads"]
    head_dim = model_args["mamba_d_head"]
    state_size = model_args["mamba_d_state"]
    n_groups = model_args["mamba_n_groups"]

    return HybridRuntimePlan(
        layers=HybridLayerPlan(
            layer_roles=tuple(
                STATE_LAYER if kind == "mamba" else ATTENTION_LAYER
                for kind in model_args["layer_types"]
            )
        ),
        family=GRANITE_FAMILY,
        geometry=RecurrentStateGeometry(
            conv_kernel_dim=model_args["mamba_d_conv"],
            conv_dim=num_heads * head_dim + 2 * n_groups * state_size,
            num_v_heads=num_heads,
            value_head_dim=head_dim,
            key_head_dim=state_size,
        ),
        state_dtypes=state_dtypes,
    )
