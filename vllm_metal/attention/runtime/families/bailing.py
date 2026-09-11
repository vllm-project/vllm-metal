# SPDX-License-Identifier: Apache-2.0
"""Bailing V3 hybrid MLA/KDA layer topology and recurrent state family."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import mlx.core as mx
import torch
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

from vllm_metal.attention.caches.gdn_cache import GDNPagedStateCache
from vllm_metal.attention.impls.kda import KDAPagedAttentionWrapper, is_bailing_kda
from vllm_metal.attention.runtime.hybrid_plan import (
    ATTENTION_LAYER,
    STATE_LAYER,
    HybridLayerPlan,
    HybridRuntimePlan,
    RecurrentStateGeometry,
    StateFamilySpec,
    StateGeometry,
)


def _create_kda_state_cache(
    *,
    geometry: StateGeometry,
    num_layers: int,
    max_seqs: int,
    initial_seqs: int,
    dtypes: tuple[mx.Dtype, ...],
) -> GDNPagedStateCache:
    if not isinstance(geometry, RecurrentStateGeometry):
        raise TypeError("KDA state cache requires recurrent state geometry")
    return GDNPagedStateCache(
        num_layers=num_layers,
        max_seqs=max_seqs,
        conv_kernel_dim=geometry.conv_kernel_dim,
        conv_dim=geometry.conv_dim,
        num_v_heads=geometry.num_v_heads,
        value_head_dim=geometry.value_head_dim,
        key_head_dim=geometry.key_head_dim,
        initial_seqs=initial_seqs,
        dtype=dtypes[0],
        recurrent_dtype=dtypes[1],
    )


BAILING_MODEL_TYPES = frozenset({"bailing_hybrid"})

BAILING_FAMILY = StateFamilySpec(
    label="kda",
    wrapper_cls=KDAPagedAttentionWrapper,
    is_state_module=is_bailing_kda,
    # KDA uses the same scheduler state layout as GDN, not its compute kernel.
    mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
    supported_cache_modes=("none", "align"),
    layer_name="linear_attn",
    create_state_cache=_create_kda_state_cache,
)


def _require_bailing_v3_architecture(model_args: Mapping[str, Any]) -> None:
    architectures = model_args.get("architectures") or ()
    if isinstance(architectures, str):
        architectures = (architectures,)
    if "BailingMoeV3ForCausalLM" not in architectures:
        raise NotImplementedError(
            "Metal's bailing_hybrid runtime requires "
            "architectures=BailingMoeV3ForCausalLM"
        )


def build_bailing_hybrid_plan(
    model_args: Mapping[str, Any],
    num_layers: int,
    state_dtypes: tuple[torch.dtype, ...],
) -> HybridRuntimePlan:
    """Resolve the supported Bailing V3 MLA/KDA layout from model args."""
    _require_bailing_v3_architecture(model_args)
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
    if not 2 <= group_size <= num_layers:
        raise ValueError(
            "Bailing V3 hybrid requires 2 <= layer_group_size <= num_layers so "
            "the model keeps both MLA and KDA layers, got "
            f"layer_group_size={group_size} with num_layers={num_layers}."
        )
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
        family=BAILING_FAMILY,
        geometry=RecurrentStateGeometry(
            conv_kernel_dim=model_args["short_conv_kernel_size"],
            conv_dim=3 * num_heads * head_dim,
            num_v_heads=num_heads,
            value_head_dim=head_dim,
            key_head_dim=head_dim,
        ),
        state_dtypes=state_dtypes,
    )
