# SPDX-License-Identifier: Apache-2.0
"""LFM2 / LFM2.5 ShortConv state family, including the MoE variants."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import mlx.core as mx
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

from vllm_metal.attention.caches.shortconv_cache import ShortConvStateCache
from vllm_metal.attention.impls.shortconv import ShortConvPagedWrapper, is_shortconv
from vllm_metal.attention.runtime.hybrid_plan import (
    ATTENTION_LAYER,
    STATE_LAYER,
    ConvStateGeometry,
    HybridLayerPlan,
    HybridRuntimePlan,
    StateFamilySpec,
    StateGeometry,
)


def _create_shortconv_state_cache(
    *,
    geometry: StateGeometry,
    num_layers: int,
    max_seqs: int,
    initial_seqs: int,
    dtype: mx.Dtype,
) -> ShortConvStateCache:
    if not isinstance(geometry, ConvStateGeometry):
        raise TypeError("ShortConv state cache requires convolution-only geometry")
    return ShortConvStateCache(
        num_layers=num_layers,
        max_seqs=max_seqs,
        conv_kernel_dim=geometry.conv_kernel_dim,
        conv_dim=geometry.conv_dim,
        initial_seqs=initial_seqs,
        dtype=dtype,
    )


SHORTCONV_MODEL_TYPES = frozenset({"lfm2", "lfm2_moe"})

SHORTCONV_FAMILY = StateFamilySpec(
    label="shortconv",
    wrapper_cls=ShortConvPagedWrapper,
    is_state_module=is_shortconv,
    mamba_type=MambaAttentionBackendEnum.SHORT_CONV,
    state_dtypes=(None,),
    supported_cache_modes=("none", "align"),
    # Served with the decode pipeline since it landed on main.
    supports_decode_pipeline=True,
    layer_name="conv",
    create_state_cache=_create_shortconv_state_cache,
)


def build_shortconv_hybrid_plan(
    model_args: Mapping[str, Any], num_layers: int
) -> HybridRuntimePlan:
    """Resolve explicit attention/conv layer roles and the pre-conv tail shape."""
    layer_types = model_args.get("layer_types")
    if layer_types is None:
        # Older LFM2 configs enumerate attention indices instead. mlx-lm's
        # decoder uses these same indices when it constructs the layers.
        attention_indices = model_args.get("full_attn_idxs")
        if attention_indices is None:
            raise ValueError(
                "ShortConv model args require layer_types or full_attn_idxs"
            )
        layer_types = [
            "full_attention" if i in attention_indices else "conv"
            for i in range(num_layers)
        ]
    if len(layer_types) != num_layers:
        raise ValueError("ShortConv layer_types must contain one entry per model layer")
    unsupported = set(layer_types) - {"conv", "full_attention"}
    if unsupported:
        raise ValueError(f"Unsupported ShortConv layer_types: {sorted(unsupported)}")
    if set(layer_types) != {"conv", "full_attention"}:
        raise ValueError(
            "ShortConv hybrid requires both attention and convolution layers"
        )
    attention_indices = model_args.get("full_attn_idxs")
    if attention_indices is not None and set(attention_indices) != {
        i for i, role in enumerate(layer_types) if role == "full_attention"
    }:
        # mlx-lm constructs layers from full_attn_idxs when both are present.
        # Do not report a different state/KV topology to the scheduler.
        raise ValueError("ShortConv layer_types and full_attn_idxs disagree")
    kernel = model_args.get("conv_L_cache")
    width = model_args.get("hidden_size")
    if type(kernel) is not int or kernel < 2:
        raise ValueError("ShortConv conv_L_cache must be an integer >= 2")
    if type(width) is not int or width <= 0:
        raise ValueError("ShortConv hidden_size must be a positive integer")
    return HybridRuntimePlan(
        layers=HybridLayerPlan(
            layer_roles=tuple(
                ATTENTION_LAYER if role == "full_attention" else STATE_LAYER
                for role in layer_types
            )
        ),
        family=SHORTCONV_FAMILY,
        geometry=ConvStateGeometry(conv_kernel_dim=kernel, conv_dim=width),
    )
