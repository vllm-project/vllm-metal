# SPDX-License-Identifier: Apache-2.0
"""GDN linear-attention state family (Qwen3.5 / Qwen3.6, Qwen3-Next)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import torch
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

from vllm_metal.attention.caches.gdn_cache import GDNPagedStateCache
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
    StateGeometry,
)


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
        try:
            return cls(
                full_attention_interval=model_args["full_attention_interval"],
                num_key_heads=model_args["linear_num_key_heads"],
                num_value_heads=model_args["linear_num_value_heads"],
                key_head_dim=model_args["linear_key_head_dim"],
                value_head_dim=model_args["linear_value_head_dim"],
                conv_kernel_dim=model_args["linear_conv_kernel_dim"],
            )
        except KeyError as exc:
            raise ValueError(
                f"GDN hybrid model args are missing required {exc.args[0]!r}."
            ) from exc

    def __post_init__(self) -> None:
        invalid_fields = [
            f"{name}={value!r}"
            for name, value in (
                ("full_attention_interval", self.full_attention_interval),
                ("linear_num_key_heads", self.num_key_heads),
                ("linear_num_value_heads", self.num_value_heads),
                ("linear_key_head_dim", self.key_head_dim),
                ("linear_value_head_dim", self.value_head_dim),
                ("linear_conv_kernel_dim", self.conv_kernel_dim),
            )
            if type(value) is not int or value <= 0
        ]
        if invalid_fields:
            raise ValueError(
                "GDN hybrid model args must be positive integers; invalid "
                f"{', '.join(invalid_fields)}."
            )

    def layer_roles(self, num_layers: int) -> tuple[LayerRole, ...]:
        interval = self.full_attention_interval
        if not 2 <= interval <= num_layers:
            raise ValueError(
                "GDN hybrid requires 2 <= full_attention_interval <= num_layers so "
                "the model keeps both attention and state layers, got "
                f"full_attention_interval={interval} with num_layers={num_layers}."
            )
        return tuple(
            ATTENTION_LAYER if (i + 1) % interval == 0 else STATE_LAYER
            for i in range(num_layers)
        )

    def state_geometry(self) -> RecurrentStateGeometry:
        return RecurrentStateGeometry(
            conv_kernel_dim=self.conv_kernel_dim,
            # GDN packs q/k at key_dim and v at value_dim into one conv stream.
            conv_dim=self.num_key_heads * self.key_head_dim * 2
            + self.num_value_heads * self.value_head_dim,
            num_v_heads=self.num_value_heads,
            value_head_dim=self.value_head_dim,
            key_head_dim=self.key_head_dim,
        )


def create_gdn_state_cache(
    *,
    geometry: StateGeometry,
    num_layers: int,
    max_seqs: int,
    initial_seqs: int,
    dtype: mx.Dtype,
) -> GDNPagedStateCache:
    if not isinstance(geometry, RecurrentStateGeometry):
        raise TypeError("GDN state cache requires recurrent state geometry")
    return GDNPagedStateCache(
        num_layers=num_layers,
        max_seqs=max_seqs,
        conv_kernel_dim=geometry.conv_kernel_dim,
        conv_dim=geometry.conv_dim,
        num_v_heads=geometry.num_v_heads,
        value_head_dim=geometry.value_head_dim,
        key_head_dim=geometry.key_head_dim,
        initial_seqs=initial_seqs,
        dtype=dtype,
    )


# mlx-lm names the text-only loads by the outer config; mlx-vlm flattens the
# nested text config, so the same family also arrives as the ``_text`` types.
GDN_MODEL_TYPES = frozenset(
    {"qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text", "qwen3_next"}
)

GDN_FAMILY = StateFamilySpec(
    label="gdn",
    wrapper_cls=GDNPagedAttentionWrapper,
    is_state_module=is_linear_attention,
    mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
    # Match the fp32 recurrent pool used for kernel accumulation.
    state_dtypes=(None, torch.float32),
    # Scheduler-side mamba caching strategies the GDN state path implements.
    supported_cache_modes=("none", "align"),
    supports_decode_pipeline=True,
    layer_name="linear_attn",
    create_state_cache=create_gdn_state_cache,
)


def build_gdn_hybrid_plan(
    model_args: Mapping[str, Any], num_layers: int
) -> HybridRuntimePlan:
    """Resolve GDN layer topology and recurrent geometry from model args."""
    gdn_config = GDNHybridConfig.from_model_args(model_args)
    return HybridRuntimePlan(
        layers=HybridLayerPlan(layer_roles=gdn_config.layer_roles(num_layers)),
        family=GDN_FAMILY,
        geometry=gdn_config.state_geometry(),
    )
