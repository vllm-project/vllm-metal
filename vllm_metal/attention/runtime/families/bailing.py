# SPDX-License-Identifier: Apache-2.0
"""Bailing V3 hybrid MLA/KDA topology and recurrent state family."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

from vllm_metal.attention.impls.kda import (
    KDAPagedAttentionWrapper,
    is_kda_attention,
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


@dataclass(frozen=True, slots=True)
class BailingV3HybridConfig:
    """Bailing V3 dimensions parsed from MLX-LM model arguments."""

    layer_group_size: int
    num_heads: int
    head_dim: int
    conv_kernel_dim: int

    @classmethod
    def from_model_args(cls, model_args: Mapping[str, Any]) -> BailingV3HybridConfig:
        for name in ("no_kda_lora", "kda_safe_gate"):
            if model_args.get(name) is not True:
                raise NotImplementedError(f"Bailing V3 requires {name}=true")
        try:
            return cls(
                layer_group_size=model_args["layer_group_size"],
                num_heads=model_args["num_attention_heads"],
                head_dim=model_args["head_dim"],
                conv_kernel_dim=model_args["short_conv_kernel_size"],
            )
        except KeyError as exc:
            raise ValueError(
                f"Bailing V3 model args are missing required {exc.args[0]!r}."
            ) from exc

    def __post_init__(self) -> None:
        invalid_fields = [
            f"{name}={value!r}"
            for name, value in (
                ("layer_group_size", self.layer_group_size),
                ("num_attention_heads", self.num_heads),
                ("head_dim", self.head_dim),
                ("short_conv_kernel_size", self.conv_kernel_dim),
            )
            if type(value) is not int or value <= 0
        ]
        if invalid_fields:
            raise ValueError(
                "Bailing V3 model args must be positive integers; invalid "
                f"{', '.join(invalid_fields)}."
            )

    def layer_roles(self, num_layers: int) -> tuple[LayerRole, ...]:
        if not 2 <= self.layer_group_size <= num_layers:
            raise ValueError(
                "Bailing V3 hybrid requires 2 <= layer_group_size <= num_layers, "
                f"got layer_group_size={self.layer_group_size} with "
                f"num_layers={num_layers}."
            )
        grouped_layers = num_layers // self.layer_group_size * self.layer_group_size
        return tuple(
            ATTENTION_LAYER
            if (i + 1) % self.layer_group_size == 0 or i >= grouped_layers
            else STATE_LAYER
            for i in range(num_layers)
        )

    def state_geometry(self) -> RecurrentStateGeometry:
        return kda_state_geometry(self.num_heads, self.head_dim, self.conv_kernel_dim)


def kda_state_geometry(
    num_heads: int, head_dim: int, conv_kernel_dim: int
) -> RecurrentStateGeometry:
    """KDA state: the q/k/v conv tail and one square recurrent state per head."""
    projection_size = num_heads * head_dim
    return RecurrentStateGeometry(
        conv_kernel_dim=conv_kernel_dim,
        conv_dim=3 * projection_size,
        num_v_heads=num_heads,
        value_head_dim=head_dim,
        key_head_dim=head_dim,
    )


BAILING_MODEL_TYPES = frozenset({"bailing_hybrid"})

BAILING_FAMILY = StateFamilySpec(
    label="kda",
    wrapper_cls=KDAPagedAttentionWrapper,
    is_state_module=is_kda_attention,
    # vLLM exposes Bailing KDA through the GDN attention backend enum.
    mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
    supported_cache_modes=("none", "align"),
    layer_name="linear_attn",
)


def build_bailing_hybrid_plan(
    model_args: Mapping[str, Any],
    num_layers: int,
    state_dtypes: tuple[torch.dtype, ...],
) -> HybridRuntimePlan:
    """Resolve Bailing V3's MLA/KDA topology and recurrent geometry."""
    config = BailingV3HybridConfig.from_model_args(model_args)
    return HybridRuntimePlan(
        layers=HybridLayerPlan(layer_roles=config.layer_roles(num_layers)),
        family=BAILING_FAMILY,
        geometry=config.state_geometry(),
        state_dtypes=state_dtypes,
    )
