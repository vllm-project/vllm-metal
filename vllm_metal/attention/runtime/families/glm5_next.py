# SPDX-License-Identifier: Apache-2.0
"""GLM-5-Next (GLM-5.3-Flash) hybrid sparse-MLA/KDA topology."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from vllm_metal.attention.runtime.families.bailing import (
    BAILING_FAMILY,
    kda_state_geometry,
)
from vllm_metal.attention.runtime.hybrid_plan import (
    ATTENTION_LAYER,
    STATE_LAYER,
    HybridLayerPlan,
    HybridRuntimePlan,
    LayerRole,
)

_LAYER_ROLES: dict[str, LayerRole] = {
    "linear_attention": STATE_LAYER,
    # Transformers 5.18 renamed deepseek_sparse_attention to indexed_attention.
    "deepseek_sparse_attention": ATTENTION_LAYER,
    "indexed_attention": ATTENTION_LAYER,
}


@dataclass(frozen=True, slots=True)
class Glm5NextHybridConfig:
    """GLM-5-Next dimensions parsed from mlx-vlm text model arguments."""

    layer_types: tuple[str, ...]
    num_heads: int
    head_dim: int
    conv_kernel_dim: int

    @classmethod
    def from_model_args(cls, model_args: Mapping[str, Any]) -> Glm5NextHybridConfig:
        try:
            return cls(
                layer_types=tuple(model_args["layer_types"]),
                num_heads=model_args["linear_num_heads"],
                head_dim=model_args["linear_head_dim"],
                conv_kernel_dim=model_args["linear_conv_kernel_dim"],
            )
        except KeyError as exc:
            raise ValueError(
                f"GLM-5-Next model args are missing required {exc.args[0]!r}."
            ) from exc

    def layer_roles(self, num_layers: int) -> tuple[LayerRole, ...]:
        if len(self.layer_types) != num_layers:
            raise ValueError(
                f"GLM-5-Next lists {len(self.layer_types)} layer_types for "
                f"{num_layers} layers."
            )
        unknown = set(self.layer_types) - _LAYER_ROLES.keys()
        if unknown:
            raise ValueError(f"Unknown GLM-5-Next layer_types: {sorted(unknown)}.")
        return tuple(_LAYER_ROLES[layer_type] for layer_type in self.layer_types)


GLM5_NEXT_MODEL_TYPES = frozenset({"glm5_next_text"})


def build_glm5_next_hybrid_plan(
    model_args: Mapping[str, Any],
    num_layers: int,
    state_dtypes: tuple[torch.dtype, ...],
) -> HybridRuntimePlan:
    """Resolve GLM-5-Next's sparse-MLA/KDA topology and KDA state geometry."""
    config = Glm5NextHybridConfig.from_model_args(model_args)
    return HybridRuntimePlan(
        layers=HybridLayerPlan(layer_roles=config.layer_roles(num_layers)),
        # The linear-attention layers are KDA, served like Bailing V3's.
        family=BAILING_FAMILY,
        geometry=kda_state_geometry(
            config.num_heads, config.head_dim, config.conv_kernel_dim
        ),
        state_dtypes=state_dtypes,
    )
