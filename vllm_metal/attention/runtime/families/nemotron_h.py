# SPDX-License-Identifier: Apache-2.0
"""Nemotron-H state family (Mamba-2 mixers, attention, stateless MLP/MoE blocks)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

from vllm_metal.attention.impls.mamba2 import Mamba2PagedStateWrapper, is_mamba2_mixer
from vllm_metal.attention.runtime.families.gdn import create_gdn_state_cache
from vllm_metal.attention.runtime.hybrid_plan import (
    ATTENTION_LAYER,
    STATE_LAYER,
    STATELESS_LAYER,
    HybridLayerPlan,
    HybridRuntimePlan,
    LayerRole,
    RecurrentStateGeometry,
    StateFamilySpec,
)

# mlx-lm's single-char block pattern; MLP and MoE blocks keep no cache.
_BLOCK_TOKEN_TO_LAYER_ROLE: Mapping[str, LayerRole] = {
    "M": STATE_LAYER,
    "*": ATTENTION_LAYER,
    "-": STATELESS_LAYER,
    "E": STATELESS_LAYER,
}


@dataclass(frozen=True, slots=True)
class NemotronHHybridConfig:
    """Nemotron-H block pattern and Mamba-2 dims parsed once from model args."""

    block_pattern: tuple[str, ...]
    mamba_num_heads: int
    mamba_head_dim: int
    ssm_state_size: int
    n_groups: int
    conv_kernel: int

    @classmethod
    def from_model_args(cls, model_args: Mapping[str, Any]) -> NemotronHHybridConfig:
        try:
            return cls(
                block_pattern=tuple(model_args["hybrid_override_pattern"]),
                mamba_num_heads=model_args["mamba_num_heads"],
                mamba_head_dim=model_args["mamba_head_dim"],
                ssm_state_size=model_args["ssm_state_size"],
                n_groups=model_args["n_groups"],
                conv_kernel=model_args["conv_kernel"],
            )
        except KeyError as exc:
            raise ValueError(
                f"Nemotron-H hybrid model args are missing required {exc.args[0]!r}."
            ) from exc

    def __post_init__(self) -> None:
        unknown = sorted(
            {
                token
                for token in self.block_pattern
                if token not in _BLOCK_TOKEN_TO_LAYER_ROLE
            }
        )
        if unknown:
            raise ValueError(
                "Nemotron-H hybrid_override_pattern must use M, *, - or E; got "
                f"{', '.join(repr(token) for token in unknown)}."
            )
        if "M" not in self.block_pattern or "*" not in self.block_pattern:
            raise ValueError(
                "Nemotron-H hybrid requires both Mamba-2 ('M') and attention ('*') "
                f"blocks, got hybrid_override_pattern={''.join(self.block_pattern)!r}."
            )
        invalid_fields = [
            f"{name}={value!r}"
            for name, value in (
                ("mamba_num_heads", self.mamba_num_heads),
                ("mamba_head_dim", self.mamba_head_dim),
                ("ssm_state_size", self.ssm_state_size),
                ("n_groups", self.n_groups),
                ("conv_kernel", self.conv_kernel),
            )
            if type(value) is not int or value <= 0
        ]
        if invalid_fields:
            raise ValueError(
                "Nemotron-H hybrid model args must be positive integers; invalid "
                f"{', '.join(invalid_fields)}."
            )

    def layer_roles(self) -> tuple[LayerRole, ...]:
        return tuple(_BLOCK_TOKEN_TO_LAYER_ROLE[token] for token in self.block_pattern)

    def state_geometry(self) -> RecurrentStateGeometry:
        return RecurrentStateGeometry(
            conv_kernel_dim=self.conv_kernel,
            # Mamba-2 packs x, B and C into one conv stream.
            conv_dim=self.mamba_num_heads * self.mamba_head_dim
            + 2 * self.n_groups * self.ssm_state_size,
            num_v_heads=self.mamba_num_heads,
            value_head_dim=self.mamba_head_dim,
            key_head_dim=self.ssm_state_size,
        )


NEMOTRON_H_MODEL_TYPES = frozenset({"nemotron_h"})

NEMOTRON_H_FAMILY = StateFamilySpec(
    label="nemotron_h",
    wrapper_cls=Mamba2PagedStateWrapper,
    is_state_module=is_mamba2_mixer,
    mamba_type=MambaAttentionBackendEnum.MAMBA2,
    # Conv tail follows the runtime dtype; match the fp32 SSM state of ssm_update.
    state_dtypes=(None, torch.float32),
    # One private slot per resident request; state is not block-keyed.
    supported_cache_modes=("none",),
    # Full-step path only; not validated on the decode pipeline.
    supports_decode_pipeline=False,
    layer_name="mixer",
    # Mamba-2 keeps the same conv tail and (heads, head_dim, state) pool as GDN.
    create_state_cache=create_gdn_state_cache,
)


def build_nemotron_h_hybrid_plan(
    model_args: Mapping[str, Any], num_layers: int
) -> HybridRuntimePlan:
    """Resolve Nemotron-H block roles and Mamba-2 geometry from model args."""
    nemotron_config = NemotronHHybridConfig.from_model_args(model_args)
    return HybridRuntimePlan(
        layers=HybridLayerPlan(layer_roles=nemotron_config.layer_roles()),
        family=NEMOTRON_H_FAMILY,
        geometry=nemotron_config.state_geometry(),
    )
