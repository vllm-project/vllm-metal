# SPDX-License-Identifier: Apache-2.0
"""Factory for hybrid state-family runtime plans."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from vllm_metal.attention.runtime.families.gdn import (
    GDN_FAMILY,
    GDN_MODEL_TYPES,
    build_gdn_hybrid_plan,
)
from vllm_metal.attention.runtime.families.nemotron_h import (
    NEMOTRON_H_FAMILY,
    NEMOTRON_H_MODEL_TYPES,
    build_nemotron_h_hybrid_plan,
)
from vllm_metal.attention.runtime.hybrid_plan import HybridRuntimePlan, StateFamilySpec


@dataclass(frozen=True, slots=True)
class StateFamilyPlanBuilder:
    model_types: frozenset[str]
    family: StateFamilySpec
    build: Callable[[Mapping[str, Any], int], HybridRuntimePlan]

    def supports(self, model_args: Mapping[str, Any]) -> bool:
        return model_args.get("model_type") in self.model_types


_STATE_FAMILY_PLAN_BUILDERS = (
    # ``ModelConfig.is_hybrid`` only says a model mixes attention and state
    # layers; the family that owns its topology and geometry is resolved here.
    StateFamilyPlanBuilder(
        model_types=GDN_MODEL_TYPES,
        family=GDN_FAMILY,
        build=build_gdn_hybrid_plan,
    ),
    StateFamilyPlanBuilder(
        model_types=NEMOTRON_H_MODEL_TYPES,
        family=NEMOTRON_H_FAMILY,
        build=build_nemotron_h_hybrid_plan,
    ),
)


def build_hybrid_runtime_plan(
    model_args: Mapping[str, Any], num_layers: int
) -> HybridRuntimePlan:
    for builder in _STATE_FAMILY_PLAN_BUILDERS:
        if builder.supports(model_args):
            return builder.build(model_args, num_layers)

    raise NotImplementedError(
        f"Metal hybrid runtime has no state family for "
        f"model_type={model_args.get('model_type')!r}."
    )
