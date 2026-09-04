# SPDX-License-Identifier: Apache-2.0
"""Factory for hybrid state-family runtime plans."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from vllm_metal.attention.runtime.families.gdn import (
    build_gdn_hybrid_plan,
    supports_gdn_hybrid,
)
from vllm_metal.attention.runtime.hybrid_plan import HybridRuntimePlan


@dataclass(frozen=True, slots=True)
class StateFamilyPlanBuilder:
    supports: Callable[[Mapping[str, Any]], bool]
    build: Callable[[Mapping[str, Any], int], HybridRuntimePlan]


_STATE_FAMILY_PLAN_BUILDERS = (
    # ``ModelConfig.is_hybrid`` only says a model mixes attention and state
    # layers; the family that owns its topology and geometry is resolved here.
    StateFamilyPlanBuilder(
        supports=supports_gdn_hybrid,
        build=build_gdn_hybrid_plan,
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
