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
from vllm_metal.attention.runtime.families.shortconv import (
    SHORTCONV_FAMILY,
    SHORTCONV_MODEL_TYPES,
    build_shortconv_hybrid_plan,
)
from vllm_metal.attention.runtime.hybrid_plan import HybridRuntimePlan, StateFamilySpec


@dataclass(frozen=True, slots=True)
class StateFamilyPlanBuilder:
    model_types: frozenset[str]
    family: StateFamilySpec
    build: Callable[[Mapping[str, Any], int], HybridRuntimePlan]


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
    StateFamilyPlanBuilder(
        model_types=SHORTCONV_MODEL_TYPES,
        family=SHORTCONV_FAMILY,
        build=build_shortconv_hybrid_plan,
    ),
)


def _builder_for_model_type(model_type: str) -> StateFamilyPlanBuilder:
    for builder in _STATE_FAMILY_PLAN_BUILDERS:
        if model_type in builder.model_types:
            return builder
    raise NotImplementedError(
        f"Metal hybrid runtime has no state family for model_type={model_type!r}."
    )


def state_family_for_model_type(model_type: str) -> StateFamilySpec:
    """Return the family owning ``model_type`` before any model is built."""
    return _builder_for_model_type(model_type).family


def build_hybrid_runtime_plan(
    model_args: Mapping[str, Any], num_layers: int
) -> HybridRuntimePlan:
    builder = _builder_for_model_type(model_args["model_type"])
    return builder.build(model_args, num_layers)
