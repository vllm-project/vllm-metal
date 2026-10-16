# SPDX-License-Identifier: Apache-2.0
"""Pooling request options Metal cannot serve, whatever the model.

Kept free of MLX imports so ``MetalPlatform.validate_request`` can reject
these options before a request reaches the engine.
"""

from __future__ import annotations

from vllm.pooling_params import PoolingParams

_CLASSIFY_TASKS = ("classify", "token_classify")


def unsupported_pooling_option(
    pooling_params: PoolingParams,
) -> tuple[str, str] | None:
    """Return ``(parameter, description)`` for the first unsupported option."""
    if pooling_params.late_interaction_params is not None:
        return "late_interaction_params", "late-interaction parameters"
    if pooling_params.requires_token_ids and pooling_params.task != "token_classify":
        return "requires_token_ids", "token-level ALL pooling outputs"
    if pooling_params.step_tag_id is not None:
        return "step_tag_id", "STEP pooling parameters"
    if pooling_params.returned_token_ids is not None:
        return "returned_token_ids", "returned_token_ids"
    if pooling_params.extra_kwargs:
        return "extra_kwargs", "extra pooling kwargs"
    if (
        pooling_params.task not in _CLASSIFY_TASKS
        and pooling_params.use_activation is False
    ):
        return "use_activation", "use_activation=False"
    return None
