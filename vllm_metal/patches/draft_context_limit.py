# SPDX-License-Identifier: Apache-2.0
"""Keep the target context when MetalModelRunner serves a shorter draft model."""

from functools import wraps
from typing import Any


def _metal_runner_falls_back(vllm_config: Any) -> bool:
    from vllm.platforms import current_platform

    from vllm_metal.platform import MetalPlatform

    speculative_config = vllm_config.speculative_config
    return (
        isinstance(current_platform, MetalPlatform)
        and speculative_config is not None
        and speculative_config.method == "draft_model"
    )


def register() -> None:
    """Skip vLLM's clamp of max_model_len to a shorter draft model on Metal.

    vLLM 0.33 lowers max_model_len to the draft's because its V2 GPU runner
    cannot skip speculation past the draft's context. MetalModelRunner decodes
    those requests without drafts (#850), so the target keeps its context.
    Remove when vLLM lets the runner declare that it skips speculation there.
    Installed as a general plugin: the clamp runs in VllmConfig.__post_init__,
    before the platform config hook.
    """
    from vllm.config import VllmConfig

    original = VllmConfig._limit_max_model_len_to_draft
    if getattr(original, "_metal_keeps_target_context", False):
        return

    @wraps(original)
    def limit_max_model_len_to_draft(self: Any) -> None:
        if not _metal_runner_falls_back(self):
            original(self)

    limit_max_model_len_to_draft._metal_keeps_target_context = True  # type: ignore[attr-defined]
    VllmConfig._limit_max_model_len_to_draft = limit_max_model_len_to_draft
