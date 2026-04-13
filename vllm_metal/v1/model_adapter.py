# SPDX-License-Identifier: Apache-2.0
"""Model-specific compatibility adapter for MetalModelRunner."""

from __future__ import annotations

from typing import Any, Protocol


class ModelAdapter(Protocol):
    """Adapter interface for model-specific behavior."""

    def should_force_text_backbone(self, hf_config: Any) -> bool:
        """Return True if the model should load via mlx_lm despite multimodal flags."""

    def resolve_max_head_dim(
        self, args: dict[str, Any], head_dim: int | None
    ) -> int | None:
        """Return the maximum head_dim across attention layer types."""

    def require_uniform_kv_heads(
        self, args: dict[str, Any], num_kv_heads: int | None
    ) -> None:
        """Validate that KV head count is uniform across layer types."""

    def text_model(self, model: Any) -> Any:
        """Return the text sub-model for a VLM, or the model itself."""


# Models that vLLM flags as multimodal but must be loaded via mlx_lm.
_TEXT_BACKBONE_OVERRIDE_TYPES: frozenset[str] = frozenset({"gemma4"})


class DefaultModelAdapter:
    """Default adapter implementation for known model quirks."""

    def should_force_text_backbone(self, hf_config: Any) -> bool:
        model_type = getattr(hf_config, "model_type", "")
        return model_type in _TEXT_BACKBONE_OVERRIDE_TYPES

    def resolve_max_head_dim(
        self, args: dict[str, Any], head_dim: int | None
    ) -> int | None:
        global_head_dim = args.get("global_head_dim")
        if global_head_dim and head_dim:
            return max(int(head_dim), int(global_head_dim))
        return head_dim

    def require_uniform_kv_heads(
        self, args: dict[str, Any], num_kv_heads: int | None
    ) -> None:
        global_kv_heads = args.get("num_global_key_value_heads")
        if (
            global_kv_heads
            and num_kv_heads
            and int(global_kv_heads) != int(num_kv_heads)
        ):
            raise ValueError(
                f"Paged attention does not support variable KV head count: "
                f"num_key_value_heads={num_kv_heads}, "
                f"num_global_key_value_heads={global_kv_heads}. "
                f"Use VLLM_METAL_USE_PAGED_ATTENTION=0 to fall back to the "
                f"non-paged path."
            )

    def text_model(self, model: Any) -> Any:
        if hasattr(model, "language_model"):
            return model.language_model
        return model
