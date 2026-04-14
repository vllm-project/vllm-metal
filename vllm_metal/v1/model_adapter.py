# SPDX-License-Identifier: Apache-2.0
"""Model-specific compatibility adapter for MetalModelRunner."""

from __future__ import annotations

from typing import Any, Protocol


class ModelAdapter(Protocol):
    """Model-specific hooks used by runner and cache orchestration."""

    def should_force_text_backbone(self, hf_config: Any) -> bool:
        """Whether a multimodal config should run on the text-only path."""

    def resolve_max_head_dim(
        self, args: dict[str, Any], head_dim: int | None
    ) -> int | None:
        """Resolve the head dimension used for cache sizing."""

    def require_uniform_kv_heads(
        self, args: dict[str, Any], num_kv_heads: int | None
    ) -> None:
        """Raise when paged attention cannot support the model's KV layout."""

    def text_model(self, model: Any) -> Any:
        """Return the callable model used for text-only execution."""


# Models that vLLM flags as multimodal but must be loaded via mlx_lm.
# gemma4: mlx_vlm forward path produces garbled output vs mlx_lm.
# Remove once mlx_vlm gemma4 parity is fixed upstream.
_TEXT_BACKBONE_OVERRIDE_TYPES: frozenset[str] = frozenset({"gemma4"})


class DefaultModelAdapter(ModelAdapter):
    """Default adapter implementation for known model quirks."""

    def should_force_text_backbone(self, hf_config: Any) -> bool:
        """Return True for models that must load via mlx_lm (Gemma4).

        mlx_vlm Gemma4 forward currently produces garbled output; remove this
        override once mlx_vlm Gemma4 parity is fixed upstream.
        """
        model_type = getattr(hf_config, "model_type", "")
        return model_type in _TEXT_BACKBONE_OVERRIDE_TYPES

    def resolve_max_head_dim(
        self, args: dict[str, Any], head_dim: int | None
    ) -> int | None:
        """Handle Gemma4 variable head dims (sliding vs full attention)."""
        global_head_dim = args.get("global_head_dim")
        if global_head_dim and head_dim:
            return max(int(head_dim), int(global_head_dim))
        return head_dim

    def require_uniform_kv_heads(
        self, args: dict[str, Any], num_kv_heads: int | None
    ) -> None:
        """Reject Gemma4 26B/31B variable KV head counts in paged attention.

        Paged KV cache assumes uniform KV head counts across layers. Remove
        once per-layer KV head allocation is supported.
        """
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
        """Return VLM text sub-model to avoid pixel_values/mask requirements."""
        if hasattr(model, "language_model"):
            return model.language_model
        return model
