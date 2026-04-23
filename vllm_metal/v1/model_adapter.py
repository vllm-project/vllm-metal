# SPDX-License-Identifier: Apache-2.0
"""Model-specific compatibility adapter for MetalModelRunner."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import ModelConfig

logger = init_logger(__name__)


class ModelAdapter(Protocol):
    """Model-specific hooks used by runner and cache orchestration."""

    def should_force_text_backbone(self, hf_config: Any) -> bool:
        """Whether a multimodal config should run on the text-only path."""

    def normalize_model_config(self, model_config: ModelConfig) -> None:
        """Apply model-specific normalisations to ``model_config`` in place.

        Called early during platform setup so the engine sees a consistent
        view of the model before constructing input processors, etc.
        """

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

    def build_yoco_cache_mapping(
        self, args: dict[str, Any]
    ) -> tuple[int, dict[int, int]] | None:
        """Build YOCO layer→cache_idx mapping, or None if not applicable."""

    def build_per_layer_kv_shapes(
        self,
        args: dict[str, Any],
        *,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> tuple[list[int], list[int]] | None:
        """Return per-layer ``(kv_heads, head_dim)`` lists, or None for uniform."""

    def build_sliding_window_per_layer(
        self, args: dict[str, Any], num_layers: int
    ) -> list[int] | None:
        """Return per-layer sliding window sizes, or None for no enforcement."""


# Models/configs that vLLM flags as multimodal but must be loaded via mlx_lm.
# gemma4: mlx_vlm forward path produces garbled output vs mlx_lm.
_TEXT_BACKBONE_OVERRIDE_TYPES: frozenset[str] = frozenset({"gemma4"})
# Qwen3.5/Qwen3.6 conditional-generation wrappers expose a multimodal config,
# but vllm-metal only serves them in text-only mode. Their FP8 checkpoints ship
# `*_weight_scale_inv` tensors that the mlx_vlm qwen3_5 loader does not
# currently sanitize, while mlx_lm's qwen3_5 text loader handles them.
_TEXT_BACKBONE_OVERRIDE_ARCHITECTURES: frozenset[str] = frozenset(
    {
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
    }
)


class DefaultModelAdapter(ModelAdapter):
    """Default adapter implementation for known model quirks."""

    def _multimodal_mode(self) -> str:
        from vllm_metal.config import get_config

        return get_config().multimodal_mode

    def _matches_auto_text_backbone_override(self, hf_config: Any) -> bool:
        """Return True for known multimodal checkpoints that need mlx_lm.

        Gemma4: mlx_vlm forward currently produces garbled output; remove this
        override once mlx_vlm Gemma4 parity is fixed upstream.

        Qwen3.5/Qwen3.6 conditional-generation wrappers: these configs are
        marked multimodal even when served text-only. Route them through
        mlx_lm's qwen3_5 text loader so FP8 `*_weight_scale_inv` tensors are
        consumed correctly instead of failing inside mlx_vlm.load().
        """
        if hf_config is None:
            return False

        model_type = getattr(hf_config, "model_type", "")
        if model_type in _TEXT_BACKBONE_OVERRIDE_TYPES:
            return True

        architectures = getattr(hf_config, "architectures", ()) or ()
        if not any(
            arch in _TEXT_BACKBONE_OVERRIDE_ARCHITECTURES for arch in architectures
        ):
            return False

        quantization_config = getattr(hf_config, "quantization_config", None)
        if isinstance(quantization_config, dict):
            quant_method = quantization_config.get("quant_method")
        else:
            quant_method = getattr(quantization_config, "quant_method", None)
        return quant_method == "fp8"

    def should_force_text_backbone(self, hf_config: Any) -> bool:
        """Whether the current serve mode should use the text-only path.

        Modes:
        - ``multimodal-native``: never force compatibility; keep native VLM
          loading active so multimodal support can be developed/tested.
        - ``text-only-compat``: force the text-only path for multimodal models.
        - ``auto``: apply the compatibility path only for known-incompatible
          checkpoints such as Gemma4 and Qwen3.5/Qwen3.6 FP8 wrappers.
        """
        multimodal_mode = self._multimodal_mode()
        if multimodal_mode == "multimodal-native":
            return False
        if multimodal_mode == "text-only-compat":
            return True
        return self._matches_auto_text_backbone_override(hf_config)

    def normalize_model_config(self, model_config: ModelConfig) -> None:
        """Clear ``multimodal_config`` for models served on the text backbone.

        When the active serve mode routes a multimodal checkpoint through the
        text-only compatibility path, leaving ``multimodal_config`` populated
        causes vLLM to eagerly initialize multimodal processors that the
        compatibility path intentionally bypasses. Clearing it here makes
        ``is_multimodal_model`` ``False`` so the input processor skips that
        setup. The ``should_force_text_backbone`` predicate is the single
        source of truth for whether the compatibility path applies.
        """
        if model_config.multimodal_config is None:
            return
        hf_config = getattr(model_config, "hf_config", None)
        if not self.should_force_text_backbone(hf_config):
            return

        multimodal_mode = self._multimodal_mode()
        model_config.multimodal_config = None
        logger.info(
            "Metal: forcing text-only backbone for model_type=%s "
            "(multimodal_mode=%s, cleared multimodal_config)",
            getattr(hf_config, "model_type", "unknown"),
            multimodal_mode,
        )

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
        """Reject configs with mismatched KV head counts under the uniform path.

        Called from :meth:`vllm_metal.v1.cache_policy.ModelCachePolicy.\
validate_paged_attention_support` only when ``kv_heads_per_layer`` has
        NOT been populated.  Models whose adapter populates per-layer shapes
        via :meth:`build_per_layer_kv_shapes` (Gemma4 26B/31B) handle
        mismatched KV counts layer-by-layer and skip this check.  Any other
        config with ``num_global_key_value_heads != num_key_value_heads``
        silently falls back to the scalar uniform path with wrong cache
        sizing, so fail fast here instead.
        """
        global_kv_heads = args.get("num_global_key_value_heads")
        if (
            global_kv_heads
            and num_kv_heads
            and int(global_kv_heads) != int(num_kv_heads)
        ):
            raise ValueError(
                f"Paged attention does not support variable KV head count "
                f"without per-layer shape support: "
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

    def build_yoco_cache_mapping(
        self, args: dict[str, Any]
    ) -> tuple[int, dict[int, int]] | None:
        """Build the layer→cache_idx mapping for YOCO KV sharing.

        Gemma4's "You Only Cache Once" architecture only caches K/V for
        the first ``N - num_kv_shared_layers`` layers.  Shared layers
        reuse the cache of the most recent unique layer of the same
        attention type (sliding or full).

        Follows the same logic as mlx_lm's ``Gemma4TextModel.previous_kvs``
        mapping.

        Returns:
            ``(num_unique_cache_layers, {layer_idx: cache_idx})`` or
            ``None`` if the model does not use KV sharing.
        """
        num_layers = args.get("num_hidden_layers", 0)
        num_shared = args.get("num_kv_shared_layers", 0)
        if not num_shared or not num_layers:
            return None

        layer_types: list[str] = args.get("layer_types", [])
        if len(layer_types) != num_layers:
            return None

        num_unique = num_layers - num_shared

        # Map each attention type to the LAST unique layer of that type,
        # matching mlx_lm's ``kvs_by_type`` logic.
        type_to_cache_idx: dict[str, int] = {}
        for i in range(num_unique):
            type_to_cache_idx[layer_types[i]] = i

        mapping: dict[int, int] = {}
        for i in range(num_layers):
            if i < num_unique:
                mapping[i] = i
            else:
                mapping[i] = type_to_cache_idx[layer_types[i]]

        return num_unique, mapping

    def build_per_layer_kv_shapes(
        self,
        args: dict[str, Any],
        *,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> tuple[list[int], list[int]] | None:
        """Return per-layer ``(kv_heads, head_dim)`` lists for Gemma4, else None.

        Gemma4 26B/31B mix sliding attention (``num_key_value_heads``,
        ``head_dim``) with full attention (``num_global_key_value_heads``,
        ``global_head_dim``), exposed via ``layer_types``.  Other models use
        a uniform KV shape across every layer, in which case this returns
        ``None`` and the cache path falls back to the scalar
        ``num_kv_heads`` / ``head_dim`` fields on the runner.

        Edge case: some Gemma4 checkpoints (e.g. ``gemma-4-E2B``) override
        only ``global_head_dim`` while reusing the sliding-layer KV-head
        count for full-attention layers.  In that case
        ``num_global_key_value_heads`` is absent and the full-attention
        KV-head count falls back to ``num_kv_heads`` — collapsing to a
        uniform layout would cause full-attention layers to write into
        under-sized cache slots.

        Args:
            args: Flattened model-config mapping.
            num_layers: Total number of transformer layers.
            num_kv_heads: Resolved sliding-layer KV-head count.
            head_dim: Resolved sliding-layer head_dim (pre max-with-global).

        Returns:
            ``(kv_heads_per_layer, head_dim_per_layer)`` of length
            ``num_layers``, or ``None`` when the model is uniform.

        Raises:
            ValueError: If a ``layer_types`` entry is neither
                ``"sliding_attention"`` nor ``"full_attention"``.  Unknown
                types surface loudly here instead of silently falling back
                to full-attention shapes.
        """
        layer_types = args.get("layer_types", [])
        global_head_dim = args.get("global_head_dim")
        if len(layer_types) != num_layers or not global_head_dim:
            return None

        global_kv_heads = args.get("num_global_key_value_heads")
        full_kv_heads = (
            int(global_kv_heads) if global_kv_heads is not None else int(num_kv_heads)
        )
        full_head_dim = int(global_head_dim)
        sliding_kv_heads = int(num_kv_heads)
        sliding_head_dim = int(head_dim)

        kv_heads_per_layer: list[int] = []
        head_dim_per_layer: list[int] = []
        for i, layer_type in enumerate(layer_types):
            if layer_type == "sliding_attention":
                kv_heads_per_layer.append(sliding_kv_heads)
                head_dim_per_layer.append(sliding_head_dim)
            elif layer_type == "full_attention":
                kv_heads_per_layer.append(full_kv_heads)
                head_dim_per_layer.append(full_head_dim)
            else:
                raise ValueError(
                    f"Unsupported Gemma4 layer_type at index {i}: "
                    f"{layer_type!r}.  Expected one of "
                    f"{{'sliding_attention', 'full_attention'}}."
                )
        return kv_heads_per_layer, head_dim_per_layer

    def build_sliding_window_per_layer(
        self, args: dict[str, Any], num_layers: int
    ) -> list[int] | None:
        """Return per-layer sliding window sizes for Gemma4, else None.

        Gemma4 sliding-attention layers enforce a local window
        (``config.sliding_window``); full-attention layers attend to the
        entire context (represented as ``-1``).  Models without
        ``layer_types`` or ``sliding_window`` in their config return
        ``None``, keeping the current disabled-everywhere behavior.
        """
        layer_types: list[str] = args.get("layer_types", [])
        sliding_window = args.get("sliding_window")
        if len(layer_types) != num_layers or not sliding_window:
            return None

        sw = int(sliding_window)
        return [sw if lt == "sliding_attention" else -1 for lt in layer_types]
