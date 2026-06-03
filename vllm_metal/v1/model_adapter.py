# SPDX-License-Identifier: Apache-2.0
"""Model-specific compatibility adapter for MetalModelRunner."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, cast

import mlx.core as mx
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import ModelConfig

    from vllm_metal.multimodal.feature_spec import MultiModalFeatureSpec

logger = init_logger(__name__)


@dataclass(frozen=True, slots=True)
class TargetModelForwardOutput:
    """Target-model forward output needed by sampling and speculative decode."""

    logits: mx.array
    hidden_states: mx.array | None = None


class MultimodalEncodeResult(Protocol):
    """Adapter-owned vision output for one multimodal feature."""

    hidden_states: mx.array
    deepstack_visual_embeds: Sequence[mx.array] | None


class MultimodalRuntimeAdapter(Protocol):
    """Model-owned behavior needed for native multimodal execution."""

    forward_ready: bool
    """Whether scheduled multimodal encoder inputs may be executed.

    ``MetalModelRunner`` fails fast on scheduled encoder inputs when this
    is ``False``, so a new adapter can land closed without disturbing the
    model already in production.
    """

    def text_model(self) -> Any:
        """Return the callable language model for text-only VLM execution."""

    def embed_tokens(self, input_ids: Any) -> Any:
        """Return the language model's input embeddings for ``input_ids``.

        Returns an MLX array of shape ``(batch, seq_len, hidden)``.
        """

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[MultiModalFeatureSpec],
    ) -> tuple[Any, int]:
        """Return ``((3, 1, seq_len) positions, mrope_position_delta)``.

        The runner stashes ``mrope_position_delta`` on ``RequestState`` so
        decode positions stay correct.
        """

    def encode_multimodal(
        self,
        features: list[MultiModalFeatureSpec],
    ) -> Sequence[MultimodalEncodeResult]:
        """Run the model's vision tower; return one result per feature.

        Results may carry optional deepstack visual embeds for models whose
        language model injects vision residuals in intermediate layers.
        """

    def call_lm(
        self,
        input_ids: Any,
        inputs_embeds: Any,
        cache: list[Any],
        position_ids: Any,
        *,
        visual_pos_masks: Any | None = None,
        deepstack_visual_embeds: Any | None = None,
    ) -> Any:
        """Invoke the language model with runner-built embeds and positions.

        ``visual_pos_masks`` and ``deepstack_visual_embeds`` are forwarded
        only when the language model declares both as explicit parameters;
        adapters omit them otherwise to avoid silently dropping the arrays
        into a ``**kwargs`` catch-all.
        """


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

    def target_forward(
        self,
        model: Any,
        input_ids: mx.array,
        *,
        cache: Any | None = None,
        collect_hidden_states: bool = False,
    ) -> TargetModelForwardOutput:
        """Run the target text model and optionally retain target hidden states."""

    def target_input_embeddings(self, model: Any, input_ids: mx.array) -> mx.array:
        """Return target/backbone-dim token embeddings for ``input_ids``."""

    def extract_logits(self, model_output: Any) -> mx.array:
        """Extract logits from raw model output."""

    def build_multimodal_adapter(
        self, model: Any, hf_config: Any
    ) -> MultimodalRuntimeAdapter | None:
        """Return a model-owned multimodal adapter for native VLM execution."""

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
        "Qwen3_6ForConditionalGeneration",
        "Qwen3_6MoeForConditionalGeneration",
    }
)
_QWEN3_VL_MODEL_TYPES: frozenset[str] = frozenset({"qwen3_5", "qwen3_vl"})
_QWEN3_VL_ARCHITECTURES: frozenset[str] = frozenset(
    {
        "Qwen3_5ForConditionalGeneration",
        "Qwen3VLForConditionalGeneration",
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

        ``multimodal-native`` disables compatibility overrides. ``auto`` and
        ``text-only-compat`` share the same compatibility allowlist.
        """
        multimodal_mode = self._multimodal_mode()
        if multimodal_mode == "multimodal-native":
            return False
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

    def target_forward(
        self,
        model: Any,
        input_ids: mx.array,
        *,
        cache: Any | None = None,
        collect_hidden_states: bool = False,
    ) -> TargetModelForwardOutput:
        """Run the target model and return logits plus optional hidden states."""
        if not collect_hidden_states:
            output = model(input_ids, cache=cache)
            return TargetModelForwardOutput(
                logits=self.extract_logits(output),
                hidden_states=None,
            )

        hidden_states = self._forward_target_hidden_states(
            model,
            input_ids,
            cache=cache,
        )
        logits = self._compute_target_logits(model, hidden_states)
        return TargetModelForwardOutput(
            logits=logits,
            hidden_states=self._flatten_target_hidden_states(hidden_states),
        )

    def target_input_embeddings(self, model: Any, input_ids: mx.array) -> mx.array:
        """Return target/backbone-dim token embeddings for Gemma4 MTP feedback."""
        backbone = self._target_backbone(model)
        embed_tokens = getattr(backbone, "embed_tokens", None)
        if embed_tokens is None:
            raise NotImplementedError(
                "Gemma4 MTP assistant drafting requires a target text model "
                "with `model.embed_tokens` so sampled target tokens can be "
                "embedded in backbone hidden size."
            )
        embeddings = embed_tokens(input_ids)
        embed_scale = getattr(backbone, "embed_scale", 1)
        return embeddings * embed_scale

    def extract_logits(self, model_output: Any) -> mx.array:
        """Extract logits from mlx-lm arrays or mlx-vlm model outputs."""
        if isinstance(model_output, TargetModelForwardOutput):
            return model_output.logits
        if hasattr(model_output, "logits"):
            return model_output.logits
        return model_output

    def _forward_target_hidden_states(
        self,
        model: Any,
        input_ids: mx.array,
        *,
        cache: Any | None,
    ) -> mx.array:
        backbone = self._target_backbone(model)
        if backbone is None:
            raise NotImplementedError(
                "Target hidden states require a text model with a `model` "
                "backbone; this model output does not expose hidden states."
            )
        return backbone(input_ids, cache=cache)

    def _compute_target_logits(self, model: Any, hidden_states: mx.array) -> mx.array:
        text_model = self.text_model(model)
        if hasattr(text_model, "compute_logits"):
            return text_model.compute_logits(hidden_states)

        args = getattr(text_model, "args", None)
        tie_word_embeddings = bool(
            getattr(text_model, "tie_word_embeddings", False)
            or getattr(args, "tie_word_embeddings", False)
        )
        if tie_word_embeddings:
            logits = self._compute_tied_embedding_logits(text_model, hidden_states)
            return self._apply_target_logit_postprocessing(text_model, logits)

        lm_head = getattr(text_model, "lm_head", None)
        if lm_head is not None:
            logits = lm_head(hidden_states)
            return self._apply_target_logit_postprocessing(text_model, logits)

        # Some Gemma4-compatible MLX loaders expose tied embedding projection
        # without setting tie_word_embeddings on the wrapper; try that shape
        # explicitly, then fail inside _compute_tied_embedding_logits if absent.
        logits = self._compute_tied_embedding_logits(text_model, hidden_states)
        return self._apply_target_logit_postprocessing(text_model, logits)

    def _compute_tied_embedding_logits(
        self, model: Any, hidden_states: mx.array
    ) -> mx.array:
        backbone = self._target_backbone(model)
        embed_tokens = getattr(backbone, "embed_tokens", None)
        as_linear = getattr(embed_tokens, "as_linear", None)
        if as_linear is None:
            raise NotImplementedError(
                "Target hidden states require either `lm_head` or tied "
                "`model.embed_tokens.as_linear` logits projection."
            )
        return as_linear(hidden_states)

    def _flatten_target_hidden_states(self, hidden_states: mx.array) -> mx.array:
        ndim = len(hidden_states.shape)
        if ndim == 2:
            return hidden_states
        if ndim == 3 and hidden_states.shape[0] == 1:
            return hidden_states[0]
        raise ValueError(
            "Target hidden states must be row-major [num_tokens, hidden_size] "
            "or single-batch [1, num_tokens, hidden_size]."
        )

    def _apply_target_logit_postprocessing(
        self, model: Any, logits: mx.array
    ) -> mx.array:
        softcap = getattr(model, "final_logit_softcapping", None)
        if softcap is None:
            return logits
        return mx.tanh(logits / softcap) * softcap

    def _target_backbone(self, model: Any) -> Any | None:
        return getattr(self.text_model(model), "model", None)

    def build_multimodal_adapter(
        self, model: Any, hf_config: Any
    ) -> MultimodalRuntimeAdapter | None:
        """Build the native multimodal adapter for supported model families."""
        if hf_config is None:
            return None

        model_type = getattr(hf_config, "model_type", "")
        architectures = getattr(hf_config, "architectures", ()) or ()
        if model_type not in _QWEN3_VL_MODEL_TYPES and not any(
            arch in _QWEN3_VL_ARCHITECTURES for arch in architectures
        ):
            return None

        from vllm_metal.multimodal.qwen3_vl import Qwen3VLMultimodalAdapter

        return cast(
            MultimodalRuntimeAdapter,
            Qwen3VLMultimodalAdapter.from_loaded_model(model),
        )

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
