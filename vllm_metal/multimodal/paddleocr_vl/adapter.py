# SPDX-License-Identifier: Apache-2.0
"""PaddleOCR-VL multimodal adapter for vLLM Metal."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import torch
from vllm.multimodal.inputs import MultiModalKwargsItem

from vllm_metal.multimodal.feature_spec import MultiModalFeatureSpec
from vllm_metal.multimodal.text_backbone import (
    resolve_backbone_embed_tokens,
    resolve_text_backbone,
)
from vllm_metal.tensor_bridge import torch_to_mlx


@dataclass(frozen=True)
class PaddleOCRVLVisionEncodeResult:
    """Vision tower output for one PaddleOCR-VL multimodal feature."""

    hidden_states: mx.array
    deepstack_visual_embeds: None = None


class PaddleOCRVLMultimodalAdapter:
    """Model-owned multimodal helpers for PaddleOCR-VL execution."""

    forward_ready: bool = True

    requires_explicit_positions: bool = True
    """The LM precomputes (cos, sin) from model-level position state; the
    paged text path would re-derive positions from zero-offset caches
    (arange from 0), so every batch — text-only included — must carry
    runner-built ``position_ids`` via the multimodal forward."""

    text_path_selective_logits_ok: bool = True
    """``text_model()`` is the very object the runner profiles for the
    split backbone/head path."""

    def __init__(
        self,
        *,
        spatial_merge_size: int,
        visual: Any | None = None,
        language_model: Any | None = None,
        embed_tokens_fn: Any | None = None,
        backbone: Any | None = None,
    ) -> None:
        if spatial_merge_size <= 0:
            raise ValueError(
                f"spatial_merge_size must be positive, got {spatial_merge_size}"
            )
        self._spatial_merge_size = spatial_merge_size
        self._visual = visual
        self._language_model = language_model
        self._embed_tokens_fn = embed_tokens_fn
        self._backbone = backbone

    @classmethod
    def from_loaded_model(cls, model: Any) -> PaddleOCRVLMultimodalAdapter:
        """Create an adapter from an mlx-vlm PaddleOCR-VL composite."""
        visual = model.visual
        language_model = model.language_model
        spatial_merge_size = int(model.config.vision_config.spatial_merge_size)
        backbone = resolve_text_backbone(language_model, owner="mlx_vlm")
        if not callable(backbone):
            raise RuntimeError(
                "language_model.model is not callable; mlx_vlm version "
                "drift detected. Expected the PaddleOCR text backbone."
            )
        embed_tokens_fn = cls._resolve_embed_tokens(language_model)
        return cls(
            spatial_merge_size=spatial_merge_size,
            visual=visual,
            language_model=language_model,
            embed_tokens_fn=embed_tokens_fn,
            backbone=backbone,
        )

    @classmethod
    def _resolve_embed_tokens(cls, language_model: Any) -> Any:
        return resolve_backbone_embed_tokens(
            resolve_text_backbone(language_model, owner="mlx_vlm"), owner="mlx_vlm"
        )

    def text_model(self) -> Any:
        return self._language_model

    def embed_tokens(self, input_ids: mx.array) -> mx.array:
        if self._embed_tokens_fn is None:
            raise RuntimeError(
                "embed_tokens_fn not resolved; construct via "
                "PaddleOCRVLMultimodalAdapter.from_loaded_model."
            )
        return self._embed_tokens_fn(input_ids)

    def encode_multimodal(
        self,
        features: list[MultiModalFeatureSpec],
    ) -> list[PaddleOCRVLVisionEncodeResult]:
        """Run PaddleOCR-VL's vision tower for each image feature."""
        if self._visual is None:
            raise RuntimeError(
                "visual tower not loaded; encode_multimodal unavailable. "
                "Construct via PaddleOCRVLMultimodalAdapter.from_loaded_model."
            )

        target_dtype = self._visual.embeddings.patch_embedding.weight.dtype
        outputs: list[PaddleOCRVLVisionEncodeResult] = []
        for feature in features:
            if feature.modality != "image":
                raise ValueError(
                    f"Feature {feature.identifier!r}: encode_multimodal only "
                    "supports image features; got "
                    f"modality={feature.modality!r}."
                )
            if feature.data is None:
                raise ValueError(
                    f"Feature {feature.identifier!r}: feature.data is required "
                    "for vision encoding."
                )

            pixel_values = self._as_mlx(
                self._feature_value(feature.data, "pixel_values")
            )
            if len(pixel_values.shape) == 4:
                pixel_values = mx.expand_dims(pixel_values, axis=0)
            if len(pixel_values.shape) != 5:
                raise ValueError(
                    f"Feature {feature.identifier!r}: pixel_values must have "
                    "shape (patches, 3, patch, patch) "
                    "or (1, patches, 3, patch, patch); got "
                    f"{pixel_values.shape}"
                )

            try:
                grid_thw = self._grid_thw(feature.data, "image_grid_thw")
            except ValueError as exc:
                raise ValueError(f"Feature {feature.identifier!r}: {exc}") from exc
            raw_patches = grid_thw[0] * grid_thw[1] * grid_thw[2]
            if int(pixel_values.shape[1]) != raw_patches:
                raise ValueError(
                    f"Feature {feature.identifier!r}: pixel_values patch "
                    "count does not match image_grid_thw: "
                    f"got {pixel_values.shape[1]}, expected {raw_patches}"
                )
            image_grid_thw = mx.array([grid_thw], dtype=mx.int32)

            hidden_states = self._visual(
                pixel_values.astype(target_dtype),
                image_grid_thw,
                output_hidden_states=False,
            )
            outputs.append(PaddleOCRVLVisionEncodeResult(hidden_states=hidden_states))

        return outputs

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[MultiModalFeatureSpec],
    ) -> tuple[mx.array, int]:
        """Return ``((3, 1, seq_len) int32 positions, mrope_position_delta)``."""
        if not input_tokens:
            return mx.zeros((3, 1, 0), dtype=mx.int32), 0
        if self._language_model is None:
            raise RuntimeError(
                "language_model not loaded; get_mrope_input_positions unavailable. "
                "Construct via PaddleOCRVLMultimodalAdapter.from_loaded_model."
            )

        sorted_features = sorted(mm_features, key=lambda f: f.mm_position.offset)
        image_grid_thw_values = self._validate_image_features_and_collect_grids(
            sorted_features
        )
        image_grid_thw = (
            mx.array(image_grid_thw_values, dtype=mx.int32)
            if image_grid_thw_values
            else None
        )
        input_ids = mx.array([input_tokens], dtype=mx.int32)
        position_ids, rope_deltas = self._language_model.get_rope_index(
            input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=None,
            attention_mask=None,
        )
        return position_ids.astype(mx.int32), self._first_scalar_int(rope_deltas)

    def call_lm(
        self,
        input_ids: mx.array,
        inputs_embeds: mx.array,
        cache: list[Any],
        position_ids: mx.array,
        *,
        visual_pos_masks: Any | None = None,
        deepstack_visual_embeds: Any | None = None,
    ) -> Any:
        """Invoke the PaddleOCR-VL language model with runner-built embeds."""
        del visual_pos_masks
        self._check_lm_ready("call_lm", deepstack_visual_embeds)
        return self._language_model(
            input_ids,
            inputs_embeds=inputs_embeds,
            cache=cache,
            position_ids=position_ids,
        )

    def _check_lm_ready(self, call: str, deepstack_visual_embeds: Any | None) -> None:
        """Guard shared by ``call_lm`` and ``call_lm_hidden_states``."""
        if self._language_model is None:
            raise RuntimeError(
                f"language_model not loaded; {call} unavailable. "
                "Construct via PaddleOCRVLMultimodalAdapter.from_loaded_model."
            )
        if deepstack_visual_embeds is not None:
            raise RuntimeError(
                "PaddleOCR-VL does not expose deepstack visual residuals."
            )

    def call_lm_hidden_states(
        self,
        input_ids: mx.array,
        inputs_embeds: mx.array,
        cache: list[Any],
        position_ids: mx.array,
        *,
        visual_pos_masks: Any | None = None,
        deepstack_visual_embeds: Any | None = None,
    ) -> mx.array:
        """:meth:`call_lm` without the output head: the backbone's final hidden states."""
        del visual_pos_masks
        self._check_lm_ready("call_lm_hidden_states", deepstack_visual_embeds)
        if self._backbone is None or not callable(self._backbone):
            raise RuntimeError(
                "language_model.model attribute missing or not callable; "
                "mlx_vlm version drift detected. Expected the PaddleOCR "
                "text backbone."
            )
        return self._backbone(
            input_ids,
            inputs_embeds=inputs_embeds,
            cache=cache,
            position_ids=position_ids,
        )

    def _validate_image_features_and_collect_grids(
        self,
        features: list[MultiModalFeatureSpec],
    ) -> list[tuple[int, int, int]]:
        image_grid_thw_values: list[tuple[int, int, int]] = []
        for feature in features:
            modality = feature.modality
            if modality == "video":
                raise NotImplementedError(
                    f"Feature {feature.identifier!r}: Video multimodal "
                    "features are not yet supported."
                )
            if modality != "image":
                raise ValueError(
                    f"Feature {feature.identifier!r}: Unsupported modality: {modality}"
                )
            if feature.data is None:
                raise ValueError(
                    f"Feature {feature.identifier!r}: Image feature data is "
                    "required to read image_grid_thw."
                )

            try:
                t, h, w = self._grid_thw(feature.data, "image_grid_thw")
            except ValueError as exc:
                raise ValueError(f"Feature {feature.identifier!r}: {exc}") from exc
            if t != 1:
                raise ValueError(
                    f"Feature {feature.identifier!r}: Multi-frame images are "
                    f"not yet supported, got t={t}"
                )
            num_grid_tokens = (
                t * (h // self._spatial_merge_size) * (w // self._spatial_merge_size)
            )
            num_embeds = feature.mm_position.get_num_embeds()
            if num_embeds != num_grid_tokens:
                raise ValueError(
                    f"Feature {feature.identifier!r}: image_grid_thw implies "
                    f"{num_grid_tokens} multimodal embeddings, got "
                    f"mm_position.get_num_embeds()={num_embeds}"
                )
            image_grid_thw_values.append((t, h, w))

        return image_grid_thw_values

    @classmethod
    def _grid_thw(cls, data: MultiModalKwargsItem, key: str) -> tuple[int, int, int]:
        value = cls._as_mlx(cls._feature_value(data, key))
        if len(value.shape) == 2 and value.shape[0] == 1:
            value = value[0]
        values = value.tolist()
        if len(values) != 3:
            raise ValueError(f"{key} must contain exactly 3 values, got {values}.")
        t, h, w = values
        return int(t), int(h), int(w)

    @staticmethod
    def _as_mlx(value: Any) -> Any:
        # vLLM's multimodal cache hands the same tensors to every later request
        # with this image, and the pixels arrive in the model dtype. A cast
        # between fp16 and bf16 (--dtype float16 against a bf16 tower) keeps
        # the item size, so on a shared import MLX may write the cast into the
        # donated input buffer and overwrite the cached pixels.
        if isinstance(value, torch.Tensor):
            return torch_to_mlx(value, copy=True)
        return value

    @staticmethod
    def _feature_value(data: MultiModalKwargsItem, key: str) -> Any:
        return data[key].data

    @staticmethod
    def _first_scalar_int(value: mx.array) -> int:
        if value.ndim == 0:
            return int(value.item())
        return int(value.reshape(-1)[0].item())
