# SPDX-License-Identifier: Apache-2.0
"""Image-only bridge between vLLM Nano features and mlx-vlm's Nano model.

Image serving is checked against a pinned direct mlx-vlm reference. Audio and
video use separate model and processor contracts and are not enabled here.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import torch
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItem

from vllm_metal.multimodal.feature_spec import MultiModalFeatureSpec, PlaceholderRange
from vllm_metal.multimodal.nemotron_nano_vl.config import image_tile_limit
from vllm_metal.tensor_bridge import torch_to_mlx


@dataclass(frozen=True)
class NemotronNanoVLVisionEncodeResult:
    """Projected image rows for exactly one scheduler-owned image feature."""

    hidden_states: mx.array
    deepstack_visual_embeds: None = None


class NemotronNanoVLImageAdapter:
    """Use Nano's vision/projector and inner Nemotron-H language model."""

    forward_ready: bool = True
    # Nemotron-H attention has no RoPE; the runner's normal text route is safe.
    requires_explicit_positions: bool = False
    supplies_segment_positions: bool = False

    def __init__(
        self,
        *,
        model: Any,
        language_model: Any,
        patch_size: int,
        downsample_ratio: float,
        image_token_id: int,
        hidden_size: int,
        tile_size: int,
        profile_tiles: int,
    ) -> None:
        if patch_size <= 0 or downsample_ratio != 0.5 or hidden_size <= 0:
            raise ValueError(
                "Nano image adapter requires positive patch_size/hidden_size "
                "and downsample_ratio=0.5."
            )
        self._model = model
        self._language_model = language_model
        self._patch_size = patch_size
        self._downsample_factor = 2
        self._image_token_id = image_token_id
        self._hidden_size = hidden_size
        self._tile_size = tile_size
        self._profile_tiles = profile_tiles

    @classmethod
    def from_loaded_model(
        cls, model: Any, *, hf_config: Any = None, mm_config: Any = None
    ) -> NemotronNanoVLImageAdapter:
        """Resolve the pinned mlx-vlm surface and configured profiling geometry."""
        config = getattr(model, "config", None)
        language_model = getattr(model, "language_model", None)
        vision_model = getattr(model, "vision_model", None)
        extract_feature = getattr(model, "_extract_feature_single", None)
        if (
            config is None
            or language_model is None
            or vision_model is None
            or not callable(extract_feature)
        ):
            raise RuntimeError(
                "Nano mlx-vlm model must expose config, language_model, "
                "vision_model, and _extract_feature_single; loader surface changed."
            )
        backbone = getattr(language_model, "backbone", None)
        if not callable(getattr(backbone, "embeddings", None)):
            raise RuntimeError(
                "Nano language_model.backbone.embeddings is missing; "
                "mlx-vlm loader surface changed."
            )
        if not hasattr(language_model, "backbone") or not hasattr(
            language_model.backbone, "layers"
        ):
            raise RuntimeError(
                "Nano language_model.backbone.layers is missing; the hybrid "
                "runtime cannot locate its layers."
            )
        try:
            signature = inspect.signature(language_model.__call__)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("Cannot inspect Nano language_model.__call__.") from exc
        explicit = {
            name
            for name, parameter in signature.parameters.items()
            if parameter.kind
            in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        }
        if not {"inputs_embeds", "cache"} <= explicit:
            raise RuntimeError(
                "Nano language_model.__call__ must explicitly accept "
                "inputs_embeds and cache; refusing a **kwargs-only fallback."
            )
        try:
            patch_size = int(config.vision_config.patch_size)
            downsample_ratio = float(config.downsample_ratio)
            image_token_id = int(config.img_context_token_id)
            hidden_size = int(config.text_config.hidden_size)
            tile_size = int(config.force_image_size)
        except (AttributeError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "Nano vision/config layout differs from pinned mlx-vlm."
            ) from exc
        if patch_size <= 0 or tile_size <= 0 or tile_size % (patch_size * 2):
            raise ValueError(
                "Nano VL V2 requires a positive, patch-aligned fixed tile size."
            )
        max_tiles = image_tile_limit(mm_config)
        profile_tiles = max_tiles + int(
            max_tiles > 1 and getattr(hf_config, "use_thumbnail", True)
        )
        if mm_config is not None and mm_config.get_limit_per_prompt("image") == 0:
            profile_tiles = 0
        return cls(
            model=model,
            language_model=language_model,
            patch_size=patch_size,
            downsample_ratio=downsample_ratio,
            image_token_id=image_token_id,
            hidden_size=hidden_size,
            tile_size=tile_size,
            profile_tiles=profile_tiles,
        )

    def text_model(self) -> Any:
        return self._language_model

    def embed_tokens(self, input_ids: mx.array) -> mx.array:
        return self._language_model.backbone.embeddings(input_ids)

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[MultiModalFeatureSpec],
    ) -> tuple[mx.array, int]:
        """Validate image placeholders; return ordinary sequential positions.

        The common runner expects three M-RoPE rows. Nano's attention ignores
        them, so replicate the standard text row and keep decode delta zero.
        """
        last_end = 0
        for feature in sorted(mm_features, key=lambda item: item.mm_position.offset):
            pixels = self._validate_feature(feature)
            position = feature.mm_position
            start = position.offset
            end = start + position.length
            count = position.get_num_embeds()
            if start < 0 or end > len(input_tokens):
                raise ValueError(
                    f"Feature {feature.identifier!r}: image placeholders "
                    "extend outside the prompt."
                )
            if start < last_end:
                raise ValueError(
                    f"Feature {feature.identifier!r}: image placeholders overlap "
                    "another image feature."
                )
            placeholder_tokens = input_tokens[start:end]
            if position.is_embed is not None:
                # vLLM includes <img> and </img> in the range, but only the
                # selected <image> tokens receive projected image rows.
                placeholder_tokens = [
                    token_id
                    for token_id, is_embed in zip(
                        placeholder_tokens, position.is_embed.tolist(), strict=True
                    )
                    if is_embed
                ]
            if any(token_id != self._image_token_id for token_id in placeholder_tokens):
                raise ValueError(
                    f"Feature {feature.identifier!r}: image placeholder token "
                    "ids do not match img_context_token_id."
                )
            self._validate_image_geometry(feature, pixels, count)
            last_end = end
        positions = mx.arange(len(input_tokens), dtype=mx.int32)
        return mx.broadcast_to(positions[None, None, :], (3, 1, len(input_tokens))), 0

    def encode_multimodal(
        self, features: list[MultiModalFeatureSpec]
    ) -> list[NemotronNanoVLVisionEncodeResult]:
        """Project pixels to LM-width rows, never re-entering composite forward."""
        results: list[NemotronNanoVLVisionEncodeResult] = []
        for feature in features:
            pixels = self._validate_feature(feature)
            count = feature.mm_position.get_num_embeds()
            self._validate_image_geometry(feature, pixels, count)
            # Match direct mlx-vlm: preserve processor pixel precision.
            # Weight storage dtype, especially packed quantized integers,
            # does not determine the dtype of the encoder's input.
            projected = self._model._extract_feature_single(pixels)
            if projected.ndim != 3:
                raise ValueError(
                    f"Feature {feature.identifier!r}: Nano vision output must "
                    f"have rank 3, got shape {projected.shape}."
                )
            rows = projected.reshape(-1, projected.shape[-1])
            if rows.shape[-1] != self._hidden_size:
                raise ValueError(
                    f"Feature {feature.identifier!r}: vision projection width "
                    f"{rows.shape[-1]} differs from LM hidden size "
                    f"{self._hidden_size}."
                )
            if rows.shape[0] != count:
                raise ValueError(
                    f"Feature {feature.identifier!r}: vision produced "
                    f"{rows.shape[0]} rows for {count} image placeholders."
                )
            results.append(NemotronNanoVLVisionEncodeResult(hidden_states=rows))
        return results

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
        """Run the inner LM once with runner-spliced image embeddings."""
        if deepstack_visual_embeds is not None:
            raise ValueError("Nano image adapter does not support deepstack embeds.")
        if position_ids.shape != (3, 1, input_ids.shape[1]):
            raise ValueError("Nano position_ids must have shape (3, 1, seq_len).")
        # Images have already been fused by merge_multimodal_embeddings.
        # Nano has position-free attention, so it does not take position_ids.
        return self._language_model(input_ids, inputs_embeds=inputs_embeds, cache=cache)

    def profile_features(self) -> list[MultiModalFeatureSpec]:
        """Profile the configured largest image, including its optional thumbnail.

        Adapted from syang-genai's PR #1160 at 96ba69519bd7c3d8ce83591fa31c33b3a29223d7.
        Encoding is per image, so the image count does not multiply this shape.
        """
        if self._profile_tiles == 0:
            return []
        tokens_per_tile = (self._tile_size // (self._patch_size * 2)) ** 2
        field = MultiModalFieldConfig.batched("image", keep_on_cpu=True)
        pixels = field.build_elems(
            "pixel_values_flat",
            torch.zeros(
                (1, self._profile_tiles, 3, self._tile_size, self._tile_size),
                dtype=torch.bfloat16,
            ),
        )[0]
        return [
            MultiModalFeatureSpec(
                data=MultiModalKwargsItem({"pixel_values_flat": pixels}),
                modality="image",
                identifier="nano-vl-profile-image",
                mm_position=PlaceholderRange(
                    offset=0, length=self._profile_tiles * tokens_per_tile
                ),
            )
        ]

    def _validate_feature(self, feature: MultiModalFeatureSpec) -> mx.array:
        if feature.modality != "image":
            raise NotImplementedError(
                f"Feature {feature.identifier!r}: Nano adapter supports image "
                f"pixels only, got modality={feature.modality!r}."
            )
        if feature.data is None:
            raise ValueError(f"Feature {feature.identifier!r}: image data is missing.")
        if "image_embeds" in feature.data:
            raise NotImplementedError(
                f"Feature {feature.identifier!r}: precomputed image_embeds "
                "are not supported."
            )
        position = feature.mm_position
        mask = position.is_embed
        if position.length <= 0 or (
            mask is not None
            and (mask.dtype != torch.bool or mask.shape != (position.length,))
        ):
            raise ValueError(
                f"Feature {feature.identifier!r}: image placeholder range must "
                "have positive length and an optional boolean mask of that length."
            )
        pixels = self._as_mlx(self._feature_value(feature.data, "pixel_values_flat"))
        if pixels.ndim == 3:
            pixels = pixels[None, ...]
        if (
            pixels.ndim != 4
            or pixels.shape[1] != 3
            or any(size <= 0 for size in pixels.shape)
        ):
            raise ValueError(
                f"Feature {feature.identifier!r}: pixel_values_flat must have "
                f"shape (tiles, 3, height, width), got {pixels.shape}."
            )
        return pixels

    def _validate_image_geometry(
        self, feature: MultiModalFeatureSpec, pixels: mx.array, count: int
    ) -> None:
        # Both callers first validate the feature and require its pixel data.
        assert feature.data is not None
        _, _, height, width = pixels.shape
        divisor = self._patch_size * self._downsample_factor
        if height % divisor or width % divisor:
            raise ValueError(
                f"Feature {feature.identifier!r}: image height and width must "
                f"be divisible by {divisor}, got {height}x{width}."
            )
        expected = pixels.shape[0] * (height // divisor) * (width // divisor)
        if expected != count:
            raise ValueError(
                f"Feature {feature.identifier!r}: pixels imply {expected} image "
                f"embeddings, got {count} placeholders."
            )
        if "num_tokens_per_image" in feature.data:
            reported = int(self._feature_value(feature.data, "num_tokens_per_image"))
            if reported != expected:
                raise ValueError(
                    f"Feature {feature.identifier!r}: num_tokens_per_image="
                    f"{reported} disagrees with pixel geometry ({expected})."
                )
        if "image_num_patches" in feature.data:
            reported = int(self._feature_value(feature.data, "image_num_patches"))
            if reported != pixels.shape[0]:
                raise ValueError(
                    f"Feature {feature.identifier!r}: image_num_patches="
                    f"{reported} disagrees with tile count {pixels.shape[0]}."
                )
        if "imgs_sizes" in feature.data:
            size = self._feature_value(feature.data, "imgs_sizes")
            if tuple(int(v) for v in size) != (height, width):
                raise ValueError(
                    f"Feature {feature.identifier!r}: imgs_sizes disagrees "
                    "with pixel geometry."
                )

    @staticmethod
    def _as_mlx(value: Any) -> mx.array:
        if isinstance(value, torch.Tensor):
            # vLLM owns/caches this tensor; MLX must not donate its storage.
            return torch_to_mlx(value, copy=True)
        if isinstance(value, mx.array):
            return value
        raise TypeError(f"Expected torch or MLX pixel tensor, got {type(value)!r}.")

    @staticmethod
    def _feature_value(data: MultiModalKwargsItem, key: str) -> Any:
        try:
            return data[key].data
        except KeyError as exc:
            raise ValueError(f"Nano image feature is missing {key!r}.") from exc
