# SPDX-License-Identifier: Apache-2.0
"""Nemotron Nano VL V2 multimodal adapter for vLLM Metal.

Wraps mlx-vlm's ``nemotron_h_nano_omni`` composite.  vLLM's own
``NanoNemotronVLMultiModalProcessor`` tiles each image into
``force_image_size`` squares (plus a thumbnail) and expands ``<image>`` into
``<img>`` + ``<image>`` * N + ``</img>``; this adapter runs the RADIO tower,
pixel shuffle and ``mlp1`` projector on those tiles.  The Nemotron-H text
model has no positional encoding, so positions are plain sequential.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import torch
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItem

from vllm_metal.multimodal.feature_spec import MultiModalFeatureSpec, PlaceholderRange
from vllm_metal.tensor_bridge import torch_to_mlx

# vLLM's processor default (``DEFAULT_NUM_TILES``); the thumbnail adds one.
_PROFILE_MAX_TILES = 12 + 1


@dataclass(frozen=True)
class NemotronNanoVLVisionEncodeResult:
    """Projected vision embeddings for one Nemotron Nano VL image."""

    hidden_states: mx.array
    deepstack_visual_embeds: None = None


class NemotronNanoVLMultimodalAdapter:
    """Model-owned multimodal helpers for Nemotron Nano VL V2 execution."""

    forward_ready: bool = True
    requires_explicit_positions: bool = False
    supplies_segment_positions: bool = False
    """Nemotron-H attention has no RoPE, so there are no positions to supply."""

    def __init__(
        self,
        *,
        model: Any,
        language_model: Any,
        embeddings: Any,
        tile_size: int,
        tokens_per_tile: int,
    ) -> None:
        self._model = model
        self._language_model = language_model
        self._embeddings = embeddings
        self._tile_size = tile_size
        self._tokens_per_tile = tokens_per_tile

    @classmethod
    def from_loaded_model(cls, model: Any) -> NemotronNanoVLMultimodalAdapter:
        """Create an adapter from an mlx-vlm ``nemotron_h_nano_omni`` composite."""
        language_model = model.language_model
        backbone = getattr(language_model, "backbone", None)
        embeddings = getattr(backbone, "embeddings", None)
        if embeddings is None or not callable(embeddings):
            raise RuntimeError(
                "language_model.backbone.embeddings missing or not callable; "
                "mlx_vlm version drift detected.  Expected the Nemotron-H "
                "backbone."
            )
        for name in ("_extract_feature_single", "vision_model", "mlp1"):
            if getattr(model, name, None) is None:
                raise RuntimeError(
                    f"mlx_vlm Nemotron Nano model has no {name!r}; "
                    "mlx_vlm version drift detected."
                )

        config = model.config
        tile_size = config.force_image_size
        if not tile_size:
            raise ValueError(
                "Nemotron Nano VL needs force_image_size for fixed-size tiles; "
                "dynamic-resolution checkpoints are not supported."
            )
        patch_size = int(
            model.vision_model.radio_model.model.patch_generator.patch_size
        )
        tokens_per_tile = int(
            (int(tile_size) // patch_size) ** 2 * config.downsample_ratio**2
        )
        return cls(
            model=model,
            language_model=language_model,
            embeddings=embeddings,
            tile_size=int(tile_size),
            tokens_per_tile=tokens_per_tile,
        )

    def text_model(self) -> Any:
        return self._language_model

    def embed_tokens(self, input_ids: mx.array) -> mx.array:
        return self._embeddings(input_ids)

    def encode_multimodal(
        self,
        features: list[MultiModalFeatureSpec],
    ) -> list[NemotronNanoVLVisionEncodeResult]:
        """Encode each image's tiles into ``(tiles * tokens_per_tile, hidden)``."""
        target_dtype = self._model.mlp1.layers[0].weight.dtype
        outputs: list[NemotronNanoVLVisionEncodeResult] = []
        for feature in features:
            self._validate_image_feature(feature, require_data=True)
            assert feature.data is not None
            pixel_values = self._as_mlx(feature.data["pixel_values_flat"].data)
            if pixel_values.ndim != 4 or tuple(pixel_values.shape[1:]) != (
                3,
                self._tile_size,
                self._tile_size,
            ):
                raise ValueError(
                    f"Feature {feature.identifier!r}: pixel_values_flat must "
                    f"have shape (tiles, 3, {self._tile_size}, {self._tile_size}); "
                    f"got {pixel_values.shape}"
                )
            projected = self._model._extract_feature_single(
                pixel_values.astype(target_dtype)
            )
            projected = projected.reshape(-1, projected.shape[-1])
            expected = feature.mm_position.get_num_embeds()
            if int(projected.shape[0]) != expected:
                raise ValueError(
                    f"Feature {feature.identifier!r}: vision tower produced "
                    f"{int(projected.shape[0])} rows for {expected} placeholder "
                    "embeddings."
                )
            outputs.append(NemotronNanoVLVisionEncodeResult(hidden_states=projected))
        return outputs

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[MultiModalFeatureSpec],
    ) -> tuple[mx.array, int]:
        """Plain sequential positions, broadcast to the ``(3, 1, L)`` M-RoPE shape."""
        for feature in mm_features:
            self._validate_image_feature(feature, require_data=False)
        seq_len = len(input_tokens)
        positions = mx.arange(seq_len, dtype=mx.int32)
        return mx.broadcast_to(positions[None, None, :], (3, 1, seq_len)), 0

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
        """Run the language model on runner-built embeddings.

        ``position_ids`` and ``visual_pos_masks`` are unused: Nemotron-H has
        no positional encoding and treats image rows like any other token.
        """
        del position_ids, visual_pos_masks
        if deepstack_visual_embeds is not None:
            raise RuntimeError(
                "Nemotron Nano VL has no deepstack residuals; refusing to drop "
                "deepstack_visual_embeds silently."
            )
        return self._language_model(input_ids, inputs_embeds=inputs_embeds, cache=cache)

    def profile_features(self) -> list[MultiModalFeatureSpec]:
        """One image at the processor's default maximum tile count."""
        num_embeds = _PROFILE_MAX_TILES * self._tokens_per_tile
        field = MultiModalFieldConfig.batched("image", keep_on_cpu=True)
        pixels_elem = field.build_elems(
            "pixel_values_flat",
            torch.zeros(
                (1, _PROFILE_MAX_TILES, 3, self._tile_size, self._tile_size),
                dtype=torch.float32,
            ),
        )[0]
        return [
            MultiModalFeatureSpec(
                data=MultiModalKwargsItem({"pixel_values_flat": pixels_elem}),
                modality="image",
                identifier="nemotron-nano-vl-profile-image",
                mm_position=PlaceholderRange(offset=0, length=num_embeds),
            )
        ]

    @staticmethod
    def _validate_image_feature(
        feature: MultiModalFeatureSpec, *, require_data: bool
    ) -> None:
        if feature.modality != "image":
            raise NotImplementedError(
                f"Feature {feature.identifier!r}: Nemotron Nano VL on Metal "
                f"serves image inputs only; got modality={feature.modality!r}."
            )
        if require_data and feature.data is None:
            raise ValueError(
                f"Feature {feature.identifier!r}: feature.data is required "
                "for vision encoding."
            )

    @staticmethod
    def _as_mlx(value: Any) -> Any:
        # Copy: vLLM's multimodal cache shares these tensors across requests,
        # and a same-itemsize cast could write into the donated buffer.
        if isinstance(value, torch.Tensor):
            return torch_to_mlx(value, copy=True)
        return value
