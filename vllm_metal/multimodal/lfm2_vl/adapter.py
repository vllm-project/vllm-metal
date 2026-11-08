# SPDX-License-Identifier: Apache-2.0
"""LFM2-VL multimodal adapter for vLLM Metal."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import mlx.core as mx
from vllm.multimodal.inputs import MultiModalKwargsItem

from vllm_metal.multimodal.feature_spec import MultiModalFeatureSpec
from vllm_metal.multimodal.lfm_text import LfmTextAdapter
from vllm_metal.multimodal.text_backbone import (
    resolve_backbone_embed_tokens,
    resolve_text_backbone,
)


@dataclass(frozen=True)
class Lfm2VlVisionEncodeResult:
    """Projected image embeddings for one LFM2-VL multimodal feature."""

    hidden_states: mx.array
    deepstack_visual_embeds: None = None


class Lfm2VlMultimodalAdapter(LfmTextAdapter):
    """Model-owned multimodal helpers for LFM2-VL execution.

    LFM2-VL splits each image into tiles plus a thumbnail, so one feature
    carries several vision-tower views.  ``encode_images`` returns one array
    per view and this adapter concatenates them in prompt order.
    """

    family: str = "LFM2-VL"

    forward_ready: bool = True

    requires_explicit_positions: bool = False
    """The LFM2 text tower derives RoPE from ``ctx.offsets``, so text-only
    batches keep the plain paged text path."""

    supplies_segment_positions: bool = False
    """LFM2 is plain 1-D RoPE: its mlx_lm ``rope(x, offset=)`` modules reject
    caller-supplied positions, so the context field stays ``None`` and the
    batched decode RoPE path is preserved."""

    text_path_selective_logits_ok: bool = True
    """``text_model()`` is the very object the runner profiles for the
    split backbone/head path."""

    @classmethod
    def from_loaded_model(cls, model: Any) -> Lfm2VlMultimodalAdapter:
        """Create an adapter from an mlx-vlm LFM2-VL composite."""
        language_model = model.language_model
        if not hasattr(model, "encode_images"):
            raise RuntimeError(
                f"{cls.family} model.encode_images missing; mlx_vlm version drift "
                "detected."
            )
        backbone = resolve_text_backbone(language_model, owner="mlx_vlm")
        if not callable(backbone):
            raise RuntimeError(
                "language_model.model is not callable; mlx_vlm version drift "
                f"detected. Expected the {cls.family} text backbone."
            )
        embed_tokens_fn = resolve_backbone_embed_tokens(backbone, owner="mlx_vlm")
        return cls(
            model=model,
            language_model=language_model,
            embed_tokens_fn=embed_tokens_fn,
            backbone=backbone,
        )

    def encode_multimodal(
        self,
        features: list[MultiModalFeatureSpec],
    ) -> list[Lfm2VlVisionEncodeResult]:
        """Run LFM2-VL's vision tower and projector for each image feature."""
        outputs: list[Lfm2VlVisionEncodeResult] = []
        for feature in features:
            if feature.modality == "video":
                raise NotImplementedError(
                    f"Feature {feature.identifier!r}: {self.family} video input is "
                    "not yet supported."
                )
            if feature.modality != "image":
                raise ValueError(
                    f"Feature {feature.identifier!r}: encode_multimodal only "
                    f"supports image features; got modality={feature.modality!r}."
                )
            if feature.data is None:
                raise ValueError(
                    f"Feature {feature.identifier!r}: feature.data is required "
                    "for vision encoding."
                )

            hidden_states = self._encode_image(feature.identifier, feature.data)

            expected = feature.mm_position.get_num_embeds()
            if int(hidden_states.shape[0]) != expected:
                raise ValueError(
                    f"Feature {feature.identifier!r}: vision tower produced "
                    f"{int(hidden_states.shape[0])} rows for {expected} "
                    "placeholder embeddings; the processor and the encoder "
                    "disagree on this image."
                )
            outputs.append(Lfm2VlVisionEncodeResult(hidden_states=hidden_states))

        return outputs

    def _encode_image(self, identifier: str, data: MultiModalKwargsItem) -> mx.array:
        pixel_values = self._as_mlx(self._feature_value(data, "pixel_values"))
        spatial_shapes = self._as_mlx(self._feature_value(data, "spatial_shapes"))
        lengths = self._tile_lengths(identifier, pixel_values, spatial_shapes)

        # vLLM hands the tiles already padded to a common patch count, and
        # the tower recovers each tile's real length from the mask rather
        # than from spatial_shapes, so synthesise it here instead of
        # repacking the pixels.
        padded = int(pixel_values.shape[1])
        mask = (
            mx.arange(padded, dtype=mx.int32)[None, :]
            < mx.array(lengths, dtype=mx.int32)[:, None]
        )

        views = self._model.encode_images(
            pixel_values,
            spatial_shapes=spatial_shapes,
            pixel_attention_mask=mask,
        )
        return mx.concatenate(views, axis=0)

    @staticmethod
    def _tile_lengths(
        identifier: str,
        pixel_values: mx.array,
        spatial_shapes: mx.array,
    ) -> list[int]:
        """Per-tile patch counts from ``spatial_shapes``, validated against the pixels."""
        if len(pixel_values.shape) != 3:
            raise ValueError(
                f"Feature {identifier!r}: pixel_values must have shape "
                f"(tiles, patches, features); got {tuple(pixel_values.shape)}"
            )
        if len(spatial_shapes.shape) != 2 or int(spatial_shapes.shape[1]) != 2:
            raise ValueError(
                f"Feature {identifier!r}: spatial_shapes must have shape "
                f"(tiles, 2); got {tuple(spatial_shapes.shape)}"
            )
        tiles = int(pixel_values.shape[0])
        if int(spatial_shapes.shape[0]) != tiles:
            raise ValueError(
                f"Feature {identifier!r}: spatial_shapes describes "
                f"{int(spatial_shapes.shape[0])} tiles but pixel_values has "
                f"{tiles}"
            )
        padded = int(pixel_values.shape[1])
        lengths: list[int] = []
        # The shape checks above pin this to (tiles, 2), which mx.array.tolist()
        # cannot express in its return type.
        rows = cast(list[list[int]], spatial_shapes.tolist())
        for index, shape in enumerate(rows):
            height, width = (int(value) for value in shape)
            if height <= 0 or width <= 0:
                raise ValueError(
                    f"Feature {identifier!r}: tile {index} has non-positive "
                    f"spatial shape ({height}, {width})"
                )
            length = height * width
            if length > padded:
                raise ValueError(
                    f"Feature {identifier!r}: tile {index} needs {length} "
                    f"patches but pixel_values only carries {padded}"
                )
            lengths.append(length)
        return lengths
