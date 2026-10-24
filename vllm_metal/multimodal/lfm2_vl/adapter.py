# SPDX-License-Identifier: Apache-2.0
"""LFM2-VL multimodal adapter for vLLM Metal."""

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
class Lfm2VlVisionEncodeResult:
    """Projected image embeddings for one LFM2-VL multimodal feature."""

    hidden_states: mx.array
    deepstack_visual_embeds: None = None


class Lfm2VlMultimodalAdapter:
    """Model-owned multimodal helpers for LFM2-VL execution.

    LFM2-VL splits each image into tiles plus a thumbnail, so one feature
    carries several vision-tower views.  ``encode_images`` returns one array
    per view and this adapter concatenates them in prompt order.
    """

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

    def __init__(
        self,
        *,
        model: Any | None = None,
        language_model: Any | None = None,
        embed_tokens_fn: Any | None = None,
        backbone: Any | None = None,
    ) -> None:
        self._model = model
        self._language_model = language_model
        self._embed_tokens_fn = embed_tokens_fn
        self._backbone = backbone

    @classmethod
    def from_loaded_model(cls, model: Any) -> Lfm2VlMultimodalAdapter:
        """Create an adapter from an mlx-vlm LFM2-VL composite."""
        language_model = model.language_model
        if not hasattr(model, "encode_images"):
            raise RuntimeError(
                "Lfm2Vl model.encode_images missing; mlx_vlm version drift detected."
            )
        if getattr(model.config, "image_token_index", None) is None:
            raise RuntimeError(
                "Lfm2Vl config.image_token_index missing; mlx_vlm version "
                "drift detected."
            )
        backbone = resolve_text_backbone(language_model, owner="mlx_vlm")
        if not callable(backbone):
            raise RuntimeError(
                "language_model.model is not callable; mlx_vlm version drift "
                "detected. Expected the LFM2 text backbone."
            )
        embed_tokens_fn = resolve_backbone_embed_tokens(backbone, owner="mlx_vlm")
        return cls(
            model=model,
            language_model=language_model,
            embed_tokens_fn=embed_tokens_fn,
            backbone=backbone,
        )

    def text_model(self) -> Any:
        return self._language_model

    def embed_tokens(self, input_ids: mx.array) -> mx.array:
        if self._embed_tokens_fn is None:
            raise RuntimeError(
                "embed_tokens_fn not resolved; construct via "
                "Lfm2VlMultimodalAdapter.from_loaded_model."
            )
        return self._embed_tokens_fn(input_ids)

    def encode_multimodal(
        self,
        features: list[MultiModalFeatureSpec],
    ) -> list[Lfm2VlVisionEncodeResult]:
        """Run LFM2-VL's vision tower and projector for each image feature."""
        if self._model is None:
            raise RuntimeError(
                "model not loaded; encode_multimodal unavailable. Construct "
                "via Lfm2VlMultimodalAdapter.from_loaded_model."
            )

        outputs: list[Lfm2VlVisionEncodeResult] = []
        for feature in features:
            if feature.modality == "video":
                raise NotImplementedError(
                    f"Feature {feature.identifier!r}: LFM2-VL video input is "
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

            pixel_values = self._as_mlx(
                self._feature_value(feature.data, "pixel_values")
            )
            spatial_shapes = self._as_mlx(
                self._feature_value(feature.data, "spatial_shapes")
            )
            lengths = self._tile_lengths(
                feature.identifier, pixel_values, spatial_shapes
            )

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
            hidden_states = (
                views
                if isinstance(views, mx.array)
                else mx.concatenate(list(views), axis=0)
            )

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

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[MultiModalFeatureSpec],
    ) -> tuple[mx.array, int]:
        """Plain sequential positions, broadcast to the ``(3, 1, L)`` M-RoPE shape.

        LFM2 has no M-RoPE section; positions come from ``ctx.offsets`` on the
        paged path, so the delta is always 0.
        """
        for feature in mm_features:
            if feature.modality == "video":
                raise NotImplementedError(
                    f"Feature {feature.identifier!r}: LFM2-VL video input is "
                    "not yet supported."
                )
        seq_len = len(input_tokens)
        if seq_len == 0:
            return mx.zeros((3, 1, 0), dtype=mx.int32), 0
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
        """Invoke the LFM2-VL language model on runner-built embeddings."""
        del visual_pos_masks, position_ids
        self._check_lm_ready("call_lm", deepstack_visual_embeds)
        # ``input_embeddings`` is the name both the LFM2 ``LanguageModel`` and
        # its ``Lfm2Model`` backbone declare; the wrapper also accepts
        # ``inputs_embeds`` but the backbone does not.
        return self._language_model(
            input_ids,
            cache=cache,
            input_embeddings=inputs_embeds,
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
        """:meth:`call_lm` without the output head: the backbone's hidden states."""
        del visual_pos_masks, position_ids
        self._check_lm_ready("call_lm_hidden_states", deepstack_visual_embeds)
        if self._backbone is None or not callable(self._backbone):
            raise RuntimeError(
                "language_model.model attribute missing or not callable; "
                "mlx_vlm version drift detected. Expected the LFM2 text "
                "backbone."
            )
        return self._backbone(
            input_ids,
            cache=cache,
            input_embeddings=inputs_embeds,
        )

    def _check_lm_ready(self, call: str, deepstack_visual_embeds: Any | None) -> None:
        if self._language_model is None:
            raise RuntimeError(
                f"language_model not loaded; {call} unavailable. Construct "
                "via Lfm2VlMultimodalAdapter.from_loaded_model."
            )
        if deepstack_visual_embeds is not None:
            raise RuntimeError("LFM2-VL does not expose deepstack visual residuals.")

    @classmethod
    def _tile_lengths(
        cls,
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
        for index, shape in enumerate(spatial_shapes.tolist()):
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

    @staticmethod
    def _as_mlx(value: Any) -> Any:
        # vLLM's multimodal cache hands the same tensors to every later request
        # with this image, so copy rather than risk MLX writing a cast into the
        # donated input buffer and corrupting the cached pixels.
        if isinstance(value, torch.Tensor):
            return torch_to_mlx(value, copy=True)
        return value

    @staticmethod
    def _feature_value(data: MultiModalKwargsItem, key: str) -> Any:
        if key not in data:
            raise ValueError(f"multimodal feature is missing {key!r}")
        return data[key].data
