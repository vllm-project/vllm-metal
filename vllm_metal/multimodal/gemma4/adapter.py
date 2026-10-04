# SPDX-License-Identifier: Apache-2.0
"""Gemma 4 multimodal adapter: vision sidecar on the mlx_lm text backbone.

The language model is the mlx_lm ``gemma4.Model`` that vllm-metal serves
for text; ``vision_tower`` and ``embed_vision`` come from
:class:`Gemma4VisionSidecar`.  mlx_lm multiplies ``input_embeddings`` by
``embed_scale`` inside its forward, so this adapter hands the runner *raw*
token embeddings and image features pre-divided by that scale: after the
model's multiplication the text carries its usual scale and the image rows
are back at the values ``embed_vision`` produced, exactly like
``mlx_vlm.Model.get_input_embeddings``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import numpy as np
import torch
from vllm.logger import init_logger
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItem

from vllm_metal.multimodal.feature_spec import MultiModalFeatureSpec, PlaceholderRange
from vllm_metal.multimodal.gemma4.sidecar import Gemma4VisionSidecar
from vllm_metal.pytorch_backend.tensor_bridge import torch_to_mlx

logger = init_logger(__name__)


def _factor_near_sqrt(n: int) -> tuple[int, int]:
    """Split ``n`` into ``(a, b)`` with ``a`` the largest divisor of ``n`` that
    is ``<= sqrt(n)`` (e.g. 280 -> (14, 20))."""
    a = math.isqrt(n)
    while a > 1 and n % a != 0:
        a -= 1
    return a, n // a


@dataclass(frozen=True)
class Gemma4VisionEncodeResult:
    """Projected image embeddings for one feature, pre-divided by embed_scale."""

    hidden_states: mx.array
    deepstack_visual_embeds: Sequence[mx.array] | None = None


class Gemma4MultimodalAdapter:
    """``MultimodalRuntimeAdapter`` for Gemma 4 on the text backbone."""

    forward_ready: bool = True
    requires_explicit_positions: bool = False
    supplies_segment_positions: bool = False
    """Gemma 4 uses plain 1-D RoPE from ``ctx.offsets``; mlx_lm ``rope``
    modules reject caller-supplied positions."""
    text_path_selective_logits_ok: bool = True
    """``text_model()`` is the very object the runner profiles for the
    split backbone/head path."""

    def __init__(
        self,
        *,
        text_model: Any,
        backbone: Any,
        sidecar: Gemma4VisionSidecar,
        embed_dtype: mx.Dtype,
        bidirectional_layer_kinds: frozenset[str] = frozenset(),
    ) -> None:
        self._text_model = text_model
        self._backbone = backbone
        self._sidecar = sidecar
        self._embed_dtype = embed_dtype
        self.bidirectional_layer_kinds = bidirectional_layer_kinds
        """Layer kinds ("sliding", "full") whose image-block rows attend
        bidirectionally (HF: only sliding layers when
        ``use_bidirectional_attention == "vision"``)."""
        # mlx_lm multiplies bf16 embeddings by the python float ``embed_scale``;
        # MLX casts that scalar to the array dtype first (bf16(53.066) == 53.0
        # for the 26B model), so divide by the same rounded value.
        self._embed_scale_rounded = float(
            mx.array(float(backbone.embed_scale), dtype=embed_dtype).item()
        )
        self._pad_token_id = int(
            getattr(getattr(backbone, "config", None), "pad_token_id", 0) or 0
        )

    @classmethod
    def from_loaded(
        cls,
        text_model: Any,
        sidecar: Gemma4VisionSidecar,
        *,
        bidirectional_attention: str | None,
    ) -> Gemma4MultimodalAdapter:
        """Resolve the mlx_lm backbone at load time; drift raises here.

        ``bidirectional_attention`` is the HF ``text_config.use_bidirectional_attention``
        value; the mlx_lm ``ModelArgs`` do not carry it.
        """
        if bidirectional_attention == "vision":
            kinds = frozenset({"sliding"})
        elif bidirectional_attention is None:
            kinds = frozenset()
            logger.info(
                "Metal: Gemma 4 checkpoint has no use_bidirectional_attention; "
                "image tokens attend causally"
            )
        else:
            raise RuntimeError(
                "Gemma 4 vision sidecar: use_bidirectional_attention="
                f"{bidirectional_attention!r} is not supported (only 'vision' or None)"
            )
        language_model = getattr(text_model, "language_model", None)
        backbone = getattr(language_model, "model", None)
        if backbone is None:
            raise RuntimeError(
                "text_model.language_model.model missing; expected the mlx_lm "
                "Gemma4TextModel (mlx-lm version drift detected)."
            )
        embed_tokens = getattr(backbone, "embed_tokens", None)
        if embed_tokens is None or not callable(embed_tokens):
            raise RuntimeError(
                "Gemma4TextModel.embed_tokens missing or not callable; "
                "mlx-lm version drift detected."
            )
        if not hasattr(backbone, "embed_scale"):
            raise RuntimeError(
                "Gemma4TextModel.embed_scale missing; mlx-lm version drift detected."
            )
        probe = embed_tokens(mx.zeros((1, 1), dtype=mx.int32))
        return cls(
            text_model=text_model,
            backbone=backbone,
            sidecar=sidecar,
            embed_dtype=probe.dtype,
            bidirectional_layer_kinds=kinds,
        )

    @property
    def embed_scale_rounded(self) -> float:
        return self._embed_scale_rounded

    def text_model(self) -> Any:
        return self._text_model

    def embed_tokens(self, input_ids: mx.array) -> mx.array:
        """Raw token embeddings; the mlx_lm forward applies ``embed_scale``."""
        return self._backbone.embed_tokens(input_ids)

    def encode_multimodal(
        self, features: list[MultiModalFeatureSpec]
    ) -> list[Gemma4VisionEncodeResult]:
        outputs: list[Gemma4VisionEncodeResult] = []
        for feature in features:
            self._validate_image_feature(feature, require_data=True)
            assert feature.data is not None
            # Encode at the tower's own dtype instead of passing the
            # processor's float32 array straight through the way mlx-vlm
            # does: the tower runs in its weight dtype either way, so the
            # wider input is dropped at the first projection, and for the
            # [0, 1] pixels the processor derives from uint8 the cast costs
            # at most half a source quantum.
            pixel_values = self._as_mlx(feature.data["pixel_values"].data).astype(
                self._sidecar.pixel_dtype
            )
            pixel_position_ids = self._as_mlx(feature.data["pixel_position_ids"].data)
            projected = self._sidecar.embed_vision(
                self._sidecar.vision_tower(pixel_values, pixel_position_ids)
            )
            if projected.ndim == 3:
                projected = projected[0]
            expected = feature.mm_position.get_num_embeds()
            if int(projected.shape[0]) != expected:
                raise ValueError(
                    f"Feature {feature.identifier!r}: vision tower produced "
                    f"{int(projected.shape[0])} rows for {expected} placeholder "
                    "embeddings; processor and pooler disagree on this image."
                )
            scaled = (projected.astype(mx.float32) / self._embed_scale_rounded).astype(
                self._embed_dtype
            )
            outputs.append(Gemma4VisionEncodeResult(hidden_states=scaled))
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
        """Run the mlx_lm model on runner-built embeddings.

        ``position_ids`` is unused: RoPE comes from the paged context.
        """
        if deepstack_visual_embeds is not None:
            raise RuntimeError(
                "Gemma 4 has no deepstack residuals; refusing to drop "
                "deepstack_visual_embeds silently."
            )
        del position_ids
        if visual_pos_masks is not None:
            # HF looks up per-layer inputs (E2B/E4B) with pad ids at image positions.
            input_ids = mx.where(visual_pos_masks, self._pad_token_id, input_ids)
        return self._text_model(input_ids, cache=cache, input_embeddings=inputs_embeds)

    def profile_features(self) -> list[MultiModalFeatureSpec]:
        """One maximal image feature for ``profile_run``, sized from the loaded tower.

        The geometry (patch count, patch dim, pooled grid) is derived from
        ``self._sidecar.vision_tower`` rather than hardcoded, so the profiled
        input always matches the checkpoint actually loaded. mlx-vlm's
        ``VisionPooler`` bins patches into a pooling grid *by position*
        (``kernel_idx = floor(x / k) + (max_x // k) * floor(y / k)``, ``k =
        pooling_kernel_size``); both grid dimensions must be exact multiples
        of ``k`` or patches from different bins alias into the same output
        row, silently pooling to fewer than the expected number of rows. Real
        images never hit this: ``Gemma4ImageProcessor`` always pads both
        dimensions to a multiple of ``patch_size * pooling_kernel_size``.
        """
        tower = self._sidecar.vision_tower
        k = int(tower.pooling_kernel_size)
        patch = int(tower.patch_size)
        num_embeds = int(tower.default_output_length)
        num_patches = int(tower.max_patches)
        a, b = _factor_near_sqrt(num_embeds)
        rows, cols = a * k, b * k
        if rows * cols != num_patches:
            raise RuntimeError(
                f"Gemma 4 vision tower geometry mismatch: a pooled grid of "
                f"{a}x{b} cells at pooling_kernel_size={k} needs {rows * cols} "
                f"patches, but the tower reports max_patches={num_patches} "
                f"(default_output_length={num_embeds})."
            )
        patch_dim = 3 * patch * patch
        grid_y, grid_x = np.mgrid[0:rows, 0:cols]
        positions = np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1).astype(np.int64)
        field = MultiModalFieldConfig.batched("image", keep_on_cpu=True)
        pixels_elem = field.build_elems(
            "pixel_values",
            torch.zeros((1, num_patches, patch_dim), dtype=torch.float32),
        )[0]
        positions_elem = field.build_elems(
            "pixel_position_ids", torch.from_numpy(positions)[None]
        )[0]
        item = MultiModalKwargsItem(
            {"pixel_values": pixels_elem, "pixel_position_ids": positions_elem}
        )
        return [
            MultiModalFeatureSpec(
                data=item,
                modality="image",
                identifier="gemma4-profile-image",
                mm_position=PlaceholderRange(offset=0, length=num_embeds),
            )
        ]

    @staticmethod
    def _validate_image_feature(
        feature: MultiModalFeatureSpec, *, require_data: bool
    ) -> None:
        if feature.modality != "image":
            raise ValueError(
                f"Feature {feature.identifier!r}: Gemma 4 adapter only supports "
                f"image features; got modality={feature.modality!r}."
            )
        if require_data and feature.data is None:
            raise ValueError(
                f"Feature {feature.identifier!r}: feature.data is required for "
                "vision encoding."
            )

    @staticmethod
    def _as_mlx(value: Any) -> mx.array:
        # vLLM's multimodal cache hands the same tensors to every later request
        # with this image, and the pixels already arrive in the tower dtype, so
        # a shared import would reach the tower's first elementwise ops, which
        # MLX may run in the donated input buffer: the cached pixels would be
        # normalized in place and every re-encode would see a different image.
        if isinstance(value, torch.Tensor):
            return torch_to_mlx(value, copy=True)
        return mx.array(value)
