# SPDX-License-Identifier: Apache-2.0
"""LFM3-VL multimodal adapter for vLLM Metal."""

from __future__ import annotations

import mlx.core as mx
from vllm.multimodal.inputs import MultiModalKwargsItem

from vllm_metal.multimodal.lfm2_vl import Lfm2VlMultimodalAdapter


class Lfm3VlMultimodalAdapter(Lfm2VlMultimodalAdapter):
    """LFM3-VL keeps LFM2-VL's text path and encodes a Qwen2-VL patch grid.

    Each image arrives as flat patches plus its ``(t, h, w)`` grid; the tower
    and projector return one row per merged 2x2 patch.
    """

    family = "LFM3-VL"

    def _encode_image(self, identifier: str, data: MultiModalKwargsItem) -> mx.array:
        pixel_values = self._as_mlx(self._feature_value(data, "pixel_values"))
        grid_thw = self._as_mlx(self._feature_value(data, "image_grid_thw"))
        if len(pixel_values.shape) != 2:
            raise ValueError(
                f"Feature {identifier!r}: pixel_values must have shape "
                f"(patches, features); got {tuple(pixel_values.shape)}"
            )
        grid_thw = grid_thw.reshape(-1, 3)
        return self._model.encode_images(pixel_values, image_grid_thw=grid_thw)
