# SPDX-License-Identifier: Apache-2.0
"""GLM-5-Next multimodal adapter: mlx-vlm's vision tower on the paged runner.

The language model reads no positions (NoPE sparse MLA, KDA) and attends
causally over image tokens, so the runner only splices the tower's rows into
the input embeddings, exactly like ``mlx_vlm.Model.get_input_embeddings``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, cast

import mlx.core as mx
import torch
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItem

from vllm_metal.multimodal.feature_spec import MultiModalFeatureSpec, PlaceholderRange
from vllm_metal.multimodal.text_backbone import (
    resolve_backbone_embed_tokens,
    resolve_text_backbone,
)
from vllm_metal.tensor_bridge import torch_to_mlx


@dataclass(frozen=True)
class Glm5NextVisionEncodeResult:
    """Vision tower rows for one GLM-5-Next image feature."""

    hidden_states: mx.array
    deepstack_visual_embeds: None = None


class Glm5NextMultimodalAdapter:
    """``MultimodalRuntimeAdapter`` for mlx-vlm's GLM-5-Next composite."""

    forward_ready: bool = True
    requires_explicit_positions: bool = False
    supplies_segment_positions: bool = False
    """No layer of the language model reads positions."""

    def __init__(
        self, *, vision_tower: Any, language_model: Any, max_encode_tokens: int
    ) -> None:
        self._vision_tower = vision_tower
        self._language_model = language_model
        backbone = resolve_text_backbone(language_model, owner="mlx_vlm")
        self._embed_tokens = resolve_backbone_embed_tokens(backbone, owner="mlx_vlm")
        self._max_encode_tokens = max_encode_tokens

    @classmethod
    def from_loaded_model(
        cls, model: Any, *, max_encode_tokens: int
    ) -> Glm5NextMultimodalAdapter:
        """Create an adapter from an mlx-vlm GLM-5-Next composite.

        ``max_encode_tokens`` is the most embeddings one item can have under the
        server's processor settings.
        """
        return cls(
            vision_tower=model.vision_tower,
            language_model=model.language_model,
            max_encode_tokens=max_encode_tokens,
        )

    def text_model(self) -> Any:
        return self._language_model

    def embed_tokens(self, input_ids: mx.array) -> mx.array:
        return self._embed_tokens(input_ids)

    def encode_multimodal(
        self, features: list[MultiModalFeatureSpec]
    ) -> list[Glm5NextVisionEncodeResult]:
        """Run the vision tower for each image feature."""
        dtype = self._vision_tower.patch_embed.proj.weight.dtype
        outputs: list[Glm5NextVisionEncodeResult] = []
        for feature in features:
            if feature.modality != "image" or feature.data is None:
                raise ValueError(
                    f"Feature {feature.identifier!r}: GLM-5-Next encodes image "
                    f"features with data; got modality={feature.modality!r}."
                )
            # vLLM's multimodal cache hands the same tensors to every later
            # request with this item; a shared import could let MLX write the
            # cast into the donated input buffer and change the cached pixels.
            pixels = torch_to_mlx(feature.data["pixel_values"].data, copy=True)
            grid = cast(torch.Tensor, feature.data["image_grid_thw"].data)
            t, h, w = grid.reshape(-1).tolist()
            hidden_states = self._encode(pixels.astype(dtype), t, h, w)
            # Evaluate here: an encode joined to the forward's graph would run in
            # its command buffers and add to its peak memory.
            mx.eval(hidden_states)
            expected = feature.mm_position.get_num_embeds()
            if hidden_states.shape[0] != expected:
                raise ValueError(
                    f"Feature {feature.identifier!r}: vision tower produced "
                    f"{hidden_states.shape[0]} rows for {expected} placeholder "
                    "embeddings."
                )
            outputs.append(Glm5NextVisionEncodeResult(hidden_states=hidden_states))
        return outputs

    def _encode(self, pixels: mx.array, t: int, h: int, w: int) -> mx.array:
        """``vision_tower`` on ``t`` temporal patches of ``h`` x ``w``.

        Each block is submitted on its own, so a command buffer holds at most
        one block and the peak memory stays near one block's activations
        whatever ``MLX_MAX_MB_PER_BUFFER`` is. The arithmetic is the tower's own.
        """
        tower = self._vision_tower
        hidden_states = tower.patch_embed(pixels)
        rotary = tower._rotary_embeddings(mx.array([[t, h, w]]))
        cu_seqlens = mx.array([i * h * w for i in range(t + 1)])
        for block in tower.blocks:
            hidden_states = block(hidden_states, cu_seqlens, rotary)
            mx.async_eval(hidden_states)
        merge = tower.spatial_merge_size
        hidden_states = tower.post_layernorm(hidden_states).reshape(
            -1, merge, merge, hidden_states.shape[-1]
        )
        hidden_states = tower.downsample(hidden_states).reshape(
            -1, tower.config.out_hidden_size
        )
        return tower.merger(hidden_states)

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[MultiModalFeatureSpec],
    ) -> tuple[mx.array, int]:
        """Plain sequential positions in the ``(3, 1, L)`` M-RoPE shape."""
        positions = mx.arange(len(input_tokens), dtype=mx.int32)
        return mx.broadcast_to(positions, (3, 1, len(input_tokens))), 0

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
        """Run the language model on runner-built embeddings."""
        del position_ids, visual_pos_masks, deepstack_visual_embeds
        return self._language_model(input_ids, cache=cache, inputs_embeds=inputs_embeds)

    def profile_features(self) -> list[MultiModalFeatureSpec]:
        """The largest image vLLM admits, for ``profile_run``: the largest square
        grid within ``max_encode_tokens``."""
        config = self._vision_tower.config
        n = math.isqrt(self._max_encode_tokens)
        side = config.spatial_merge_size * n
        patch_dim = (
            config.in_channels * config.temporal_patch_size * config.patch_size**2
        )
        field = MultiModalFieldConfig.batched("image", keep_on_cpu=True)
        tensors = {
            "pixel_values": torch.zeros((1, side * side, patch_dim)),
            "image_grid_thw": torch.tensor([[1, side, side]]),
        }
        item = MultiModalKwargsItem(
            {key: field.build_elems(key, value)[0] for key, value in tensors.items()}
        )
        return [
            MultiModalFeatureSpec(
                data=item,
                modality="image",
                identifier="glm5-next-profile-image",
                mm_position=PlaceholderRange(offset=0, length=n * n),
            )
        ]
