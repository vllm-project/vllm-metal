# SPDX-License-Identifier: Apache-2.0
"""Shared text execution for LFM image and audio adapters."""

from typing import Any

import mlx.core as mx
import torch
from vllm.multimodal.inputs import MultiModalKwargsItem

from vllm_metal.multimodal.feature_spec import MultiModalFeatureSpec
from vllm_metal.tensor_bridge import torch_to_mlx


class LfmTextAdapter:
    """Text embeddings, positions and language calls shared by LFM adapters."""

    family: str
    forward_ready = True
    requires_explicit_positions = False
    supplies_segment_positions = False
    text_path_selective_logits_ok = True

    def __init__(
        self,
        *,
        model: Any,
        language_model: Any,
        embed_tokens_fn: Any,
        backbone: Any,
    ) -> None:
        self._model = model
        self._language_model = language_model
        self._embed_tokens_fn = embed_tokens_fn
        self._backbone = backbone

    def text_model(self) -> Any:
        return self._language_model

    def embed_tokens(self, input_ids: mx.array) -> mx.array:
        return self._embed_tokens_fn(input_ids)

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[MultiModalFeatureSpec],
    ) -> tuple[mx.array, int]:
        """Plain sequential positions, broadcast to the ``(3, 1, L)`` M-RoPE shape.

        These text towers use 1-D RoPE; positions come from ``ctx.offsets`` on the
        paged path, so the delta is always 0.
        """
        for feature in mm_features:
            if feature.modality == "video":
                raise NotImplementedError(
                    f"Feature {feature.identifier!r}: {self.family} video input is "
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
        """Invoke the LFM language model on runner-built embeddings."""
        del visual_pos_masks, position_ids
        self._check_deepstack(deepstack_visual_embeds)
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
        self._check_deepstack(deepstack_visual_embeds)
        return self._backbone(
            input_ids,
            cache=cache,
            input_embeddings=inputs_embeds,
        )

    def _check_deepstack(self, deepstack_visual_embeds: Any | None) -> None:
        if deepstack_visual_embeds is not None:
            raise RuntimeError(
                f"{self.family} does not expose deepstack visual residuals."
            )

    @staticmethod
    def _as_mlx(value: Any) -> Any:
        # vLLM's multimodal cache hands the same tensors to every later request
        # with this feature, so copy rather than risk MLX writing a cast into the
        # donated input buffer and corrupting the cached input.
        if isinstance(value, torch.Tensor):
            return torch_to_mlx(value, copy=True)
        return value

    @staticmethod
    def _feature_value(data: MultiModalKwargsItem, key: str) -> Any:
        if key not in data:
            raise ValueError(f"multimodal feature is missing {key!r}")
        return data[key].data
