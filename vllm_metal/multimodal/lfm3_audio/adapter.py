# SPDX-License-Identifier: Apache-2.0
"""LFM3 audio input with text output on the MLX runner."""

from dataclasses import dataclass
from typing import Any, Self

import mlx.core as mx

from vllm_metal.multimodal import MultiModalFeatureSpec
from vllm_metal.multimodal.lfm_text import LfmTextAdapter
from vllm_metal.multimodal.text_backbone import (
    resolve_backbone_embed_tokens,
    resolve_text_backbone,
)


@dataclass(frozen=True)
class Lfm3AudioEncodeResult:
    """Projected audio embeddings for one feature."""

    hidden_states: mx.array
    deepstack_visual_embeds: None = None


class Lfm3AudioMultimodalAdapter(LfmTextAdapter):
    family = "LFM3-Audio"
    # The language wrapper masks speech placeholders after projecting logits.
    text_path_selective_logits_ok = False

    @classmethod
    def from_loaded_model(cls, model: Any) -> Self:
        # Match mlx-vlm's explicit text generation mode. Its wrapper applies
        # this mask to both multimodal prefill and subsequent text decoding.
        model.language_model._placeholder_ids = list(
            dict.fromkeys(
                [
                    *model.language_model._placeholder_ids,
                    model.config.audio_start_token_id,
                    model.config.mixed_start_token_id,
                ]
            )
        )
        backbone = resolve_text_backbone(model.language_model, owner="mlx_vlm")
        return cls(
            model=model,
            language_model=model.language_model,
            backbone=backbone,
            embed_tokens_fn=resolve_backbone_embed_tokens(backbone, owner="mlx_vlm"),
        )

    def encode_multimodal(
        self, features: list[MultiModalFeatureSpec]
    ) -> list[Lfm3AudioEncodeResult]:
        outputs = []
        for feature in features:
            if feature.modality != "audio":
                raise ValueError("LFM3-Audio supports audio input only")
            if feature.data is None:
                raise ValueError("LFM3-Audio requires audio features")
            mel = self._as_mlx(self._feature_value(feature.data, "input_features"))
            mask = self._as_mlx(
                self._feature_value(feature.data, "feature_attention_mask")
            )
            if mel.ndim != 2 or mask.ndim != 1 or mel.shape[-1] != mask.shape[0]:
                raise ValueError(
                    "LFM3-Audio expects (mel_bins, frames) and a frame mask"
                )
            encoded = self._model.audio_projector(
                self._model.audio_tower(mel[None], mask.sum()[None])
            )
            if encoded.shape[0] != feature.mm_position.get_num_embeds():
                raise ValueError("LFM3-Audio encoder and placeholder lengths disagree")
            outputs.append(Lfm3AudioEncodeResult(hidden_states=encoded))
        return outputs
