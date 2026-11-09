# SPDX-License-Identifier: Apache-2.0
"""LFM3-Omni image/audio input with text output."""

from typing import Any, Self

from vllm_metal.multimodal import MultiModalFeatureSpec
from vllm_metal.multimodal.lfm3_audio import Lfm3AudioMultimodalAdapter
from vllm_metal.multimodal.lfm3_vl import Lfm3VlMultimodalAdapter
from vllm_metal.multimodal.lfm_text import LfmTextAdapter
from vllm_metal.multimodal.text_backbone import resolve_text_backbone


class Lfm3OmniMultimodalAdapter(LfmTextAdapter):
    family = "LFM3-Omni"
    text_path_selective_logits_ok = False

    @classmethod
    def from_loaded_model(cls, model: Any) -> Self:
        audio = Lfm3AudioMultimodalAdapter.from_loaded_model(model)
        vision = Lfm3VlMultimodalAdapter.from_loaded_model(model)
        adapter = cls(
            model=model,
            language_model=audio.text_model(),
            backbone=resolve_text_backbone(model.language_model, owner="mlx_vlm"),
            embed_tokens_fn=audio.embed_tokens,
        )
        adapter._audio = audio
        adapter._vision = vision
        return adapter

    def encode_multimodal(self, features: list[MultiModalFeatureSpec]) -> list[Any]:
        outputs = []
        for feature in features:
            if feature.modality == "image":
                outputs.extend(self._vision.encode_multimodal([feature]))
            elif feature.modality == "audio":
                outputs.extend(self._audio.encode_multimodal([feature]))
            else:
                raise ValueError("LFM3-Omni supports image and audio input only")
        return outputs
