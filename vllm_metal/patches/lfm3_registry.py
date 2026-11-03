# SPDX-License-Identifier: Apache-2.0
"""LFM3 scheduler metadata; mlx-vlm supplies the executable model."""

from collections.abc import Mapping
from typing import TYPE_CHECKING

from mlx_vlm.models.lfm3_vl.processing_lfm3_vl import Lfm3VlProcessor
from vllm.exceptions import VLLMValidationError
from vllm.inputs import MultiModalDataDict
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.lfm2 import Lfm2ForCausalLM
from vllm.model_executor.models.qwen2_vl import (
    Qwen2VLDummyInputsBuilder,
    Qwen2VLMultiModalProcessor,
    Qwen2VLProcessingInfo,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.parse import (
    DictEmbeddingItems,
    EmbeddingItems,
    MultiModalDataItems,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig


# TODO: Once the pinned vLLM registers Lfm3ForCausalLM with native hybrid
# capabilities and ShortConv shape/dtype/copy hooks, remove this shim and its
# LFM3 branch in compat.register_model_compat. Keep the shared hook and
# metal_model_metadata entry point while Flint or other fallbacks need them.
# Re-run batched and prefix-cache LFM3 parity against the native registration.
class Lfm3ForCausalLM(Lfm2ForCausalLM):
    """Reuse LFM2's hybrid capabilities and short-convolution state policy."""

    def __init__(self, *, vllm_config: "VllmConfig", prefix: str = "") -> None:
        raise NotImplementedError("LFM3 execution requires the Metal MLX model runner.")

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: "VllmConfig"
    ) -> tuple[tuple[int, int]]:
        config = vllm_config.model_config.hf_text_config
        return MambaStateShapeCalculator.short_conv_state_shape(
            tp_world_size=vllm_config.parallel_config.tensor_parallel_size,
            intermediate_size=config.hidden_size,
            conv_kernel=config.conv_L_cache,
        )


class Lfm3VlProcessingInfo(Qwen2VLProcessingInfo):
    """LFM3-VL lays images out as Qwen2-VL does: a patch grid merged 2x2.

    The config comes from the checkpoint's remote code and the processor from
    mlx-vlm, so skip Qwen2-VL's type checks. Video input is not supported yet.
    """

    def get_hf_config(self):
        return self.ctx.get_hf_config()

    def get_hf_processor(self, **kwargs: object):
        return self.ctx.get_hf_processor(Lfm3VlProcessor, **kwargs)

    def parse_mm_data(
        self, mm_data: MultiModalDataDict, *, validate: bool = True
    ) -> MultiModalDataItems:
        items = super().parse_mm_data(mm_data, validate=validate)
        if validate and isinstance(
            items.get("image"), (EmbeddingItems, DictEmbeddingItems)
        ):
            raise VLLMValidationError(
                "LFM3-VL requires image pixels. Precomputed image embeddings are unsupported.",
                parameter="image",
            )
        return items

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None}

    def get_mm_max_tokens_per_item(
        self, seq_len: int, mm_counts: Mapping[str, int]
    ) -> Mapping[str, int]:
        return {"image": self.get_max_image_tokens()}


@MULTIMODAL_REGISTRY.register_processor(
    Qwen2VLMultiModalProcessor,
    info=Lfm3VlProcessingInfo,
    dummy_inputs=Qwen2VLDummyInputsBuilder,
)
class Lfm3VlForConditionalGeneration(Lfm3ForCausalLM, SupportsMultiModal):
    """LFM3 text metadata plus Qwen2-VL-style image placeholders."""

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<|vision_start|><|image|><|vision_end|>"
        raise ValueError("Only image modality is supported")
