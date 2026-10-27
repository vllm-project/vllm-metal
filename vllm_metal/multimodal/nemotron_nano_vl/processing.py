# SPDX-License-Identifier: Apache-2.0
"""VL V2 image preprocessing and image-only admission for Metal.

The resize/rescale order follows NVIDIA's VL V2 image_processing.py. The
generic Omni processor uses a different, dynamic-resolution image contract.
"""

from __future__ import annotations

import numpy as np
import torch
import torchvision.transforms.functional as tvf
from PIL import Image
from torchvision.transforms import InterpolationMode
from vllm.exceptions import VLLMValidationError
from vllm.inputs import MultiModalDataDict, MultiModalInput
from vllm.model_executor.models.nano_nemotron_vl import (
    NanoNemotronVLDummyInputsBuilder,
    NanoNemotronVLMultiModalProcessor,
    NanoNemotronVLProcessingInfo,
    NemotronH_Nano_VL_V2,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.parse import ImageEmbeddingItems, MultiModalDataItems
from vllm.multimodal.processing import ProcessorInputs, TimingContext
from vllm.transformers_utils.processors.internvl import (
    calculate_internvl_targets,
    get_internvl_target_ratios,
)
from vllm.transformers_utils.processors.nano_nemotron_vl import NanoNemotronVLProcessor

from vllm_metal.multimodal.nemotron_nano_vl.config import image_tile_limit


class NemotronNanoVLProcessor(NanoNemotronVLProcessor):
    """Use VL V2's fixed-size tiles, thumbnail, and BF16 encoder input."""

    @property
    def supports_video(self) -> bool:
        return False

    def _images_to_pixel_values_lst(
        self, images: list[Image.Image], max_num_tiles: int
    ) -> list[torch.Tensor]:
        if max_num_tiles < 1:
            raise ValueError("max_num_tiles must be positive.")
        ratios = get_internvl_target_ratios(1, max_num_tiles)
        results = []
        for image in images:
            # NVIDIA rescales before tensor bicubic resizing. Moving /255
            # after interpolation changes rounding even after a BF16 cast.
            pixels = torch.from_numpy(np.array(image.convert("RGB"), copy=True))
            pixels = pixels.permute(2, 0, 1).float() / 255.0
            _, width, height = calculate_internvl_targets(
                orig_width=image.width,
                orig_height=image.height,
                target_ratios=ratios,
                image_size=self.image_size,
                use_thumbnail=False,
            )
            resized = tvf.resize(
                pixels, [height, width], InterpolationMode.BICUBIC, antialias=True
            )
            size = self.image_size
            tiles = [
                resized[:, top : top + size, left : left + size]
                for top in range(0, height, size)
                for left in range(0, width, size)
            ]
            if self.use_thumbnail and len(tiles) > 1:
                tiles.append(
                    tvf.resize(
                        pixels, [size, size], InterpolationMode.BICUBIC, antialias=True
                    )
                )
            normalized = (torch.stack(tiles) - self.norm_mean) / self.norm_std
            results.append(normalized.to(self.dtype))
        return results


class NemotronNanoVLProcessingInfo(NanoNemotronVLProcessingInfo):
    def parse_mm_data(
        self, mm_data: MultiModalDataDict, *, validate: bool = True
    ) -> MultiModalDataItems:
        unsupported = set(mm_data) - {"image"}
        if unsupported:
            raise VLLMValidationError(
                f"Nano VL V2 image serving does not support {', '.join(sorted(unsupported))}.",
                parameter="multi_modal_data",
            )
        return super().parse_mm_data(mm_data, validate=validate)

    @property
    def supports_video(self) -> bool:
        return False

    @property
    def supports_audio(self) -> bool:
        return False

    def get_hf_processor(self, **kwargs: object) -> NemotronNanoVLProcessor:
        # vLLM passes tokenizer truncation options to processor construction.
        kwargs.pop("truncation", None)
        config = self.get_hf_config()
        if NemotronNanoVLProcessor.use_dynamic_resolution(config):
            raise ValueError(
                "Nano VL V2 image serving requires the static tiling checkpoint."
            )
        return self.ctx.init_processor(
            NemotronNanoVLProcessor,
            config=config,
            tokenizer=self.get_tokenizer(),
            max_model_len=self.ctx.model_config.max_model_len,
            **kwargs,
        )


class NemotronNanoVLMultiModalProcessor(NanoNemotronVLMultiModalProcessor):
    def apply(
        self, inputs: ProcessorInputs, timing_ctx: TimingContext
    ) -> MultiModalInput:
        limit = image_tile_limit(self.info.ctx.get_mm_config())
        requested = inputs.hf_processor_mm_kwargs.get("max_num_tiles")
        if requested is not None and (
            isinstance(requested, bool)
            or not isinstance(requested, int)
            or not 1 <= requested <= limit
        ):
            raise VLLMValidationError(
                f"Nano VL V2 max_num_tiles must be between 1 and the configured "
                f"limit {limit}; larger inputs were not profiled at startup.",
                parameter="mm_processor_kwargs.max_num_tiles",
            )
        if "image" in inputs.mm_data_items and isinstance(
            inputs.mm_data_items["image"], ImageEmbeddingItems
        ):
            raise VLLMValidationError(
                "Nano VL V2 accepts image pixels, not precomputed image embeddings.",
                parameter="multi_modal_data.image",
            )
        return super().apply(inputs, timing_ctx)


@MULTIMODAL_REGISTRY.register_processor(
    NemotronNanoVLMultiModalProcessor,
    info=NemotronNanoVLProcessingInfo,
    dummy_inputs=NanoNemotronVLDummyInputsBuilder,
)
class NemotronNanoVLForProcessing(NemotronH_Nano_VL_V2):
    """Reuse vLLM architecture metadata; model execution is owned by MLX-VLM."""
