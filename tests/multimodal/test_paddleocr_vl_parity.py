# SPDX-License-Identifier: Apache-2.0
"""End-to-end parity: ``adapter.call_lm`` vs ``mlx_vlm.Model.__call__``.

Marked ``slow`` — opt in with ``pytest -m slow`` (matches the existing
real-model convention in the shared parity tool).  Skips when
the model is not pre-pulled into the HF cache; pre-pull locally with::

    hf download mlx-community/PaddleOCR-VL-4bit

Override via ``PADDLEOCR_VL_PARITY_MODEL`` env var.
"""

from __future__ import annotations

import os

import mlx.core as mx
import numpy as np
import pytest
import torch
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItem

from tests.multimodal.vl_parity import (
    adapter_logits,
    assert_logits_match,
    build_image_inputs,
    reference_logits,
    skip_unless_cached,
)
from vllm_metal.multimodal.paddleocr_vl import PaddleOCRVLMultimodalAdapter

MODEL_ID = os.environ.get(
    "PADDLEOCR_VL_PARITY_MODEL", "mlx-community/PaddleOCR-VL-4bit"
)

skip_unless_cached(MODEL_ID)


@pytest.fixture(scope="module")
def loaded():
    from mlx_vlm import load

    model, processor = load(MODEL_ID)
    return model, processor


def _kwargs_item(
    pixel_values: mx.array, image_grid_thw: mx.array
) -> MultiModalKwargsItem:
    """Wrap mlx-vlm processor output into the MultiModalKwargsItem the adapter expects.

    The processor emits ``pixel_values`` already batched per image —
    ``(1, patches, 3, patch, patch)`` — and ``image_grid_thw`` as ``(1, 3)``.
    """
    pixels_t = torch.from_numpy(np.asarray(pixel_values))
    grid_t = torch.from_numpy(np.asarray(image_grid_thw))
    field_cfg = MultiModalFieldConfig.batched("image", keep_on_cpu=True)
    return MultiModalKwargsItem(
        {
            "pixel_values": field_cfg.build_elems("pixel_values", pixels_t)[0],
            "image_grid_thw": field_cfg.build_elems("image_grid_thw", grid_t)[0],
        }
    )


@pytest.mark.slow
def test_call_lm_logits_match_reference(loaded):
    model, processor = loaded
    input_ids, pixel_values, image_grid_thw = build_image_inputs(model, processor)

    ref = reference_logits(model, input_ids, pixel_values, image_grid_thw)
    got = adapter_logits(
        PaddleOCRVLMultimodalAdapter.from_loaded_model(model),
        model,
        input_ids,
        image_token_id=int(model.config.image_token_id),
        item=_kwargs_item(pixel_values, image_grid_thw),
    )
    assert_logits_match(got, ref)
