# SPDX-License-Identifier: Apache-2.0
"""End-to-end parity: ``adapter.call_lm`` vs ``mlx_vlm.Model.__call__``.

Marked ``slow`` — opt in with ``pytest -m slow`` (matches the existing
real-model convention in ``tests/test_qwen35_smoke.py``).  Skips when
the model is not pre-pulled into the HF cache; pre-pull locally with::

    hf download mlx-community/Qwen3-VL-4B-Instruct-4bit

Override via ``QWEN3_VL_PARITY_MODEL`` env var.
"""

from __future__ import annotations

import os
from typing import Any

import mlx.core as mx
import numpy as np
import pytest
import torch
from PIL import Image
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItem

from vllm_metal.multimodal import (
    MultiModalFeatureSpec,
    PlaceholderRange,
    merge_multimodal_embeddings,
)
from vllm_metal.multimodal.qwen3_vl import Qwen3VLMultimodalAdapter

MODEL_ID = os.environ.get(
    "QWEN3_VL_PARITY_MODEL", "mlx-community/Qwen3-VL-4B-Instruct-4bit"
)


def _model_in_cache(model_id: str) -> bool:
    from huggingface_hub import scan_cache_dir
    from huggingface_hub.errors import CacheNotFound

    try:
        info = scan_cache_dir()
    except CacheNotFound:
        return False
    for repo in info.repos:
        if repo.repo_id != model_id:
            continue
        for rev in repo.revisions:
            if rev.size_on_disk > 100 * 1024 * 1024:
                return True
    return False


if not _model_in_cache(MODEL_ID):
    pytest.skip(
        f"{MODEL_ID} not in HF cache; pre-pull with `hf download {MODEL_ID}`",
        allow_module_level=True,
    )


@pytest.fixture(scope="module")
def loaded():
    from mlx_vlm import load

    model, processor = load(MODEL_ID)
    return model, processor


def _build_image_inputs(model, processor):
    """Encode 'describe the image' + one deterministic dummy image."""
    from mlx_vlm.prompt_utils import apply_chat_template

    rng = np.random.default_rng(0)
    image = Image.fromarray((rng.random((128, 128, 3)) * 255).astype(np.uint8))
    prompt = apply_chat_template(
        processor, model.config, "describe the image", num_images=1
    )
    proc = processor(text=[prompt], images=[image], return_tensors="np")
    input_ids = mx.array(np.asarray(proc["input_ids"]))
    pixel_values = mx.array(np.asarray(proc["pixel_values"]))
    image_grid_thw = mx.array(np.asarray(proc["image_grid_thw"]))
    return input_ids, pixel_values, image_grid_thw


def _build_feature(
    pixel_values: mx.array,
    image_grid_thw: mx.array,
    offset: int,
    length: int,
) -> MultiModalFeatureSpec:
    """Wrap mlx-vlm processor output into the MultiModalKwargsItem the adapter expects.

    Mirrors vLLM's Qwen2-VL field factory: pixel_values uses ``flat_from_sizes``
    (the leading axis is patch count, not batch); image_grid_thw uses ``batched``.
    """
    pixels_t = torch.from_numpy(np.asarray(pixel_values))
    grid_t = torch.from_numpy(np.asarray(image_grid_thw))  # (1, 3)
    pixel_grid_sizes = grid_t.prod(-1)  # (1,) — patches per image
    pixel_cfg = MultiModalFieldConfig.flat_from_sizes("image", pixel_grid_sizes)
    grid_cfg = MultiModalFieldConfig.batched("image", keep_on_cpu=True)
    pixel_elem = pixel_cfg.build_elems("pixel_values", pixels_t)[0]
    grid_elem = grid_cfg.build_elems("image_grid_thw", grid_t)[0]
    return MultiModalFeatureSpec(
        data=MultiModalKwargsItem(
            {"pixel_values": pixel_elem, "image_grid_thw": grid_elem}
        ),
        modality="image",
        identifier="img-0",
        mm_position=PlaceholderRange(offset=offset, length=length),
    )


def _reference_logits(model, input_ids, pixel_values, image_grid_thw) -> mx.array:
    out: Any = model(
        input_ids, pixel_values=pixel_values, image_grid_thw=image_grid_thw
    )
    logits = getattr(out, "logits", out)
    mx.eval(logits)
    return logits


def _adapter_logits(model, input_ids, pixel_values, image_grid_thw) -> mx.array:
    adapter = Qwen3VLMultimodalAdapter.from_loaded_model(model)
    image_token_id = int(model.config.image_token_index)

    ids_np = np.asarray(input_ids)[0]
    is_image_np = ids_np == image_token_id
    placeholder_positions = np.where(is_image_np)[0]
    assert placeholder_positions.size > 0, "processor produced no image placeholders"
    placeholder_offset = int(placeholder_positions[0])
    placeholder_len = int(placeholder_positions.size)

    feature = _build_feature(
        pixel_values, image_grid_thw, placeholder_offset, placeholder_len
    )
    encode_result = adapter.encode_multimodal([feature])[0]

    inputs_embeds = adapter.embed_tokens(input_ids)
    is_image_mx = mx.array(is_image_np)
    spliced = merge_multimodal_embeddings(
        inputs_embeds, [encode_result.hidden_states], is_image_mx
    )

    positions, _delta = adapter.get_mrope_input_positions(ids_np.tolist(), [feature])

    n_layers = len(model.language_model.model.layers)
    out: Any = adapter.call_lm(
        input_ids,
        inputs_embeds=spliced,
        cache=[None] * n_layers,
        position_ids=positions,
        visual_pos_masks=is_image_mx[None, :],
        deepstack_visual_embeds=encode_result.deepstack_visual_embeds,
    )
    logits = getattr(out, "logits", out)
    mx.eval(logits)
    return logits


@pytest.mark.slow
def test_call_lm_logits_match_reference(loaded):
    model, processor = loaded
    input_ids, pixel_values, image_grid_thw = _build_image_inputs(model, processor)

    ref = _reference_logits(model, input_ids, pixel_values, image_grid_thw)
    got = _adapter_logits(model, input_ids, pixel_values, image_grid_thw)

    assert got.shape == ref.shape, (
        f"adapter logits shape {got.shape} differs from reference {ref.shape}"
    )

    diff = mx.abs(got - ref)
    abs_max = float(mx.max(diff).item())
    rel = diff / (mx.abs(ref) + 1e-6)
    rel_max = float(mx.max(rel).item())

    # Compare next-token argmax (deterministic check, looser than logits parity)
    ref_argmax = int(mx.argmax(ref[0, -1]).item())
    got_argmax = int(mx.argmax(got[0, -1]).item())

    assert got_argmax == ref_argmax, (
        f"next-token argmax mismatch: adapter={got_argmax} ref={ref_argmax} "
        f"(abs_max={abs_max:.4g}, rel_max={rel_max:.4g})"
    )

    # Logits parity: 4-bit quant + accumulation order leaves ~1e-2 absolute noise.
    assert abs_max < 5e-2, (
        f"logits abs_max {abs_max:.4g} exceeds 5e-2 (rel_max={rel_max:.4g})"
    )
