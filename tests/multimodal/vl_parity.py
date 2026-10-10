# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the ``adapter.call_lm`` vs ``mlx_vlm.Model.__call__``
parity tests. Each test module keeps only its model id, cache fixture and
the ``MultiModalKwargsItem`` layout its processor/adapter expects."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import mlx.core as mx
import numpy as np
import pytest
from PIL import Image
from vllm.multimodal.inputs import MultiModalKwargsItem

from vllm_metal.multimodal import (
    MultiModalFeatureSpec,
    PlaceholderRange,
    merge_multimodal_embeddings,
)


def skip_unless_cached(model_id: str) -> None:
    """Skip the calling module unless ``model_id`` is pre-pulled in the HF cache."""
    from huggingface_hub import scan_cache_dir
    from huggingface_hub.errors import CacheNotFound

    try:
        info = scan_cache_dir()
    except CacheNotFound:
        in_cache = False
    else:
        in_cache = any(
            rev.size_on_disk > 100 * 1024 * 1024
            for repo in info.repos
            if repo.repo_id == model_id
            for rev in repo.revisions
        )
    if not in_cache:
        pytest.skip(
            f"{model_id} not in HF cache; pre-pull with `hf download {model_id}`",
            allow_module_level=True,
        )


def build_image_inputs(model, processor) -> tuple[mx.array, mx.array, mx.array]:
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


def reference_logits(model, input_ids, pixel_values, image_grid_thw) -> mx.array:
    out: Any = model(
        input_ids, pixel_values=pixel_values, image_grid_thw=image_grid_thw
    )
    logits = getattr(out, "logits", out)
    mx.eval(logits)
    return logits


def adapter_logits(
    adapter,
    model,
    input_ids: mx.array,
    *,
    image_token_id: int,
    item: MultiModalKwargsItem,
    extra_lm_kwargs: Callable[[Any, mx.array], dict[str, Any]] | None = None,
) -> mx.array:
    ids_np = np.asarray(input_ids)[0]
    is_image_np = ids_np == image_token_id
    placeholder_positions = np.where(is_image_np)[0]
    assert placeholder_positions.size > 0, "processor produced no image placeholders"
    placeholder_offset = int(placeholder_positions[0])
    placeholder_len = int(placeholder_positions.size)

    feature = MultiModalFeatureSpec(
        data=item,
        modality="image",
        identifier="img-0",
        mm_position=PlaceholderRange(offset=placeholder_offset, length=placeholder_len),
    )
    encode_result = adapter.encode_multimodal([feature])[0]

    inputs_embeds = adapter.embed_tokens(input_ids)
    is_image_mx = mx.array(is_image_np)
    spliced = merge_multimodal_embeddings(
        inputs_embeds, [encode_result.hidden_states], is_image_mx
    )

    positions, _delta = adapter.get_mrope_input_positions(ids_np.tolist(), [feature])

    extra = extra_lm_kwargs(encode_result, is_image_mx) if extra_lm_kwargs else {}
    n_layers = len(model.language_model.model.layers)
    out: Any = adapter.call_lm(
        input_ids,
        inputs_embeds=spliced,
        cache=[None] * n_layers,
        position_ids=positions,
        **extra,
    )
    logits = getattr(out, "logits", out)
    mx.eval(logits)
    return logits


def assert_logits_match(got: mx.array, ref: mx.array) -> None:
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
