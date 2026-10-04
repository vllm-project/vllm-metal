# SPDX-License-Identifier: Apache-2.0
"""End-to-end parity: the Gemma 4 vision sidecar vs ``mlx_vlm``'s composite.

The sidecar's contract is that ``embed_tokens`` + ``encode_multimodal`` +
``merge_multimodal_embeddings`` reproduce the input embeddings that
``mlx_vlm.models.gemma4.Model.get_input_embeddings`` builds from the same
checkpoint and the same processor output.  Everything downstream is the
mlx_lm text model this plugin already serves for text.

Both sides are compared *before* the language model, which is what makes the
comparison sharp: the two conventions differ (mlx-vlm scatters raw vision
features into embeddings it has already multiplied by ``embed_scale``, while
the adapter hands the runner raw token embeddings and pre-divides the image
rows so the mlx_lm forward's own multiplication lands them in the same
place), so any drift in weight mapping, the vision tower, the scale rounding
or the splice offsets shows up here as a large error rather than a rounding
difference.

One deliberate alignment: the adapter casts pixels to the vision tower's own
weight dtype before encoding, while ``get_input_embeddings`` feeds the
processor's float32 array straight in.  On a bfloat16 tower that cast is worth
about 7% of the image-row magnitude — far more than everything this file is
trying to measure — so the reference is given pixels of the same dtype and the
embedding assertion stays about weight mapping, the scale round trip and the
splice.  The next-token check below runs against the unmodified mlx-vlm path.

``adapter.call_lm`` builds its own dense causal mask through mlx_lm, so this
file does not exercise the paged attention mask at all.

Two levels:

* :class:`TestTinyCheckpointParity` builds the tiny synthetic checkpoint from
  ``tools/gemma4_tiny_checkpoint.py``.  It runs anywhere the source repo's
  tokenizer and processor files are reachable (cached or downloadable) and
  pins the embedding contract on random weights.
* :class:`TestRealModelParity` runs the same comparison, plus the next-token
  argmax, against a real checkpoint.  Skipped unless the model is pre-pulled::

      hf download mlx-community/unsloth-gemma-4-26B-A4B-it-qat-oQ4

  ``GEMMA4_PARITY_MODEL`` overrides it with another repo id or with the path to
  a local checkout.  Pick a checkpoint the sidecar actually accepts: of the
  26B-A4B repos only this one ships the ``video_processor`` block in
  ``processor_config.json`` that ``Gemma4Processor`` needs, and the
  ``mlx-community/gemma-4-e2b-it-4bit`` / ``gemma-4-e4b-it-4bit`` repos ship it too.
"""

from __future__ import annotations

import gc
import importlib.util
import os
from pathlib import Path
from types import ModuleType
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

MODEL_ID = os.environ.get(
    "GEMMA4_PARITY_MODEL", "mlx-community/unsloth-gemma-4-26B-A4B-it-qat-oQ4"
)

# Text rows must be reproduced bit-for-bit: the splice may not touch them, and
# both sides take the embed_scale product in the same dtype.
#
# Image rows additionally survive the scale round trip -- the adapter divides by
# the rounded scale and the mlx_lm forward multiplies it back -- which costs
# nothing when the scale is a power of two (the tiny checkpoint's 8.0, measured
# at exactly 0) and 0.24% of the image-row magnitude when it is not (53.0 on the
# 26B). The bound below leaves that an order of magnitude of headroom while
# staying well under the ~7% a change in the pixel dtype would cost and far
# under the factor-of-scale error a broken convention would produce.
_IMAGE_ROW_RTOL = 0.02


def _load_tiny_checkpoint_tool() -> ModuleType:
    """Load ``tools/gemma4_tiny_checkpoint.py`` directly; ``tools/`` isn't a package."""
    path = Path(__file__).resolve().parents[2] / "tools" / "gemma4_tiny_checkpoint.py"
    spec = importlib.util.spec_from_file_location("gemma4_tiny_checkpoint", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _model_available(model_id: str) -> bool:
    """A local checkpoint directory, or a repo id already pulled into the HF cache.

    Mirrors what the engine accepts: ``_local_checkpoint_dir`` takes either.
    """
    if Path(model_id).is_dir():
        return True
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


def _processor_inputs(model_dir: Path, seed: int) -> dict[str, np.ndarray]:
    """Encode 'describe the image' plus one deterministic dummy image.

    The processor class is named explicitly rather than resolved through
    ``AutoProcessor``: importing ``mlx_vlm`` registers its own
    ``Gemma4Processor`` over the transformers one, and the two disagree on the
    image layout — transformers patchifies to ``(1, patches, 3 * p * p)`` and
    emits ``image_position_ids``, mlx-vlm hands back a raw ``(1, 3, H, W)``
    tensor.  The engine consumes the transformers layout (vLLM builds the
    processor before the sidecar's lazy mlx-vlm import), so the test pins it
    instead of inheriting whatever the import order of the wider test session
    happens to leave registered.
    """
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config
    from transformers.models.gemma4 import Gemma4Processor

    processor = Gemma4Processor.from_pretrained(str(model_dir))
    config = load_config(model_dir)
    rng = np.random.default_rng(seed)
    image = Image.fromarray((rng.random((128, 128, 3)) * 255).astype(np.uint8))
    prompt = apply_chat_template(processor, config, "describe the image", num_images=1)
    encoded = processor(text=[prompt], images=[image], return_tensors="np")
    assert "image_position_ids" in encoded, (
        "the transformers Gemma4Processor stopped emitting image_position_ids; "
        f"got {sorted(encoded.keys())} — the engine's pixel layout changed"
    )
    return {
        "input_ids": np.asarray(encoded["input_ids"]),
        "pixel_values": np.asarray(encoded["pixel_values"]),
        "image_position_ids": np.asarray(encoded["image_position_ids"]),
    }


def _build_feature(
    pixel_values: np.ndarray,
    image_position_ids: np.ndarray,
    offset: int,
    length: int,
) -> MultiModalFeatureSpec:
    """Wrap processor output into the ``MultiModalKwargsItem`` the adapter expects.

    Both Gemma 4 image fields are batched, and the adapter reads the position
    ids under ``pixel_position_ids`` (the processor emits them as
    ``image_position_ids``) — the same renaming vLLM's field factory does.
    """
    field = MultiModalFieldConfig.batched("image", keep_on_cpu=True)
    pixels_elem = field.build_elems("pixel_values", torch.from_numpy(pixel_values))[0]
    positions_elem = field.build_elems(
        "pixel_position_ids", torch.from_numpy(image_position_ids)
    )[0]
    return MultiModalFeatureSpec(
        data=MultiModalKwargsItem(
            {"pixel_values": pixels_elem, "pixel_position_ids": positions_elem}
        ),
        modality="image",
        identifier="img-0",
        mm_position=PlaceholderRange(offset=offset, length=length),
    )


def _reference(
    model_dir: Path, inputs: dict[str, np.ndarray]
) -> tuple[Any, Any, int, float]:
    """mlx-vlm composite: input embeddings, logits, image token id, embed scale.

    ``mm_token_type_ids`` is deliberately not passed: mlx-vlm only overlays its
    bidirectional image-block mask when it is, so both sides run a plain causal
    mask and the comparison stays about the embeddings, not the mask.
    """
    from mlx_vlm.utils import load_model

    model = load_model(model_dir, lazy=False)
    input_ids = mx.array(inputs["input_ids"])
    pixel_values = mx.array(inputs["pixel_values"])
    position_ids = mx.array(inputs["image_position_ids"])

    # The dtype the sidecar encodes in: the patch projection's weight dtype.
    pixel_dtype = model.vision_tower.patch_embedder.input_proj.weight.dtype
    embeds = model.get_input_embeddings(
        input_ids=input_ids,
        pixel_values=pixel_values.astype(pixel_dtype),
        image_position_ids=position_ids,
    )
    inputs_embeds = getattr(embeds, "inputs_embeds", embeds)
    logits = model(
        input_ids, pixel_values=pixel_values, image_position_ids=position_ids
    )
    logits = getattr(logits, "logits", logits)
    mx.eval(inputs_embeds, logits)
    # The reference's own scale, rounded the way its embedding dtype rounds it.
    # Scaling our side by *this* keeps the comparison independent of the value
    # the adapter computed, so a wrong scale in the adapter lands on the image
    # rows instead of smearing across every row.
    scale = float(
        mx.array(
            float(model.language_model.model.embed_scale), dtype=inputs_embeds.dtype
        ).item()
    )
    image_token_id = int(model.config.image_token_id)
    # A real checkpoint is tens of gigabytes and the adapter side loads its own
    # copy of the same weights; on a machine whose working set is not twice the
    # checkpoint, the two only fit one at a time.
    del model
    gc.collect()
    mx.clear_cache()
    return inputs_embeds, logits, image_token_id, scale


def _adapter_side(
    model_dir: Path, inputs: dict[str, np.ndarray], image_token_id: int
) -> tuple[Any, Any, float]:
    """Sidecar path: pre-scale input embeddings, logits, and the rounded scale."""
    import json

    from mlx_lm import load as mlx_lm_load

    from vllm_metal.multimodal.gemma4 import Gemma4VisionSidecar
    from vllm_metal.multimodal.gemma4.adapter import Gemma4MultimodalAdapter

    text_model, _tokenizer = mlx_lm_load(str(model_dir))
    sidecar = Gemma4VisionSidecar.load(model_dir)
    # Same field the engine reads.  It only selects the layer kinds the paged
    # path applies the image-block mask on; ``call_lm`` builds its own dense
    # causal mask, so it does not affect this comparison either way.
    text_config = json.loads((model_dir / "config.json").read_text()).get(
        "text_config", {}
    )
    adapter = Gemma4MultimodalAdapter.from_loaded(
        text_model,
        sidecar,
        bidirectional_attention=text_config.get("use_bidirectional_attention"),
    )

    ids = inputs["input_ids"][0]
    placeholders = np.where(ids == image_token_id)[0]
    assert placeholders.size > 0, "processor produced no image placeholders"
    feature = _build_feature(
        inputs["pixel_values"],
        inputs["image_position_ids"],
        int(placeholders[0]),
        int(placeholders.size),
    )

    input_ids = mx.array(inputs["input_ids"])
    image_mask = mx.array(ids == image_token_id)
    encoded = adapter.encode_multimodal([feature])[0]
    spliced = merge_multimodal_embeddings(
        adapter.embed_tokens(input_ids),
        [encoded.hidden_states],
        image_mask,
    )
    positions, _delta = adapter.get_mrope_input_positions(ids.tolist(), [feature])
    num_layers = len(text_model.language_model.model.layers)
    logits = adapter.call_lm(
        input_ids,
        inputs_embeds=spliced,
        cache=[None] * num_layers,
        position_ids=positions,
        visual_pos_masks=image_mask[None, :],
    )
    logits = getattr(logits, "logits", logits)
    mx.eval(spliced, logits)
    adapter_scale = adapter.embed_scale_rounded
    del adapter, sidecar, text_model
    gc.collect()
    mx.clear_cache()
    return spliced, logits, adapter_scale


def _assert_embeddings_match(
    spliced: Any,
    embed_scale: float,
    reference: Any,
    image_mask: np.ndarray,
) -> None:
    """The adapter's rows, once scaled, must equal mlx-vlm's input embeddings.

    ``embed_scale`` is the *reference's* scale, so a wrong scale on the adapter
    side lands on the image rows rather than on every row at once.
    """
    # Multiply in the embedding dtype: that is what mlx-vlm and the mlx_lm
    # forward both do, and for a scale that is not a power of two (53.0 on the
    # 26B, against 8.0 on the tiny checkpoint) a float32 product rounds
    # differently and shows up as a spurious text-row mismatch.
    ours = np.asarray((spliced * embed_scale).astype(mx.float32))[0]
    ref = np.asarray(reference.astype(mx.float32))[0]
    assert ours.shape == ref.shape, f"shape {ours.shape} differs from {ref.shape}"

    diff = np.abs(ours - ref)
    text_max = float(diff[~image_mask].max())
    assert text_max == 0.0, (
        f"text rows must be untouched by the splice; max diff {text_max:.6g}"
    )

    image_diff = float(diff[image_mask].max())
    budget = _IMAGE_ROW_RTOL * float(np.abs(ref[image_mask]).max())
    assert image_diff <= budget, (
        f"image-row embeddings diverge: max diff {image_diff:.6g} exceeds "
        f"{budget:.6g} ({_IMAGE_ROW_RTOL:.0%} of the reference magnitude); "
        f"reference embed_scale={embed_scale}"
    )


@pytest.mark.slow
@pytest.mark.network
class TestTinyCheckpointParity:
    """Pins the embedding contract on the tiny synthetic checkpoint.

    Random weights make the logits near-degenerate, so only the embeddings are
    asserted here; the next-token check lives in the real-model class.
    """

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_input_embeddings_match_reference(self, tmp_path: Path, seed: int) -> None:
        model_dir = _load_tiny_checkpoint_tool().build_tiny_checkpoint(
            tmp_path / "tiny"
        )
        inputs = _processor_inputs(model_dir, seed)
        reference, _ref_logits, image_token_id, scale = _reference(model_dir, inputs)
        spliced, _logits, _adapter_scale = _adapter_side(
            model_dir, inputs, image_token_id
        )
        _assert_embeddings_match(
            spliced, scale, reference, inputs["input_ids"][0] == image_token_id
        )


@pytest.fixture(scope="module")
def prepared() -> tuple[Path, dict[str, np.ndarray], Any, Any, int, float]:
    """Load the real checkpoint once and keep only arrays (never the model)."""
    from vllm_metal.utils import get_model_download_path

    model_dir = (
        Path(MODEL_ID)
        if Path(MODEL_ID).is_dir()
        else Path(get_model_download_path(MODEL_ID))
    )
    inputs = _processor_inputs(model_dir, seed=0)
    reference, ref_logits, image_token_id, scale = _reference(model_dir, inputs)
    return model_dir, inputs, reference, ref_logits, image_token_id, scale


@pytest.mark.slow
@pytest.mark.skipif(
    not _model_available(MODEL_ID),
    reason=(
        f"{MODEL_ID} is neither a local directory nor in the HF cache; "
        f"pre-pull with `hf download {MODEL_ID}`, or point GEMMA4_PARITY_MODEL "
        "at a checkout"
    ),
)
class TestRealModelParity:
    """Same comparison on a real checkpoint, plus the next-token argmax."""

    def test_input_embeddings_match_reference(
        self, prepared: tuple[Path, dict[str, np.ndarray], Any, Any, int, float]
    ) -> None:
        model_dir, inputs, reference, _ref_logits, image_token_id, scale = prepared
        spliced, _logits, _adapter_scale = _adapter_side(
            model_dir, inputs, image_token_id
        )
        _assert_embeddings_match(
            spliced, scale, reference, inputs["input_ids"][0] == image_token_id
        )

    def test_next_token_argmax_matches_reference(
        self, prepared: tuple[Path, dict[str, np.ndarray], Any, Any, int, float]
    ) -> None:
        model_dir, inputs, _reference, ref_logits, image_token_id, _scale = prepared
        _spliced, logits, _embed_scale = _adapter_side(
            model_dir, inputs, image_token_id
        )
        assert logits.shape == ref_logits.shape, (
            f"logits shape {logits.shape} differs from reference {ref_logits.shape}"
        )
        ours = int(mx.argmax(logits[0, -1]).item())
        reference_argmax = int(mx.argmax(ref_logits[0, -1]).item())
        abs_max = float(
            mx.max(
                mx.abs(logits.astype(mx.float32) - ref_logits.astype(mx.float32))
            ).item()
        )
        assert ours == reference_argmax, (
            f"next-token argmax mismatch: adapter={ours} "
            f"reference={reference_argmax} (abs_max={abs_max:.4g})"
        )
