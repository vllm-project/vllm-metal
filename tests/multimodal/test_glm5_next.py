# SPDX-License-Identifier: Apache-2.0
"""Tests for the GLM-5-Next multimodal adapter against mlx-vlm's own splice."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import mlx.core as mx
import pytest
import torch
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItem

from tests.test_pp_pipeline import _micro_glm5_next
from vllm_metal.multimodal import (
    MultiModalFeatureSpec,
    PlaceholderRange,
    merge_multimodal_embeddings,
)
from vllm_metal.multimodal.glm5_next import Glm5NextMultimodalAdapter
from vllm_metal.v1.model_adapter import DefaultModelAdapter

_IMAGE, _BOI, _EOI = 60, 56, 57


def _micro_model() -> Any:
    from mlx_vlm.models.glm5_next import Model, ModelConfig, VisionConfig

    return Model(
        ModelConfig(
            text_config=_micro_glm5_next().config,
            vision_config=VisionConfig(
                depth=2,
                hidden_size=16,
                num_heads=2,
                intermediate_size=16,
                out_hidden_size=32,
                projection_intermediate_size=16,
            ),
            image_token_id=_IMAGE,
        )
    )


def _feature(
    grid: tuple[int, int, int], *, length: int | None = None
) -> MultiModalFeatureSpec:
    t, h, w = grid
    pixels = torch.rand((1, t * h * w, 3 * 2 * 14 * 14))
    grid_thw = torch.tensor([grid])
    field = MultiModalFieldConfig.batched("image", keep_on_cpu=True)
    item = MultiModalKwargsItem(
        {
            "pixel_values": field.build_elems("pixel_values", pixels)[0],
            "image_grid_thw": field.build_elems("image_grid_thw", grid_thw)[0],
        }
    )
    return MultiModalFeatureSpec(
        data=item,
        modality="image",
        identifier=f"image-{h}x{w}",
        mm_position=PlaceholderRange(
            offset=0, length=t * h * w // 4 if length is None else length
        ),
    )


def test_spliced_prompt_matches_mlx_vlm_logits() -> None:
    model = _micro_model()
    adapter = Glm5NextMultimodalAdapter.from_loaded_model(model, max_encode_tokens=64)
    # A non-square grid, so the tower cannot swap height and width unnoticed.
    ids = [1, _BOI, *[_IMAGE] * 8, _EOI, 9, 2]
    feature = _feature((1, 8, 4))
    assert feature.data is not None
    media = {
        "pixel_values": mx.array(feature.data["pixel_values"].data.numpy()),
        "image_grid_thw": mx.array([[1, 8, 4]]),
    }
    input_ids = mx.array([ids])

    [rows] = adapter.encode_multimodal([feature])
    embeds = merge_multimodal_embeddings(
        adapter.embed_tokens(input_ids), [rows.hidden_states], input_ids == _IMAGE
    )
    positions, _ = adapter.get_mrope_input_positions(ids, [feature])
    logits = adapter.call_lm(input_ids, embeds, None, positions).logits

    expected = model(input_ids, **media).logits
    assert mx.allclose(logits, expected, atol=1e-5).item()


def test_rejects_rows_that_do_not_fill_the_placeholder() -> None:
    adapter = Glm5NextMultimodalAdapter.from_loaded_model(
        _micro_model(), max_encode_tokens=64
    )

    with pytest.raises(ValueError, match="4 rows for 5 placeholder"):
        adapter.encode_multimodal([_feature((1, 4, 4), length=5)])


def test_build_profiles_one_maximal_image_and_needs_media(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The adapter encodes one item at a time, so the startup profile encodes
    the largest image vLLM admits. With zero media limits nothing is built."""
    import vllm.multimodal.encoder_budget as encoder_budget

    monkeypatch.setattr(
        encoder_budget,
        "MultiModalBudget",
        lambda *args, **kwargs: SimpleNamespace(mm_max_toks_per_item={"image": 20}),
    )
    model = _micro_model()

    def build(supports_media: bool) -> Any:
        model_config = SimpleNamespace(supports_multimodal_inputs=supports_media)
        return DefaultModelAdapter().build_multimodal_adapter(
            model,
            SimpleNamespace(model_type="glm5_next"),
            vllm_config=SimpleNamespace(model_config=model_config),
        )

    assert build(False) is None
    adapter = build(True)
    [result] = adapter.encode_multimodal(adapter.profile_features())

    # The largest square merge grid within 20 tokens is 4 x 4.
    assert result.hidden_states.shape == (16, 32)
