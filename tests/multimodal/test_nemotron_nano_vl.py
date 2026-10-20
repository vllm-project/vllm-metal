# SPDX-License-Identifier: Apache-2.0
"""Tests for the Nemotron Nano VL V2 multimodal adapter."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
import torch
from mlx.utils import tree_flatten
from mlx_vlm import utils as mlx_vlm_utils
from mlx_vlm.models.nemotron_h_nano_omni import Model, ModelConfig
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItem

from vllm_metal import compat
from vllm_metal.multimodal import (
    MultiModalFeatureSpec,
    PlaceholderRange,
    merge_multimodal_embeddings,
)
from vllm_metal.multimodal.nemotron_nano_vl import NemotronNanoVLMultimodalAdapter

_TILE = 64
_PATCH = 16
# (64 / 16) ** 2 patches per tile, pixel-shuffled by 0.5 in each dimension.
_TOKENS_PER_TILE = 4
_IMAGE_TOKEN_ID = 7
_IMG_START_ID = 5
_IMG_END_ID = 6


def _tiny_config() -> ModelConfig:
    return ModelConfig.from_dict(
        {
            "model_type": "NemotronH_Nano_VL_V2",
            "force_image_size": _TILE,
            "downsample_ratio": 0.5,
            "ps_version": "v2",
            "projector_hidden_size": 48,
            "vit_hidden_size": 32,
            "img_context_token_id": _IMAGE_TOKEN_ID,
            "llm_config": {
                "model_type": "nemotron_h",
                "vocab_size": 64,
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 3,
                "hybrid_override_pattern": "M*-",
                "max_position_embeddings": 256,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 8,
                "attention_bias": False,
                "mamba_num_heads": 4,
                "mamba_head_dim": 8,
                "mamba_proj_bias": False,
                "ssm_state_size": 8,
                "conv_kernel": 4,
                "n_groups": 2,
                "mlp_bias": False,
                "layer_norm_epsilon": 1e-5,
                "use_bias": False,
                "use_conv_bias": True,
            },
            "vision_config": {
                "hidden_size": 32,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "intermediate_size": 64,
                "patch_size": _PATCH,
                "max_resolution": _TILE,
                "args": {"register_multiple": 4},
            },
        }
    )


@pytest.fixture(scope="module")
def model() -> Model:
    mx.random.seed(0)
    model = Model(_tiny_config())
    # Random weights: the zero-initialised position embedding and class
    # tokens would otherwise make every tile encode the same way.
    params = [
        (name, mx.random.normal(value.shape) * 0.1)
        for name, value in tree_flatten(model.parameters())
    ]
    model.load_weights(params)
    mx.eval(model.parameters())
    return model


def _image_feature(
    pixels: np.ndarray, *, offset: int, num_embeds: int, modality: str = "image"
) -> MultiModalFeatureSpec:
    field = MultiModalFieldConfig.batched(modality)
    elem = field.build_elems("pixel_values_flat", torch.from_numpy(pixels)[None])[0]
    # <img> and </img> sit inside the placeholder but take no embedding.
    is_embed = torch.tensor([False] + [True] * num_embeds + [False])
    return MultiModalFeatureSpec(
        data=MultiModalKwargsItem({"pixel_values_flat": elem}),
        modality=modality,
        identifier=f"{modality}-{offset}",
        mm_position=PlaceholderRange(
            offset=offset, length=num_embeds + 2, is_embed=is_embed
        ),
    )


def _prompt(num_tiles: int) -> list[int]:
    image = [_IMG_START_ID] + [_IMAGE_TOKEN_ID] * num_tiles * _TOKENS_PER_TILE
    return [1, 9, 10] + image + [_IMG_END_ID, 11, 12]


class TestNemotronNanoVLAdapter:
    def test_paged_inputs_reproduce_mlx_vlm_forward(self, model: Model) -> None:
        """The adapter's encode + splice + LM call matches mlx-vlm's composite.

        This is the path the runner takes: vLLM tiles the image, the adapter
        encodes the tiles, and the runner splices the rows into the
        ``is_embed`` positions before calling the language model.
        """
        num_tiles = 2
        pixels = np.random.default_rng(0).standard_normal(
            (num_tiles, 3, _TILE, _TILE), dtype=np.float32
        )
        tokens = _prompt(num_tiles)
        feature = _image_feature(
            pixels, offset=3, num_embeds=num_tiles * _TOKENS_PER_TILE
        )
        adapter = NemotronNanoVLMultimodalAdapter.from_loaded_model(model)

        (encoded,) = adapter.encode_multimodal([feature])
        input_ids = mx.array([tokens])
        embeds = merge_multimodal_embeddings(
            adapter.embed_tokens(input_ids),
            encoded.hidden_states,
            input_ids == _IMAGE_TOKEN_ID,
        )
        positions, delta = adapter.get_mrope_input_positions(tokens, [feature])
        logits = adapter.call_lm(
            input_ids,
            embeds,
            model.language_model.make_cache(),
            positions,
        ).logits

        expected = model(
            input_ids,
            pixel_values=mx.array(pixels),
            cache=model.language_model.make_cache(),
        ).logits
        assert encoded.hidden_states.shape == (num_tiles * _TOKENS_PER_TILE, 32)
        assert delta == 0
        assert positions.tolist() == [[list(range(len(tokens)))]] * 3
        np.testing.assert_allclose(
            np.array(logits), np.array(expected), rtol=1e-4, atol=1e-4
        )

    def test_rejects_row_count_that_disagrees_with_placeholders(
        self, model: Model
    ) -> None:
        pixels = np.zeros((1, 3, _TILE, _TILE), dtype=np.float32)
        feature = _image_feature(pixels, offset=0, num_embeds=_TOKENS_PER_TILE + 1)
        adapter = NemotronNanoVLMultimodalAdapter.from_loaded_model(model)

        with pytest.raises(ValueError, match="placeholder"):
            adapter.encode_multimodal([feature])

    def test_rejects_non_image_modalities(self, model: Model) -> None:
        pixels = np.zeros((1, 3, _TILE, _TILE), dtype=np.float32)
        feature = _image_feature(
            pixels, offset=0, num_embeds=_TOKENS_PER_TILE, modality="video"
        )
        adapter = NemotronNanoVLMultimodalAdapter.from_loaded_model(model)

        with pytest.raises(NotImplementedError, match="image inputs only"):
            adapter.encode_multimodal([feature])
        with pytest.raises(NotImplementedError, match="image inputs only"):
            adapter.get_mrope_input_positions([1, 2], [feature])

    def test_profile_feature_encodes_at_the_maximum_tile_count(
        self, model: Model
    ) -> None:
        adapter = NemotronNanoVLMultimodalAdapter.from_loaded_model(model)

        (feature,) = adapter.profile_features()
        (encoded,) = adapter.encode_multimodal([feature])

        assert encoded.hidden_states.shape[0] == 13 * _TOKENS_PER_TILE


class TestNemotronNanoVLV2CompatPatch:
    _VIDEO_EMBEDDER_KEY = (
        "vision_model.radio_model.model.patch_generator.video_embedder.weight"
    )

    def test_v2_model_type_resolves_to_the_omni_module(self, monkeypatch) -> None:
        remapping = dict(mlx_vlm_utils.MODEL_REMAPPING)
        remapping.pop("nemotronh_nano_vl_v2", None)
        monkeypatch.setattr(mlx_vlm_utils, "MODEL_REMAPPING", remapping)
        config = {"model_type": "NemotronH_Nano_VL_V2"}
        with pytest.raises(ValueError, match="nemotronh_nano_vl_v2"):
            mlx_vlm_utils.get_model_and_args(config)

        compat._patch_mlx_vlm_nemotron_nano_vl_v2()

        module, _ = mlx_vlm_utils.get_model_and_args(config)
        assert module.__name__ == "mlx_vlm.models.nemotron_h_nano_omni"

    def test_strict_load_accepts_a_checkpoint_without_video_embedder(self) -> None:
        """Nano VL V2 checkpoints ship no Omni tubelet video embedder."""
        compat._patch_mlx_vlm_nemotron_nano_vl_v2()
        model = Model(_tiny_config())
        weights = {
            key: value
            for key, value in tree_flatten(model.parameters())
            if key != self._VIDEO_EMBEDDER_KEY
        }

        model.load_weights(list(model.sanitize(weights).items()), strict=True)

        patch_generator = model.vision_model.radio_model.model.patch_generator
        assert "video_embedder" not in patch_generator

    def test_keeps_a_shipped_video_embedder(self) -> None:
        compat._patch_mlx_vlm_nemotron_nano_vl_v2()
        model = Model(_tiny_config())
        weights = dict(tree_flatten(model.parameters()))

        model.load_weights(list(model.sanitize(weights).items()), strict=True)

        assert "video_embedder" in model.vision_model.radio_model.model.patch_generator
