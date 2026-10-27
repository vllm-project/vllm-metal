# SPDX-License-Identifier: Apache-2.0
"""Offline image-only Nano adapter contract tests (no model weights)."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import mlx.core as mx
import pytest
import torch
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItem

from vllm_metal.attention.attention_contracts import attention_contract_for
from vllm_metal.multimodal import MultiModalFeatureSpec, PlaceholderRange
from vllm_metal.multimodal.embeddings import merge_multimodal_embeddings
from vllm_metal.multimodal.nemotron_nano_vl import NemotronNanoVLImageAdapter
from vllm_metal.v1.model_adapter import DefaultModelAdapter

_IMAGE_TOKEN = 42
_HIDDEN = 8


class _LanguageModel:
    def __init__(self) -> None:
        self.backbone = SimpleNamespace(
            layers=[object()], embeddings=self.get_input_embeddings
        )
        self.calls: list[dict[str, Any]] = []

    def get_input_embeddings(self, input_ids: mx.array) -> mx.array:
        return mx.ones((*input_ids.shape, _HIDDEN), dtype=mx.float32)

    def __call__(
        self,
        input_ids: mx.array,
        *,
        inputs_embeds: mx.array,
        cache: list[Any],
    ) -> Any:
        self.calls.append(
            {"input_ids": input_ids, "inputs_embeds": inputs_embeds, "cache": cache}
        )
        return SimpleNamespace(logits=mx.zeros((*input_ids.shape, 16)))


class _Model:
    def __init__(
        self, *, output_rows_delta: int = 0, output_hidden_size: int = _HIDDEN
    ) -> None:
        self.config = SimpleNamespace(
            text_config=SimpleNamespace(hidden_size=_HIDDEN),
            vision_config=SimpleNamespace(patch_size=16),
            downsample_ratio=0.5,
            force_image_size=32,
            img_context_token_id=_IMAGE_TOKEN,
        )
        patch_generator = SimpleNamespace(
            embedder=SimpleNamespace(weight=mx.zeros((1,), dtype=mx.float16))
        )
        self.vision_model = SimpleNamespace(
            radio_model=SimpleNamespace(
                model=SimpleNamespace(patch_generator=patch_generator)
            )
        )
        self.language_model = _LanguageModel()
        self.output_rows_delta = output_rows_delta
        self.output_hidden_size = output_hidden_size
        self.pixels_seen: list[mx.array] = []
        self.composite_calls = 0

    def _extract_feature_single(self, pixels: mx.array) -> mx.array:
        self.pixels_seen.append(pixels)
        rows_per_tile = (pixels.shape[-2] // 32) * (pixels.shape[-1] // 32)
        return mx.ones(
            (
                pixels.shape[0],
                rows_per_tile + self.output_rows_delta,
                self.output_hidden_size,
            ),
            dtype=mx.float16,
        )

    def __call__(self, *_args: Any, **_kwargs: Any) -> Any:
        self.composite_calls += 1
        raise AssertionError("Composite forward would re-encode the image")


def _item(fields: dict[str, torch.Tensor], *, tiles: int) -> MultiModalKwargsItem:
    elements = {}
    for key, value in fields.items():
        if key == "pixel_values_flat" and tiles > 1:
            config = MultiModalFieldConfig.flat_from_sizes(
                "image", torch.tensor([tiles])
            )
        else:
            config = MultiModalFieldConfig.batched("image", keep_on_cpu=True)
        elements[key] = config.build_elems(key, value)[0]
    return MultiModalKwargsItem(elements)


def _feature(
    *,
    offset: int = 1,
    height: int = 64,
    width: int = 32,
    tiles: int = 1,
    length: int | None = None,
    modality: str = "image",
    reported_tokens: int | None = None,
    reported_tiles: int | None = None,
    image_size: tuple[int, int] | None = None,
) -> MultiModalFeatureSpec:
    count = tiles * (height // 32) * (width // 32)
    fields = {
        "pixel_values_flat": torch.zeros((tiles, 3, height, width)),
    }
    if reported_tokens is not None:
        fields["num_tokens_per_image"] = torch.tensor([reported_tokens])
    if reported_tiles is not None:
        fields["image_num_patches"] = torch.tensor([reported_tiles])
    if image_size is not None:
        fields["imgs_sizes"] = torch.tensor([image_size])
    return MultiModalFeatureSpec(
        data=_item(fields, tiles=tiles),
        modality=modality,
        identifier=f"{modality}-{offset}",
        mm_position=PlaceholderRange(
            offset=offset, length=count if length is None else length
        ),
    )


def _tokens(feature: MultiModalFeatureSpec) -> list[int]:
    count = feature.mm_position.get_num_embeds()
    return [1] * feature.mm_position.offset + [_IMAGE_TOKEN] * count + [2]


def test_exact_nano_architecture_registers_image_adapter() -> None:
    model = _Model()
    hf_config = SimpleNamespace(
        model_type="nemotron_h_nano_omni",
        architectures=["NemotronH_Nano_VL_V2"],
    )
    adapter = DefaultModelAdapter().build_multimodal_adapter(model, hf_config)

    assert isinstance(adapter, NemotronNanoVLImageAdapter)
    assert adapter.text_model() is model.language_model
    assert adapter.forward_ready is True
    assert adapter.requires_explicit_positions is False


def test_omni_or_other_architecture_is_not_auto_enabled() -> None:
    model = _Model()
    hf_config = SimpleNamespace(
        model_type="nemotron_h_nano_omni",
        architectures=["NemotronH_Nano_Omni"],
    )
    assert DefaultModelAdapter().build_multimodal_adapter(model, hf_config) is None


def test_dynamic_image_positions_and_projected_rows() -> None:
    model = _Model()
    adapter = NemotronNanoVLImageAdapter.from_loaded_model(model)
    feature = _feature(reported_tokens=2, image_size=(64, 32))

    positions, delta = adapter.get_mrope_input_positions(_tokens(feature), [feature])
    encoded = adapter.encode_multimodal([feature])

    assert positions.tolist() == [[[0, 1, 2, 3]]] * 3
    assert delta == 0
    assert len(encoded) == 1
    assert encoded[0].hidden_states.shape == (2, _HIDDEN)
    assert encoded[0].deepstack_visual_embeds is None
    assert model.pixels_seen[0].shape == (1, 3, 64, 32)
    assert model.pixels_seen[0].dtype == mx.float32
    assert model.composite_calls == 0


def test_static_tiled_image_uses_all_projected_rows() -> None:
    model = _Model()
    adapter = NemotronNanoVLImageAdapter.from_loaded_model(model)
    feature = _feature(
        height=32,
        width=32,
        tiles=3,
        reported_tiles=3,
        length=3,
    )

    adapter.get_mrope_input_positions(_tokens(feature), [feature])
    encoded = adapter.encode_multimodal([feature])

    assert encoded[0].hidden_states.shape == (3, _HIDDEN)
    assert model.pixels_seen[0].shape == (3, 3, 32, 32)


def test_multiple_images_preserve_feature_order() -> None:
    adapter = NemotronNanoVLImageAdapter.from_loaded_model(_Model())
    first = _feature(offset=1, height=32, width=32)
    second = _feature(offset=4, height=64, width=32)
    tokens = [1, _IMAGE_TOKEN, 2, 3, _IMAGE_TOKEN, _IMAGE_TOKEN, 4]

    positions, delta = adapter.get_mrope_input_positions(tokens, [second, first])
    encoded = adapter.encode_multimodal([first, second])

    assert positions.shape == (3, 1, len(tokens))
    assert delta == 0
    assert [result.hidden_states.shape[0] for result in encoded] == [1, 2]


def _masked_feature() -> tuple[MultiModalFeatureSpec, list[int]]:
    """Use vLLM's Nano replacement and scheduler range conversion."""
    from vllm.multimodal.processing.processor import PlaceholderFeaturesInfo
    from vllm.transformers_utils.processors.nano_nemotron_vl import (
        NanoNemotronVLProcessor,
    )

    # Only the tokenizer is synthetic. The marker selection and full-span
    # range come from the installed vLLM processor, without a checkpoint.
    tokenizer = Mock()
    tokenizer.encode.return_value = [40, _IMAGE_TOKEN, _IMAGE_TOKEN, 41]
    processor = SimpleNamespace(
        tokenizer=tokenizer, _img_context_token_ids=[_IMAGE_TOKEN]
    )
    update = NanoNemotronVLProcessor.get_image_repl(processor, 2, None)
    tokenizer.encode.assert_called_once_with(
        "<img><image><image></img>", truncation=None, add_special_tokens=False
    )
    assert update.is_embed is not None
    position = PlaceholderFeaturesInfo(
        modality="image",
        item_idx=0,
        start_idx=1,
        tokens=update.full,
        is_embed=update.is_embed(update.full),
    ).to_range()
    return replace(_feature(), mm_position=position), [1, *update.full, 2]


def test_vllm_nano_markers_keep_their_text_embeddings() -> None:
    feature, tokens = _masked_feature()
    adapter = NemotronNanoVLImageAdapter.from_loaded_model(_Model())

    positions, delta = adapter.get_mrope_input_positions(tokens, [feature])
    assert positions.shape == (3, 1, 6)
    assert delta == 0
    assert feature.mm_position.length == 4
    assert feature.mm_position.get_num_embeds() == 2

    encoded = adapter.encode_multimodal([feature])[0].hidden_states
    text_embeds = mx.arange(6 * _HIDDEN).reshape(1, 6, _HIDDEN).astype(mx.float32)
    merged = merge_multimodal_embeddings(
        text_embeds,
        encoded,
        mx.array([False, *feature.mm_position.is_embed.tolist(), False]),
    )
    assert mx.array_equal(merged[:, 2:4], encoded[None]).item()
    for index in (0, 1, 4, 5):
        assert mx.array_equal(merged[:, index], text_embeds[:, index]).item()


def test_masked_placeholder_bounds_include_the_end_marker() -> None:
    feature, tokens = _masked_feature()
    adapter = NemotronNanoVLImageAdapter.from_loaded_model(_Model())
    with pytest.raises(ValueError, match="outside the prompt"):
        adapter.get_mrope_input_positions(tokens[:4], [feature])


def test_masked_placeholder_overlap_includes_the_end_marker() -> None:
    feature, tokens = _masked_feature()
    overlapping = _feature(offset=4, height=32, width=32)
    adapter = NemotronNanoVLImageAdapter.from_loaded_model(_Model())
    with pytest.raises(ValueError, match="overlap"):
        adapter.get_mrope_input_positions(tokens, [feature, overlapping])


@pytest.mark.parametrize(
    "mask",
    [
        torch.ones(3, dtype=torch.bool),
        torch.ones((4, 1), dtype=torch.bool),
        torch.ones(4),
    ],
)
def test_malformed_placeholder_mask_is_rejected_before_encoding(
    mask: torch.Tensor,
) -> None:
    feature, _ = _masked_feature()
    feature = replace(feature, mm_position=replace(feature.mm_position, is_embed=mask))
    model = _Model()
    adapter = NemotronNanoVLImageAdapter.from_loaded_model(model)
    with pytest.raises(ValueError, match="boolean mask"):
        adapter.encode_multimodal([feature])
    assert not model.pixels_seen


def test_masked_non_image_token_is_rejected() -> None:
    feature, tokens = _masked_feature()
    tokens[2] = 7
    adapter = NemotronNanoVLImageAdapter.from_loaded_model(_Model())
    with pytest.raises(ValueError, match="img_context_token_id"):
        adapter.get_mrope_input_positions(tokens, [feature])


def test_empty_image_is_rejected_before_encoding() -> None:
    model = _Model()
    adapter = NemotronNanoVLImageAdapter.from_loaded_model(model)
    with pytest.raises(ValueError, match="shape"):
        adapter.encode_multimodal([_feature(height=0, length=1)])
    assert not model.pixels_seen


@pytest.mark.parametrize(
    ("feature", "match"),
    [
        (_feature(length=1), "pixels imply 2"),
        (_feature(reported_tokens=3), "num_tokens_per_image=3"),
        (_feature(tiles=2, height=32, reported_tiles=1), "image_num_patches=1"),
        (_feature(image_size=(32, 64)), "imgs_sizes disagrees"),
        (_feature(height=48, width=32), "divisible by 32"),
    ],
)
def test_invalid_geometry_or_metadata_fails_closed(
    feature: MultiModalFeatureSpec, match: str
) -> None:
    adapter = NemotronNanoVLImageAdapter.from_loaded_model(_Model())
    with pytest.raises(ValueError, match=match):
        adapter.get_mrope_input_positions(_tokens(feature), [feature])


def test_wrong_placeholder_token_is_rejected() -> None:
    adapter = NemotronNanoVLImageAdapter.from_loaded_model(_Model())
    feature = _feature()
    tokens = _tokens(feature)
    tokens[feature.mm_position.offset] = 0
    with pytest.raises(ValueError, match="img_context_token_id"):
        adapter.get_mrope_input_positions(tokens, [feature])


def test_overlapping_image_ranges_are_rejected() -> None:
    adapter = NemotronNanoVLImageAdapter.from_loaded_model(_Model())
    first = _feature(offset=1, height=32, width=32)
    second = _feature(offset=1, height=32, width=32)
    with pytest.raises(ValueError, match="overlap"):
        adapter.get_mrope_input_positions(_tokens(first), [first, second])


@pytest.mark.parametrize("modality", ["audio", "video", "image_embeds"])
def test_non_image_pixels_are_not_accepted(modality: str) -> None:
    adapter = NemotronNanoVLImageAdapter.from_loaded_model(_Model())
    feature = _feature(modality=modality)
    with pytest.raises(NotImplementedError, match="image pixels only"):
        adapter.encode_multimodal([feature])


def test_precomputed_image_embeds_are_not_accepted() -> None:
    adapter = NemotronNanoVLImageAdapter.from_loaded_model(_Model())
    field = MultiModalFieldConfig.batched("image")
    embeds = field.build_elems("image_embeds", torch.zeros((1, 1, _HIDDEN)))[0]
    feature = MultiModalFeatureSpec(
        data=MultiModalKwargsItem({"image_embeds": embeds}),
        modality="image",
        identifier="image-embeds",
        mm_position=PlaceholderRange(offset=1, length=1),
    )
    with pytest.raises(NotImplementedError, match="precomputed image_embeds"):
        adapter.encode_multimodal([feature])


def test_vision_row_mismatch_fails_before_cache_insert() -> None:
    adapter = NemotronNanoVLImageAdapter.from_loaded_model(_Model(output_rows_delta=1))
    with pytest.raises(ValueError, match="vision produced 3 rows for 2"):
        adapter.encode_multimodal([_feature()])


def test_vision_width_mismatch_fails_before_merge() -> None:
    adapter = NemotronNanoVLImageAdapter.from_loaded_model(
        _Model(output_hidden_size=_HIDDEN + 1)
    )
    with pytest.raises(ValueError, match="vision projection width 9"):
        adapter.encode_multimodal([_feature()])


def test_call_lm_uses_inner_model_with_premerged_embeddings() -> None:
    model = _Model()
    adapter = NemotronNanoVLImageAdapter.from_loaded_model(model)
    input_ids = mx.array([[1, _IMAGE_TOKEN, 2]])
    inputs_embeds = adapter.embed_tokens(input_ids)
    cache = [object()]
    positions = mx.zeros((3, 1, 3), dtype=mx.int32)

    output = adapter.call_lm(input_ids, inputs_embeds, cache, positions)

    assert output.logits.shape == (1, 3, 16)
    call = model.language_model.calls[0]
    assert call["input_ids"] is input_ids
    assert call["inputs_embeds"] is inputs_embeds
    assert call["cache"] is cache
    assert model.composite_calls == 0


def test_nano_attention_contract_is_position_free() -> None:
    attention_cls = type(
        "NemotronHAttention",
        (),
        {"__module__": "mlx_vlm.models.nemotron_h.language"},
    )
    assert attention_contract_for(attention_cls()).use_rope is False


@pytest.mark.parametrize(
    "max_tiles,thumbnail,images,expected",
    [
        (None, True, 2, 13),
        (2, True, 2, 3),
        (16, True, 1, 17),
        (16, False, 1, 16),
        (1, True, 1, 1),
        (2, True, 0, 0),
    ],
)
def test_profile_matches_configured_image_limits(
    max_tiles, thumbnail, images, expected
):
    from vllm.config.multimodal import MultiModalConfig

    mm_config = MultiModalConfig(
        limit_per_prompt={"image": images},
        mm_processor_kwargs={"max_num_tiles": max_tiles},
    )
    adapter = DefaultModelAdapter().build_multimodal_adapter(
        _Model(),
        SimpleNamespace(
            architectures=["NemotronH_Nano_VL_V2"], use_thumbnail=thumbnail
        ),
        model_config=SimpleNamespace(multimodal_config=mm_config),
    )
    encoded = adapter.encode_multimodal(adapter.profile_features())
    assert [item.hidden_states.shape for item in encoded] == (
        [(expected, _HIDDEN)] if expected else []
    )


def test_encoding_cannot_mutate_cached_torch_pixels(monkeypatch):
    model = _Model()
    pixels = torch.ones((1, 3, 32, 32))
    original = pixels.clone()
    feature = replace(
        _feature(height=32), data=_item({"pixel_values_flat": pixels}, tiles=1)
    )
    encode = model._extract_feature_single

    def consume(pixels):
        pixels[0, 0, 0, 0] = 99
        mx.eval(pixels)
        return encode(pixels)

    monkeypatch.setattr(model, "_extract_feature_single", consume)
    adapter = NemotronNanoVLImageAdapter.from_loaded_model(model)
    mx.eval(adapter.encode_multimodal([feature])[0].hidden_states)
    assert torch.equal(pixels, original)


@pytest.fixture
def tiny_nano_model() -> tuple[Any, Any]:
    """A randomly initialized, weight-free Nano model from the pinned mlx-vlm."""
    pytest.importorskip("mlx_vlm.models.nemotron_h.language")
    from mlx_vlm.models.nemotron_h_nano_omni.config import (
        ModelConfig,
        TextConfig,
        VisionConfig,
    )
    from mlx_vlm.models.nemotron_h_nano_omni.nemotron_h_nano_omni import Model

    config = ModelConfig(
        text_config=TextConfig(
            model_type="nemotron_h",
            vocab_size=100,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            max_position_embeddings=512,
            num_attention_heads=4,
            num_key_value_heads=2,
            attention_bias=False,
            mamba_num_heads=4,
            mamba_head_dim=8,
            mamba_proj_bias=False,
            ssm_state_size=32,
            conv_kernel=4,
            n_groups=2,
            mlp_bias=False,
            layer_norm_epsilon=1e-5,
            use_bias=False,
            use_conv_bias=True,
            hybrid_override_pattern=list("M*"),
        ),
        vision_config=VisionConfig(
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            image_size=32,
            max_resolution=32,
            patch_size=16,
        ),
        model_type="NemotronH_Nano_VL_V2",
        force_image_size=32,
        ps_version="v2",
        downsample_ratio=0.5,
        projector_hidden_size=64,
        vit_hidden_size=16,
        img_context_token_id=_IMAGE_TOKEN,
    )
    mx.random.seed(0)
    model = Model(config)
    return model, config


@pytest.mark.parametrize("weight_dtype", [mx.float16, mx.bfloat16])
def test_tiny_vision_preserves_direct_mlx_vlm_pixel_precision(
    tiny_nano_model: tuple[Any, Any], weight_dtype: mx.Dtype
) -> None:
    model, _ = tiny_nano_model
    model.set_dtype(weight_dtype)
    adapter = NemotronNanoVLImageAdapter.from_loaded_model(model)
    pixels = torch.linspace(-1.0, 1.0, 3 * 32 * 32).reshape(1, 3, 32, 32)
    feature = replace(
        _feature(height=32), data=_item({"pixel_values_flat": pixels}, tiles=1)
    )

    direct_rows = model.extract_feature(mx.array(pixels.numpy())).reshape(1, -1)
    adapter_rows = adapter.encode_multimodal([feature])[0].hidden_states
    mx.eval(direct_rows, adapter_rows)

    # Identical model, pixels and encoder calls should agree exactly, even
    # when checkpoint weights use less precision than the processor pixels.
    assert mx.array_equal(adapter_rows, direct_rows).item()


def test_tiny_real_mlx_vlm_model_matches_decomposed_image_forward(
    tiny_nano_model: tuple[Any, Any],
) -> None:
    """Compare adapter embeddings/logits with the same tiny mlx-vlm model.

    This checks API decomposition, not checkpoint quality or serving parity.
    """
    model, config = tiny_nano_model
    assert (
        attention_contract_for(model.language_model.layers[1].mixer).use_rope is False
    )
    adapter = NemotronNanoVLImageAdapter.from_loaded_model(model)
    feature = _feature(height=32, width=32)
    input_ids = mx.array([[1, _IMAGE_TOKEN, 2]])
    pixels = mx.zeros((1, 3, 32, 32), dtype=mx.float32)

    direct_embeds = model.get_input_embeddings(
        input_ids, pixel_values=pixels
    ).inputs_embeds
    encoded = adapter.encode_multimodal([feature])[0].hidden_states
    runner_embeds = merge_multimodal_embeddings(
        adapter.embed_tokens(input_ids), encoded, mx.array([[False, True, False]])
    )
    mx.eval(direct_embeds, runner_embeds)
    assert mx.allclose(runner_embeds, direct_embeds).item()

    direct_output = model(input_ids, pixel_values=pixels, cache=None)
    adapter_output = adapter.call_lm(
        input_ids,
        runner_embeds,
        [None] * config.text_config.num_hidden_layers,
        mx.zeros((3, 1, 3), dtype=mx.int32),
    )
    mx.eval(direct_output.logits, adapter_output.logits)
    assert mx.allclose(adapter_output.logits, direct_output.logits).item()


def test_tiny_quantized_projector_preserves_direct_pixel_precision(tiny_nano_model):
    import mlx.nn as nn

    model, _ = tiny_nano_model
    model.set_dtype(mx.bfloat16)
    nn.quantize(model.mlp1, group_size=32, bits=4)
    assert model.mlp1.layers[1].weight.dtype == mx.uint32
    pixels = torch.linspace(-1, 1, 3 * 32 * 32).reshape(1, 3, 32, 32)
    feature = replace(
        _feature(height=32), data=_item({"pixel_values_flat": pixels}, tiles=1)
    )
    adapter = NemotronNanoVLImageAdapter.from_loaded_model(model)
    direct = model.extract_feature(mx.array(pixels.numpy())).reshape(1, -1)
    actual = adapter.encode_multimodal([feature])[0].hidden_states
    assert mx.array_equal(actual, direct).item()


def test_tiny_vision_profiles_thirteen_tiles(tiny_nano_model):
    # PR #1160's maximal-image regression, using our existing tiny model.
    model, _ = tiny_nano_model
    adapter = NemotronNanoVLImageAdapter.from_loaded_model(model)
    (encoded,) = adapter.encode_multimodal(adapter.profile_features())
    assert encoded.hidden_states.shape == (13, 32)


@pytest.mark.parametrize("model_type", ["NemotronH_Nano_VL_V2", "nemotron_h_nano_omni"])
@pytest.mark.parametrize("omit", [None, "video_embedder.weight", "embedder.weight"])
def test_nano_compat_strict_loading_does_not_hide_missing_omni_weights(
    tiny_nano_model, model_type, omit
):
    from mlx.utils import tree_flatten

    from vllm_metal.compat import _patch_mlx_vlm_nemotron_nano_vl_v2

    _patch_mlx_vlm_nemotron_nano_vl_v2()
    model, config = tiny_nano_model
    config.model_type = model_type
    prefix = "vision_model.radio_model.model.patch_generator."
    weights = dict(tree_flatten(model.parameters()))
    if omit is not None:
        del weights[prefix + omit]
    sanitized = model.sanitize(weights)
    allowed = omit is None or (
        model_type == "NemotronH_Nano_VL_V2" and omit == "video_embedder.weight"
    )
    if allowed:
        model.load_weights(list(sanitized.items()), strict=True)
        assert (
            "video_embedder" in model.vision_model.radio_model.model.patch_generator
        ) == (omit is None)
    else:
        with pytest.raises(ValueError, match=omit):
            model.load_weights(list(sanitized.items()), strict=True)


def test_tiny_real_nano_hybrid_layers_accept_paged_wrappers(
    tiny_nano_model: tuple[Any, Any],
) -> None:
    # The VLM's nested Nemotron-H blocks must be found by the same hybrid
    # runtime that owns text-only Mamba state and paged attention.
    from importlib.metadata import version

    from packaging.version import Version

    if Version(version("vllm").split("+")[0]) < Version("0.29.0"):
        pytest.skip("Hybrid wrapper integration needs the pinned vLLM 0.29 API")

    model, config = tiny_nano_model
    from vllm_metal.attention.impls.mamba2 import Mamba2PagedStateWrapper
    from vllm_metal.attention.impls.sdpa_wrapper import SDPAPagedAttentionWrapper
    from vllm_metal.attention.runtime.factory import build_hybrid_runtime_plan
    from vllm_metal.attention.runtime.hybrid import HybridPagedAttentionRuntime

    runtime = HybridPagedAttentionRuntime(
        hybrid_plan=build_hybrid_runtime_plan(
            vars(config.text_config), 2, (torch.float32, torch.float32)
        ),
        dtype=mx.float32,
    )
    from tests.stub_runner import initialize_hybrid_runtime

    initialize_hybrid_runtime(runtime, 4, block_size=8, num_kv_heads=2, head_dim=8)
    assert runtime.patch_model(model) == 2
    assert isinstance(model.language_model.layers[0].mixer, Mamba2PagedStateWrapper)
    assert isinstance(model.language_model.layers[1].mixer, SDPAPagedAttentionWrapper)


def test_tiny_nano_mamba_state_follows_requests_through_reorder_and_reuse(
    tiny_nano_model: tuple[Any, Any],
) -> None:
    """Exercise the VLM's real mixer with upstream's per-request state pool."""
    from mlx_vlm.models.cache import ArraysCache

    from tests.stub_runner import make_state_cache
    from vllm_metal.attention.context import (
        PagedAttentionContext,
        clear_context,
        set_context,
    )
    from vllm_metal.attention.impls.mamba2 import Mamba2PagedStateWrapper

    model, _ = tiny_nano_model
    mixer = model.language_model.layers[0].mixer
    pool = make_state_cache(
        num_layers=1,
        max_seqs=2,
        conv_kernel_dim=mixer.conv_kernel_size,
        conv_dim=mixer.conv_dim,
        num_v_heads=mixer.num_heads,
        value_head_dim=mixer.head_dim,
        key_head_dim=mixer.ssm_state_size,
        dtype=mx.float32,
    )
    wrapper = Mamba2PagedStateWrapper(mixer, 0, 0, pool)
    first, second = mx.random.normal((1, 3, 32)), mx.random.normal((1, 5, 32))
    first_cache, second_cache = ArraysCache(size=2), ArraysCache(size=2)
    first_ref = mixer(first, mask=None, cache=first_cache)
    second_ref = mixer(second, mask=None, cache=second_cache)

    def set_step(cu_seqlens: list[int], slots: list[int], num_decode: int) -> None:
        set_context(
            PagedAttentionContext(
                slot_mapping=[0] * cu_seqlens[-1],
                cu_seqlens=cu_seqlens,
                state_slot_mapping=slots,
                num_decode_requests=num_decode,
            )
        )

    def assert_state(slot: int, reference: ArraysCache, row: int = 0) -> None:
        assert mx.array_equal(pool.conv_states[0][slot], reference[0][row]).item()
        assert mx.array_equal(pool.recurrent_states[0][slot], reference[1][row]).item()

    clear_context()
    try:
        set_step([0, 3, 8], [1, 0], 0)
        prefill = wrapper(mx.concatenate([first, second], axis=1))
        assert mx.array_equal(prefill, mx.concatenate([first_ref, second_ref], axis=1))
        assert_state(1, first_cache)
        assert_state(0, second_cache)

        # The next scheduler batch reverses request order. Compare with the
        # same batched direct mixer so batching-related rounding is identical.
        reference = ArraysCache(size=2)
        for index in (0, 1):
            reference[index] = mx.concatenate(
                [second_cache[index], first_cache[index]], axis=0
            )
        next_tokens = mx.random.normal((2, 1, 32))
        direct_decode = mixer(next_tokens, mask=None, cache=reference)
        set_step([0, 1, 2], [0, 1], 2)
        decode = wrapper(next_tokens.reshape(1, 2, 32))
        assert mx.array_equal(decode, direct_decode.reshape(1, 2, 32))
        assert_state(0, reference, 0)
        assert_state(1, reference, 1)

        # Simulate the runtime releasing one request slot for a new prompt.
        pool.zero_slots([1], [0])
        fresh = mx.random.normal((1, 4, 32))
        fresh_cache = ArraysCache(size=2)
        fresh_ref = mixer(fresh, mask=None, cache=fresh_cache)
        set_step([0, 4], [1], 0)
        assert mx.array_equal(wrapper(fresh), fresh_ref)
        assert_state(1, fresh_cache)
        assert_state(0, reference, 0)
    finally:
        clear_context()
