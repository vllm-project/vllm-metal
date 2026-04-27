# SPDX-License-Identifier: Apache-2.0
"""Tests for YOCO fast-prefill indexing helpers."""

from __future__ import annotations

import pytest

from vllm_metal.yoco_fast_prefill import (
    build_yoco_reduced_context_from_full_metadata,
    get_yoco_fast_prefill_ineligibility_reason,
    is_yoco_fast_prefill_eligible,
    patch_gemma4_yoco_fast_prefill,
)


@pytest.mark.parametrize("model_type", ["gemma4", "gemma4_text"])
def test_eligible_for_gemma4_yoco_paged_attention(model_type: str) -> None:
    assert is_yoco_fast_prefill_eligible(
        {
            "model_type": model_type,
            "num_hidden_layers": 42,
            "num_kv_shared_layers": 18,
        },
        use_paged_attention=True,
    )


@pytest.mark.parametrize(
    "model_args",
    [
        {"model_type": "qwen3", "num_hidden_layers": 32, "num_kv_shared_layers": 8},
        {"model_type": "gemma4", "num_hidden_layers": 42, "num_kv_shared_layers": 0},
        {"model_type": "gemma4", "num_hidden_layers": 42},
        {"model_type": "gemma4", "num_hidden_layers": 18, "num_kv_shared_layers": 18},
        {"model_type": "gemma4", "num_hidden_layers": "bad", "num_kv_shared_layers": 1},
        {"num_hidden_layers": 42, "num_kv_shared_layers": 18},
    ],
)
def test_ineligible_without_supported_gemma4_yoco_shape(model_args) -> None:
    assert not is_yoco_fast_prefill_eligible(
        model_args,
        use_paged_attention=True,
    )


def test_ineligible_without_paged_attention() -> None:
    assert not is_yoco_fast_prefill_eligible(
        {
            "model_type": "gemma4",
            "num_hidden_layers": 42,
            "num_kv_shared_layers": 18,
        },
        use_paged_attention=False,
    )


def test_ineligibility_reason_reports_missing_num_hidden_layers() -> None:
    assert (
        get_yoco_fast_prefill_ineligibility_reason(
            {
                "model_type": "gemma4",
                "num_kv_shared_layers": 18,
            },
            use_paged_attention=True,
        )
        == "num_hidden_layers is missing or not an int"
    )


def test_reduced_context_from_full_paged_metadata() -> None:
    meta = build_yoco_reduced_context_from_full_metadata(
        slot_mapping=[
            2 * 4 + 2,
            3 * 4 + 1,
            4 * 4 + 0,
            4 * 4 + 1,
            4 * 4 + 2,
            4 * 4 + 3,
            5 * 4 + 0,
            7 * 4 + 1,
            7 * 4 + 2,
            7 * 4 + 3,
        ],
        block_tables=[
            [1, 2],
            [3],
            [4, 5],
            [6, 7, 8],
        ],
        context_lens=[7, 2, 5, 8],
        offsets=[6, 1, 0, 5],
        cu_seqlens=[0, 1, 2, 7, 10],
    )

    assert meta.selected_query_indices == [0, 1, 6, 9]
    assert meta.slot_mapping == [2 * 4 + 2, 3 * 4 + 1, 5 * 4 + 0, 7 * 4 + 3]
    assert meta.block_tables == [[1, 2], [3], [4, 5], [6, 7, 8]]
    assert meta.context_lens == [7, 2, 5, 8]
    assert meta.offsets == [6, 1, 4, 7]
    assert meta.cu_seqlens == [0, 1, 2, 3, 4]


def test_mlx_slice_assignment_writes_in_place() -> None:
    import mlx.core as mx

    h = mx.zeros((1, 5, 2), dtype=mx.float32)
    selected = mx.array([1, 3], dtype=mx.int32)
    cross_h = mx.array([[[7.0, 8.0], [9.0, 10.0]]], dtype=mx.float32)

    h[:, selected, :] = cross_h
    mx.eval(h)

    assert h[0, 0, 0].item() == 0.0
    assert h[0, 1, 0].item() == 7.0
    assert h[0, 3, 0].item() == 9.0


def test_reduced_context_from_empty_full_paged_metadata() -> None:
    meta = build_yoco_reduced_context_from_full_metadata(
        slot_mapping=[],
        block_tables=[],
        context_lens=[],
        offsets=[],
        cu_seqlens=[0],
    )

    assert meta.selected_query_indices == []
    assert meta.slot_mapping == []
    assert meta.block_tables == []
    assert meta.context_lens == []
    assert meta.offsets == []
    assert meta.cu_seqlens == [0]


def test_patch_gemma4_yoco_fast_prefill_reduces_shared_layer_queries() -> None:
    import mlx.core as mx

    from vllm_metal.paged_attention_common import (
        PagedAttentionContext,
        clear_context,
        get_context,
        set_context,
    )

    class _Config:
        num_hidden_layers = 4
        num_kv_shared_layers = 2

    class _Layer:
        def __init__(self, layer_idx: int) -> None:
            self.layer_idx = layer_idx
            self.calls = []

        def __call__(
            self,
            h,
            mask,
            cache,
            *,
            per_layer_input=None,
            shared_kv=None,
            offset=None,
        ):
            ctx = get_context()
            self.calls.append(
                {
                    "tokens": h.shape[1],
                    "cu_seqlens": tuple(ctx.cu_seqlens),
                    "offsets": tuple(ctx.offsets),
                    "gdn_slot_mapping": ctx.gdn_slot_mapping,
                }
            )
            return h + (self.layer_idx + 1), (f"k{self.layer_idx}", "v"), offset

    class _TextModel:
        config = _Config()
        embed_scale = 1
        hidden_size_per_layer_input = 0

        def __init__(self) -> None:
            self.layers = [_Layer(i) for i in range(4)]
            self.previous_kvs = [0, 1, 0, 1]
            self.embed_tokens = lambda inputs: inputs
            self.norm = lambda h: h

        def _get_per_layer_inputs(self, input_ids, input_embeddings=None):
            raise AssertionError("per-layer input lookup should not run")

        def _project_per_layer_inputs(self, input_embeddings, per_layer_inputs=None):
            raise AssertionError("per-layer input projection should not run")

        def _make_masks(self, h, cache):
            return [None] * len(self.layers)

        def __call__(
            self,
            inputs=None,
            cache=None,
            input_embeddings=None,
            per_layer_inputs=None,
        ):
            return input_embeddings + 100

    text_model = _TextModel()
    top_model = type("_TopModel", (), {"model": text_model})()

    assert patch_gemma4_yoco_fast_prefill(
        top_model,
        {
            "model_type": "gemma4_text",
            "num_hidden_layers": 4,
            "num_kv_shared_layers": 2,
        },
        use_paged_attention=True,
    )

    inputs = mx.zeros((1, 5, 2), dtype=mx.float32)
    fallback = text_model(input_embeddings=inputs)
    mx.eval(fallback)
    assert fallback[0, 0, 0].item() == 100

    ctx = PagedAttentionContext(
        slot_mapping=[0, 1, 2, 3, 4],
        block_tables=[[0, 1]],
        context_lens=[5],
        offsets=[0],
        cu_seqlens=[0, 5],
        gdn_slot_mapping=[17],
    )
    set_context(ctx)
    try:
        out = text_model(input_embeddings=inputs, cache=[object()] * 4)
        assert get_context() is ctx
    finally:
        clear_context()

    mx.eval(out)
    assert [layer.calls[0]["tokens"] for layer in text_model.layers] == [5, 5, 1, 1]
    assert text_model.layers[2].calls[0]["cu_seqlens"] == (0, 1)
    assert text_model.layers[2].calls[0]["offsets"] == (4,)
    assert text_model.layers[0].calls[0]["gdn_slot_mapping"] == [17]
    assert text_model.layers[2].calls[0]["gdn_slot_mapping"] is None
    assert text_model.layers[3].calls[0]["gdn_slot_mapping"] is None
    assert out[0, 0, 0].item() == 3
    assert out[0, 4, 0].item() == 10


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            {
                "slot_mapping": [],
                "block_tables": [],
                "context_lens": [],
                "offsets": [],
                "cu_seqlens": [],
            },
            "cu_seqlens must start with 0",
        ),
        (
            {
                "slot_mapping": [],
                "block_tables": [],
                "context_lens": [],
                "offsets": [],
                "cu_seqlens": [1],
            },
            "cu_seqlens must start with 0",
        ),
        (
            {
                "slot_mapping": [1],
                "block_tables": [[1]],
                "context_lens": [1],
                "offsets": [],
                "cu_seqlens": [0, 1],
            },
            "block_tables, context_lens, and offsets must match",
        ),
        (
            {
                "slot_mapping": [1, 2],
                "block_tables": [[1]],
                "context_lens": [1],
                "offsets": [0],
                "cu_seqlens": [0, 1],
            },
            "slot_mapping length must match final cu_seqlens",
        ),
        (
            {
                "slot_mapping": [],
                "block_tables": [[1]],
                "context_lens": [1],
                "offsets": [0],
                "cu_seqlens": [0, 1],
            },
            "slot_mapping shorter than cu_seqlens",
        ),
        (
            {
                "slot_mapping": [1],
                "block_tables": [[1]],
                "context_lens": [1],
                "offsets": [0],
                "cu_seqlens": [0, 0],
            },
            "cu_seqlens segments must be positive and increasing",
        ),
    ],
)
def test_reduced_context_from_full_metadata_validates_inputs(
    kwargs,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        build_yoco_reduced_context_from_full_metadata(**kwargs)
