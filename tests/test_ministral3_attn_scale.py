# SPDX-License-Identifier: Apache-2.0
"""Ministral 3's position-dependent query scale on the paged path (#1130).

mlx_lm's ``ministral3`` calls attention as ``(x, attn_scale, mask, cache)``
and scales queries by ``1 + beta * log(1 + floor(pos / original_max))``. The
model computes that scale from one offset for the whole batch, so packed
requests need it per row from their own positions.
"""

import math

import mlx.core as mx
import pytest
from mlx_lm.models.cache import KVCache
from mlx_lm.models.ministral3 import Model, ModelArgs

from tests.stub_runner import make_kv_cache
from vllm_metal.attention.context import (
    OffsetCache,
    PagedAttentionContext,
    clear_context,
    prepare_unified,
)
from vllm_metal.attention.impls.sdpa import llama4_query_scale
from vllm_metal.attention.impls.sdpa_wrapper import patch_sdpa_attention

_BLOCK_SIZE = 8
_ORIGINAL_MAX = 4


def _tiny_model() -> Model:
    mx.random.seed(0)
    model = Model(
        ModelArgs(
            model_type="ministral3",
            hidden_size=128,
            num_hidden_layers=2,
            intermediate_size=256,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=64,
            rms_norm_eps=1e-5,
            vocab_size=64,
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 10000.0,
                "llama_4_scaling_beta": 0.5,
                "original_max_position_embeddings": _ORIGINAL_MAX,
            },
        )
    )
    mx.eval(model.parameters())
    return model


def _reference_logits(model: Model, tokens: list[int]) -> mx.array:
    cache = [KVCache() for _ in model.layers]
    return model(mx.array([tokens]), cache=cache)[0]


def test_packed_prefill_and_decode_match_mlx_lm() -> None:
    model = _tiny_model()
    prompt_a = [3, 14, 15, 9, 26, 5, 35, 8, 9, 7]
    prompt_b = [2, 7, 18, 28, 18, 28]
    next_a, next_b = 11, 13
    blocks_a, blocks_b = [0, 1], [2, 3]

    kv_cache = make_kv_cache(
        num_layers=2,
        num_kv_heads=1,
        head_dim=64,
        num_blocks=4,
        block_size=_BLOCK_SIZE,
        dtype=mx.float32,
    )
    assert patch_sdpa_attention(model, kv_cache, _BLOCK_SIZE) == 2

    # Without a paged context the wrapper delegates to mlx_lm unchanged.
    expected_a = _reference_logits(model, prompt_a + [next_a])
    expected_b = _reference_logits(model, prompt_b + [next_b])

    def paged_forward(tokens: list[int]) -> mx.array:
        try:
            caches = [OffsetCache(0) for _ in model.layers]
            logits = model(mx.array([tokens]), cache=caches)[0]
            mx.eval(logits)
            return logits
        finally:
            clear_context()

    prepare_unified(
        [],
        [(blocks_a, len(prompt_a), 0), (blocks_b, len(prompt_b), 0)],
        _BLOCK_SIZE,
    )
    prefill = paged_forward(prompt_a + prompt_b)

    prepare_unified(
        [(blocks_a, len(prompt_a)), (blocks_b, len(prompt_b))], [], _BLOCK_SIZE
    )
    decode = paged_forward([next_a, next_b])

    tol = {"rtol": 1e-4, "atol": 1e-4}
    assert mx.allclose(prefill[: len(prompt_a)], expected_a[:-1], **tol).item()
    assert mx.allclose(prefill[len(prompt_a) :], expected_b[:-1], **tol).item()
    assert mx.allclose(decode[0], expected_a[-1], **tol).item()
    assert mx.allclose(decode[1], expected_b[-1], **tol).item()


@pytest.mark.parametrize(
    ("offsets", "cu_seqlens", "expected"),
    [
        # Every row stays below original_max: the scale is exactly one.
        ([0, 3], [0, 4, 5], None),
        # A decode row at position 9, then a prefill chunk at positions 3, 4.
        (
            [9, 3],
            [0, 1, 3],
            [1.0 + 0.5 * math.log(3.0), 1.0, 1.0 + 0.5 * math.log(2.0)],
        ),
    ],
)
def test_query_scale_follows_each_row_position(offsets, cu_seqlens, expected) -> None:
    ctx = PagedAttentionContext(slot_mapping=[], offsets=offsets, cu_seqlens=cu_seqlens)
    scale = llama4_query_scale(ctx, 0.5, _ORIGINAL_MAX)
    if expected is None:
        assert scale is None
        return
    assert scale is not None
    assert scale.shape == (1, 1, len(expected), 1)
    assert mx.allclose(scale.reshape(-1), mx.array(expected)).item()
