# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import mlx.core as mx
import numpy as np
import pytest
from mlx_lm.models.bailing_moe_v3 import BailingKDA, ModelArgs
from mlx_lm.models.cache import ArraysCache
from mlx_vlm.models.cache import ArraysCache as VLMArraysCache
from mlx_vlm.models.glm5_next.config import TextConfig as Glm5NextTextConfig
from mlx_vlm.models.glm5_next.language import Glm5NextLinearAttention

from tests.stub_runner import make_state_cache
from vllm_metal.attention.context import (
    PagedAttentionContext,
    clear_context,
    set_context,
)
from vllm_metal.attention.impls.kda import KDAPagedAttentionWrapper

_HIDDEN_SIZE = 16
_NUM_HEADS = 2
_HEAD_DIM = 4
_CONV_KERNEL = 3


def _bailing_kda() -> tuple[Any, ...]:
    """Inner layer, reference cache factory, reference -> (packed conv, recurrent)."""
    args = ModelArgs(
        hidden_size=_HIDDEN_SIZE,
        num_attention_heads=_NUM_HEADS,
        head_dim=_HEAD_DIM,
        short_conv_kernel_size=_CONV_KERNEL,
    )
    return (
        BailingKDA(args),
        lambda: ArraysCache(size=4),
        lambda ref: (mx.concatenate(ref.cache[:3], axis=-1), ref[3]),
    )


def _glm5_next_kda() -> tuple[Any, ...]:
    config = Glm5NextTextConfig(
        hidden_size=_HIDDEN_SIZE,
        linear_num_heads=_NUM_HEADS,
        linear_head_dim=_HEAD_DIM,
        linear_conv_kernel_dim=_CONV_KERNEL,
    )
    return (
        Glm5NextLinearAttention(config),
        lambda: VLMArraysCache(size=2),
        lambda ref: (ref[0], ref[1]),
    )


@pytest.mark.parametrize("make_kda", [_bailing_kda, _glm5_next_kda])
def test_kda_paged_state_matches_reference_with_reordering_and_mixed_steps(
    make_kda: Callable[[], tuple[Any, ...]],
) -> None:
    mx.random.seed(17)
    inner, make_reference, reference_state = make_kda()
    projection_size = _NUM_HEADS * _HEAD_DIM
    state_cache = make_state_cache(
        num_layers=1,
        max_seqs=2,
        conv_kernel_dim=_CONV_KERNEL,
        conv_dim=3 * projection_size,
        num_v_heads=_NUM_HEADS,
        value_head_dim=_HEAD_DIM,
        key_head_dim=_HEAD_DIM,
        dtype=mx.float32,
        recurrent_dtype=mx.float32,
    )
    wrapper = KDAPagedAttentionWrapper(inner, 0, 0, state_cache)
    references = [make_reference(), make_reference()]

    for request_order, lengths, grouped_slots, num_decode in (
        ((0, 1), (3, 2), False, 0),
        ((1, 0), (1, 2), True, 1),
        ((0, 1), (1, 1), False, 2),
    ):
        cu_seqlens = [0, lengths[0], sum(lengths)]
        slots = list(request_order)
        x = mx.random.normal((1, cu_seqlens[-1], _HIDDEN_SIZE)).astype(mx.float32)
        expected = mx.concatenate(
            [
                inner(x[:, start:end], cache=references[request_idx])
                for start, end, request_idx in zip(
                    cu_seqlens[:-1], cu_seqlens[1:], request_order, strict=True
                )
            ],
            axis=1,
        )

        set_context(
            PagedAttentionContext(
                slot_mapping=[],
                cu_seqlens=cu_seqlens,
                state_slot_mapping=None if grouped_slots else slots,
                state_group_slot_mappings=(slots,) if grouped_slots else None,
                num_decode_requests=num_decode,
            )
        )
        try:
            actual = wrapper(x)
        finally:
            clear_context()

        mx.eval(actual, expected)
        np.testing.assert_allclose(
            np.array(actual), np.array(expected), rtol=1e-5, atol=1e-5
        )

    expected_conv, expected_recurrent = (
        mx.concatenate(states)
        for states in zip(*map(reference_state, references), strict=True)
    )
    np.testing.assert_allclose(
        np.array(state_cache.conv_states[0]),
        np.array(expected_conv),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.array(state_cache.recurrent_states[0]),
        np.array(expected_recurrent),
        rtol=1e-6,
        atol=1e-6,
    )
