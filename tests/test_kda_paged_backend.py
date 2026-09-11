# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import mlx.core as mx
import numpy as np
from mlx_lm.models.bailing_moe_v3 import BailingKDA, ModelArgs
from mlx_lm.models.cache import ArraysCache

from vllm_metal.attention.caches.gdn_cache import GDNPagedStateCache
from vllm_metal.attention.context import (
    PagedAttentionContext,
    clear_context,
    set_context,
)
from vllm_metal.attention.impls.kda import KDAPagedAttentionWrapper


class TestKDAPagedAttentionWrapper:
    def test_multi_request_prefill_and_decode_match_mlx_lm(self) -> None:
        mx.random.seed(17)
        args = ModelArgs(
            hidden_size=16,
            num_attention_heads=2,
            head_dim=4,
            short_conv_kernel_size=3,
        )
        inner = BailingKDA(args)
        projection_size = args.num_attention_heads * args.head_dim
        state_cache = GDNPagedStateCache(
            num_layers=1,
            max_seqs=2,
            conv_kernel_dim=args.short_conv_kernel_size,
            conv_dim=3 * projection_size,
            num_v_heads=args.num_attention_heads,
            value_head_dim=args.head_dim,
            key_head_dim=args.head_dim,
            initial_seqs=2,
            dtype=mx.float32,
        )
        wrapper = KDAPagedAttentionWrapper(inner, 0, 0, state_cache)
        references = [ArraysCache(size=4), ArraysCache(size=4)]

        for cu_seqlens, grouped_slots in (([0, 3, 5], False), ([0, 1, 2], True)):
            x = mx.random.normal((1, cu_seqlens[-1], args.hidden_size)).astype(
                mx.float32
            )
            expected = mx.concatenate(
                [
                    inner(x[:, start:end], cache=reference)
                    for start, end, reference in zip(
                        cu_seqlens[:-1], cu_seqlens[1:], references, strict=True
                    )
                ],
                axis=1,
            )

            set_context(
                PagedAttentionContext(
                    slot_mapping=[],
                    cu_seqlens=cu_seqlens,
                    state_slot_mapping=None if grouped_slots else [0, 1],
                    state_group_slot_mappings=([0, 1],) if grouped_slots else None,
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

        expected_conv = mx.concatenate(
            [mx.concatenate(reference.cache[:3], axis=-1) for reference in references]
        )
        expected_recurrent = mx.concatenate([reference[3] for reference in references])
        np.testing.assert_array_equal(
            np.array(state_cache.conv_states[0]), np.array(expected_conv)
        )
        np.testing.assert_array_equal(
            np.array(state_cache.recurrent_states[0]), np.array(expected_recurrent)
        )
