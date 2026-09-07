# SPDX-License-Identifier: Apache-2.0
"""ShortConvPagedWrapper math vs the unmodified mlx_lm LFM2 module.

The wrapper must reproduce ``mlx_lm.models.lfm2.ShortConv`` exactly: packed
multi-request forwards through the paged state pool have to match running
each request sequentially against mlx_lm's own ``ArraysCache``.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
from mlx_lm.models.cache import ArraysCache
from mlx_lm.models.lfm2 import ModelArgs, ShortConv

from vllm_metal.attention.caches.shortconv_cache import ShortConvStateCache
from vllm_metal.attention.context import (
    PagedAttentionContext,
    clear_context,
    set_context,
)
from vllm_metal.attention.impls.shortconv import ShortConvPagedWrapper

HIDDEN = 32
L_CACHE = 3
MAX_SEQS = 8

ATOL = 1e-5
RTOL = 1e-5


@pytest.fixture(autouse=True)
def _cpu_device_and_clean_context():
    # Pin the math-equivalence comparison to the CPU stream: the GPU fp32
    # matmul kernels choose different precision paths by row count (M=1 gemv
    # is exact, M>=2 tiles accumulate at reduced precision on M5), so packed
    # vs sequential shapes would differ by kernel choice, not by wrapper math.
    default_device = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(default_device)
    clear_context()


def _make_module() -> ShortConv:
    args = ModelArgs(
        model_type="lfm2",
        vocab_size=64,
        hidden_size=HIDDEN,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        norm_eps=1e-5,
        conv_bias=True,
        conv_L_cache=L_CACHE,
        block_dim=HIDDEN,
        block_ff_dim=64,
        block_multiple_of=8,
        block_ffn_dim_multiplier=1.0,
        block_auto_adjust_ff_dim=False,
        layer_types=["conv", "full_attention"],
    )
    return ShortConv(args, layer_idx=0)


def _make_wrapper(
    module: ShortConv,
) -> tuple[ShortConvPagedWrapper, ShortConvStateCache]:
    state_cache = ShortConvStateCache(
        num_layers=1,
        max_seqs=MAX_SEQS,
        conv_kernel_dim=L_CACHE,
        conv_dim=HIDDEN,
        initial_seqs=MAX_SEQS,
        dtype=mx.float32,
    )
    wrapper = ShortConvPagedWrapper(
        module, layer_idx=0, cache_idx=0, state_cache=state_cache
    )
    return wrapper, state_cache


def _set_packed_context(
    seg_lens: list[int], slot_ids: list[int], grouped: bool
) -> None:
    cu_seqlens = [0]
    for seg_len in seg_lens:
        cu_seqlens.append(cu_seqlens[-1] + seg_len)
    # Mirror prepare_grouped: decode requests (single-token segments) are
    # packed first, so a leading run of 1-token segments is the decode set.
    # This drives the wrapper's batched pure-decode fast path on all-decode
    # steps, exactly like the real runner.
    num_decode = 0
    for seg_len in seg_lens:
        if seg_len != 1:
            break
        num_decode += 1
    set_context(
        PagedAttentionContext(
            slot_mapping=[],
            cu_seqlens=cu_seqlens,
            state_slot_mapping=None if grouped else list(slot_ids),
            state_group_slot_mappings=(list(slot_ids),) if grouped else None,
            num_decode_requests=num_decode,
        )
    )


def _run_wrapper_step(
    wrapper: ShortConvPagedWrapper,
    chunks: list[mx.array],
    slot_ids: list[int],
    grouped: bool,
) -> list[mx.array]:
    """Run one packed forward and split the output back per request."""
    seg_lens = [chunk.shape[1] for chunk in chunks]
    _set_packed_context(seg_lens, slot_ids, grouped)
    packed = mx.concatenate(chunks, axis=1)
    out = wrapper(packed)
    clear_context()
    outputs = []
    start = 0
    for seg_len in seg_lens:
        outputs.append(out[:, start : start + seg_len, :])
        start += seg_len
    return outputs


def _assert_close(actual: mx.array, expected: mx.array) -> None:
    np.testing.assert_allclose(
        np.array(actual, copy=False),
        np.array(expected, copy=False),
        atol=ATOL,
        rtol=RTOL,
    )


class TestShortConvPagedWrapper:
    @pytest.mark.parametrize(
        "grouped", [False, True], ids=["request-slots", "block-slots"]
    )
    def test_packed_multi_request_matches_sequential_reference(self, grouped) -> None:
        """Three requests packed per step vs mlx_lm cache-by-cache reference.

        Runs a prefill step and two decode steps; both the per-step outputs
        and the final conv states must match the reference to fp tolerance.
        """
        mx.random.seed(0)
        module = _make_module()
        wrapper, state_cache = _make_wrapper(module)

        slot_ids = [2, 0, 5]
        # Per request: prefill chunk then two single-token decode chunks.
        request_chunks = [
            [mx.random.normal((1, t, HIDDEN)) for t in (5, 1, 1)],
            [mx.random.normal((1, t, HIDDEN)) for t in (3, 1, 1)],
            [mx.random.normal((1, t, HIDDEN)) for t in (7, 1, 1)],
        ]

        # Reference: each request sequentially with mlx_lm's own cache.
        ref_outputs: list[list[mx.array]] = []
        ref_caches: list[ArraysCache] = []
        for chunks in request_chunks:
            cache = ArraysCache(size=1)
            ref_outputs.append(
                [module(chunk, mask=None, cache=cache) for chunk in chunks]
            )
            ref_caches.append(cache)

        # Wrapper: the three requests packed together, one step at a time.
        for step in range(3):
            chunks = [request_chunks[req][step] for req in range(3)]
            outputs = _run_wrapper_step(wrapper, chunks, slot_ids, grouped)
            for req in range(3):
                _assert_close(outputs[req], ref_outputs[req][step])

        # Final per-request conv state must match the mlx_lm cache tail.
        for req, slot in enumerate(slot_ids):
            _assert_close(
                state_cache.conv_states[0][slot : slot + 1],
                ref_caches[req][0],
            )
