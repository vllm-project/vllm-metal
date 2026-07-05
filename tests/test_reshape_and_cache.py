# SPDX-License-Identifier: Apache-2.0
"""Parity test for the fused reshape_and_cache K/V paged scatter.

The fused Metal op must produce byte-identical caches to the MLX scatter it
replaces on the decode path (flat[slot_mapping] = kv).
"""

import mlx.core as mx
import pytest

from vllm_metal.metal import get_ops


def _reference_scatter(key, value, num_blocks, block_size, slot_mapping):
    h, d = key.shape[1], key.shape[2]
    kc = mx.zeros((num_blocks, block_size, h, d), dtype=key.dtype)
    vc = mx.zeros((num_blocks, block_size, h, d), dtype=value.dtype)
    fk = kc.reshape(-1, h, d)
    fk[slot_mapping] = key
    fv = vc.reshape(-1, h, d)
    fv[slot_mapping] = value
    kc = fk.reshape(num_blocks, block_size, h, d)
    vc = fv.reshape(num_blocks, block_size, h, d)
    mx.eval(kc, vc)
    return kc, vc


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32])
@pytest.mark.parametrize("num_kv_heads,head_dim", [(2, 64), (4, 128)])
def test_reshape_and_cache_matches_scatter(dtype, num_kv_heads, head_dim):
    if not mx.metal.is_available():
        pytest.skip("Metal is not available")
    num_blocks, block_size, num_tokens = 4, 16, 5
    mx.random.seed(0)
    key = mx.random.normal((num_tokens, num_kv_heads, head_dim)).astype(dtype)
    value = mx.random.normal((num_tokens, num_kv_heads, head_dim)).astype(dtype)
    slot = mx.array([3, 20, 7, 40, 1], dtype=mx.int64)  # all < num_blocks*block_size

    kc = mx.zeros((num_blocks, block_size, num_kv_heads, head_dim), dtype=dtype)
    vc = mx.zeros((num_blocks, block_size, num_kv_heads, head_dim), dtype=dtype)
    new_k, new_v = get_ops().reshape_and_cache(key, value, kc, vc, slot)
    mx.eval(new_k, new_v)

    ref_k, ref_v = _reference_scatter(key, value, num_blocks, block_size, slot)
    assert mx.array_equal(new_k, ref_k)
    assert mx.array_equal(new_v, ref_v)


def test_reshape_and_cache_skips_negative_slots():
    if not mx.metal.is_available():
        pytest.skip("Metal is not available")
    num_blocks, block_size, num_kv_heads, head_dim = 4, 16, 4, 128
    mx.random.seed(1)
    key = mx.random.normal((3, num_kv_heads, head_dim)).astype(mx.float16)
    value = mx.random.normal((3, num_kv_heads, head_dim)).astype(mx.float16)
    slot = mx.array([5, -1, 9], dtype=mx.int64)  # padding slot must be ignored

    kc = mx.zeros((num_blocks, block_size, num_kv_heads, head_dim), dtype=mx.float16)
    vc = mx.zeros((num_blocks, block_size, num_kv_heads, head_dim), dtype=mx.float16)
    new_k, _ = get_ops().reshape_and_cache(key, value, kc, vc, slot)
    mx.eval(new_k)

    ref_k, _ = _reference_scatter(
        key[[0, 2]],
        value[[0, 2]],
        num_blocks,
        block_size,
        mx.array([5, 9], dtype=mx.int64),
    )
    assert mx.array_equal(new_k, ref_k)
