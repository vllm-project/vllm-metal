# SPDX-License-Identifier: Apache-2.0
"""Native SDPA is a test-only numeric/performance ceiling.

Production decode always goes through ``paged_attention_primitive`` (the
GQA-shared flash-decode pass at long context).  These tests keep MLX native
SDPA around as a reference: contiguous paged GQA output must match it, and
the shipped attention module must not dispatch to it.
"""

from __future__ import annotations

import inspect

import mlx.core as mx
import numpy as np
import pytest

import vllm_metal.attention.impls.sdpa as sdpa_mod
from tools.attention_bench_utils import native_sdpa_contiguous_decode
from vllm_metal.metal import get_ops

NUM_QUERY_HEADS = 32
NUM_KV_HEADS = 8
HEAD_SIZE = 128
BLOCK_SIZE = 16

_TOLERANCES = {
    mx.bfloat16: (3e-2, 2e-2),
    mx.float16: (1.5e-2, 2e-2),
}


def test_production_decode_does_not_call_native_sdpa() -> None:
    src = inspect.getsource(sdpa_mod.sdpa_forward)
    assert "_native_sdpa_decode_fast_path" not in src
    assert "scaled_dot_product_attention" not in src
    assert not hasattr(sdpa_mod, "_native_sdpa_decode_fast_path")
    assert not hasattr(sdpa_mod, "_gqa_decode_kernels")


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_paged_gqa_matches_native_sdpa_reference(dtype: mx.Dtype) -> None:
    """Contiguous long decode: shipped primitive vs MLX native SDPA."""
    ops = get_ops()
    cores = ops.detected_gpu_core_count()
    if cores <= 0:
        pytest.skip("GPU core count unavailable: GQA conservatively disabled")
    n_grid = ((3 * cores + NUM_KV_HEADS - 1) // NUM_KV_HEADS) * 512
    seq = max(32768, n_grid)
    if seq > 131072:
        pytest.skip("No context inside the scoped range meets the GPU grid guard")
    assert ops.has_gqa_decode_kernel()
    mx.random.seed(11)
    n_blocks = seq // BLOCK_SIZE
    first = 2
    num_cache = first + n_blocks + 2
    kc = mx.random.normal((num_cache, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)).astype(
        dtype
    )
    vc = mx.random.normal((num_cache, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)).astype(
        dtype
    )
    q = mx.random.normal((1, NUM_QUERY_HEADS, HEAD_SIZE)).astype(dtype)
    scale = HEAD_SIZE**-0.5
    table = list(range(first, first + n_blocks))
    bt = mx.array([table], dtype=mx.int32)
    sl = mx.array([seq], dtype=mx.int32)
    cu = mx.array([0, 1], dtype=mx.int32)
    mx.eval(kc, vc, q, bt, sl, cu)

    out = mx.array(0)
    ops.paged_attention_primitive(
        q, kc, vc, NUM_KV_HEADS, scale, 0.0, bt, sl, cu, BLOCK_SIZE, seq, -1, out
    )
    mx.eval(out)
    assert ops.last_paged_dispatch() == "gqa_decode"
    ref = native_sdpa_contiguous_decode(q, kc, vc, first, seq, scale)
    mx.eval(ref)
    atol, rtol = _TOLERANCES[dtype]
    np.testing.assert_allclose(
        np.array(out.astype(mx.float32)),
        np.array(ref.astype(mx.float32)),
        atol=atol,
        rtol=rtol,
    )
