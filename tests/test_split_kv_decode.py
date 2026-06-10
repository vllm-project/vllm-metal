# SPDX-License-Identifier: Apache-2.0
"""Split-KV (flash-decoding) decode correctness.

``dispatch_paged_attention_v2_online`` routes pure-decode batches to the
partitioned kernel (``_ps512`` + ``paged_attention_v2_reduce``) when the base
grid (num_q_heads * num_seqs) underfills the GPU — fewer than 8 threadgroups
per core — and the longest context spans >= 2 partitions (> 512 tokens).

Every split case engages on any hardware: base_grid <= 32, while the gate
threshold is >= 56 (8 x 7 cores on the smallest Apple GPU) and 112 on the
IORegistry fallback used in VMs — and each test asserts that premise via the
exported ``min_decode_grid()`` so a gate regression fails loudly instead of
silently degrading every cell to single-pass-vs-reference.  TurboQuant and
sliding-window batches are excluded from the split until their interaction is
validated; their single-pass coverage lives in test_turboquant.py and
test_primitive_and_donation.py.

Run with:
    python -m pytest tests/test_split_kv_decode.py -v
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from tools.attention_bench_utils import ref_paged_attn
from vllm_metal.metal import get_ops

NUM_QUERY_HEADS = 16  # Qwen3-0.6B GQA shape: 16 query / 8 KV heads
NUM_KV_HEADS = 8
HEAD_SIZE = 128
BLOCK_SIZE = 16
NUM_BLOCKS = 256

# atol/rtol per dtype.  float32: both paths compute in fp32, only
# kernel-order error is left.
_TOLERANCES = {
    mx.bfloat16: (3e-2, 2e-2),
    mx.float16: (1.5e-2, 2e-2),
    mx.float32: (1e-3, 1e-3),
}


def _run_paged_decode(
    kv_lens: list[int], dtype: mx.Dtype, seed: int
) -> tuple[mx.array, mx.array]:
    """Run one decode step through the primitive; return (out, reference)."""
    mx.random.seed(seed)
    num_seqs = len(kv_lens)
    max_kv_len = max(kv_lens)
    scale = HEAD_SIZE**-0.5

    key_cache = mx.random.normal(
        shape=(NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)
    ).astype(dtype)
    value_cache = mx.random.normal(
        shape=(NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)
    ).astype(dtype)
    query = mx.random.normal(shape=(num_seqs, NUM_QUERY_HEADS, HEAD_SIZE)).astype(dtype)

    max_blocks_per_seq = (max_kv_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    block_tables = mx.random.randint(
        0, NUM_BLOCKS, shape=(num_seqs, max_blocks_per_seq)
    ).astype(mx.int32)
    kv_lens_arr = mx.array(kv_lens, dtype=mx.int32)
    cu_seqlens_q = mx.arange(num_seqs + 1, dtype=mx.int32)
    mx.eval(key_cache, value_cache, query, block_tables, kv_lens_arr, cu_seqlens_q)

    out = mx.array(0)
    get_ops().paged_attention_primitive(
        query,
        key_cache,
        value_cache,
        NUM_KV_HEADS,
        scale,
        0.0,  # softcap
        block_tables,
        kv_lens_arr,
        cu_seqlens_q,
        BLOCK_SIZE,
        max_kv_len,
        -1,  # sliding_window (disabled; windowed batches skip the split)
        out,
    )
    mx.eval(out)

    ref = ref_paged_attn(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=[1] * num_seqs,
        kv_lens=kv_lens,
        block_tables=np.array(block_tables),
        scale=scale,
    )
    mx.eval(ref)
    return out, ref


def _assert_close(out: mx.array, ref: mx.array, dtype: mx.Dtype) -> None:
    atol, rtol = _TOLERANCES[dtype]
    np.testing.assert_allclose(
        np.array(out.astype(mx.float32)),
        np.array(ref.astype(mx.float32)),
        atol=atol,
        rtol=rtol,
    )


@pytest.mark.parametrize(
    "kv_lens",
    [
        [8192],  # headline single-stream regime: 16 partitions
        [100, 2048],  # 1 valid partition inside a 4-partition grid
        [700, 4096],  # non-block-aligned lengths: 2 vs 8 valid partitions
        [1300],  # ODD partition count (3): reduce shmem 16-byte alignment
    ],
)
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32])
def test_split_decode_vs_reference(kv_lens: list[int], dtype: mx.Dtype) -> None:
    """Partitioned decode matches the pure-MLX reference."""
    ops = get_ops()
    # Engagement premise: the gate is invisible from Python, so assert the
    # inputs that make it take the split path — otherwise a gate regression
    # would silently turn every cell into single-pass-vs-reference.
    assert NUM_QUERY_HEADS * len(kv_lens) < ops.min_decode_grid()
    assert max(kv_lens) > ops.PARTITION_SIZE

    out, ref = _run_paged_decode(kv_lens, dtype, seed=0)
    _assert_close(out, ref, dtype)


def test_high_occupancy_stays_single_pass() -> None:
    """The gate's OFF side: a batch whose base grid meets this machine's
    threshold must stay on the single-pass path and still match the reference.
    num_seqs is derived from the exported threshold, so the construction is
    machine-robust (engages nowhere)."""
    ops = get_ops()
    num_seqs = -(-ops.min_decode_grid() // NUM_QUERY_HEADS)  # ceil division
    kv_lens = [1024] * num_seqs  # > PARTITION_SIZE: only the grid term gates
    assert NUM_QUERY_HEADS * num_seqs >= ops.min_decode_grid()

    out, ref = _run_paged_decode(kv_lens, mx.float16, seed=1)
    _assert_close(out, ref, mx.float16)
