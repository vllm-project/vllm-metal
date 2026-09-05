# SPDX-License-Identifier: Apache-2.0
"""GQA-shared flash-decode pass inside ``paged_attention_primitive``.

Eligible long single-sequence decode (no TQ/sinks/softcap/window, head size
in {64,96,128,256}, GQA group <= 8, context >= ``GQA_DECODE_MIN_SEQ_LEN``)
is dispatched to ``paged_attention_gqa_decode`` and merged by the existing
``paged_attention_v2_reduce``.  Shorter contexts, multi-sequence batches,
and spec-decode verify windows stay on the established per-token / split-KV
family — a different kernel family would break
``tests/test_spec_window_parity.py`` bitwise identity.

These tests drive the shipped primitive (not a reimplementation) against
``ref_paged_attn``.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from tools.attention_bench_utils import ref_paged_attn
from vllm_metal.metal import get_ops

NUM_QUERY_HEADS = 16
NUM_KV_HEADS = 8
HEAD_SIZE = 128
BLOCK_SIZE = 16

_TOLERANCES = {
    mx.bfloat16: (3e-2, 2e-2),
    mx.float16: (1.5e-2, 2e-2),
}


def _interleaved_table(n_blocks: int) -> list[int]:
    """Run-of-2, skip-1 pattern produced by hybrid GDN block interleave."""
    table: list[int] = []
    b = 3
    while len(table) < n_blocks:
        table += [b, b + 1]
        b += 3
    return table[:n_blocks]


def _assert_close(out: mx.array, ref: mx.array, dtype: mx.Dtype) -> None:
    atol, rtol = _TOLERANCES[dtype]
    np.testing.assert_allclose(
        np.array(out.astype(mx.float32)),
        np.array(ref.astype(mx.float32)),
        atol=atol,
        rtol=rtol,
    )


def _run_primitive(
    kv_lens: list[int],
    dtype: mx.Dtype,
    *,
    interleaved: bool,
    seed: int,
    window_seqlen_q: int = 1,
    query_lens: list[int] | None = None,
) -> tuple[mx.array, mx.array]:
    mx.random.seed(seed)
    num_seqs = len(kv_lens)
    max_kv_len = max(kv_lens)
    scale = HEAD_SIZE**-0.5
    if query_lens is None:
        query_lens = [1] * num_seqs
    total_q = sum(query_lens)
    n_blocks_needed = (max_kv_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    tables = []
    max_blk = 0
    for _ in range(num_seqs):
        if interleaved:
            table = _interleaved_table(n_blocks_needed)
        else:
            table = list(range(n_blocks_needed))
        tables.append(table)
        max_blk = max(max_blk, max(table))
    num_cache_blocks = max_blk + 4
    key_cache = mx.random.normal(
        (num_cache_blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)
    ).astype(dtype)
    value_cache = mx.random.normal(
        (num_cache_blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)
    ).astype(dtype)
    query = mx.random.normal((total_q, NUM_QUERY_HEADS, HEAD_SIZE)).astype(dtype)
    padded = max(len(t) for t in tables)
    block_tables = mx.array(
        [t + [0] * (padded - len(t)) for t in tables], dtype=mx.int32
    )
    kv_lens_arr = mx.array(kv_lens, dtype=mx.int32)
    cu = [0]
    for qlen in query_lens:
        cu.append(cu[-1] + qlen)
    cu_seqlens_q = mx.array(cu, dtype=mx.int32)
    mx.eval(key_cache, value_cache, query, block_tables, kv_lens_arr, cu_seqlens_q)

    out = mx.array(0)
    get_ops().paged_attention_primitive(
        query,
        key_cache,
        value_cache,
        NUM_KV_HEADS,
        scale,
        0.0,
        block_tables,
        kv_lens_arr,
        cu_seqlens_q,
        BLOCK_SIZE,
        max_kv_len,
        -1,
        out,
        window_seqlen_q=window_seqlen_q,
    )
    mx.eval(out)
    ref = ref_paged_attn(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=query_lens,
        kv_lens=kv_lens,
        block_tables=np.array(block_tables),
        scale=scale,
    )
    mx.eval(ref)
    return out, ref


def test_gqa_decode_kernel_is_in_default_library() -> None:
    ops = get_ops()
    assert ops.has_gqa_decode_kernel(), (
        "default shader library is missing paged_attention_gqa_decode; "
        "rebuild with `python -m vllm_metal.metal.build`"
    )
    assert ops.GQA_DECODE_MIN_SEQ_LEN == 16384


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize(
    "kv_len,interleaved",
    [
        (16384, True),  # crossover: first length that must take GQA-decode
        (16384 + 17, True),  # partial last block
        (20000, True),  # hybrid-style interleaved table
        (16384, False),  # contiguous table still valid through the primitive
    ],
)
def test_gqa_decode_matches_reference(
    kv_len: int, interleaved: bool, dtype: mx.Dtype
) -> None:
    ops = get_ops()
    assert ops.has_gqa_decode_kernel()
    assert kv_len >= ops.GQA_DECODE_MIN_SEQ_LEN
    assert NUM_QUERY_HEADS < ops.min_decode_grid()
    out, ref = _run_primitive([kv_len], dtype, interleaved=interleaved, seed=0)
    _assert_close(out, ref, dtype)


def test_below_crossover_stays_on_split_kv() -> None:
    """Contexts under GQA_DECODE_MIN_SEQ_LEN must not take the new pass.

    They still match the reference via the established split-KV kernel.
    """
    ops = get_ops()
    kv_len = ops.GQA_DECODE_MIN_SEQ_LEN - 1
    assert kv_len > ops.PARTITION_SIZE
    assert NUM_QUERY_HEADS < ops.min_decode_grid()
    out, ref = _run_primitive([kv_len], mx.float16, interleaved=True, seed=1)
    _assert_close(out, ref, mx.float16)


def test_multi_seq_does_not_switch_kernel_family() -> None:
    """Two long sequences stay on split-KV (GQA-decode is single-seq only).

    A multi-sequence decode batch is indistinguishable from an expanded
    spec-decode verify window; switching family would break bitwise parity.
    """
    ops = get_ops()
    kv_lens = [ops.GQA_DECODE_MIN_SEQ_LEN, ops.GQA_DECODE_MIN_SEQ_LEN + 64]
    assert NUM_QUERY_HEADS * len(kv_lens) < ops.min_decode_grid()
    out, ref = _run_primitive(kv_lens, mx.float16, interleaved=True, seed=2)
    _assert_close(out, ref, mx.float16)


def test_spec_window_does_not_switch_kernel_family() -> None:
    """A K+1 verify window on a long context stays on the window/split path."""
    ops = get_ops()
    ctx = ops.GQA_DECODE_MIN_SEQ_LEN
    window = 4
    kv_len = ctx + window
    out, ref = _run_primitive(
        [kv_len],
        mx.float16,
        interleaved=True,
        seed=3,
        window_seqlen_q=window,
        query_lens=[window],
    )
    _assert_close(out, ref, mx.float16)
