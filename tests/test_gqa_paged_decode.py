# SPDX-License-Identifier: Apache-2.0
"""GQA-shared flash-decode pass inside ``paged_attention_primitive``.

Eligible single-request, long-context decode (no TQ/sinks/softcap/window,
head size in {64,96,128,256}, GQA group <= 8, ``window_seqlen_q <= 1``,
``num_seqs == 1`` with the scheduler confirming one real decode request,
KV ``max_seq_len >= GQA_DECODE_MIN_SEQ_LEN``) dispatches to
``paged_attention_gqa_decode`` and merges through ``paged_attention_v2_reduce``.
Multi-request batches, spec-decode verify windows, and everything under the
crossover stay on the established per-token / split-KV family.
``VLLM_METAL_DISABLE_GQA_DECODE`` (mirror kwarg ``gqa_disabled``) forces
eligible batches back to the established kernels.

The routing tests assert the dispatch family via ``ops.last_paged_dispatch()``
(the gate lives below the Python boundary, so the family name is the only
faithful record of which kernel ran); parity is checked against
``ref_paged_attn`` on every path.
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


def _dispatch_family() -> str:
    """Kernel family chosen by the most recent primitive eval."""
    return get_ops().last_paged_dispatch()


def _run_primitive(
    kv_lens: list[int],
    dtype: mx.Dtype,
    *,
    interleaved: bool,
    seed: int,
    window_seqlen_q: int = 1,
    query_lens: list[int] | None = None,
    num_decode_requests: int = -1,
    gqa_disabled: bool = False,
    num_query_heads: int = NUM_QUERY_HEADS,
    num_kv_heads: int = NUM_KV_HEADS,
    head_size: int = HEAD_SIZE,
) -> tuple[mx.array, mx.array]:
    mx.random.seed(seed)
    num_seqs = len(kv_lens)
    max_kv_len = max(kv_lens)
    scale = head_size**-0.5
    if query_lens is None:
        query_lens = [1] * num_seqs
    total_q = sum(query_lens)
    n_blocks_needed = (max_kv_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    tables = []
    max_blk = 0
    # Offset each sequence's pages so multi-seq batches do not alias.
    page_stride = n_blocks_needed * 3
    for s in range(num_seqs):
        if interleaved:
            table = [b + s * page_stride for b in _interleaved_table(n_blocks_needed)]
        else:
            table = list(range(s * n_blocks_needed, (s + 1) * n_blocks_needed))
        tables.append(table)
        max_blk = max(max_blk, max(table))
    num_cache_blocks = max_blk + 4
    key_cache = mx.random.normal(
        (num_cache_blocks, BLOCK_SIZE, num_kv_heads, head_size)
    ).astype(dtype)
    value_cache = mx.random.normal(
        (num_cache_blocks, BLOCK_SIZE, num_kv_heads, head_size)
    ).astype(dtype)
    query = mx.random.normal((total_q, num_query_heads, head_size)).astype(dtype)
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
        num_kv_heads,
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
        num_decode_requests=num_decode_requests,
        gqa_disabled=gqa_disabled,
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
    # num_decode_requests omitted (-1): the conservative default still
    # routes a single long-context sequence to the GQA kernel.
    out, ref = _run_primitive([kv_len], dtype, interleaved=interleaved, seed=0)
    assert _dispatch_family() == "gqa_decode"
    _assert_close(out, ref, dtype)


def test_explicit_one_decode_request_takes_gqa() -> None:
    """Production passes num_decode_requests=1 for a lone decode row."""
    ops = get_ops()
    kv_len = ops.GQA_DECODE_MIN_SEQ_LEN
    out, ref = _run_primitive(
        [kv_len],
        mx.bfloat16,
        interleaved=True,
        seed=4,
        num_decode_requests=1,
    )
    assert _dispatch_family() == "gqa_decode"
    _assert_close(out, ref, mx.bfloat16)


def test_qwen38_gqa_shape_takes_kernel() -> None:
    """The measured 27B shape (24q/4kv/hs256) at the 16k crossover."""
    ops = get_ops()
    kv_len = ops.GQA_DECODE_MIN_SEQ_LEN
    out, ref = _run_primitive(
        [kv_len],
        mx.bfloat16,
        interleaved=True,
        seed=5,
        num_decode_requests=1,
        num_query_heads=24,
        num_kv_heads=4,
        head_size=256,
    )
    assert _dispatch_family() == "gqa_decode"
    _assert_close(out, ref, mx.bfloat16)


def test_below_crossover_stays_on_split_kv() -> None:
    """A single sequence under GQA_DECODE_MIN_SEQ_LEN stays on split-KV."""
    ops = get_ops()
    kv_len = ops.GQA_DECODE_MIN_SEQ_LEN - 1
    assert kv_len > ops.PARTITION_SIZE
    assert NUM_QUERY_HEADS < ops.min_decode_grid()
    out, ref = _run_primitive([kv_len], mx.float16, interleaved=True, seed=1)
    assert _dispatch_family() == "per_token_ps512"
    _assert_close(out, ref, mx.float16)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("num_decode_requests", [2, -1])
def test_multi_request_batch_stays_on_established_kernels(
    dtype: mx.Dtype, num_decode_requests: int
) -> None:
    """Multi-request decode never takes the GQA kernel (review on #715).

    The batch's aggregate KV meets the 16k budget with two 8k+ sequences,
    which the earlier draft routed to GQA-decode; the gate is now scoped to
    ``num_seqs == 1``.  Both the explicit scheduler count and the omitted
    default must stay on the established split-KV family, with unchanged
    numerics.
    """
    ops = get_ops()
    half = ops.GQA_DECODE_MIN_SEQ_LEN // 2
    kv_lens = [half, half + 64]
    assert len(kv_lens) * max(kv_lens) >= ops.GQA_DECODE_MIN_SEQ_LEN
    assert NUM_QUERY_HEADS * len(kv_lens) < ops.min_decode_grid()
    out, ref = _run_primitive(
        kv_lens,
        dtype,
        interleaved=True,
        seed=2,
        num_decode_requests=num_decode_requests,
    )
    assert _dispatch_family() == "per_token_ps512"
    _assert_close(out, ref, dtype)


def test_two_individually_eligible_requests_stay_off_gqa() -> None:
    """Two 16k decode rows: each would take GQA alone; the batch must not.

    This is the production-serving shape the first-cut gate refuses
    (review on #715). Kernel microbench of 4x64k GQA is follow-up work.
    """
    ops = get_ops()
    kv_len = ops.GQA_DECODE_MIN_SEQ_LEN
    out, ref = _run_primitive(
        [kv_len, kv_len],
        mx.float16,
        interleaved=True,
        seed=6,
        num_decode_requests=2,
    )
    assert _dispatch_family() == "per_token_ps512"
    _assert_close(out, ref, mx.float16)


def test_spec_window_does_not_switch_kernel_family() -> None:
    """A K+1 verify window on a long context stays on the window/split path.

    Production expanded verify windows pass window_seqlen_q=K+1 even when
    packed as length-1 segments; this keeps them off GQA-decode so they
    stay bitwise with the windowed dispatch.
    """
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
    assert _dispatch_family().startswith("window_")
    _assert_close(out, ref, mx.float16)


def test_gqa_disable_flag_forces_established_kernels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """gqa_disabled=True keeps eligible batches off the GQA kernel.

    This mirrors VLLM_METAL_DISABLE_GQA_DECODE (issue #713 benchmarking
    escape hatch): results must be unchanged on the fallback family.
    """
    from vllm_metal import envs

    ops = get_ops()
    assert envs.VLLM_METAL_DISABLE_GQA_DECODE is False  # default off
    kv_len = 2 * ops.GQA_DECODE_MIN_SEQ_LEN
    out_on, ref = _run_primitive([kv_len], mx.bfloat16, interleaved=True, seed=7)
    assert _dispatch_family() == "gqa_decode"
    out_off, _ = _run_primitive(
        [kv_len], mx.bfloat16, interleaved=True, seed=7, gqa_disabled=True
    )
    assert _dispatch_family() == "per_token_ps512"
    _assert_close(out_on, ref, mx.bfloat16)
    _assert_close(out_off, ref, mx.bfloat16)  # numerics identical either way

    monkeypatch.setenv("VLLM_METAL_DISABLE_GQA_DECODE", "1")
    assert envs.VLLM_METAL_DISABLE_GQA_DECODE is True
    out_env, _ = _run_primitive(
        [kv_len],
        mx.bfloat16,
        interleaved=True,
        seed=7,
        gqa_disabled=envs.VLLM_METAL_DISABLE_GQA_DECODE,
    )
    assert _dispatch_family() == "per_token_ps512"
    _assert_close(out_env, ref, mx.bfloat16)
