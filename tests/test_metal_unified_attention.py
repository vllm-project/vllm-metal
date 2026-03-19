# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Adapted from vLLM's test_triton_unified_attention.py for Metal/MLX.
#
# Compares metal_unified_attention
# against ref_paged_attn (a naive pure-MLX loop implementation that is
# trivially correct).  Both receive the same paged KV cache and query
# inputs; the test asserts their outputs match within FP tolerance.

import mlx.core as mx
import numpy as np
import pytest

from tools.attention_bench_utils import ref_paged_attn, run_v1_paged_attention
from vllm_metal.metal import PARTITION_THRESHOLD, metal_unified_attention

# Original upstream parameters (vLLM Triton/CUDA test_triton_unified_attention.py):
#   HEAD_SIZES = [128, 256]
#   NUM_BLOCKS = [32768, 2048]
#   sliding_window = [None, 64, 128, 256]
#   DTYPES = [torch.bfloat16]
#   Also tested: FP8 output quantization (QDTYPES), 3D decode kernel
#     (SEQ_THRESHOLD_3D) — both CUDA-specific, omitted here.
# Current sizes are reduced for Apple Silicon unified memory.
# TODO: try head_size=256 and larger num_blocks as @pytest.mark.slow variants.
NUM_HEADS = [(4, 4), (8, 2), (5, 1)]
HEAD_SIZES = [128]
BLOCK_SIZES = [16]
DTYPES = [mx.float16]


# ---------------------------------------------------------------------------
# Shared reference / decode helpers
# ---------------------------------------------------------------------------
#
# The pure-MLX textbook attention reference and the kernel_v1 wrapper are
# implemented in a shared module so the correctness tests and the benchmark
# script exercise the same logic.
#


# ---------------------------------------------------------------------------
# Triangle edge: v1 == ref (decode-only)
#
# Validates that the v1 kernel and the pure-MLX reference produce the same
# results for decode-only inputs. This test also validates ref_paged_attn
# itself, so we can trust it as ground truth.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "seq_lens",
    [
        [(1, 523), (1, 37), (1, 2011)],
        [(1, 1), (1, 128), (1, 2048)],
    ],
)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("num_blocks", [256])
def test_v1_kernel_vs_reference(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    dtype: mx.Dtype,
    block_size: int,
    num_blocks: int,
) -> None:
    """v1 kernel == reference for decode-only inputs.

    Completes the triangle: if v1 == ref and v2 == v1, then v2 == ref.
    Also serves as a smoke test for ref_paged_attn correctness.
    """
    mx.random.seed(0)
    num_seqs = len(seq_lens)
    query_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]
    assert all(q == 1 for q in query_lens)
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0
    max_kv_len = max(kv_lens)
    scale = head_size**-0.5

    query = mx.random.normal(shape=(num_seqs, num_query_heads, head_size)).astype(dtype)
    key_cache = mx.random.normal(
        shape=(num_blocks, block_size, num_kv_heads, head_size)
    ).astype(dtype)
    value_cache = mx.random.normal(
        shape=(num_blocks, block_size, num_kv_heads, head_size)
    ).astype(dtype)
    kv_lens_arr = mx.array(kv_lens, dtype=mx.int32)

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = mx.random.randint(
        0, num_blocks, shape=(num_seqs, max_num_blocks_per_seq)
    ).astype(mx.int32)

    v1_output = run_v1_paged_attention(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        num_kv_heads=num_kv_heads,
        scale=scale,
        block_tables=block_tables,
        seq_lens=kv_lens_arr,
        block_size=block_size,
        max_seq_len=max_kv_len,
    )

    ref_output = ref_paged_attn(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=query_lens,
        kv_lens=kv_lens,
        block_tables=np.array(block_tables),
        scale=scale,
    )

    atol, rtol = 1.5e-2, 1e-2
    np.testing.assert_allclose(
        np.array(v1_output), np.array(ref_output), atol=atol, rtol=rtol
    )


# ---------------------------------------------------------------------------
# Scaffolding
# Triangle edge: v2 == v1 (decode-only scaffolding)
#
# Freezes parameters to the subset that the existing paged_attention_v1
# already handles: every sequence has q_len=1, no sliding window, no
# soft_cap. Compares the v2 kernel output against the v1 kernel output
# to prove v2 is a drop-in replacement for decode. This remains useful as
# a focused decode-only regression test alongside the full varlen test below.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "seq_lens",
    [
        [(1, 523), (1, 37), (1, 2011)],
        [(1, 1), (1, 128), (1, 2048)],
    ],
)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("num_blocks", [256])
def test_metal_unified_attn_decode_only(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    dtype: mx.Dtype,
    block_size: int,
    num_blocks: int,
) -> None:
    """Decode-only: all q_len=1, no sliding window, no soft cap.

    Compares the v2 unified kernel against the existing v1 paged_attention
    kernel to prove v2 is a drop-in replacement for the decode path.
    """
    mx.random.seed(0)
    num_seqs = len(seq_lens)
    query_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]
    assert all(q == 1 for q in query_lens), "Scaffolding test requires q_len=1"
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0
    max_kv_len = max(kv_lens)
    scale = head_size**-0.5

    query = mx.random.normal(shape=(num_seqs, num_query_heads, head_size)).astype(dtype)
    key_cache = mx.random.normal(
        shape=(num_blocks, block_size, num_kv_heads, head_size)
    ).astype(dtype)
    value_cache = mx.random.normal(
        shape=(num_blocks, block_size, num_kv_heads, head_size)
    ).astype(dtype)
    cu_query_lens = mx.cumsum(mx.array([0] + query_lens, dtype=mx.int32))
    kv_lens_arr = mx.array(kv_lens, dtype=mx.int32)

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = mx.random.randint(
        0, num_blocks, shape=(num_seqs, max_num_blocks_per_seq)
    ).astype(mx.int32)

    # --- v1 kernel output (known-correct, production code) ---
    v1_output = run_v1_paged_attention(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        num_kv_heads=num_kv_heads,
        scale=scale,
        block_tables=block_tables,
        seq_lens=kv_lens_arr,
        block_size=block_size,
        max_seq_len=max_kv_len,
    )

    # --- v2 kernel output ---
    v2_output = mx.zeros_like(query)

    metal_unified_attention(
        q=query,
        k=key_cache,
        v=value_cache,
        out=v2_output,
        cu_seqlens_q=cu_query_lens,
        seqused_k=kv_lens_arr,
        max_seqlen_q=1,
        max_seqlen_k=max_kv_len,
        softmax_scale=scale,
        causal=True,
        window_size=(-1, -1),
        block_table=block_tables,
        softcap=0,
    )

    # Partitioned decode changes the reduction order versus v1, so long-context
    # cases need a slightly looser tolerance than the no-partition exact-match
    # path.
    atol = rtol = 1e-4
    if max_kv_len >= PARTITION_THRESHOLD:
        atol = rtol = 3e-4
    np.testing.assert_allclose(
        np.array(v2_output), np.array(v1_output), atol=atol, rtol=rtol
    )


# ---------------------------------------------------------------------------
# Triangle edge: v2 == ref (full varlen unified attention)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "seq_lens", [[(1, 1328), (5, 18), (129, 463)], [(1, 523), (1, 37), (1, 2011)]]
)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("sliding_window", [None, 128])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("soft_cap", [None, 50.0])
@pytest.mark.parametrize("num_blocks", [256])
def test_metal_unified_attn(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    sliding_window: int | None,
    dtype: mx.Dtype,
    block_size: int,
    soft_cap: float | None,
    num_blocks: int,
) -> None:
    mx.random.seed(0)
    num_seqs = len(seq_lens)
    query_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]

    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0
    max_query_len = max(query_lens)
    max_kv_len = max(kv_lens)
    window_size = (sliding_window - 1, 0) if sliding_window is not None else (-1, -1)
    scale = head_size**-0.5

    query = mx.random.normal(
        shape=(sum(query_lens), num_query_heads, head_size)
    ).astype(dtype)
    key_cache = mx.random.normal(
        shape=(num_blocks, block_size, num_kv_heads, head_size)
    ).astype(dtype)
    value_cache = mx.random.normal(
        shape=(num_blocks, block_size, num_kv_heads, head_size)
    ).astype(dtype)
    cu_query_lens = mx.cumsum(mx.array([0] + query_lens, dtype=mx.int32))
    kv_lens_arr = mx.array(kv_lens, dtype=mx.int32)

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = mx.random.randint(
        0, num_blocks, shape=(num_seqs, max_num_blocks_per_seq)
    ).astype(mx.int32)

    output = mx.zeros_like(query)

    metal_unified_attention(
        q=query,
        k=key_cache,
        v=value_cache,
        out=output,
        cu_seqlens_q=cu_query_lens,
        seqused_k=kv_lens_arr,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_kv_len,
        softmax_scale=scale,
        causal=True,
        window_size=window_size,
        block_table=block_tables,
        softcap=soft_cap if soft_cap is not None else 0,
    )

    ref_output = ref_paged_attn(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=query_lens,
        kv_lens=kv_lens,
        block_tables=np.array(block_tables),
        scale=scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
    )

    atol, rtol = 1.5e-2, 1e-2
    np.testing.assert_allclose(
        np.array(output), np.array(ref_output), atol=atol, rtol=rtol
    )
