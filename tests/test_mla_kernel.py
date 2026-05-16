# SPDX-License-Identifier: Apache-2.0
"""Direct unit tests for the MLA Metal kernel (RFC #360).

Single-pass decode kernel. The kernel handles ``ctx_len`` of any size
that fits the caller-provided block_tables row, with two-pass softmax
and a cross-simdgroup merge.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from vllm_metal.metal import metal_mla_paged_attention

# Production shapes — only kv_lora_rank=512, qk_rope_head_dim=64 are
# instantiated in mla.metal.
_KV_LORA_RANK = 512
_QK_ROPE_HEAD_DIM = 64
_LATENT_DIM = _KV_LORA_RANK + _QK_ROPE_HEAD_DIM


def _tolerance(dtype: mx.Dtype) -> tuple[float, float]:
    """Per-dtype (rtol, atol).

    bf16 has 7 mantissa bits — 1 ULP near magnitude 1.0 is ~7.8e-3, so a
    512-dim dot product accumulated through fp32 then cast back to bf16
    routinely shows 1–2 ULP drift vs an einsum-ordered reference. fp16
    has 10 mantissa bits and is much tighter.
    """
    if dtype == mx.bfloat16:
        return 1e-2, 2e-2
    return 1e-3, 1e-3


def _absorbed_mla_dense_reference(
    q_nope: mx.array,  # [num_q, num_heads, kv_lora_rank], fp32
    q_pe: mx.array,  # [num_q, num_heads, qk_rope_head_dim], fp32
    kv_norm: mx.array,  # [num_q, ctx_len, kv_lora_rank], fp32
    k_pe: mx.array,  # [num_q, ctx_len, qk_rope_head_dim], fp32
    scale: float,
) -> mx.array:
    """Pure-MLX absorbed-MLA single attention step.

    All inputs are expected fp32. Returns fp32 output of shape
    [num_q, num_heads, kv_lora_rank].
    """
    nope_scores = mx.einsum("qhd,qtd->qht", q_nope, kv_norm)
    pe_scores = mx.einsum("qhd,qtd->qht", q_pe, k_pe)
    scores = scale * (nope_scores + pe_scores)
    weights = mx.softmax(scores, axis=-1)
    out = mx.einsum("qht,qtd->qhd", weights, kv_norm)
    return out


def _make_inputs(
    *,
    num_seqs: int,
    num_heads: int,
    ctx_len: int,
    block_size: int,
    dtype: mx.Dtype,
    seed: int = 0,
):
    """Build a decode input set that fits ``ctx_len`` worth of valid
    context per sequence. Each sequence is allocated
    ``ceil(ctx_len / block_size)`` blocks; out-of-context slots in the
    last block hold garbage that the kernel must ignore via the ctx_len
    bound. Block tables are contiguous per-seq for simplicity."""
    mx.random.seed(seed)

    n_blocks_per_seq = max(1, (ctx_len + block_size - 1) // block_size)
    num_blocks = n_blocks_per_seq * num_seqs

    q_nope = mx.random.normal(shape=(num_seqs, num_heads, _KV_LORA_RANK)).astype(dtype)
    q_pe = mx.random.normal(shape=(num_seqs, num_heads, _QK_ROPE_HEAD_DIM)).astype(
        dtype
    )
    latent_cache = mx.random.normal(shape=(num_blocks, block_size, _LATENT_DIM)).astype(
        dtype
    )

    block_tables_np = np.arange(num_blocks, dtype=np.int32).reshape(
        num_seqs, n_blocks_per_seq
    )
    block_tables = mx.array(block_tables_np)

    context_lens = mx.array([ctx_len] * num_seqs, dtype=mx.uint32)
    cu_seqlens_q = mx.array(list(range(num_seqs + 1)), dtype=mx.int32)

    return (
        q_nope,
        q_pe,
        latent_cache,
        block_tables,
        context_lens,
        cu_seqlens_q,
        block_tables_np,
    )


def _expected_output(
    q_nope: mx.array,
    q_pe: mx.array,
    latent_cache: mx.array,
    block_tables_np: np.ndarray,
    ctx_lens: list[int],
    scale: float,
) -> mx.array:
    """Run the dense reference per-request, gathering the valid-context
    window across however many blocks the request occupies. Casts to
    fp32 for the reference math then back to the original dtype."""
    num_seqs = q_nope.shape[0]
    block_size = latent_cache.shape[1]
    in_dtype = q_nope.dtype

    outs = []
    for i in range(num_seqs):
        ctx_len = ctx_lens[i]
        n_blocks = (ctx_len + block_size - 1) // block_size
        # Concatenate all valid blocks then slice down to ctx_len.
        gathered = mx.concatenate(
            [latent_cache[int(block_tables_np[i, b]), :, :] for b in range(n_blocks)],
            axis=0,
        )[:ctx_len, :].astype(mx.float32)
        kv_norm = gathered[:, :_KV_LORA_RANK].reshape(1, ctx_len, _KV_LORA_RANK)
        k_pe = gathered[:, _KV_LORA_RANK:].reshape(1, ctx_len, _QK_ROPE_HEAD_DIM)
        out_i = _absorbed_mla_dense_reference(
            q_nope[i : i + 1, :, :].astype(mx.float32),
            q_pe[i : i + 1, :, :].astype(mx.float32),
            kv_norm,
            k_pe,
            scale,
        )
        outs.append(out_i)
    return mx.concatenate(outs, axis=0).astype(in_dtype)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("block_size", [16, 32])
def test_decode_single_block(dtype: mx.Dtype, block_size: int) -> None:
    """ctx_len strictly less than block_size — exercises the masked-tail
    code path: out-of-context slots in the BLOCK_SIZE-wide score buffer
    must not contribute to the softmax."""
    ctx_len = max(1, block_size // 2)  # 8 or 16
    (
        q_nope,
        q_pe,
        latent_cache,
        block_tables,
        context_lens,
        cu_seqlens_q,
        block_tables_np,
    ) = _make_inputs(
        num_seqs=2,
        num_heads=4,
        ctx_len=ctx_len,
        block_size=block_size,
        dtype=dtype,
    )

    out = metal_mla_paged_attention(
        q_nope=q_nope,
        q_pe=q_pe,
        latent_cache=latent_cache,
        block_tables=block_tables,
        context_lens=context_lens,
        cu_seqlens_q=cu_seqlens_q,
        scale=0.125,
    )

    expected = _expected_output(
        q_nope,
        q_pe,
        latent_cache,
        block_tables_np,
        ctx_lens=[ctx_len, ctx_len],
        scale=0.125,
    )

    rtol, atol = _tolerance(dtype)
    diff = mx.abs(out.astype(mx.float32) - expected.astype(mx.float32))
    max_abs = mx.max(diff).item()
    assert mx.allclose(
        out.astype(mx.float32), expected.astype(mx.float32), rtol=rtol, atol=atol
    ).item(), (
        f"single-block mismatch (dtype={dtype}, block_size={block_size}, "
        f"ctx_len={ctx_len}): max_abs_diff={max_abs:.5f}"
    )


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("block_size", [16, 32])
def test_decode_full_block(dtype: mx.Dtype, block_size: int) -> None:
    """ctx_len == block_size — every slot in the block is valid context.
    Catches off-by-one errors in the ctx_len boundary check."""
    ctx_len = block_size
    (
        q_nope,
        q_pe,
        latent_cache,
        block_tables,
        context_lens,
        cu_seqlens_q,
        block_tables_np,
    ) = _make_inputs(
        num_seqs=3,
        num_heads=4,
        ctx_len=ctx_len,
        block_size=block_size,
        dtype=dtype,
    )

    out = metal_mla_paged_attention(
        q_nope=q_nope,
        q_pe=q_pe,
        latent_cache=latent_cache,
        block_tables=block_tables,
        context_lens=context_lens,
        cu_seqlens_q=cu_seqlens_q,
        scale=0.125,
    )

    expected = _expected_output(
        q_nope,
        q_pe,
        latent_cache,
        block_tables_np,
        ctx_lens=[ctx_len] * 3,
        scale=0.125,
    )

    rtol, atol = _tolerance(dtype)
    diff = mx.abs(out.astype(mx.float32) - expected.astype(mx.float32))
    max_abs = mx.max(diff).item()
    assert mx.allclose(
        out.astype(mx.float32), expected.astype(mx.float32), rtol=rtol, atol=atol
    ).item(), (
        f"full-block mismatch (dtype={dtype}, block_size={block_size}): "
        f"max_abs_diff={max_abs:.5f}"
    )


def test_decode_single_token_context(dtype: mx.Dtype = mx.float16) -> None:
    """ctx_len = 1 — degenerate case. softmax of a single score is
    always 1.0, output should equal kv_norm[0, :]. Catches obviously-wrong
    softmax / accumulation bugs."""
    block_size = 16
    (
        q_nope,
        q_pe,
        latent_cache,
        block_tables,
        context_lens,
        cu_seqlens_q,
        block_tables_np,
    ) = _make_inputs(
        num_seqs=1,
        num_heads=2,
        ctx_len=1,
        block_size=block_size,
        dtype=dtype,
    )

    out = metal_mla_paged_attention(
        q_nope=q_nope,
        q_pe=q_pe,
        latent_cache=latent_cache,
        block_tables=block_tables,
        context_lens=context_lens,
        cu_seqlens_q=cu_seqlens_q,
        scale=0.125,
    )

    # Expected: out[0, h, :] = kv_norm[block_idx, 0, :KV_LORA_RANK] for all h
    expected_row = latent_cache[
        int(block_tables_np[0, 0]), 0, :_KV_LORA_RANK
    ]  # [KV_LORA_RANK]
    expected = mx.broadcast_to(
        expected_row.reshape(1, 1, _KV_LORA_RANK), out.shape
    ).astype(dtype)

    rtol, atol = _tolerance(dtype)
    assert mx.allclose(
        out.astype(mx.float32), expected.astype(mx.float32), rtol=rtol, atol=atol
    ).item()


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("block_size", [16, 32])
def test_decode_two_blocks_with_partial_tail(dtype: mx.Dtype, block_size: int) -> None:
    """ctx_len = block_size + 1 — minimal multi-block case. Exercises the
    block-table walk (vs. step 7's hardcoded block 0), the partial last
    block, and the cross-warp merge with one warp doing real work and
    seven idle (their state must be the identity element)."""
    ctx_len = block_size + 1
    (
        q_nope,
        q_pe,
        latent_cache,
        block_tables,
        context_lens,
        cu_seqlens_q,
        block_tables_np,
    ) = _make_inputs(
        num_seqs=2,
        num_heads=4,
        ctx_len=ctx_len,
        block_size=block_size,
        dtype=dtype,
    )

    out = metal_mla_paged_attention(
        q_nope=q_nope,
        q_pe=q_pe,
        latent_cache=latent_cache,
        block_tables=block_tables,
        context_lens=context_lens,
        cu_seqlens_q=cu_seqlens_q,
        scale=0.125,
    )

    expected = _expected_output(
        q_nope,
        q_pe,
        latent_cache,
        block_tables_np,
        ctx_lens=[ctx_len, ctx_len],
        scale=0.125,
    )
    rtol, atol = _tolerance(dtype)
    diff = mx.abs(out.astype(mx.float32) - expected.astype(mx.float32))
    max_abs = mx.max(diff).item()
    assert mx.allclose(
        out.astype(mx.float32), expected.astype(mx.float32), rtol=rtol, atol=atol
    ).item(), (
        f"two-block mismatch (dtype={dtype}, block_size={block_size}, "
        f"ctx_len={ctx_len}): max_abs_diff={max_abs:.5f}"
    )


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("block_size", [16, 32])
def test_decode_one_block_per_warp(dtype: mx.Dtype, block_size: int) -> None:
    """ctx_len = block_size * NUM_WARPS — every warp processes exactly one
    block, all NUM_WARPS=8 warp states participate in the merge."""
    num_warps = 8
    ctx_len = block_size * num_warps
    (
        q_nope,
        q_pe,
        latent_cache,
        block_tables,
        context_lens,
        cu_seqlens_q,
        block_tables_np,
    ) = _make_inputs(
        num_seqs=1,
        num_heads=4,
        ctx_len=ctx_len,
        block_size=block_size,
        dtype=dtype,
    )

    out = metal_mla_paged_attention(
        q_nope=q_nope,
        q_pe=q_pe,
        latent_cache=latent_cache,
        block_tables=block_tables,
        context_lens=context_lens,
        cu_seqlens_q=cu_seqlens_q,
        scale=0.125,
    )

    expected = _expected_output(
        q_nope,
        q_pe,
        latent_cache,
        block_tables_np,
        ctx_lens=[ctx_len],
        scale=0.125,
    )
    rtol, atol = _tolerance(dtype)
    assert mx.allclose(
        out.astype(mx.float32), expected.astype(mx.float32), rtol=rtol, atol=atol
    ).item()


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_decode_many_blocks_per_warp(dtype: mx.Dtype) -> None:
    """ctx_len = 4096 — each warp does ctx_len / (block_size * NUM_WARPS)
    iterations, tests the warp-strided loop's iteration count and the
    online softmax accumulator stability over many blocks."""
    block_size = 32
    ctx_len = 4096
    (
        q_nope,
        q_pe,
        latent_cache,
        block_tables,
        context_lens,
        cu_seqlens_q,
        block_tables_np,
    ) = _make_inputs(
        num_seqs=1,
        num_heads=2,
        ctx_len=ctx_len,
        block_size=block_size,
        dtype=dtype,
    )

    out = metal_mla_paged_attention(
        q_nope=q_nope,
        q_pe=q_pe,
        latent_cache=latent_cache,
        block_tables=block_tables,
        context_lens=context_lens,
        cu_seqlens_q=cu_seqlens_q,
        scale=0.125,
    )

    expected = _expected_output(
        q_nope,
        q_pe,
        latent_cache,
        block_tables_np,
        ctx_lens=[ctx_len],
        scale=0.125,
    )
    rtol, atol = _tolerance(dtype)
    diff = mx.abs(out.astype(mx.float32) - expected.astype(mx.float32))
    max_abs = mx.max(diff).item()
    assert mx.allclose(
        out.astype(mx.float32), expected.astype(mx.float32), rtol=rtol, atol=atol
    ).item(), (
        f"long-context mismatch (dtype={dtype}, ctx_len={ctx_len}): "
        f"max_abs_diff={max_abs:.5f}"
    )


def test_decode_mixed_ctx_batch() -> None:
    """Batch with three sequences at different ctx_lens — exercises the
    per-seq context_lens read and ensures the kernel doesn't accidentally
    use one sequence's ctx_len for another (which would happen if the
    block-iteration bound was hoisted out of the threadgroup-local lookup)."""
    block_size = 32
    ctx_lens = [1, 200, 1024]  # single-block, multi-block, many-warp
    max_ctx = max(ctx_lens)
    n_blocks_per_seq = (max_ctx + block_size - 1) // block_size
    num_seqs = len(ctx_lens)
    num_heads = 4
    dtype = mx.float16

    mx.random.seed(13)
    num_blocks = n_blocks_per_seq * num_seqs
    q_nope = mx.random.normal(shape=(num_seqs, num_heads, _KV_LORA_RANK)).astype(dtype)
    q_pe = mx.random.normal(shape=(num_seqs, num_heads, _QK_ROPE_HEAD_DIM)).astype(
        dtype
    )
    latent_cache = mx.random.normal(shape=(num_blocks, block_size, _LATENT_DIM)).astype(
        dtype
    )
    block_tables_np = np.arange(num_blocks, dtype=np.int32).reshape(
        num_seqs, n_blocks_per_seq
    )
    block_tables = mx.array(block_tables_np)
    context_lens = mx.array(ctx_lens, dtype=mx.uint32)
    cu_seqlens_q = mx.array(list(range(num_seqs + 1)), dtype=mx.int32)

    out = metal_mla_paged_attention(
        q_nope=q_nope,
        q_pe=q_pe,
        latent_cache=latent_cache,
        block_tables=block_tables,
        context_lens=context_lens,
        cu_seqlens_q=cu_seqlens_q,
        scale=0.125,
    )

    expected = _expected_output(
        q_nope,
        q_pe,
        latent_cache,
        block_tables_np,
        ctx_lens=ctx_lens,
        scale=0.125,
    )
    rtol, atol = _tolerance(dtype)
    assert mx.allclose(
        out.astype(mx.float32), expected.astype(mx.float32), rtol=rtol, atol=atol
    ).item()


def test_unsupported_kv_lora_rank_raises() -> None:
    """Phase 1 only instantiates kv_lora_rank=512. Anything else must
    raise at dispatch time, not silently dispatch a wrong kernel."""
    q_nope = mx.zeros((1, 2, 16), dtype=mx.float16)
    q_pe = mx.zeros((1, 2, 4), dtype=mx.float16)
    latent_cache = mx.zeros((1, 16, 20), dtype=mx.float16)
    block_tables = mx.zeros((1, 1), dtype=mx.int32)
    context_lens = mx.array([1], dtype=mx.uint32)
    cu_seqlens_q = mx.array([0, 1], dtype=mx.int32)

    with pytest.raises(RuntimeError, match="kv_lora_rank=512"):
        out = metal_mla_paged_attention(
            q_nope=q_nope,
            q_pe=q_pe,
            latent_cache=latent_cache,
            block_tables=block_tables,
            context_lens=context_lens,
            cu_seqlens_q=cu_seqlens_q,
            scale=0.125,
        )
        mx.eval(out)


# ---------------------------------------------------------------------------
# Cross-head amortization (HEADS_PER_TG > 1)
# ---------------------------------------------------------------------------
# G=2 packs 2 query heads into one threadgroup so each K/V load is reused
# across 2 dot products (2× KV bandwidth amortization). num_heads must be
# divisible by G; num_threads is 512 instead of 1024 so the per-thread
# register footprint stays comparable.


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("block_size", [16, 32])
@pytest.mark.parametrize("num_heads", [2, 8, 128])
def test_g2_matches_dense(dtype: mx.Dtype, block_size: int, num_heads: int) -> None:
    """G=2 single-pass kernel must match the dense reference within fp16
    tolerance — same math as G=1, just two heads sharing each K/V load."""
    ctx_len = 384  # spans multiple blocks at both bs=16 and bs=32
    (
        q_nope,
        q_pe,
        latent_cache,
        block_tables,
        context_lens,
        cu_seqlens_q,
        block_tables_np,
    ) = _make_inputs(
        num_seqs=2,
        num_heads=num_heads,
        ctx_len=ctx_len,
        block_size=block_size,
        dtype=dtype,
    )

    out = metal_mla_paged_attention(
        q_nope=q_nope,
        q_pe=q_pe,
        latent_cache=latent_cache,
        block_tables=block_tables,
        context_lens=context_lens,
        cu_seqlens_q=cu_seqlens_q,
        scale=0.125,
        heads_per_tg=2,
    )

    expected = _expected_output(
        q_nope,
        q_pe,
        latent_cache,
        block_tables_np,
        ctx_lens=[ctx_len, ctx_len],
        scale=0.125,
    )

    rtol, atol = _tolerance(dtype)
    diff = mx.abs(out.astype(mx.float32) - expected.astype(mx.float32))
    max_abs = mx.max(diff).item()
    assert mx.allclose(
        out.astype(mx.float32), expected.astype(mx.float32), rtol=rtol, atol=atol
    ).item(), (
        f"G=2 mismatch (dtype={dtype}, bs={block_size}, H={num_heads}): "
        f"max_abs_diff={max_abs:.5f}"
    )


def test_g2_matches_g1() -> None:
    """G=2 and G=1 must produce identical outputs (up to fp32 rounding) on
    the same workload — different parallelism, same math."""
    ctx_len = 1024
    (
        q_nope,
        q_pe,
        latent_cache,
        block_tables,
        context_lens,
        cu_seqlens_q,
        _,
    ) = _make_inputs(
        num_seqs=2,
        num_heads=128,
        ctx_len=ctx_len,
        block_size=16,
        dtype=mx.float16,
    )

    out_g2 = metal_mla_paged_attention(
        q_nope=q_nope,
        q_pe=q_pe,
        latent_cache=latent_cache,
        block_tables=block_tables,
        context_lens=context_lens,
        cu_seqlens_q=cu_seqlens_q,
        scale=0.125,
        heads_per_tg=2,
    )
    out_g1 = metal_mla_paged_attention(
        q_nope=q_nope,
        q_pe=q_pe,
        latent_cache=latent_cache,
        block_tables=block_tables,
        context_lens=context_lens,
        cu_seqlens_q=cu_seqlens_q,
        scale=0.125,
        heads_per_tg=1,
    )

    diff = mx.abs(out_g2.astype(mx.float32) - out_g1.astype(mx.float32))
    max_abs = mx.max(diff).item()
    # Same dtype, same data, same math — only parallelism differs. Reduction
    # order can differ (different number of simdgroups merging), so allow
    # a small tolerance from accumulation reordering.
    assert max_abs < 1e-2, f"G=2 vs G=1 divergence: max_abs_diff={max_abs:.5f}"


def test_g_invalid_raises() -> None:
    """num_heads not divisible by G should raise at the dispatch boundary,
    not silently produce garbage."""
    q_nope = mx.zeros((1, 5, _KV_LORA_RANK), dtype=mx.float16)
    q_pe = mx.zeros((1, 5, _QK_ROPE_HEAD_DIM), dtype=mx.float16)
    latent_cache = mx.zeros((1, 16, _LATENT_DIM), dtype=mx.float16)
    block_tables = mx.zeros((1, 1), dtype=mx.int32)
    context_lens = mx.array([1], dtype=mx.uint32)
    cu_seqlens_q = mx.array([0, 1], dtype=mx.int32)

    with pytest.raises(RuntimeError, match="divisible by heads_per_tg"):
        out = metal_mla_paged_attention(
            q_nope=q_nope,
            q_pe=q_pe,
            latent_cache=latent_cache,
            block_tables=block_tables,
            context_lens=context_lens,
            cu_seqlens_q=cu_seqlens_q,
            scale=0.125,
            heads_per_tg=2,  # 5 % 2 != 0
        )
        mx.eval(out)


def test_g_unsupported_raises() -> None:
    """heads_per_tg outside {1, 2} should raise — only G=1 and G=2 are
    currently instantiated. Catches accidental routing to a non-existent
    PSO (e.g. a future picker change that emits G=4 without restoring
    the instantiations)."""
    q_nope = mx.zeros((1, 4, _KV_LORA_RANK), dtype=mx.float16)
    q_pe = mx.zeros((1, 4, _QK_ROPE_HEAD_DIM), dtype=mx.float16)
    latent_cache = mx.zeros((1, 16, _LATENT_DIM), dtype=mx.float16)
    block_tables = mx.zeros((1, 1), dtype=mx.int32)
    context_lens = mx.array([1], dtype=mx.uint32)
    cu_seqlens_q = mx.array([0, 1], dtype=mx.int32)

    with pytest.raises(RuntimeError, match=r"heads_per_tg must be in \{1, 2\}"):
        out = metal_mla_paged_attention(
            q_nope=q_nope,
            q_pe=q_pe,
            latent_cache=latent_cache,
            block_tables=block_tables,
            context_lens=context_lens,
            cu_seqlens_q=cu_seqlens_q,
            scale=0.125,
            heads_per_tg=4,
        )
        mx.eval(out)


# ---------------------------------------------------------------------------
# Mixed-dtype rejection: every MLA dispatcher picks one Metal specialization
# from q_nope.dtype() but the kernel template binds the same `T` to all
# fp16/bf16 buffers (q_nope, q_pe, latent_cache, out, tmp_out). If they
# disagree the shader silently reinterprets bytes — e.g. a bf16 cache read
# as fp16 — and produces corrupt attention. Validate that we reject up
# front instead.
# ---------------------------------------------------------------------------


def _mixed_dtype_inputs(dtype_q: mx.Dtype, dtype_kv: mx.Dtype):
    """Build a minimal valid input set with mismatched query / cache dtypes.
    Production shapes (kv_lora_rank=512) so the dispatcher gets past the
    shape check and reaches dtype validation."""
    block_size = 16
    num_seqs = 1
    num_heads = 4
    q_nope = mx.zeros((num_seqs, num_heads, _KV_LORA_RANK), dtype=dtype_q)
    q_pe = mx.zeros((num_seqs, num_heads, _QK_ROPE_HEAD_DIM), dtype=dtype_q)
    latent_cache = mx.zeros((1, block_size, _LATENT_DIM), dtype=dtype_kv)
    block_tables = mx.zeros((num_seqs, 1), dtype=mx.int32)
    context_lens = mx.array([1], dtype=mx.uint32)
    cu_seqlens_q = mx.array([0, 1], dtype=mx.int32)
    return q_nope, q_pe, latent_cache, block_tables, context_lens, cu_seqlens_q


def test_mla_rejects_mixed_dtypes_single_pass() -> None:
    """fp16 queries vs bf16 latent cache must raise, not silently corrupt."""
    q_nope, q_pe, latent_cache, btab, ctx_lens, cu_q = _mixed_dtype_inputs(
        dtype_q=mx.float16, dtype_kv=mx.bfloat16
    )
    with pytest.raises(RuntimeError, match="must share the same dtype"):
        out = metal_mla_paged_attention(
            q_nope=q_nope,
            q_pe=q_pe,
            latent_cache=latent_cache,
            block_tables=btab,
            context_lens=ctx_lens,
            cu_seqlens_q=cu_q,
            scale=0.125,
        )
        mx.eval(out)
