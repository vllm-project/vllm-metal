# SPDX-License-Identifier: Apache-2.0
"""Tests for the paged attention primitive and scatter-based cache donation.

Covers two of Eric's review concerns on PR #225:
  1. No test coverage for the primitive path (paged_attention_primitive).
  2. Buffer donation may silently fail, making scatter-based cache writes
     O(entire_cache) instead of O(new_tokens).

Run with:
    python -m pytest tests/test_primitive_and_donation.py -v -s
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from tools.attention_bench_utils import ref_paged_attn
from vllm_metal.metal import get_ops
from vllm_metal.metal_kernel_backend.cache import MetalPagedKVCache

# ── Shared fixtures ──────────────────────────────────────────────────────────

NUM_KV_HEADS_CASES = [2, 4]
NUM_QUERY_HEADS_CASES = [4, 8]  # must be divisible by corresponding kv heads
HEAD_SIZE = 128
BLOCK_SIZE = 16
DTYPE = mx.float16


def _make_cache_and_inputs(
    num_blocks: int,
    num_kv_heads: int,
    num_query_heads: int,
    seq_lens: list[tuple[int, int]],
    *,
    dtype: mx.Dtype = DTYPE,
):
    """Build a populated cache and matching query/metadata tensors."""
    block_size = BLOCK_SIZE
    head_size = HEAD_SIZE
    num_seqs = len(seq_lens)
    query_lens = [s[0] for s in seq_lens]
    kv_lens = [s[1] for s in seq_lens]
    total_q = sum(query_lens)
    max_kv_len = max(kv_lens)
    scale = head_size**-0.5

    key_cache = mx.random.normal(
        shape=(num_blocks, block_size, num_kv_heads, head_size)
    ).astype(dtype)
    value_cache = mx.random.normal(
        shape=(num_blocks, block_size, num_kv_heads, head_size)
    ).astype(dtype)
    query = mx.random.normal(shape=(total_q, num_query_heads, head_size)).astype(dtype)

    max_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = mx.random.randint(
        0, num_blocks, shape=(num_seqs, max_blocks_per_seq)
    ).astype(mx.int32)

    kv_lens_arr = mx.array(kv_lens, dtype=mx.int32)
    cu_seqlens_q = mx.cumsum(mx.array([0] + query_lens, dtype=mx.int32))

    mx.eval(key_cache, value_cache, query, block_tables, kv_lens_arr, cu_seqlens_q)

    return {
        "query": query,
        "key_cache": key_cache,
        "value_cache": value_cache,
        "num_kv_heads": num_kv_heads,
        "scale": scale,
        "block_tables": block_tables,
        "kv_lens_arr": kv_lens_arr,
        "cu_seqlens_q": cu_seqlens_q,
        "query_lens": query_lens,
        "kv_lens": kv_lens,
        "max_kv_len": max_kv_len,
    }


# ── 1. Primitive correctness tests ──────────────────────────────────────────


@pytest.mark.parametrize(
    "seq_lens",
    [
        [(1, 523), (1, 37), (1, 2011)],
        [(1, 1), (1, 128), (1, 2048)],
    ],
)
@pytest.mark.parametrize(
    "num_heads",
    [(4, 4), (8, 2)],
)
@pytest.mark.parametrize("num_blocks", [256])
def test_primitive_vs_reference_decode(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    num_blocks: int,
) -> None:
    """paged_attention_primitive matches the pure-MLX reference (decode)."""
    mx.random.seed(0)
    num_query_heads, num_kv_heads = num_heads
    d = _make_cache_and_inputs(num_blocks, num_kv_heads, num_query_heads, seq_lens)

    ops = get_ops()
    out = mx.array(0)
    ops.paged_attention_primitive(
        d["query"],
        d["key_cache"],
        d["value_cache"],
        d["num_kv_heads"],
        d["scale"],
        0.0,  # softcap
        d["block_tables"],
        d["kv_lens_arr"],
        d["cu_seqlens_q"],
        BLOCK_SIZE,
        d["max_kv_len"],
        -1,  # sliding_window
        out,
    )
    mx.eval(out)

    ref = ref_paged_attn(
        query=d["query"],
        key_cache=d["key_cache"],
        value_cache=d["value_cache"],
        query_lens=d["query_lens"],
        kv_lens=d["kv_lens"],
        block_tables=np.array(d["block_tables"]),
        scale=d["scale"],
    )
    mx.eval(ref)

    np.testing.assert_allclose(
        np.array(out),
        np.array(ref),
        atol=1.5e-2,
        rtol=1e-2,
    )


@pytest.mark.parametrize(
    "seq_lens",
    [
        [(1, 1328), (5, 18), (129, 463)],
        [(1, 523), (1, 37), (1, 2011)],
    ],
)
@pytest.mark.parametrize(
    "num_heads",
    [(4, 4), (8, 2)],
)
@pytest.mark.parametrize("num_blocks", [256])
def test_primitive_vs_reference_varlen(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    num_blocks: int,
) -> None:
    """paged_attention_primitive matches reference for mixed prefill+decode."""
    mx.random.seed(0)
    num_query_heads, num_kv_heads = num_heads
    d = _make_cache_and_inputs(num_blocks, num_kv_heads, num_query_heads, seq_lens)

    ops = get_ops()
    out = mx.array(0)
    ops.paged_attention_primitive(
        d["query"],
        d["key_cache"],
        d["value_cache"],
        d["num_kv_heads"],
        d["scale"],
        0.0,  # softcap
        d["block_tables"],
        d["kv_lens_arr"],
        d["cu_seqlens_q"],
        BLOCK_SIZE,
        d["max_kv_len"],
        -1,  # sliding_window
        out,
    )
    mx.eval(out)

    ref = ref_paged_attn(
        query=d["query"],
        key_cache=d["key_cache"],
        value_cache=d["value_cache"],
        query_lens=d["query_lens"],
        kv_lens=d["kv_lens"],
        block_tables=np.array(d["block_tables"]),
        scale=d["scale"],
    )
    mx.eval(ref)

    np.testing.assert_allclose(
        np.array(out),
        np.array(ref),
        atol=1.5e-2,
        rtol=1e-2,
    )


# ── 2. Buffer donation test ─────────────────────────────────────────────────


@pytest.mark.parametrize("num_blocks", [128, 256])
def test_scatter_cache_donation(num_blocks: int) -> None:
    """Verify that scatter-based cache write reuses the buffer (donation).

    MLX's buffer donation is an optimisation, not a contract.  This test
    acts as an early-warning: if donation stops happening (due to MLX
    changes or unexpected reference leaks), the memory delta will spike
    and the assertion will fail.
    """
    num_kv_heads = 4
    head_dim = HEAD_SIZE
    num_layers = 1
    block_size = BLOCK_SIZE
    dtype = DTYPE

    cache = MetalPagedKVCache(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        num_blocks=num_blocks,
        block_size=block_size,
        dtype=dtype,
    )
    # Already eval'd by MetalPagedKVCache.__init__

    cache_nbytes = cache.key_caches[0].nbytes  # size of one K or V cache

    # Simulate a decode step: scatter 3 tokens into random slots
    num_tokens = 3
    slot_indices = mx.array(
        np.random.choice(num_blocks * block_size, size=num_tokens, replace=False),
        dtype=mx.int64,
    )
    new_k = mx.random.normal(shape=(num_tokens, num_kv_heads, head_dim)).astype(dtype)
    new_v = mx.random.normal(shape=(num_tokens, num_kv_heads, head_dim)).astype(dtype)
    mx.eval(slot_indices, new_k, new_v)

    # Warm up: run multiple rounds so the MLX memory pool stabilises.
    # IMPORTANT: delete all locals afterwards — any stray reference to the
    # old cache array bumps use_count and defeats buffer donation.
    for _ in range(5):
        _fk = cache.key_caches[0].reshape(-1, num_kv_heads, head_dim)
        _fk[slot_indices] = new_k
        _wk = _fk.reshape(cache.key_caches[0].shape)
        cache.key_caches[0] = _wk
        _fv = cache.value_caches[0].reshape(-1, num_kv_heads, head_dim)
        _fv[slot_indices] = new_v
        _wv = _fv.reshape(cache.value_caches[0].shape)
        cache.value_caches[0] = _wv
        mx.eval(cache.key_caches[0], cache.value_caches[0])
        del _fk, _wk, _fv, _wv

    # ── Measure at steady state (average of several rounds) ──
    total_delta = 0
    num_rounds = 5
    for _ in range(num_rounds):
        mem_before = mx.get_active_memory()

        # K cache scatter + rebind
        flat_k = cache.key_caches[0].reshape(-1, num_kv_heads, head_dim)
        flat_k[slot_indices] = new_k
        new_k_cache = flat_k.reshape(cache.key_caches[0].shape)
        cache.key_caches[0] = new_k_cache

        # V cache scatter + rebind
        flat_v = cache.value_caches[0].reshape(-1, num_kv_heads, head_dim)
        flat_v[slot_indices] = new_v
        new_v_cache = flat_v.reshape(cache.value_caches[0].shape)
        cache.value_caches[0] = new_v_cache

        mx.eval(cache.key_caches[0], cache.value_caches[0])

        mem_after = mx.get_active_memory()
        total_delta += mem_after - mem_before
        del flat_k, new_k_cache, flat_v, new_v_cache

    avg_delta = total_delta / num_rounds

    # If donation works: avg_delta ≈ 0 (buffers reused in-place).
    # If donation fails: avg_delta ≈ 2 * cache_nbytes (full copy for K + V).
    # Allow generous headroom (one full cache) for pool fluctuations.
    threshold = cache_nbytes
    assert avg_delta < threshold, (
        f"Buffer donation likely failed: avg memory growth {avg_delta:,.0f} "
        f"bytes/round over {num_rounds} rounds, "
        f"but each cache is only {cache_nbytes:,} bytes. "
        f"Expected near-zero growth with donation."
    )
