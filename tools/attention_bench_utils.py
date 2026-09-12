# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for attention correctness tests and benchmarks."""

from __future__ import annotations

import mlx.core as mx
import numpy as np


def native_sdpa_contiguous_decode(
    query: mx.array,
    key_cache: mx.array,
    value_cache: mx.array,
    first_block: int,
    seq: int,
    scale: float,
) -> mx.array:
    """Test-only reference: MLX native SDPA over a contiguous paged run.

    ``query`` is ``(1, n_heads, head_dim)``.  The sequence occupies
    ``key_cache`` rows ``[first_block * block_size, first_block * block_size + seq)``.
    Production decode does **not** call this; it is the performance/numeric
    ceiling used to check the paged GQA-decode kernel.
    """
    n_heads = int(query.shape[1])
    dim = int(query.shape[2])
    n_kv_heads = int(key_cache.shape[2])
    block = int(key_cache.shape[1])
    group = n_heads // n_kv_heads
    flat_k = key_cache.reshape(-1, n_kv_heads, dim)
    flat_v = value_cache.reshape(-1, n_kv_heads, dim)
    row0 = first_block * block
    k_view = flat_k[row0 : row0 + seq].transpose(1, 0, 2)[:, None]
    v_view = flat_v[row0 : row0 + seq].transpose(1, 0, 2)[:, None]
    q_sdpa = query.reshape(n_kv_heads, group, 1, dim)
    out = mx.fast.scaled_dot_product_attention(q_sdpa, k_view, v_view, scale=scale)
    return out.reshape(1, n_heads, dim)


def ref_paged_attn(
    query: mx.array,
    key_cache: mx.array,
    value_cache: mx.array,
    query_lens: list[int],
    kv_lens: list[int],
    block_tables: np.ndarray,
    scale: float,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
) -> mx.array:
    """Pure-MLX reference: gather K/V from paged cache, compute attention."""
    _, block_size, num_kv_heads, head_size = key_cache.shape

    outputs: list[mx.array] = []
    start_idx = 0
    for i, query_len in enumerate(query_lens):
        kv_len = kv_lens[i]
        q = query[start_idx : start_idx + query_len] * scale

        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = mx.array(block_tables[i, :num_kv_blocks])

        k = key_cache[block_indices].reshape(-1, num_kv_heads, head_size)[:kv_len]
        v = value_cache[block_indices].reshape(-1, num_kv_heads, head_size)[:kv_len]

        if q.shape[1] != k.shape[1]:
            n_rep = q.shape[1] // k.shape[1]
            k = mx.repeat(k, n_rep, axis=1)
            v = mx.repeat(v, n_rep, axis=1)

        attn = mx.einsum("qhd,khd->hqk", q, k).astype(mx.float32)

        empty_mask = mx.ones((query_len, kv_len))
        mask = mx.triu(empty_mask, k=kv_len - query_len + 1).astype(mx.bool_)

        if sliding_window is not None:
            sliding_window_mask = mx.logical_not(
                mx.triu(empty_mask, k=kv_len - (query_len + sliding_window) + 1).astype(
                    mx.bool_
                )
            )
            mask = mx.logical_or(mask, sliding_window_mask)

        if soft_cap is not None and soft_cap > 0:
            attn = soft_cap * mx.tanh(attn / soft_cap)

        attn = mx.where(mask, float("-inf"), attn)
        attn = mx.softmax(attn, axis=-1).astype(v.dtype)
        outputs.append(mx.einsum("hqk,khd->qhd", attn, v))
        start_idx += query_len

    return mx.concatenate(outputs, axis=0)
