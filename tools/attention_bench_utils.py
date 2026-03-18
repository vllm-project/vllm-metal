# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for attention correctness tests and benchmarks."""

from __future__ import annotations

import mlx.core as mx
import numpy as np


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


def run_v1_paged_attention(
    query: mx.array,
    key_cache: mx.array,
    value_cache: mx.array,
    num_kv_heads: int,
    scale: float,
    block_tables: mx.array,
    seq_lens: mx.array,
    block_size: int,
    max_seq_len: int,
) -> mx.array:
    """Run kernel_v1 paged attention."""
    from vllm_metal.metal import get_ops

    ops = get_ops()
    out = mx.zeros_like(query)
    mx.eval(out, query, key_cache, value_cache, block_tables, seq_lens)
    ops.paged_attention_v1(
        out,
        query,
        key_cache,
        value_cache,
        num_kv_heads,
        scale,
        block_tables,
        seq_lens,
        block_size,
        max_seq_len,
    )
    mx.synchronize()
    return out
