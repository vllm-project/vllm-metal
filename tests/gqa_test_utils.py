# SPDX-License-Identifier: Apache-2.0
"""Shared GQA fixtures and independent numerical references."""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from tools.attention_bench_utils import attention_tolerances
from vllm_metal.metal import get_ops

NUM_QUERY_HEADS, NUM_KV_HEADS, HEAD_SIZE, BLOCK_SIZE = 32, 8, 128, 16


def _primitive(
    query,
    keys,
    values,
    tables,
    lengths,
    block,
    maximum,
    host_lengths,
    *,
    gqa_disabled=False,
    max_decode_context_len=None,
):
    """Build an ordinary decode node without evaluating its lazy output."""
    ops = get_ops()
    batch, _, head = query.shape
    routing = {"gqa_length_plan": ops.gqa_decode_length_plan(host_lengths)}
    out = ops.paged_attention_primitive(
        query,
        keys,
        values,
        keys.shape[2],
        head**-0.5,
        0.0,
        tables,
        lengths,
        mx.arange(batch + 1, dtype=mx.int32),
        block,
        maximum,
        -1,
        num_decode_requests=batch,
        num_decode_tokens=batch,
        max_decode_context_len=(
            maximum if max_decode_context_len is None else max_decode_context_len
        ),
        gqa_disabled=gqa_disabled,
        **routing,
    )
    return out


def _interleaved_table(n_blocks: int) -> list[int]:
    """Non-contiguous logical pages within a compact physical allocation."""
    return np.random.default_rng(715).permutation(n_blocks).tolist()


def _assert_close(out: mx.array, ref: mx.array, dtype: mx.Dtype) -> None:
    atol, rtol = attention_tolerances(dtype, float32_tolerance=2e-4)
    np.testing.assert_allclose(
        np.array(out.astype(mx.float32)),
        np.array(ref.astype(mx.float32)),
        atol=atol,
        rtol=rtol,
    )
    # Long-context outputs are small: absolute tolerances alone could accept
    # an incorrectly zero-filled result, so also bound the relative L2 error.
    out_np = np.array(out.astype(mx.float32))
    ref_np = np.array(ref.astype(mx.float32))
    relative_l2 = np.linalg.norm(out_np - ref_np) / max(np.linalg.norm(ref_np), 1e-10)
    assert (
        relative_l2
        < {
            mx.bfloat16: 2e-2,
            mx.float16: 6e-3,
            mx.float32: 1e-3,
        }[dtype]
    )


def _dispatch_family() -> str:
    return get_ops().last_paged_dispatch()


def _assert_fallback() -> None:
    # Existing occupancy routing chooses ps0 or ps512 depending on the GPU.
    assert _dispatch_family() in {"per_token_ps0", "per_token_ps512"}


def _grouped_paged_reference(
    *,
    query: mx.array,
    key_cache: mx.array,
    value_cache: mx.array,
    query_lens: list[int],
    kv_lens: list[int],
    block_tables: np.ndarray,
    scale: float,
    sliding_window: int | None = None,
    soft_cap: float = 0.0,
) -> mx.array:
    """FP32 attention without repeating K/V for each grouped query head.

    Fold the query-token and GQA-group axes into a matrix row axis. Each KV
    head then owns one ordinary QK/PV matrix product, so the long-context reference
    retains only the unique gathered K/V instead of allocating a copy for
    every query head. This is a high-level full-softmax oracle, independent
    of the paged kernel's partitioning and online softmax implementation.
    """
    # GPU matrix-matrix and matrix-vector paths can use different effective
    # precision even for float32 arrays. Keep the oracle's arithmetic on CPU
    # so grouping changes neither its precision nor its error allowance.
    with mx.stream(mx.cpu):
        _, block_size, kv_heads, head_size = key_cache.shape
        query_heads = query.shape[1]
        group = query_heads // kv_heads
        outputs = []
        offset = 0
        for index, (query_len, kv_len) in enumerate(
            zip(query_lens, kv_lens, strict=True)
        ):
            pages = mx.array(
                block_tables[index, : (kv_len + block_size - 1) // block_size]
            )
            keys = (
                key_cache[pages]
                .reshape(-1, kv_heads, head_size)[:kv_len]
                .astype(mx.float32)
                .transpose(1, 0, 2)
            )
            values = (
                value_cache[pages]
                .reshape(-1, kv_heads, head_size)[:kv_len]
                .astype(mx.float32)
                .transpose(1, 0, 2)
            )
            queries = (
                query[offset : offset + query_len]
                .astype(mx.float32)
                .reshape(query_len, kv_heads, group, head_size)
                .transpose(1, 0, 2, 3)
                .reshape(kv_heads, query_len * group, head_size)
            )
            scores = mx.einsum("krd,knd->krn", queries * scale, keys)
            scores = scores.reshape(kv_heads, query_len, group, kv_len)
            if soft_cap > 0:
                scores = soft_cap * mx.tanh(scores / soft_cap)
            query_positions = (kv_len - query_len + mx.arange(query_len))[:, None]
            key_positions = mx.arange(kv_len)[None, :]
            allowed = key_positions <= query_positions
            if sliding_window is not None:
                allowed = allowed & (key_positions > query_positions - sliding_window)
            scores = mx.where(allowed[None, :, None, :], scores, float("-inf"))
            probabilities = mx.softmax(scores, axis=-1).reshape(
                kv_heads, query_len * group, kv_len
            )
            result = mx.einsum("krn,knd->krd", probabilities, values)
            outputs.append(
                result.reshape(kv_heads, query_len, group, head_size)
                .transpose(1, 0, 2, 3)
                .reshape(query_len, query_heads, head_size)
            )
            offset += query_len
        return mx.concatenate(outputs, axis=0)


def _run_primitive(
    kv_lens: list[int],
    dtype: mx.Dtype,
    *,
    interleaved: bool,
    seed: int,
    window_seqlen_q: int = 1,
    query_lens: list[int] | None = None,
    num_decode_requests: int = -1,
    num_decode_tokens: int = 0,
    max_decode_context_len: int = 0,
    gqa_disabled: bool = False,
    num_query_heads: int = NUM_QUERY_HEADS,
    num_kv_heads: int = NUM_KV_HEADS,
    head_size: int = HEAD_SIZE,
    block_size: int = BLOCK_SIZE,
    softcap: float = 0.0,
    sliding_window: int = -1,
    sink_value: float | None = None,
    turboquant: bool = False,
    native_reference: bool = False,
    test_partition: int | None = None,
    gqa_lengths: list[int] | None = None,
) -> tuple[mx.array, mx.array]:
    mx.random.seed(seed)
    num_seqs = len(kv_lens)
    max_kv_len = max(kv_lens)
    scale = head_size**-0.5
    if query_lens is None:
        query_lens = [1] * num_seqs
    n_blocks_needed = (max_kv_len + block_size - 1) // block_size
    tables = []
    max_blk = 0
    for s in range(num_seqs):
        if interleaved:
            table = [
                b + s * n_blocks_needed for b in _interleaved_table(n_blocks_needed)
            ]
        else:
            table = list(range(s * n_blocks_needed, (s + 1) * n_blocks_needed))
        tables.append(table)
        max_blk = max(max_blk, max(table))
    cache_shape = (max_blk + 4, block_size, num_kv_heads, head_size)
    key_cache = mx.random.normal(cache_shape).astype(dtype)
    value_cache = mx.random.normal(cache_shape).astype(dtype)
    query = mx.random.normal((sum(query_lens), num_query_heads, head_size)).astype(
        dtype
    )
    block_tables = mx.array(tables, dtype=mx.int32)
    kv_lens_arr = mx.array(kv_lens, dtype=mx.int32)
    cu = [0]
    for qlen in query_lens:
        cu.append(cu[-1] + qlen)
    cu_seqlens_q = mx.array(cu, dtype=mx.int32)
    sinks = (
        None
        if sink_value is None
        else mx.full((num_query_heads,), sink_value, dtype=mx.float32)
    )
    mx.eval(key_cache, value_cache, query, block_tables, kv_lens_arr, cu_seqlens_q)

    key_ref, value_ref = key_cache, value_cache
    quant_kwargs = {}
    if gqa_lengths is not None:
        quant_kwargs["gqa_length_plan"] = get_ops().gqa_decode_length_plan(gqa_lengths)
    if turboquant:
        from vllm_metal.attention.caches.turboquant import (
            get_v_centroids,
            turbo_quant_decode,
            turbo_quant_encode,
        )

        (key_cache, k_scale, k_zero), (value_cache, v_scale) = turbo_quant_encode(
            key_cache, value_cache, "q8_0"
        )
        key_ref, value_ref = turbo_quant_decode(
            (key_cache, k_scale, k_zero),
            (value_cache, v_scale),
            output_dtype=dtype,
            key_quant_type="q8_0",
        )
        quant_kwargs.update(
            {
                "key_scale_cache": k_scale,
                "value_scale_cache": v_scale,
                "key_zero_cache": k_zero,
                "v_centroids": get_v_centroids(3),
                "use_turboquant": True,
                "quant_type": "q8_0",
                "v_bits": 3,
            }
        )
        mx.eval(key_cache, value_cache, k_scale, k_zero, v_scale, key_ref, value_ref)

    if test_partition is not None:
        out = get_ops()._gqa_paged_attention_for_test(
            query,
            key_cache,
            value_cache,
            scale,
            block_tables,
            kv_lens_arr,
            block_size,
            max_kv_len,
            test_partition,
            **quant_kwargs,
        )
    else:
        out = get_ops().paged_attention_primitive(
            query,
            key_cache,
            value_cache,
            num_kv_heads,
            scale,
            softcap,
            block_tables,
            kv_lens_arr,
            cu_seqlens_q,
            block_size,
            max_kv_len,
            sliding_window,
            window_seqlen_q=window_seqlen_q,
            num_decode_requests=num_decode_requests,
            num_decode_tokens=num_decode_tokens,
            max_decode_context_len=max_decode_context_len,
            gqa_disabled=gqa_disabled,
            sinks=sinks,
            **quant_kwargs,
        )
    mx.eval(out)
    if sinks is None and not native_reference:
        ref = _grouped_paged_reference(
            query=query,
            key_cache=key_ref,
            value_cache=value_ref,
            query_lens=query_lens,
            kv_lens=kv_lens,
            block_tables=np.array(block_tables),
            scale=scale,
            sliding_window=None if sliding_window < 0 else sliding_window,
            soft_cap=softcap,
        )
    else:
        assert query_lens == [1] and softcap == 0 and sliding_window < 0
        flat_k = key_cache[block_tables[0]].reshape(-1, num_kv_heads, head_size)
        flat_v = value_cache[block_tables[0]].reshape(-1, num_kv_heads, head_size)
        ref = mx.fast.scaled_dot_product_attention(
            query.transpose(1, 0, 2)[None],
            flat_k[:max_kv_len].transpose(1, 0, 2)[None],
            flat_v[:max_kv_len].transpose(1, 0, 2)[None],
            scale=scale,
            sinks=None if sinks is None else sinks.astype(dtype),
        ).reshape(out.shape)
    mx.eval(ref)
    return out, ref
