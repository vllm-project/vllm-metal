# SPDX-License-Identifier: Apache-2.0
"""Parity and routing tests for fused GQA speculative-verification windows."""

from __future__ import annotations

from collections.abc import Iterator

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from tools.attention_bench_utils import attention_tolerances, ref_paged_attn
from vllm_metal.attention.caches.kv_cache import MetalPagedKVCache
from vllm_metal.attention.context import clear_context, get_context, prepare_unified
from vllm_metal.attention.impls.sdpa import sdpa_forward
from vllm_metal.metal import get_ops

HEAD_SIZE = 128
BLOCK_SIZE = 16


@pytest.fixture(autouse=True)
def _enable_dispatch_diagnostics() -> Iterator[None]:
    ops = get_ops()
    previous = ops._set_paged_dispatch_diagnostics(True)
    try:
        yield
    finally:
        ops._set_paged_dispatch_diagnostics(previous)


def _cache(
    num_blocks: int,
    num_kv_heads: int,
    dtype: mx.Dtype,
    head_size: int = HEAD_SIZE,
    block_size: int = BLOCK_SIZE,
) -> mx.array:
    return mx.random.normal(
        shape=(num_blocks, block_size, num_kv_heads, head_size)
    ).astype(dtype)


def _run_window_pair(
    num_q_heads: int,
    num_kv_heads: int,
    dtype: mx.Dtype,
    ctx_lens: list[int],
    windows: list[int],
    *,
    sliding_window: int = -1,
    head_size: int = HEAD_SIZE,
    block_size: int = BLOCK_SIZE,
    seed: int = 0,
) -> tuple[mx.array, mx.array, mx.array]:
    """Run reference, standard window mode, and fused gqa_window on identical data."""
    mx.random.seed(seed)
    total_q = sum(windows)
    total_lens = [c + w for c, w in zip(ctx_lens, windows, strict=True)]
    max_kv = max(total_lens)
    blocks_per_seq = max(1, (max_kv + block_size - 1) // block_size)
    num_seqs = len(windows)

    # Non-trivial interleaved physical block mapping to test block_table indexing.
    all_blocks = blocks_per_seq * num_seqs
    key_cache = _cache(all_blocks, num_kv_heads, dtype, head_size, block_size)
    value_cache = _cache(all_blocks, num_kv_heads, dtype, head_size, block_size)
    query = mx.random.normal(shape=(total_q, num_q_heads, head_size)).astype(dtype)
    mx.eval(key_cache, value_cache, query)

    tables = [
        [(s + b * num_seqs) % all_blocks for b in range(blocks_per_seq)]
        for s in range(num_seqs)
    ]
    cu = [0]
    for w in windows:
        cu.append(cu[-1] + w)
    win_q = max(windows)
    scale = head_size**-0.5

    ops = get_ops()
    tables_a = mx.array(tables, dtype=mx.int32)
    lens_a = mx.array(total_lens, dtype=mx.int32)
    cu_a = mx.array(cu, dtype=mx.int32)

    base_out = ops.paged_attention_primitive(
        query,
        key_cache,
        value_cache,
        num_kv_heads,
        scale,
        0.0,
        tables_a,
        lens_a,
        cu_a,
        block_size,
        max_kv,
        sliding_window,
        window_seqlen_q=win_q,
        gqa_spec_verify=False,
    )
    mx.eval(base_out)
    assert ops.last_paged_dispatch().startswith("window_")

    gqa_out = ops.paged_attention_primitive(
        query,
        key_cache,
        value_cache,
        num_kv_heads,
        scale,
        0.0,
        tables_a,
        lens_a,
        cu_a,
        block_size,
        max_kv,
        sliding_window,
        window_seqlen_q=win_q,
        gqa_spec_verify=True,
    )
    mx.eval(gqa_out)
    assert ops.last_paged_dispatch() == "gqa_window"
    assert ops.last_gqa_partition_size() == 256

    ref_out = ref_paged_attn(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=windows,
        kv_lens=total_lens,
        block_tables=np.array(tables, dtype=np.int32),
        scale=scale,
        sliding_window=None if sliding_window < 0 else sliding_window,
    )
    mx.eval(ref_out)
    return gqa_out, base_out, ref_out


def _assert_close(actual: mx.array, expected: mx.array, dtype: mx.Dtype) -> None:
    atol, rtol = attention_tolerances(dtype)
    np.testing.assert_allclose(
        np.array(actual.astype(mx.float32)),
        np.array(expected.astype(mx.float32)),
        atol=atol,
        rtol=rtol,
    )


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32])
@pytest.mark.parametrize(
    "num_q_heads,num_kv_heads",
    [(32, 8), (24, 4), (16, 2), (28, 4), (8, 8)],
)
@pytest.mark.parametrize("ctx", [0, 37, 255, 513, 4096])
@pytest.mark.parametrize("window", [2, 4, 5, 9])
def test_gqa_window_matches_reference_and_baseline(
    dtype: mx.Dtype,
    num_q_heads: int,
    num_kv_heads: int,
    ctx: int,
    window: int,
) -> None:
    gqa_out, base_out, ref_out = _run_window_pair(
        num_q_heads, num_kv_heads, dtype, [ctx], [window], seed=17
    )
    _assert_close(gqa_out, ref_out, dtype)
    _assert_close(gqa_out, base_out, dtype)


@pytest.mark.parametrize("head_size", [64, 80, 96, 112, 128, 192, 256])
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_gqa_window_non_power_of_two_and_wide_head_dims(
    head_size: int, dtype: mx.Dtype
) -> None:
    gqa_out, base_out, ref_out = _run_window_pair(
        24,
        4,
        dtype,
        [777],
        [6],
        head_size=head_size,
        seed=23,
    )
    _assert_close(gqa_out, ref_out, dtype)
    _assert_close(gqa_out, base_out, dtype)


@pytest.mark.parametrize("block_size", [8, 16, 32])
def test_gqa_window_block_sizes_and_ragged_batched_decode(block_size: int) -> None:
    gqa_out, base_out, ref_out = _run_window_pair(
        32,
        8,
        mx.float16,
        [19, 254, 1500, 3100],
        [1, 3, 5, 7],
        block_size=block_size,
        seed=31,
    )
    _assert_close(gqa_out, ref_out, mx.float16)
    _assert_close(gqa_out, base_out, mx.float16)


@pytest.mark.parametrize("sliding_window", [1, 17, 128, 512])
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_gqa_window_sliding_window_masks(sliding_window: int, dtype: mx.Dtype) -> None:
    gqa_out, base_out, ref_out = _run_window_pair(
        16,
        2,
        dtype,
        [1200, 4096],
        [4, 6],
        sliding_window=sliding_window,
        seed=41,
    )
    _assert_close(gqa_out, ref_out, dtype)
    _assert_close(gqa_out, base_out, dtype)


def test_gqa_window_skips_masked_inf_v() -> None:
    """Masked future KV slots holding ±inf must not poison earlier window rows."""
    mx.random.seed(53)
    ctx, window = 48, 5
    total_len = ctx + window
    blocks = (total_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    key_cache = _cache(blocks, 4, mx.float16)
    value_cache = _cache(blocks, 4, mx.float16)
    poison_pos = ctx + 3  # visible only to rows 3 and 4
    value_cache[poison_pos // BLOCK_SIZE, poison_pos % BLOCK_SIZE, :, :] = float("inf")
    query = mx.random.normal(shape=(window, 16, HEAD_SIZE)).astype(mx.float16)
    mx.eval(key_cache, value_cache, query)

    ops = get_ops()
    tables = mx.array([list(range(blocks))], dtype=mx.int32)
    lens = mx.array([total_len], dtype=mx.int32)
    cu = mx.array([0, window], dtype=mx.int32)

    base = ops.paged_attention_primitive(
        query,
        key_cache,
        value_cache,
        4,
        HEAD_SIZE**-0.5,
        0.0,
        tables,
        lens,
        cu,
        BLOCK_SIZE,
        total_len,
        -1,
        window_seqlen_q=window,
        gqa_spec_verify=False,
    )
    gqa = ops.paged_attention_primitive(
        query,
        key_cache,
        value_cache,
        4,
        HEAD_SIZE**-0.5,
        0.0,
        tables,
        lens,
        cu,
        BLOCK_SIZE,
        total_len,
        -1,
        window_seqlen_q=window,
        gqa_spec_verify=True,
    )
    mx.eval(base)
    assert ops.last_paged_dispatch().startswith("window_")
    mx.eval(gqa)
    assert ops.last_paged_dispatch() == "gqa_window"
    assert bool(mx.all(mx.isfinite(gqa[:3].astype(mx.float32))))
    _assert_close(gqa[:3], base[:3], mx.float16)


def test_gqa_window_fallbacks_when_disabled_or_unsupported() -> None:
    """gqa_disabled, softcap, or sinks fall back cleanly to window_* kernels."""
    mx.random.seed(61)
    ctx, window = 600, 4
    total_len = ctx + window
    blocks = (total_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    key_cache = _cache(blocks, 8, mx.float16)
    value_cache = _cache(blocks, 8, mx.float16)
    query = mx.random.normal(shape=(window, 32, HEAD_SIZE)).astype(mx.float16)
    sinks = mx.zeros((32,), dtype=mx.float32)
    mx.eval(key_cache, value_cache, query, sinks)

    ops = get_ops()
    tables = mx.array([list(range(blocks))], dtype=mx.int32)
    lens = mx.array([total_len], dtype=mx.int32)
    cu = mx.array([0, window], dtype=mx.int32)

    for extra in (
        {"gqa_disabled": True},
        {"softcap": 30.0},
        {"sinks": sinks},
    ):
        out = ops.paged_attention_primitive(
            query,
            key_cache,
            value_cache,
            8,
            HEAD_SIZE**-0.5,
            extra.get("softcap", 0.0),
            tables,
            lens,
            cu,
            BLOCK_SIZE,
            total_len,
            -1,
            window_seqlen_q=window,
            sinks=extra.get("sinks"),
            gqa_disabled=extra.get("gqa_disabled", False),
            gqa_spec_verify=True,
        )
        mx.eval(out)
        assert ops.last_paged_dispatch().startswith("window_")


def test_sdpa_forward_env_flag_ab_toggle(monkeypatch: pytest.MonkeyPatch) -> None:
    """VLLM_METAL_GQA_SPEC_VERIFY=1 routes sdpa_forward verify windows to
    gqa_window and matches VLLM_METAL_GQA_SPEC_VERIFY=0."""

    class TinyGqaAttention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.n_heads = 16
            self.n_kv_heads = 4
            self.head_dim = 64
            self.scale = 64**-0.5
            dim = 16 * 64
            kv_dim = 4 * 64
            self.q_proj = nn.Linear(dim, dim, bias=False)
            self.k_proj = nn.Linear(dim, kv_dim, bias=False)
            self.v_proj = nn.Linear(dim, kv_dim, bias=False)
            self.o_proj = nn.Linear(dim, dim, bias=False)
            self.rope = nn.RoPE(64, traditional=False, base=10000.0)

    mx.random.seed(71)
    attn = TinyGqaAttention()
    attn.set_dtype(mx.float16)
    x_prefill = mx.random.normal((1, 96, 16 * 64)).astype(mx.float16)
    x_verify = mx.random.normal((1, 5, 16 * 64)).astype(mx.float16)
    mx.eval(attn.parameters(), x_prefill, x_verify)

    def run_arm(enable_gqa_verify: str) -> tuple[mx.array, str]:
        monkeypatch.setenv("VLLM_METAL_GQA_SPEC_VERIFY", enable_gqa_verify)
        monkeypatch.delenv("VLLM_METAL_DISABLE_GQA_DECODE", raising=False)
        cache = MetalPagedKVCache(
            num_layers=1,
            num_kv_heads=4,
            head_dim=64,
            num_blocks=16,
            block_size=16,
            dtype=mx.float16,
        )
        cache.kv_heads_per_layer = [4]
        cache.head_dim_per_layer = [64]
        cache.sliding_window_per_layer = [-1]
        cache._upstream_group_indices = [0]
        cache._upstream_block_sizes = [16]
        cache.key_caches = [mx.zeros((16, 16, 4, 64), dtype=mx.float16)]
        cache.value_caches = [mx.zeros((16, 16, 4, 64), dtype=mx.float16)]
        blocks = list(range(8))
        prepare_unified([], [(blocks, 96, 0)], 16, merge_verify_windows=True)
        try:
            ctx = get_context()
            assert ctx is not None
            prefill_out, _ = sdpa_forward(attn, x_prefill, ctx, cache, 0)
            mx.eval(prefill_out)
        finally:
            clear_context()

        prepare_unified([(blocks, 96, 5)], [], 16, merge_verify_windows=True)
        try:
            ctx = get_context()
            assert ctx is not None
            verify_out, _ = sdpa_forward(attn, x_verify, ctx, cache, 0)
            mx.eval(verify_out)
            family = get_ops().last_paged_dispatch()
        finally:
            clear_context()
        return verify_out, family

    ref_out, ref_family = run_arm("0")
    fast_out, fast_family = run_arm("1")
    assert ref_family.startswith("window_")
    assert fast_family == "gqa_window"
    _assert_close(fast_out, ref_out, mx.float16)
