# SPDX-License-Identifier: Apache-2.0
"""Isolation benchmark for speculative-verification attention (RFC #465).

A speculative-decode verification step runs attention for a K+1-token query
window over a long cached prefix. ``prepare_unified`` currently emits that
window as K+1 single-token ``cu_seqlens`` segments, so the kernel re-reads
the prefix once per token; the obvious alternative is one (K+1)-token
segment, which however routes past the ``pure_decode`` gate in
``paged_ops.cpp`` and loses split-KV partitioning. This tool measures both
segmentations against each other and against the single-token decode floor,
so kernel work on the verification shape can be judged with numbers:

  A. split (current) vs fused (one segment) per prefix length and window,
     with both variants parity-checked against ``ref_paged_attn``.
  B. Single-token decode on the same prefixes — the lower bound for an
     ideal fused-window kernel that reads each KV block once.

Timings are medians over REPEATS runs with min/max spread printed
alongside: sub-millisecond cells at short prefixes flip ordering between
runs (observed +-25% on an M4 Pro), so single-shot readings below ~4k
prefix are not meaningful.

Run with:
    uv run python tools/spec_verify_window_bench.py
"""

from __future__ import annotations

import time

import mlx.core as mx
import numpy as np

from tools.attention_bench_utils import ref_paged_attn
from vllm_metal.metal import get_ops

NUM_Q_HEADS = 16  # Qwen3-0.6B GQA shape, matches test_split_kv_decode
NUM_KV_HEADS = 8
HEAD_DIM = 128
BLOCK_SIZE = 16
DTYPE = mx.float16
SCALE = HEAD_DIM**-0.5

PREFIXES = (2048, 8192, 16384, 32768)
WINDOWS = (4, 6)  # K+1 for num_speculative_tokens = 3 and 5
WARMUP = 5
ITERS = 50
REPEATS = 5
PARITY_ATOL = 2e-2  # fp16 kernel-order error, same budget as kernel tests


def _build(prefix: int, window: int, variant: str):
    num_blocks = (prefix + window + BLOCK_SIZE - 1) // BLOCK_SIZE
    mx.random.seed(42)
    key_cache = mx.random.normal(
        shape=(num_blocks + 4, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM)
    ).astype(DTYPE)
    value_cache = mx.random.normal(
        shape=(num_blocks + 4, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM)
    ).astype(DTYPE)
    query = mx.random.normal(shape=(window, NUM_Q_HEADS, HEAD_DIM)).astype(DTYPE)
    table = mx.arange(num_blocks, dtype=mx.int32)
    if variant == "split":
        block_tables = mx.stack([table] * window)
        kv_lens = mx.array([prefix + 1 + j for j in range(window)], dtype=mx.int32)
        cu_seqlens = mx.arange(window + 1, dtype=mx.int32)
    else:
        block_tables = table[None, :]
        kv_lens = mx.array([prefix + window], dtype=mx.int32)
        cu_seqlens = mx.array([0, window], dtype=mx.int32)
    max_kv_len = int(mx.max(kv_lens).item())
    mx.eval(key_cache, value_cache, query, block_tables, kv_lens, cu_seqlens)
    return query, key_cache, value_cache, block_tables, kv_lens, cu_seqlens, max_kv_len


def _run_once(args) -> mx.array:
    query, key_cache, value_cache, block_tables, kv_lens, cu_seqlens, max_kv = args
    out = mx.array(0)
    get_ops().paged_attention_primitive(
        query,
        key_cache,
        value_cache,
        NUM_KV_HEADS,
        SCALE,
        0.0,  # softcap
        block_tables,
        kv_lens,
        cu_seqlens,
        BLOCK_SIZE,
        max_kv,
        -1,  # sliding_window disabled
        out,
    )
    mx.eval(out)
    return out


def _time_gpu(args) -> float:
    for _ in range(WARMUP):
        _run_once(args)
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        _run_once(args)
    mx.synchronize()
    return (time.perf_counter() - t0) / ITERS * 1e6


def _median_spread(args) -> tuple[float, float, float]:
    ts = sorted(_time_gpu(args) for _ in range(REPEATS))
    return ts[REPEATS // 2], ts[0], ts[-1]


def _check_parity(prefix: int, window: int) -> float:
    """Max abs deviation of both segmentations from the reference."""
    worst = 0.0
    for variant, query_lens, kv_lens in (
        ("split", [1] * window, [prefix + 1 + j for j in range(window)]),
        ("fused", [window], [prefix + window]),
    ):
        args = _build(prefix, window, variant)
        out = _run_once(args)
        query, key_cache, value_cache, block_tables = args[0], args[1], args[2], args[3]
        ref = ref_paged_attn(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            query_lens=query_lens,
            kv_lens=kv_lens,
            block_tables=np.array(block_tables),
            scale=SCALE,
        )
        mx.eval(ref)
        diff = float(
            mx.max(mx.abs(out.astype(mx.float32) - ref.astype(mx.float32))).item()
        )
        worst = max(worst, diff)
    return worst


def main() -> int:
    print(
        f"shapes: GQA {NUM_Q_HEADS}q/{NUM_KV_HEADS}kv, head {HEAD_DIM}, "
        f"block {BLOCK_SIZE}, {DTYPE}; median of {REPEATS} x {ITERS} iters"
    )
    print(
        f"\n{'prefix':>7} {'K+1':>4} {'split µs (min..max)':>24} "
        f"{'fused µs (min..max)':>24} {'fused/split':>11} {'ref |Δ|':>9}"
    )
    for window in WINDOWS:
        for prefix in PREFIXES:
            parity = _check_parity(prefix, window)
            s_med, s_min, s_max = _median_spread(_build(prefix, window, "split"))
            f_med, f_min, f_max = _median_spread(_build(prefix, window, "fused"))
            flag = "" if parity <= PARITY_ATOL else "  PARITY FAIL"
            print(
                f"{prefix:>7} {window:>4} "
                f"{s_med:>10.1f} ({s_min:>6.1f}..{s_max:>6.1f}) "
                f"{f_med:>10.1f} ({f_min:>6.1f}..{f_max:>6.1f}) "
                f"{f_med / s_med:>10.2f}x {parity:>9.4f}{flag}"
            )

    print(
        f"\n{'prefix':>7} {'1-token decode µs (min..max)':>30}   (ideal fused-window floor)"
    )
    for prefix in PREFIXES:
        med, lo, hi = _median_spread(_build(prefix, 1, "split"))
        print(f"{prefix:>7} {med:>16.1f} ({lo:>6.1f}..{hi:>6.1f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
