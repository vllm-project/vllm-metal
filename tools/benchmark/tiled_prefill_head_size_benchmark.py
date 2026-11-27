# SPDX-License-Identifier: Apache-2.0
"""Tiled prefill kernel time for one full-attention layer, per head size.

Causal prefill of the whole prompt (q = kv = seq_len), bf16, 16 query heads,
block 16, NAX off so the tiled kernel runs on every chip.  KV heads follow
gemma-4-26B-A4B: 8 for the 256-wide sliding layers, 2 for the 512-wide full
layers.  Twice the head size is twice the MMA work, so the 512/256 column
shows how well the HEAD_SIZE=512 instantiation keeps the GPU busy.

Run:
    .venv-vllm-metal/bin/python tools/benchmark/tiled_prefill_head_size_benchmark.py
    .venv-vllm-metal/bin/python tools/benchmark/tiled_prefill_head_size_benchmark.py \
        --seq-lens 8192,16384,32768
"""

from __future__ import annotations

import argparse
import time

import mlx.core as mx

from vllm_metal.metal import get_ops

BLOCK = 16
HEADS = 16
KV_HEADS = {128: 8, 256: 8, 512: 2}
DTYPE = mx.bfloat16


def _median_ms(head_size: int, seq_len: int, repeats: int) -> float:
    kv_heads = KV_HEADS[head_size]
    mx.random.seed(7)
    blocks = (seq_len + BLOCK - 1) // BLOCK
    shape = (blocks + 1, BLOCK, kv_heads, head_size)
    key_cache = mx.random.normal(shape).astype(DTYPE)
    value_cache = mx.random.normal(shape).astype(DTYPE)
    query = mx.random.normal((seq_len, HEADS, head_size)).astype(DTYPE)
    table = mx.array([list(range(1, blocks + 1))], dtype=mx.int32)
    kv_lens = mx.array([seq_len], dtype=mx.int32)
    cu_seqlens_q = mx.array([0, seq_len], dtype=mx.int32)
    mx.eval(key_cache, value_cache, query, table, kv_lens, cu_seqlens_q)
    ops = get_ops()

    def once() -> None:
        out = ops.paged_attention_primitive(
            query,
            key_cache,
            value_cache,
            kv_heads,
            head_size**-0.5,
            0.0,
            table,
            kv_lens,
            cu_seqlens_q,
            BLOCK,
            seq_len,
            -1,
        )
        mx.eval(out)

    once()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        once()
        samples.append((time.perf_counter() - t0) * 1e3)
    return sorted(samples)[len(samples) // 2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seq-lens", default="4096,8192,16384")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    seq_lens = [int(s) for s in args.seq_lens.split(",")]

    get_ops().set_nax_enabled(False)
    print(f"{'seq_len':>8} {'128 ms':>9} {'256 ms':>9} {'512 ms':>9} {'512/256':>8}")
    for seq_len in seq_lens:
        ms = {hd: _median_ms(hd, seq_len, args.repeats) for hd in KV_HEADS}
        print(
            f"{seq_len:>8} {ms[128]:>9.1f} {ms[256]:>9.1f} {ms[512]:>9.1f}"
            f" {ms[512] / ms[256]:>7.2f}x"
        )


if __name__ == "__main__":
    main()
