# SPDX-License-Identifier: Apache-2.0
"""Time ordinary GQA decode: direct output vs P256/P512 vs established kernels.

The ``established`` arm runs the per-token / split-KV kernels with GQA
disabled. The ``p256``, ``p512`` and ``direct`` arms force the GQA producer
through the private test entry. Every arm is checked against a CPU FP32
reference and its executed dispatch family before timing. Arms run
interleaved in a random order per repetition; each timed call consumes the
previous output, so independent attention calls cannot overlap on the GPU.

    PYTHONPATH=. python tools/benchmark/gqa_nosplit_benchmark.py \\
        --dtypes float16 bfloat16 --requests 21 24 32 64 \\
        --contexts 128 129 192 255 256 --output uniform.jsonl
    PYTHONPATH=. python tools/benchmark/gqa_nosplit_benchmark.py \\
        --requests 21 24 32 64 --contexts 129 192 255 256 \\
        --length-patterns alternating one-long --short-context 128
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import statistics
import time

import mlx.core as mx
import numpy as np

from tools.attention_bench_utils import (
    attention_tolerances,
    compare,
    package_versions,
    ref_paged_attn,
)
from vllm_metal.metal import get_ops

DTYPES = {"float16": mx.float16, "bfloat16": mx.bfloat16}
HEADS, KV_HEADS, HEAD_SIZE, BLOCK_SIZE = 32, 8, 128, 16
SCALE = HEAD_SIZE**-0.5


def _lengths(requests, context, pattern, short):
    if pattern == "uniform":
        return [context] * requests
    if pattern == "alternating":
        return [context if i % 2 == 0 else short for i in range(requests)]
    return [context] + [short] * (requests - 1)


def _arms(ops, lengths, context, dtype, seed):
    requests = len(lengths)
    pages = -(-context // BLOCK_SIZE)
    mx.random.seed(seed)
    shape = (requests * pages, BLOCK_SIZE, KV_HEADS, HEAD_SIZE)
    keys = mx.random.normal(shape).astype(dtype)
    values = mx.random.normal(shape).astype(dtype)
    query = mx.random.normal((requests, HEADS, HEAD_SIZE)).astype(dtype)
    # Interleave pages across requests, as a long-running server does.
    order = np.random.default_rng(seed).permutation(requests * pages)
    tables = mx.array(order.astype(np.int32).reshape(requests, pages))
    seq_lens = mx.array(lengths, dtype=mx.int32)
    cu = mx.arange(requests + 1, dtype=mx.int32)
    plan = ops.gqa_decode_length_plan(lengths)
    mx.eval(keys, values, query, tables, seq_lens, cu)

    def established(q):
        return ops.paged_attention_primitive(
            q, keys, values, KV_HEADS, SCALE, 0.0, tables, seq_lens, cu,
            BLOCK_SIZE, context, -1, num_decode_requests=requests,
            num_decode_tokens=requests, max_decode_context_len=context,
            gqa_length_plan=plan, gqa_disabled=True,
        )  # fmt: skip

    def gqa(partition):
        return lambda q: ops._gqa_paged_attention_for_test(
            q, keys, values, SCALE, tables, seq_lens, BLOCK_SIZE, context,
            partition, plan,
        )  # fmt: skip

    with mx.stream(mx.cpu):
        reference = ref_paged_attn(
            query.astype(mx.float32), keys.astype(mx.float32),
            values.astype(mx.float32), [1] * requests, lengths,
            np.array(tables), SCALE,
        )  # fmt: skip
        mx.eval(reference)
    arms = {
        "established": established,
        "p256": gqa(256),
        "p512": gqa(512),
        "direct": gqa(0),
    }
    return query, reference, arms


def _time(fn, query, calls: int) -> float:
    mx.synchronize()
    start = time.perf_counter()
    out = query
    for _ in range(calls):
        out = fn(out)
    mx.eval(out)
    mx.synchronize()
    return (time.perf_counter() - start) * 1e6 / calls


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--requests", nargs="+", type=int, default=[21, 32, 64])
    parser.add_argument("--contexts", nargs="+", type=int, default=[128, 192, 256])
    parser.add_argument("--dtypes", nargs="+", choices=DTYPES, default=["bfloat16"])
    parser.add_argument(
        "--length-patterns",
        nargs="+",
        choices=["uniform", "alternating", "one-long"],
        default=["uniform"],
    )
    parser.add_argument(
        "--short-context", type=int, help="short-row length (default: context / 8)"
    )
    parser.add_argument("--calls", type=int, default=32, help="calls per timed eval")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--reps", type=int, default=15)
    parser.add_argument("--seed", type=int, default=713)
    parser.add_argument("--output", help="append one JSON line per shape")
    args = parser.parse_args()

    ops = get_ops()
    previous = ops._set_paged_dispatch_diagnostics(True)
    device = {
        "device": mx.device_info().get("device_name"),
        "gpu_cores": ops.detected_gpu_core_count(),
        **package_versions("mlx", "vllm"),
    }
    rng = random.Random(args.seed)
    shapes = itertools.product(
        args.dtypes, args.requests, args.contexts, args.length_patterns
    )
    try:
        for dtype, requests, context, pattern in shapes:
            short = min(context, args.short_context or max(1, context // 8))
            lengths = _lengths(requests, context, pattern, short)
            query, reference, arms = _arms(
                ops, lengths, context, DTYPES[dtype], args.seed
            )
            atol, rtol = attention_tolerances(DTYPES[dtype])
            checks, families = {}, {}
            for name, fn in arms.items():
                out = fn(query)
                mx.eval(out)
                families[name] = ops.last_paged_dispatch()
                checks[name] = compare(out, reference, atol=atol, rtol=rtol)
            assert families["direct"] == "gqa_decode_nosplit", families
            assert families["p256"] == families["p512"] == "gqa_decode", families
            assert families["established"].startswith("per_token"), families
            ops._set_paged_dispatch_diagnostics(False)
            for fn in arms.values():
                for _ in range(args.warmup):
                    _time(fn, query, args.calls)
            samples = {name: [] for name in arms}
            for _ in range(args.reps):
                names = list(arms)
                rng.shuffle(names)
                for name in names:
                    samples[name].append(_time(arms[name], query, args.calls))
            ops._set_paged_dispatch_diagnostics(True)
            median = {name: statistics.median(v) for name, v in samples.items()}
            best_other = min(v for n, v in median.items() if n != "direct")
            print(
                f"{dtype} B={requests} L={context} {pattern}: "
                + " ".join(f"{n}={m:.2f}" for n, m in median.items())
                + f" us, direct speedup vs best other = {best_other / median['direct']:.3f}",
                flush=True,
            )
            if args.output:
                row = {
                    **device,
                    "dtype": dtype,
                    "requests": requests,
                    "context": context,
                    "length_pattern": pattern,
                    "lengths": lengths,
                    "median_us": median,
                    "dispatch": families,
                    "checks": checks,
                }
                with open(args.output, "a") as handle:
                    handle.write(json.dumps(row) + "\n")
    finally:
        mx.synchronize()
        ops._set_paged_dispatch_diagnostics(previous)


if __name__ == "__main__":
    main()
