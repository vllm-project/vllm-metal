# SPDX-License-Identifier: Apache-2.0
"""Time TurboQuant decode attention: established kernels vs the GQA producer.

The ``established`` arm runs the per-token / split-KV TurboQuant kernels with
GQA disabled. The ``gqa_p256`` / ``gqa_p512`` arms force the GQA producer
through the private test entry on the same quantized cache. Arms run interleaved in a random order
per repetition, and every arm's output is checked against ``established``
before timing. One JSON line per shape records the chip, the detected GPU core
count, and the executed dispatch family of every arm.

    PYTHONPATH=. python tools/benchmark/tq_gqa_decode_benchmark.py
    PYTHONPATH=. python tools/benchmark/tq_gqa_decode_benchmark.py \\
        --geometries 32:8:128:16 --contexts 2048 4096 8192 --requests 1 4 \\
        --k-quants q8_0 q4_0 --output tq_gqa.jsonl
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

from tools.attention_bench_utils import package_versions
from vllm_metal.attention.caches.turboquant import get_v_centroids, turbo_quant_encode
from vllm_metal.metal import get_ops

DTYPES = {"float16": mx.float16, "bfloat16": mx.bfloat16}


def _parse_geometry(text: str) -> tuple[int, int, int, int]:
    heads, kv_heads, head_size, block_size = (int(x) for x in text.split(":"))
    return heads, kv_heads, head_size, block_size


def _arms(ops, geometry, context, requests, dtype, k_quant, v_bits, seed):
    heads, kv_heads, head_size, block_size = geometry
    mx.random.seed(seed)
    pages = -(-context // block_size)
    shape = (requests * pages + 1, block_size, kv_heads, head_size)
    keys = mx.random.normal(shape).astype(dtype)
    values = mx.random.normal(shape).astype(dtype)
    query = mx.random.normal((requests, heads, head_size)).astype(dtype)
    (k_cache, k_scale, k_zero), (v_cache, v_scale) = turbo_quant_encode(
        keys, values, k_quant, value_bits=v_bits
    )
    centroids = get_v_centroids(v_bits)
    # Interleave pages across requests, as a long-running server does.
    order = np.random.default_rng(seed).permutation(requests * pages)
    tables = mx.array(order.astype(np.int32).reshape(requests, pages))
    lengths = mx.array([context] * requests, dtype=mx.int32)
    cu = mx.arange(requests + 1, dtype=mx.int32)
    plan = ops.gqa_decode_length_plan([context] * requests)
    scale = head_size**-0.5
    mx.eval(query, k_cache, k_scale, k_zero, v_cache, v_scale, tables, lengths)

    def established():
        return ops.paged_attention_primitive(
            query, k_cache, v_cache, kv_heads, scale, 0.0, tables, lengths, cu,
            block_size, context, -1,
            key_scale_cache=k_scale, value_scale_cache=v_scale,
            key_zero_cache=k_zero, v_centroids=centroids, use_turboquant=True,
            quant_type=k_quant, v_bits=v_bits, num_decode_requests=requests,
            num_decode_tokens=requests, max_decode_context_len=context,
            gqa_length_plan=plan, gqa_disabled=True,
        )  # fmt: skip

    def gqa(partition):
        return lambda: ops._gqa_turboquant_paged_attention_for_test(
            query, k_cache, v_cache, k_scale, v_scale, k_zero, centroids,
            k_quant, v_bits, scale, tables, lengths, block_size, context,
            partition, plan,
        )  # fmt: skip

    return {"established": established, "gqa_p256": gqa(256), "gqa_p512": gqa(512)}


def _time(fn, calls: int) -> float:
    mx.synchronize()
    start = time.perf_counter()
    mx.eval(*[fn() for _ in range(calls)])
    mx.synchronize()
    return (time.perf_counter() - start) / calls * 1e3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--geometries",
        nargs="+",
        default=["32:8:128:16", "16:2:256:32"],
        help="query_heads:kv_heads:head_size:block_size",
    )
    parser.add_argument(
        "--contexts", nargs="+", type=int, default=[2048, 4096, 8192, 16384, 32768]
    )
    parser.add_argument("--requests", nargs="+", type=int, default=[1])
    parser.add_argument("--dtypes", nargs="+", choices=DTYPES, default=["float16"])
    parser.add_argument("--k-quants", nargs="+", default=["q8_0"])
    parser.add_argument("--v-bits", nargs="+", type=int, default=[3])
    parser.add_argument("--calls", type=int, default=32, help="calls per timed eval")
    parser.add_argument("--reps", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
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
        args.geometries, args.dtypes, args.k_quants, args.v_bits, args.requests,
        args.contexts,
    )  # fmt: skip
    try:
        for geometry, dtype, k_quant, v_bits, requests, context in shapes:
            arms = _arms(
                ops, _parse_geometry(geometry), context, requests, DTYPES[dtype],
                k_quant, v_bits, args.seed,
            )  # fmt: skip
            outputs, families = {}, {}
            for name, fn in arms.items():
                outputs[name] = fn()
                mx.eval(outputs[name])
                families[name] = ops.last_paged_dispatch()
            reference = outputs["established"].astype(mx.float32)
            max_error = {
                name: mx.abs(out.astype(mx.float32) - reference).max().item()
                for name, out in outputs.items()
            }
            for fn in arms.values():
                _time(fn, args.calls)
            samples = {name: [] for name in arms}
            for _ in range(args.reps):
                names = list(arms)
                rng.shuffle(names)
                for name in names:
                    samples[name].append(_time(arms[name], args.calls))
            median = {name: statistics.median(v) for name, v in samples.items()}
            ratio = min(median["gqa_p256"], median["gqa_p512"]) / median["established"]
            print(
                f"{geometry} {dtype} K={k_quant} V={v_bits}b B={requests} "
                f"ctx={context}: "
                + " ".join(f"{n}={m:.3f}" for n, m in median.items())
                + f" ms, best GQA / established = {ratio:.2f}",
                flush=True,
            )
            if args.output:
                row = {
                    **device,
                    "geometry": geometry,
                    "dtype": dtype,
                    "k_quant": k_quant,
                    "v_bits": v_bits,
                    "requests": requests,
                    "context": context,
                    "median_ms": median,
                    "dispatch": families,
                    "max_abs_error_vs_established": max_error,
                }
                with open(args.output, "a") as handle:
                    handle.write(json.dumps(row) + "\n")
    finally:
        mx.synchronize()
        ops._set_paged_dispatch_diagnostics(previous)


if __name__ == "__main__":
    main()
