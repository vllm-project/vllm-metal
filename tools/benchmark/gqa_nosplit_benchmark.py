# SPDX-License-Identifier: Apache-2.0
"""Compare ordinary GQA direct output, P256/P512, and established decode.

Checks an independent CPU FP32 attention reference and the executed route
before timing. Samples include host graph construction and GPU completion,
not isolated shader time or serving throughput. Timed calls form a dependency
chain, preventing identical-node elimination or overlap between independent
attention calls from overstating the occupancy of a single decode step.

    PYTHONPATH=. python tools/benchmark/gqa_nosplit_benchmark.py \\
        --requests 1 8 16 24 32 64 --contexts 64 128 256 512 1024 2048 \\
        --output gqa_nosplit.jsonl
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np

from tools.attention_bench_utils import (
    attention_tolerances,
    compare,
    package_versions,
    ref_paged_attn,
    source_file_hashes,
)
from vllm_metal.metal import get_ops

DTYPES = {"float16": mx.float16, "bfloat16": mx.bfloat16}
ARMS = {"established": None, "p256": 256, "p512": 512, "direct": 0}
HEADS, KV_HEADS, HEAD_SIZE, BLOCK_SIZE = 32, 8, 128, 16


@dataclass
class DecodeCase:
    ops: object
    query: mx.array
    keys: mx.array
    values: mx.array
    tables: mx.array
    lengths: mx.array
    host_lengths: list[int]
    cu: mx.array
    plan: object
    maximum: int

    def call(self, query, partition):
        if partition is not None:
            return self.ops._gqa_paged_attention_for_test(
                query, self.keys, self.values, HEAD_SIZE**-0.5, self.tables,
                self.lengths, BLOCK_SIZE, self.maximum, partition, self.plan,
            )  # fmt: skip
        batch = len(self.host_lengths)
        return self.ops.paged_attention_primitive(
            query, self.keys, self.values, KV_HEADS, HEAD_SIZE**-0.5, 0.0,
            self.tables, self.lengths, self.cu, BLOCK_SIZE, self.maximum, -1,
            num_decode_requests=batch, num_decode_tokens=batch,
            max_decode_context_len=self.maximum, gqa_length_plan=self.plan,
            gqa_disabled=True,
        )  # fmt: skip

    def reference(self):
        with mx.stream(mx.cpu):
            reference = ref_paged_attn(
                self.query.astype(mx.float32),
                self.keys.astype(mx.float32), self.values.astype(mx.float32),
                [1] * len(self.host_lengths), self.host_lengths,
                np.array(self.tables), HEAD_SIZE**-0.5,
            )  # fmt: skip
            mx.eval(reference)
        return reference


def _case(ops, requests, context, pattern, dtype, seed, short_context):
    short = min(context, short_context or max(1, context // 8))
    if pattern == "uniform":
        lengths = [context] * requests
    elif pattern == "alternating":
        lengths = [context if i % 2 == 0 else short for i in range(requests)]
    else:
        lengths = [context] + [short] * (requests - 1)
    pages = -(-context // BLOCK_SIZE)
    mx.random.seed(seed)
    shape = (requests * pages, BLOCK_SIZE, KV_HEADS, HEAD_SIZE)
    keys = mx.random.normal(shape).astype(dtype)
    values = mx.random.normal(shape).astype(dtype)
    query = mx.random.normal((requests, HEADS, HEAD_SIZE)).astype(dtype)
    tables = mx.array(
        np.random.default_rng(seed)
        .permutation(requests * pages)
        .astype(np.int32)
        .reshape(requests, pages)
    )
    device_lengths = mx.array(lengths, dtype=mx.int32)
    cu = mx.arange(requests + 1, dtype=mx.int32)
    mx.eval(keys, values, query, tables, device_lengths, cu)
    return DecodeCase(
        ops, query, keys, values, tables, device_lengths, lengths, cu,
        ops.gqa_decode_length_plan(lengths), context,
    )  # fmt: skip


def _check(case, dtype):
    reference = case.reference()
    atol, rtol = attention_tolerances(dtype)
    checks, routes = {}, {}
    for name, partition in ARMS.items():
        output = case.call(case.query, partition)
        mx.eval(output)
        family = case.ops.last_paged_dispatch()
        executed_partition = case.ops.last_gqa_partition_size()
        if partition is None:
            assert family in {"per_token_ps0", "per_token_ps512"}, family
        else:
            expected = "gqa_decode_nosplit" if partition == 0 else "gqa_decode"
            assert family == expected, (name, family)
            assert executed_partition == partition, (name, executed_partition)
            assert case.ops.last_gqa_num_requests() == len(case.host_lengths)
        checks[name] = compare(output, reference, atol=atol, rtol=rtol)
        routes[name] = {"family": family, "partition": executed_partition}
    return checks, routes


def _time(case, partition, calls):
    mx.synchronize()
    start = time.perf_counter()
    output = case.query
    for _ in range(calls):
        output = case.call(output, partition)
    mx.eval(output)
    mx.synchronize()
    return (time.perf_counter() - start) * 1e6 / calls


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--requests", nargs="+", type=int, default=[1, 8, 16, 24, 32, 64]
    )
    parser.add_argument(
        "--contexts", nargs="+", type=int, default=[64, 128, 256, 512, 1024, 2048]
    )
    parser.add_argument("--dtypes", nargs="+", choices=DTYPES, default=["bfloat16"])
    parser.add_argument(
        "--length-patterns",
        nargs="+",
        choices=["uniform", "alternating", "one-long"],
        default=["uniform"],
    )
    parser.add_argument(
        "--short-context",
        type=int,
        help="short-row length for ragged patterns (default: one eighth of context)",
    )
    parser.add_argument("--calls", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--reps", type=int, default=15)
    parser.add_argument("--seed", type=int, default=713)
    parser.add_argument("--output", help="append one JSON line per shape")
    args = parser.parse_args()
    if (
        min(*args.requests, *args.contexts, args.calls, args.reps) < 1
        or args.warmup < 0
        or (args.short_context is not None and args.short_context < 1)
    ):
        parser.error(
            "requests, contexts, calls, reps and short-context must be positive; warmup must be nonnegative"
        )

    ops = get_ops()
    root = Path(__file__).resolve().parents[2]
    metal = root / "vllm_metal/metal"
    metadata = {
        "device": mx.device_info().get("device_name"),
        "gpu_cores": ops.detected_gpu_core_count(),
        **package_versions("mlx", "vllm", "vllm-metal"),
        "source_and_artifact_sha256": source_file_hashes(
            root,
            [
                Path(__file__).resolve(),
                metal / "paged_attention.cpp",
                metal / "ops_internal.h",
                metal / "mlx_bindings.cpp",
                metal / "kernels_v2/pagedattention.metal",
                metal / "paged_attention_v2_kern.metallib",
                Path(ops.__file__).resolve(),
            ],
        ),
        "timing": "host graph construction and dependent GPU completion, microseconds per call",
        "geometry": [HEADS, KV_HEADS, HEAD_SIZE, BLOCK_SIZE],
        "calls": args.calls,
        "warmup": args.warmup,
        "reps": args.reps,
        "seed": args.seed,
    }
    rng = random.Random(args.seed)
    previous = ops._set_paged_dispatch_diagnostics(True)
    try:
        for dtype, batch, context, pattern in itertools.product(
            args.dtypes, args.requests, args.contexts, args.length_patterns
        ):
            case = _case(
                ops,
                batch,
                context,
                pattern,
                DTYPES[dtype],
                args.seed,
                args.short_context,
            )
            mx.synchronize()
            ops._set_paged_dispatch_diagnostics(True)
            checks, routes = _check(case, DTYPES[dtype])
            mx.synchronize()
            ops._set_paged_dispatch_diagnostics(False)
            for partition in ARMS.values():
                for _ in range(args.warmup):
                    _time(case, partition, args.calls)
            samples = {name: [] for name in ARMS}
            orders = []
            for _ in range(args.reps):
                order = list(ARMS)
                rng.shuffle(order)
                orders.append(order)
                for name in order:
                    samples[name].append(_time(case, ARMS[name], args.calls))
            medians = {
                name: statistics.median(values) for name, values in samples.items()
            }
            paired = {
                name: [a / b for a, b in zip(values, samples["direct"], strict=True)]
                for name, values in samples.items()
                if name != "direct"
            }
            row = {
                **metadata,
                "dtype": dtype,
                "requests": batch,
                "context": context,
                "length_pattern": pattern,
                "lengths": case.host_lengths,
                "checks": checks,
                "dispatch": routes,
                "median_us": medians,
                "samples_us": samples,
                "orders": orders,
                "paired_direct_speedups": paired,
                "median_paired_direct_speedup": {
                    name: statistics.median(values) for name, values in paired.items()
                },
            }
            print(
                f"{dtype} B={batch} L={context} {pattern}: "
                + " ".join(f"{name}={value:.2f}us" for name, value in medians.items()),
                flush=True,
            )
            if args.output:
                with open(args.output, "a") as handle:
                    handle.write(json.dumps(row) + "\n")
    finally:
        mx.synchronize()
        ops._set_paged_dispatch_diagnostics(previous)


if __name__ == "__main__":
    main()
