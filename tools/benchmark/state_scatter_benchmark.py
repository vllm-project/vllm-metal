#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare state-scatter kernel candidates; does not change serving dispatch."""

import argparse
import json
import math
import platform
import statistics
import time
from importlib.metadata import version
from pathlib import Path

import mlx.core as mx
import numpy as np

from vllm_metal.metal import get_ops
from vllm_metal.metal.state_scatter import _native_scatter
from vllm_metal.metal.tilelang_state_scatter import scatter as tilelang_scatter

# name, slots, row shape, dtype, update count, padding, offset, zero
CASES = [
    ("small", 8, (1, 5), mx.float32, 3, 0, 0, False),
    ("conv_b1", 20, (3, 6144), mx.float16, 1, 0, 0, False),
    ("conv_b2", 20, (3, 6144), mx.float16, 2, 0, 0, False),
    ("conv_padded_offset", 20, (3, 6144), mx.float16, 2, 64, 1, False),
    ("recurrent_b1", 20, (16, 128, 128), mx.float32, 1, 0, 0, False),
    ("recurrent_b2", 20, (16, 128, 128), mx.float32, 2, 0, 0, False),
    ("recurrent_zero", 20, (16, 128, 128), mx.float32, 2, 64, 0, True),
]


def run(case, samples, iterations):
    name, slots, row_shape, dtype, count, padding, offset, zero = case
    width = math.prod(row_shape)
    stride = width + padding
    strides = (stride, *(math.prod(row_shape[i + 1 :]) for i in range(len(row_shape))))
    ids = mx.array(list(range(1, count + 1)), dtype=mx.int32)
    np_dtype = np.float16 if dtype == mx.float16 else np.float32
    rows = mx.array(np.full((count, *row_shape), 3, dtype=np_dtype))
    states = {}
    for backend in ("native", "tilelang"):
        raw = mx.full((offset + slots * stride + 1,), 7, dtype=dtype)
        pool = get_ops().as_strided(raw, (slots, *row_shape), strides, offset)
        mx.eval(raw, pool, rows, ids)
        states[backend] = [raw, pool]
    expected = np.full((offset + slots * stride + 1,), 7, dtype=np_dtype)
    for slot in range(1, count + 1):
        expected[offset + slot * stride : offset + slot * stride + width] = (
            0 if zero else 3
        )

    result = {
        "case": name,
        "shape": [slots, *row_shape],
        "dtype": str(dtype),
        "count": count,
        "strides": strides,
        "offset": offset,
        "zero": zero,
        "modes": {},
    }
    kernels = {"native": _native_scatter, "tilelang": tilelang_scatter}
    for mode in ("synchronized", "lazy_chain"):
        timings = {b: [] for b in kernels}
        for b, kernel in kernels.items():
            for _ in range(3):
                states[b][1] = kernel(
                    states[b][1], rows, ids, strides=strides, zero=zero
                )
                mx.eval(states[b][1])
        for sample in range(samples):
            order = (
                ("native", "tilelang") if sample % 2 == 0 else ("tilelang", "native")
            )
            for b in order:
                pool = states[b][1]
                start = time.perf_counter_ns()
                for _ in range(iterations):
                    pool = kernels[b](pool, rows, ids, strides=strides, zero=zero)
                    if mode == "synchronized":
                        mx.eval(pool)
                submitted = time.perf_counter_ns()
                mx.eval(pool)
                end = time.perf_counter_ns()
                states[b][1] = pool
                timings[b].append(
                    {
                        "total_us": (end - start) / iterations / 1000,
                        "loop_us": (submitted - start) / iterations / 1000,
                        "final_wait_us": (end - submitted) / iterations / 1000,
                    }
                )
                np.testing.assert_array_equal(np.array(states[b][0]), expected)
        result["modes"][mode] = {
            b: {
                "samples": values,
                "median_us": statistics.median(x["total_us"] for x in values),
                "p10_p90_us": np.percentile(
                    [x["total_us"] for x in values], [10, 90]
                ).tolist(),
            }
            for b, values in timings.items()
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--iterations", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 3 or args.iterations < 1:
        parser.error("Use at least three samples and one iteration")
    results = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "mlx": version("mlx"),
        "samples": args.samples,
        "iterations": args.iterations,
        "cases": [],
    }
    for case in CASES:
        result = run(case, args.samples, args.iterations)
        results["cases"].append(result)
        args.output.write_text(json.dumps(results, indent=2))
        print(
            case[0],
            {
                m: {b: round(v["median_us"], 2) for b, v in pairs.items()}
                for m, pairs in result["modes"].items()
            },
            flush=True,
        )


if __name__ == "__main__":
    main()
