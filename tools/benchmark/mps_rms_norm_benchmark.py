# SPDX-License-Identifier: Apache-2.0
"""Time MPS residual-add RMSNorm against the pre-fusion PyTorch sequence.

Reports host dispatch plus GPU completion time, not isolated GPU kernel time
or serving throughput. Warm both arms, then interleave their order randomly.
Each JSONL record includes every timing sample and paired speedup.

    PYTHONPATH=. PYTORCH_ENABLE_MPS_FALLBACK=0 python \\
        tools/benchmark/mps_rms_norm_benchmark.py --output rms_norm.jsonl
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import platform
import random
import statistics
import subprocess
import time
from pathlib import Path

import torch
from vllm import ir

from tools.attention_bench_utils import (
    native_source_hashes,
    package_versions,
    source_file_hashes,
)
from vllm_metal.pytorch_backend import normalization

DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16}


def torch_sequence(x, residual, weight, epsilon):
    """The original torch_mps provider, including its FP32 ones allocation."""
    summed = x.float() + residual.float()
    normalized = torch.nn.functional.rms_norm(
        summed,
        (summed.shape[-1],),
        torch.ones_like(weight, dtype=torch.float32),
        epsilon,
    )
    return normalized.to(x.dtype) * weight, summed.to(x.dtype)


def _time(fn, inputs, calls):
    torch.mps.synchronize()
    start = time.perf_counter()
    for _ in range(calls):
        fn(*inputs)
    torch.mps.synchronize()
    return (time.perf_counter() - start) * 1e6 / calls


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--batch-sizes", nargs="+", type=int, default=[1, 2, 8, 32, 512]
    )
    parser.add_argument(
        "--hidden-sizes", nargs="+", type=int, default=[1024, 2048, 4096]
    )
    parser.add_argument("--dtypes", nargs="+", choices=DTYPES, default=["bfloat16"])
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--calls", type=int, default=64, help="calls per timed window")
    parser.add_argument(
        "--warmup", type=int, default=3, help="discarded windows per arm"
    )
    parser.add_argument("--reps", type=int, default=15)
    parser.add_argument(
        "--compile", action="store_true", help="also time torch.compile of the baseline"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", help="append one JSON line per shape")
    args = parser.parse_args()
    if (
        min(*args.batch_sizes, *args.hidden_sizes, args.calls, args.reps) < 1
        or args.warmup < 0
    ):
        parser.error(
            "shapes, calls and reps must be positive; warmup must be nonnegative"
        )
    if not torch.backends.mps.is_available():
        parser.error("MPS is unavailable")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        parser.error("set PYTORCH_ENABLE_MPS_FALLBACK=0 before running")

    # Time the provider itself, excluding IR dispatch overhead from both arms.
    fused_add_rms_norm = normalization.fused_add_rms_norm.impl_fn
    root = Path(__file__).resolve().parents[2]
    device = {
        "chip": subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip(),
        "platform": platform.platform(),
        **package_versions("torch", "vllm", "vllm-metal", "mlx"),
        "source_hashes": native_source_hashes(fused_add_rms_norm, torch_sequence),
        "timing": "host dispatch plus GPU completion, microseconds per call",
        "calls": args.calls,
        "warmup": args.warmup,
        "reps": args.reps,
        "seed": args.seed,
        "epsilon": args.epsilon,
    }
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    arms = {"torch_sequence": torch_sequence, "fused": fused_add_rms_norm}
    if args.compile:
        arms["compiled"] = torch.compile(torch_sequence, fullgraph=True, dynamic=True)
    for dtype_name, hidden, batch in itertools.product(
        args.dtypes, args.hidden_sizes, args.batch_sizes
    ):
        dtype = DTYPES[dtype_name]
        inputs = (
            torch.randn((batch, hidden), dtype=dtype, device="mps"),
            torch.randn((batch, hidden), dtype=dtype, device="mps"),
            torch.randn((hidden,), dtype=dtype, device="mps"),
            args.epsilon,
        )
        expected, expected_residual = torch_sequence(*inputs)
        max_error = {}
        for name, fn in arms.items():
            actual, actual_residual = fn(*inputs)
            torch.testing.assert_close(
                actual, expected, **ir.ops.fused_add_rms_norm.get_tolerance(dtype)
            )
            torch.testing.assert_close(
                actual_residual, expected_residual, atol=0, rtol=0
            )
            max_error[name] = (actual.float() - expected.float()).abs().max().item()
        for fn in arms.values():
            for _ in range(args.warmup):
                _time(fn, inputs, args.calls)
        samples = {name: [] for name in arms}
        orders = []
        for _ in range(args.reps):
            order = list(arms)
            rng.shuffle(order)
            orders.append(order)
            for name in order:
                samples[name].append(_time(arms[name], inputs, args.calls))
        medians = {name: statistics.median(values) for name, values in samples.items()}
        paired = {
            name: [
                before / after
                for before, after in zip(samples["torch_sequence"], values, strict=True)
            ]
            for name, values in samples.items()
            if name != "torch_sequence"
        }
        row = {
            **device,
            "shader_hashes": source_file_hashes(
                root,
                [
                    root / "vllm_metal/metal/kernels_v2/normalization.metal",
                    root / "vllm_metal/metal/normalization_kern.metallib",
                ],
            ),
            "dtype": dtype_name,
            "batch": batch,
            "hidden_size": hidden,
            "layout": "contiguous 2D input and residual, contiguous 1D weight",
            "max_abs_error": max_error,
            "median_us": medians,
            "samples_us": samples,
            "orders": orders,
            "paired_speedups": paired,
            "median_paired_speedup": {
                name: statistics.median(values) for name, values in paired.items()
            },
        }
        print(
            f"{dtype_name} B={batch} H={hidden}: "
            + " ".join(f"{name}={value:.2f} us" for name, value in medians.items())
            + f" fused speedup={row['median_paired_speedup']['fused']:.2f}x",
            flush=True,
        )
        if args.output:
            with open(args.output, "a") as handle:
                handle.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
