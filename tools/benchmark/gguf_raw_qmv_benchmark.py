# SPDX-License-Identifier: Apache-2.0
"""Microbenchmark for GGUF raw-block QMV and gather-QMV kernels."""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import statistics
import subprocess
import time

import mlx.core as mx
import numpy as np
from gguf import GGML_QUANT_SIZES, GGMLQuantizationType

from tests.gguf_kquant_fixtures import build_kquant_blocks
from vllm_metal.gguf.raw_block import GGUFRawBlockTensor

_DTYPES = {"float16": mx.float16, "bfloat16": mx.bfloat16}
_ELEM_BYTES = {"float16": 2, "bfloat16": 2}


def _parse_linear_shape(spec: str) -> tuple[str, int, int]:
    parts = spec.split(":")
    if len(parts) == 3:
        return parts[0], int(parts[1]), int(parts[2])
    in_f, out_f = int(parts[0]), int(parts[1])
    return f"{in_f}x{out_f}", in_f, out_f


def _time_linear_arm(
    qt: GGUFRawBlockTensor, x: mx.array, fast: bool, calls: int
) -> float:
    os.environ["VLLM_METAL_GGUF_FAST_QMV"] = "1" if fast else "0"
    mx.synchronize()
    t0 = time.perf_counter()
    outs = [qt.matmul(x) for _ in range(calls)]
    mx.eval(*outs)
    mx.synchronize()
    return (time.perf_counter() - t0) / calls * 1e3


def _time_moe_arm(
    qt: GGUFRawBlockTensor,
    x: mx.array,
    indices: mx.array,
    num_experts: int,
    fast: bool,
    calls: int,
) -> float:
    os.environ["VLLM_METAL_GGUF_FAST_QMV"] = "1" if fast else "0"
    mx.synchronize()
    t0 = time.perf_counter()
    outs = [qt.gather_matmul(x, indices, num_experts=num_experts) for _ in range(calls)]
    mx.eval(*outs)
    mx.synchronize()
    return (time.perf_counter() - t0) / calls * 1e3


def _metadata() -> dict[str, str | None]:
    def sh(cmd: list[str]) -> str | None:
        try:
            return subprocess.check_output(cmd, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    return {
        "chip": sh(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "device": mx.device_info().get("device_name"),
        "macos": platform.mac_ver()[0],
        "mlx": mx.__version__,
        "commit": sh(["git", "rev-parse", "--short", "HEAD"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--qtypes", nargs="+", default=["Q2_K", "Q3_K", "Q6_K"])
    parser.add_argument(
        "--dtypes", nargs="+", choices=list(_DTYPES), default=["float16", "bfloat16"]
    )
    parser.add_argument(
        "--shapes",
        nargs="+",
        default=[
            "attn_o:4096:4096",
            "mlp_up:4096:12288",
            "mlp_down:12288:4096",
            "lm_head:2048:65536",
        ],
        help="label:in_features:out_features",
    )
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 4, 8])
    parser.add_argument("--moe-tokens", nargs="+", type=int, default=[1, 4, 16])
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--moe-hidden", type=int, default=2048)
    parser.add_argument("--moe-inter", type=int, default=768)
    parser.add_argument("--calls", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--reps", type=int, default=11)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", help="optional JSON output path")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    mx.random.seed(args.seed)
    meta = _metadata()
    print(json.dumps(meta), flush=True)
    records: list[dict] = []

    print(
        f"\n{'qtype':<6} {'dtype':<9} {'shape':<20} {'B':>2} | "
        f"{'ref ms':>8} {'fast ms':>8} | {'ref GB/s':>9} {'fast GB/s':>9} | "
        f"{'speedup':>7} {'max_err':>9}"
    )
    print("-" * 102)

    for qname in args.qtypes:
        qtype = GGMLQuantizationType[qname]
        _, block_bytes = GGML_QUANT_SIZES[qtype]
        for label, in_f, out_f in (_parse_linear_shape(s) for s in args.shapes):
            raw = build_kquant_blocks(out_f, in_f, qtype, seed=args.seed)
            qt = GGUFRawBlockTensor.from_raw_blocks(raw, (out_f, in_f), qtype)
            qt.eval_arrays()
            weight_bytes = out_f * (in_f // 256) * block_bytes

            for dname in args.dtypes:
                dtype = _DTYPES[dname]
                elem_b = _ELEM_BYTES[dname]
                for batch in args.batches:
                    x = mx.random.normal((batch, in_f)).astype(dtype)
                    mx.eval(x)

                    os.environ["VLLM_METAL_GGUF_FAST_QMV"] = "0"
                    ref_out = qt.matmul(x)
                    os.environ["VLLM_METAL_GGUF_FAST_QMV"] = "1"
                    fast_out = qt.matmul(x)
                    mx.eval(ref_out, fast_out)
                    max_err = float(
                        np.max(
                            np.abs(
                                np.array(fast_out.astype(mx.float32))
                                - np.array(ref_out.astype(mx.float32))
                            )
                        )
                    )

                    for _ in range(args.warmup):
                        _time_linear_arm(qt, x, False, args.calls)
                        _time_linear_arm(qt, x, True, args.calls)

                    samples = {"ref": [], "fast": []}
                    for _ in range(args.reps):
                        order = ["ref", "fast"]
                        rng.shuffle(order)
                        for arm in order:
                            samples[arm].append(
                                _time_linear_arm(qt, x, arm == "fast", args.calls)
                            )

                    ref_ms = statistics.median(samples["ref"])
                    fast_ms = statistics.median(samples["fast"])
                    io_bytes = weight_bytes + batch * (in_f + out_f) * elem_b
                    ref_gbs = io_bytes / (ref_ms * 1e-3) / 1e9
                    fast_gbs = io_bytes / (fast_ms * 1e-3) / 1e9
                    speedup = ref_ms / fast_ms

                    shape_desc = f"{label}({in_f}->{out_f})"
                    print(
                        f"{qname:<6} {dname:<9} {shape_desc:<20} {batch:>2} | "
                        f"{ref_ms:>8.3f} {fast_ms:>8.3f} | "
                        f"{ref_gbs:>9.1f} {fast_gbs:>9.1f} | "
                        f"{speedup:>6.2f}x {max_err:>9.2e}",
                        flush=True,
                    )
                    records.append(
                        {
                            "kind": "qmv",
                            "qtype": qname,
                            "dtype": dname,
                            "label": label,
                            "in_features": in_f,
                            "out_features": out_f,
                            "batch": batch,
                            "ref_ms": ref_ms,
                            "fast_ms": fast_ms,
                            "ref_gb_s": ref_gbs,
                            "fast_gb_s": fast_gbs,
                            "speedup": speedup,
                            "max_abs_err": max_err,
                        }
                    )
            del qt, raw
            mx.clear_cache()

        # MoE gather_matmul benchmark (Qwen3-30B-A3B expert geometry)
        moe_rows = args.num_experts * args.moe_inter
        raw_moe = build_kquant_blocks(moe_rows, args.moe_hidden, qtype, seed=args.seed)
        qt_moe = GGUFRawBlockTensor.from_raw_blocks(
            raw_moe, (moe_rows, args.moe_hidden), qtype
        )
        qt_moe.eval_arrays()
        expert_bytes = args.moe_inter * (args.moe_hidden // 256) * block_bytes

        for dname in args.dtypes:
            dtype = _DTYPES[dname]
            elem_b = _ELEM_BYTES[dname]
            for tokens in args.moe_tokens:
                pairs = tokens * args.top_k
                x = mx.random.normal((1, tokens, 1, 1, args.moe_hidden)).astype(dtype)
                scores = mx.random.normal((1, tokens, args.num_experts))
                indices = mx.argpartition(-scores, args.top_k - 1, axis=-1)[
                    ..., : args.top_k
                ].astype(mx.uint32)
                mx.eval(x, indices)

                os.environ["VLLM_METAL_GGUF_FAST_QMV"] = "0"
                ref_out = qt_moe.gather_matmul(x, indices, num_experts=args.num_experts)
                os.environ["VLLM_METAL_GGUF_FAST_QMV"] = "1"
                fast_out = qt_moe.gather_matmul(
                    x, indices, num_experts=args.num_experts
                )
                mx.eval(ref_out, fast_out)
                max_err = float(
                    np.max(
                        np.abs(
                            np.array(fast_out.astype(mx.float32))
                            - np.array(ref_out.astype(mx.float32))
                        )
                    )
                )

                for _ in range(args.warmup):
                    _time_moe_arm(
                        qt_moe, x, indices, args.num_experts, False, args.calls
                    )
                    _time_moe_arm(
                        qt_moe, x, indices, args.num_experts, True, args.calls
                    )

                samples = {"ref": [], "fast": []}
                for _ in range(args.reps):
                    order = ["ref", "fast"]
                    rng.shuffle(order)
                    for arm in order:
                        samples[arm].append(
                            _time_moe_arm(
                                qt_moe,
                                x,
                                indices,
                                args.num_experts,
                                arm == "fast",
                                args.calls,
                            )
                        )

                ref_ms = statistics.median(samples["ref"])
                fast_ms = statistics.median(samples["fast"])
                io_bytes = (
                    pairs * expert_bytes
                    + tokens * args.moe_hidden * elem_b
                    + pairs * args.moe_inter * elem_b
                )
                ref_gbs = io_bytes / (ref_ms * 1e-3) / 1e9
                fast_gbs = io_bytes / (fast_ms * 1e-3) / 1e9
                speedup = ref_ms / fast_ms

                shape_desc = f"moe_up(T={tokens},P={pairs})"
                print(
                    f"{qname:<6} {dname:<9} {shape_desc:<20} {tokens:>2} | "
                    f"{ref_ms:>8.3f} {fast_ms:>8.3f} | "
                    f"{ref_gbs:>9.1f} {fast_gbs:>9.1f} | "
                    f"{speedup:>6.2f}x {max_err:>9.2e}",
                    flush=True,
                )
                records.append(
                    {
                        "kind": "gather_qmv",
                        "qtype": qname,
                        "dtype": dname,
                        "tokens": tokens,
                        "pairs": pairs,
                        "ref_ms": ref_ms,
                        "fast_ms": fast_ms,
                        "ref_gb_s": ref_gbs,
                        "fast_gb_s": fast_gbs,
                        "speedup": speedup,
                        "max_abs_err": max_err,
                    }
                )
        del qt_moe, raw_moe
        mx.clear_cache()

    os.environ.pop("VLLM_METAL_GGUF_FAST_QMV", None)
    if args.json:
        with open(args.json, "w") as handle:
            json.dump({"metadata": meta, "results": records}, handle, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
