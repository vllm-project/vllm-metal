# SPDX-License-Identifier: Apache-2.0
"""Microbenchmark for GQA speculative-verification window paged attention."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import statistics
import subprocess
import time

import mlx.core as mx
import numpy as np

from vllm_metal.metal import get_ops

_DTYPES = {"float16": mx.float16, "bfloat16": mx.bfloat16}
_ELEM_BYTES = {"float16": 2, "bfloat16": 2}
_CONFIGS = [
    ("Llama3-8B (32q/8kv, d=128)", 32, 8, 128),
    ("Qwen3-30B-A3B (32q/4kv, d=128)", 32, 4, 128),
    ("Qwen2.5-7B (28q/4kv, d=128)", 28, 4, 128),
    ("Gemma/Qwen35 (16q/2kv, d=256)", 16, 2, 256),
]


def _time_arm(
    ops,
    query: mx.array,
    kc: mx.array,
    vc: mx.array,
    nkv: int,
    scale: float,
    tables: mx.array,
    seq_lens: mx.array,
    cu: mx.array,
    block_size: int,
    max_kv: int,
    window_seqlen_q: int,
    gqa_spec_verify: bool,
    layers: int,
    warmup: int,
    reps: int,
) -> tuple[float, mx.array]:
    def run_once() -> list[mx.array]:
        outs = [
            ops.paged_attention_primitive(
                query,
                kc,
                vc,
                nkv,
                scale,
                0.0,
                tables,
                seq_lens,
                cu,
                block_size,
                max_kv,
                -1,
                window_seqlen_q=window_seqlen_q,
                gqa_spec_verify=gqa_spec_verify,
            )
            for _ in range(layers)
        ]
        mx.eval(*outs)
        return outs

    for _ in range(warmup):
        outs = run_once()
    mx.synchronize()
    samples: list[float] = []
    for _ in range(reps):
        t0 = time.perf_counter()
        outs = run_once()
        mx.synchronize()
        samples.append((time.perf_counter() - t0) / layers * 1e3)
    return statistics.median(samples), outs[0]


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
    parser.add_argument(
        "--dtypes",
        nargs="+",
        choices=list(_DTYPES),
        default=["float16", "bfloat16"],
    )
    parser.add_argument("--concs", nargs="+", type=int, default=[1, 4, 8])
    parser.add_argument("--ctxs", nargs="+", type=int, default=[2048, 8192, 16384])
    parser.add_argument("--k-spec", type=int, default=5)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--layers", type=int, default=24)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--reps", type=int, default=15)
    parser.add_argument("--json", help="optional JSON output path")
    args = parser.parse_args()

    ops = get_ops()
    mx.random.seed(0)
    meta = _metadata()
    print(json.dumps(meta), flush=True)
    records: list[dict] = []

    window = args.k_spec + 1
    for dname in args.dtypes:
        dtype = _DTYPES[dname]
        elem_b = _ELEM_BYTES[dname]
        for label, nq, nkv, hd in _CONFIGS:
            print(f"\n=== {label} | dtype={dname} | K={args.k_spec} (W={window}) ===")
            header = (
                f"{'B':>2} {'ctx':>6} | {'exp ms':>7} {'win ms':>7} {'gqa ms':>7} | "
                f"{'win GB/s':>8} {'gqa GB/s':>8} | {'vs exp':>7} {'vs win':>7} | max_err"
            )
            print(header)
            print("-" * len(header))
            scale = hd**-0.5

            for conc in args.concs:
                for ctx in args.ctxs:
                    total_len = ctx + window
                    bps = (total_len + args.block_size - 1) // args.block_size
                    kc = mx.random.normal(
                        (bps * conc, args.block_size, nkv, hd)
                    ).astype(dtype)
                    vc = mx.random.normal(
                        (bps * conc, args.block_size, nkv, hd)
                    ).astype(dtype)
                    query = mx.random.normal((conc * window, nq, hd)).astype(dtype)
                    mx.eval(kc, vc, query)

                    tables = [list(range(r * bps, (r + 1) * bps)) for r in range(conc)]
                    e_tables = mx.array(
                        [t for t in tables for _ in range(window)], dtype=mx.int32
                    )
                    e_seq = mx.array(
                        [ctx + j + 1 for _ in range(conc) for j in range(window)],
                        dtype=mx.int32,
                    )
                    e_cu = mx.array(list(range(conc * window + 1)), dtype=mx.int32)

                    w_tables = mx.array(tables, dtype=mx.int32)
                    w_seq = mx.array([total_len] * conc, dtype=mx.int32)
                    w_cu = mx.array(
                        [i * window for i in range(conc + 1)], dtype=mx.int32
                    )
                    mx.eval(e_tables, e_seq, e_cu, w_tables, w_seq, w_cu)

                    exp_ms, exp_out = _time_arm(
                        ops,
                        query,
                        kc,
                        vc,
                        nkv,
                        scale,
                        e_tables,
                        e_seq,
                        e_cu,
                        args.block_size,
                        total_len,
                        1,
                        False,
                        args.layers,
                        args.warmup,
                        args.reps,
                    )
                    win_ms, win_out = _time_arm(
                        ops,
                        query,
                        kc,
                        vc,
                        nkv,
                        scale,
                        w_tables,
                        w_seq,
                        w_cu,
                        args.block_size,
                        total_len,
                        window,
                        False,
                        args.layers,
                        args.warmup,
                        args.reps,
                    )
                    gqa_ms, gqa_out = _time_arm(
                        ops,
                        query,
                        kc,
                        vc,
                        nkv,
                        scale,
                        w_tables,
                        w_seq,
                        w_cu,
                        args.block_size,
                        total_len,
                        window,
                        True,
                        args.layers,
                        args.warmup,
                        args.reps,
                    )

                    max_err = float(
                        np.max(
                            np.abs(
                                np.array(gqa_out.astype(mx.float32))
                                - np.array(win_out.astype(mx.float32))
                            )
                        )
                    )
                    # Minimum algorithmic I/O: read K+V once per sequence + Q + O.
                    kv_io_bytes = (
                        conc * total_len * 2 * nkv * hd * elem_b
                        + 2 * conc * window * nq * hd * elem_b
                    )
                    win_gbs = kv_io_bytes / (win_ms * 1e-3) / 1e9
                    gqa_gbs = kv_io_bytes / (gqa_ms * 1e-3) / 1e9

                    print(
                        f"{conc:>2} {ctx:>6} | {exp_ms:>7.3f} {win_ms:>7.3f} {gqa_ms:>7.3f} | "
                        f"{win_gbs:>8.1f} {gqa_gbs:>8.1f} | "
                        f"{exp_ms / gqa_ms:>6.2f}x {win_ms / gqa_ms:>6.2f}x | {max_err:.2e}",
                        flush=True,
                    )
                    records.append(
                        {
                            "config": label,
                            "dtype": dname,
                            "conc": conc,
                            "ctx": ctx,
                            "window": window,
                            "exp_ms": exp_ms,
                            "win_ms": win_ms,
                            "gqa_ms": gqa_ms,
                            "win_gb_s": win_gbs,
                            "gqa_gb_s": gqa_gbs,
                            "speedup_vs_exp": exp_ms / gqa_ms,
                            "speedup_vs_win": win_ms / gqa_ms,
                            "max_abs_err": max_err,
                        }
                    )
                    del kc, vc, query, exp_out, win_out, gqa_out
                    gc.collect()
                    mx.clear_cache()

    if args.json:
        with open(args.json, "w") as handle:
            json.dump({"metadata": meta, "results": records}, handle, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
