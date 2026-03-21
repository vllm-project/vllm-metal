# SPDX-License-Identifier: Apache-2.0
"""Benchmark GatedDeltaNet (GDN) linear attention kernels.

Compares three backends:
  - metal:  mlx_lm's Metal kernel (pre-computed g, beta + kernel dispatch)
  - fused:  vllm-metal fused kernel (gating + recurrence in one dispatch)
  - ops:    mlx_lm ops reference (mx.compile'd Python)

Usage:
    python tools/bench_gdn_kernel.py
    python tools/bench_gdn_kernel.py --batch 1 4 8 --seq-lens 1 64
    python tools/bench_gdn_kernel.py --check  # correctness only
"""

from __future__ import annotations

import argparse
import sys
import time

import mlx.core as mx

# Qwen3.5 GDN dimensions (shared across all model sizes)
DK = 128  # key head dim
DV = 128  # value head dim
HK = 16   # key heads


def _make_inputs(  # noqa: N803
    batch, seq_len, n_k_heads, n_v_heads, key_dim, val_dim, dtype,
):
    """Create synthetic inputs for benchmarking.

    Uses small values to avoid fp16 overflow in multi-step recurrence.
    """
    mx.random.seed(42)
    scale = 0.1
    q = (mx.random.normal((batch, seq_len, n_k_heads, key_dim)) * scale).astype(dtype)
    k = (mx.random.normal((batch, seq_len, n_k_heads, key_dim)) * scale).astype(dtype)
    v = (mx.random.normal((batch, seq_len, n_v_heads, val_dim)) * scale).astype(dtype)
    a = (mx.random.normal((batch, seq_len, n_v_heads)) * scale).astype(dtype)
    b = (mx.random.normal((batch, seq_len, n_v_heads)) * scale).astype(dtype)
    a_log = (mx.random.normal((n_v_heads,)) * scale).astype(dtype)
    dt_bias = (mx.random.normal((n_v_heads,)) * scale).astype(dtype)
    state = mx.zeros((batch, n_v_heads, val_dim, key_dim), dtype=dtype)
    mx.eval(q, k, v, a, b, a_log, dt_bias, state)
    return q, k, v, a, b, a_log, dt_bias, state


def bench_one(  # noqa: N803
    *, backend, batch, seq_len, n_k_heads, n_v_heads, key_dim, val_dim,
    warmup, iters, dtype,
) -> float:
    """Run one benchmark config and return median ms per call."""
    from mlx_lm.models.gated_delta import (
        compute_g,
        gated_delta_kernel,
        gated_delta_ops,
    )

    q, k, v, a, b, a_log, dt_bias, state = _make_inputs(
        batch, seq_len, n_k_heads, n_v_heads, key_dim, val_dim, dtype,
    )

    if backend == "fused":
        from vllm_metal.metal.linear_attention import fused_gdn_decode

        def _fused_fn():
            return fused_gdn_decode(q, k, v, a, b, a_log, dt_bias, state)

        fn = _fused_fn
    elif backend == "metal":
        # Include compute_g + sigmoid in timing (fair comparison with fused)
        def _metal_fn():
            g = compute_g(a_log, a, dt_bias)
            beta = mx.sigmoid(b)
            return gated_delta_kernel(q, k, v, g, beta, state)

        fn = _metal_fn
    elif backend == "metal_precomp":
        # Pre-computed gating (kernel-only timing)
        g = compute_g(a_log, a, dt_bias)
        beta = mx.sigmoid(b)
        mx.eval(g, beta)

        def _precomp_fn():
            return gated_delta_kernel(q, k, v, g, beta, state)

        fn = _precomp_fn
    elif backend == "ops":
        def _ops_fn():
            g = compute_g(a_log, a, dt_bias)
            beta = mx.sigmoid(b)
            return gated_delta_ops(q, k, v, g, beta, state)

        fn = _ops_fn
    else:
        raise ValueError(f"Unknown backend: {backend}")

    for _ in range(warmup):
        y, s = fn()
        mx.eval(y, s)

    times = []
    for _ in range(iters):
        start = time.perf_counter()
        y, s = fn()
        mx.eval(y, s)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)

    times.sort()
    return times[len(times) // 2]


def check_correctness(n_v_heads=32, dtype=mx.float16):
    """Verify fused kernel matches mlx_lm's Metal kernel output."""
    from mlx_lm.models.gated_delta import compute_g, gated_delta_kernel

    from vllm_metal.metal.linear_attention import fused_gdn_decode

    print(f"Correctness check (Hv={n_v_heads}, dtype={dtype})...")

    for batch, seq_len in [(1, 1), (1, 16), (4, 1), (2, 8)]:
        q, k, v, a, b, a_log, dt_bias, state = _make_inputs(
            batch, seq_len, HK, n_v_heads, DK, DV, dtype,
        )

        # Reference: mlx_lm Metal kernel (pre-computed gating)
        g = compute_g(a_log, a, dt_bias)
        beta = mx.sigmoid(b)
        mx.eval(g, beta)
        state_copy = mx.array(state)
        mx.eval(state_copy)
        y_ref, s_ref = gated_delta_kernel(q, k, v, g, beta, state_copy)
        mx.eval(y_ref, s_ref)

        # Fused kernel (use original state, not the copy)
        y_fused, s_fused = fused_gdn_decode(q, k, v, a, b, a_log, dt_bias, state)
        mx.eval(y_fused, s_fused)

        # Compare
        y_abs = mx.abs(y_ref.astype(mx.float32) - y_fused.astype(mx.float32))
        s_abs = mx.abs(s_ref.astype(mx.float32) - s_fused.astype(mx.float32))
        y_diff = y_abs.max().item()
        s_diff = s_abs.max().item()

        # fp16 gating order differences and near-zero outputs cause max_rel noise.
        # Use absolute tolerance: for scaled inputs (0.1) outputs are O(0.01-0.1).
        status = "PASS" if y_diff < 0.05 and s_diff < 0.05 else "FAIL"
        print(
            f"  B={batch} T={seq_len}: "
            f"y_maxabs={y_diff:.6f} s_maxabs={s_diff:.6f} [{status}]"
        )

        if status == "FAIL":
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Benchmark GDN kernel")
    parser.add_argument(
        "--batch", type=int, nargs="+", default=[1, 4, 8],
    )
    parser.add_argument(
        "--seq-lens", type=int, nargs="+", default=[1, 16, 64],
    )
    parser.add_argument(
        "--hv", type=int, nargs="+", default=[32, 48],
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--dtype", choices=["float16", "bfloat16"], default="float16",
    )
    parser.add_argument(
        "--check", action="store_true", help="Run correctness check only",
    )
    args = parser.parse_args()

    dtype = mx.float16 if args.dtype == "float16" else mx.bfloat16

    # Always run correctness check first
    ok = check_correctness(n_v_heads=args.hv[0], dtype=dtype)
    if not ok:
        print("CORRECTNESS CHECK FAILED — aborting benchmark")
        sys.exit(1)
    print()

    if args.check:
        return

    print(f"GDN Kernel Benchmark (Dk={DK}, Dv={DV}, Hk={HK})")
    print(f"dtype={args.dtype}  warmup={args.warmup}  iters={args.iters}")
    print()

    header = (
        f"{'Hv':>4} | {'B':>3} | {'T':>5} | "
        f"{'fused(ms)':>10} | {'metal(ms)':>10} | {'precomp(ms)':>11} | "
        f"{'ops(ms)':>10} | {'f/m':>6}"
    )
    print(header)
    print("-" * len(header))

    for n_v_heads in args.hv:
        for batch in args.batch:
            for seq_len in args.seq_lens:
                common = {
                    "batch": batch,
                    "seq_len": seq_len,
                    "n_k_heads": HK,
                    "n_v_heads": n_v_heads,
                    "key_dim": DK,
                    "val_dim": DV,
                    "warmup": args.warmup,
                    "iters": args.iters,
                    "dtype": dtype,
                }

                t_fused = bench_one(backend="fused", **common)
                t_metal = bench_one(backend="metal", **common)
                t_precomp = bench_one(backend="metal_precomp", **common)
                t_ops = bench_one(backend="ops", **common)

                print(
                    f"{n_v_heads:4d} | {batch:3d} | {seq_len:5d} | "
                    f"{t_fused:10.3f} | {t_metal:10.3f} | {t_precomp:10.3f} | "
                    f"{t_ops:10.3f} | {t_fused/t_metal:6.2f}x"
                )

    print()
    print("Backends:")
    print("  fused   = vllm-metal fused kernel (gating + recurrence, 1 dispatch)")
    print("  metal   = mlx_lm full path (compute_g + sigmoid + kernel)")
    print("  precomp = mlx_lm kernel only (gating pre-computed, excluded from timing)")
    print("  ops     = mlx_lm ops reference (mx.compile'd Python loops)")
    print("  f/m     = fused / metal ratio (< 1.0 means fused wins)")


if __name__ == "__main__":
    main()
