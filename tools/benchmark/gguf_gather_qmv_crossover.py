# SPDX-License-Identifier: Apache-2.0
"""Crossover benchmark for GGUF raw-block MoE experts (Q2_K/Q3_K/Q6_K).

``GGUFRawBlockTensor.gather_matmul`` runs one of two arms:

  dequant  dequantize every expert once, then ``mx.gather_mm`` (the baseline).
  kernel   gather QMV: each (token, expert) pair decodes only its expert's rows.

It picks the kernel while ``pairs <= limit * num_experts``, with separate
limits for the SIMD and legacy kernels. This script forces each arm in turn
through a full mlx-lm ``SwitchGLU`` (gate, up, down; sorting at >= 64 pairs as
in serving) and derives the per-call time every candidate constant would give.

Each timed evaluation queues ``--layers`` SwitchGLU calls, as a forward pass
does, and reports milliseconds per layer. Arms alternate in random order each
rep. Both arms' outputs are checked against each other before timing.

Run from the repository root:

    python -m tools.benchmark.gguf_gather_qmv_crossover
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import statistics
import subprocess
import time

import mlx.core as mx
import mlx.nn as nn
from gguf import GGMLQuantizationType
from mlx_lm.models.switch_layers import SwiGLU, SwitchGLU

import vllm_metal.gguf.raw_block as rb
from tests.gguf_kquant_fixtures import build_kquant_blocks
from vllm_metal.gguf.wrappers import GGUFSwitchLinear

_ARMS = {"dequant": 0, "kernel": 10**9}
_DTYPES = {"bfloat16": mx.bfloat16, "float16": mx.float16}


def _expert_tensor(qtype, num_experts: int, rows: int, cols: int):
    raw = build_kquant_blocks(num_experts * rows, cols, qtype)
    t = rb.GGUFRawBlockTensor.from_raw_blocks(raw, (num_experts * rows, cols), qtype)
    mx.eval(t.qweight)
    return GGUFSwitchLinear(t, num_experts)


def _switch_glu(qtype, num_experts: int, hidden: int, inter: int) -> SwitchGLU:
    # Skip SwitchGLU.__init__: it allocates the full-precision experts.
    glu = SwitchGLU.__new__(SwitchGLU)
    nn.Module.__init__(glu)
    glu.gate_proj = _expert_tensor(qtype, num_experts, inter, hidden)
    glu.up_proj = _expert_tensor(qtype, num_experts, inter, hidden)
    glu.down_proj = _expert_tensor(qtype, num_experts, hidden, inter)
    glu.activation = SwiGLU()
    return glu


def _routing(tokens: int, num_experts: int, top_k: int) -> mx.array:
    scores = mx.random.normal((1, tokens, num_experts))
    return mx.argpartition(-scores, top_k - 1, axis=-1)[..., :top_k].astype(mx.uint32)


def _run(glu, x, idx, arm: str, layers: int) -> list[mx.array]:
    rb._GATHER_QMV_MAX_ASSIGNMENTS_PER_EXPERT = _ARMS[arm]
    rb._GATHER_QMV_LEGACY_MAX_ASSIGNMENTS_PER_EXPERT = _ARMS[arm]
    outs = [glu(x, idx) for _ in range(layers)]
    mx.eval(outs)
    return outs


def _measure(glu, tokens: int, args) -> dict[str, float]:
    x = mx.random.normal((1, tokens, args.hidden)).astype(_DTYPES[args.dtype])
    idx = _routing(tokens, args.num_experts, args.top_k)
    mx.eval(x, idx)

    (ref,) = _run(glu, x, idx, "dequant", 1)
    (out,) = _run(glu, x, idx, "kernel", 1)
    ref32, out32 = ref.astype(mx.float32), out.astype(mx.float32)
    err = float(mx.max(mx.abs(out32 - ref32))) / float(mx.max(mx.abs(ref32)))
    if err > args.tolerance:
        raise AssertionError(f"{tokens} tokens: arms disagree, rel err {err:.2e}")

    for arm in _ARMS:
        for _ in range(args.warmup):
            _run(glu, x, idx, arm, args.layers)
    samples: dict[str, list[float]] = {arm: [] for arm in _ARMS}
    for _ in range(args.reps):
        for arm in random.sample(list(_ARMS), len(_ARMS)):
            t0 = time.perf_counter()
            _run(glu, x, idx, arm, args.layers)
            samples[arm].append((time.perf_counter() - t0) * 1e3 / args.layers)
    return {arm: statistics.median(v) for arm, v in samples.items()} | {"err": err}


def _table(qtype: str, rows: list[dict], args) -> str:
    cs = args.constants
    head = (
        f"\n### {qtype}  (ms per SwitchGLU layer, median of {args.reps})\n\n"
        "| tokens | pairs | pairs/E | dequant | kernel | kernel speedup | "
        + " | ".join(f"c={c}" for c in cs)
        + " |\n|"
        + "---|" * (6 + len(cs))
    )
    lines = [head]
    for r in rows:
        pairs = r["tokens"] * args.top_k
        cells = []
        for c in cs:
            arm = "kernel" if pairs <= c * args.num_experts else "dequant"
            cells.append(f"{r[arm]:.2f} {arm[0]}")
        lines.append(
            f"| {r['tokens']} | {pairs} | {pairs / args.num_experts:.2f} | "
            f"{r['dequant']:.2f} | {r['kernel']:.2f} | "
            f"{r['dequant'] / r['kernel']:.2f}x | " + " | ".join(cells) + " |"
        )
    losing = [r for r in rows if r["kernel"] > r["dequant"]]
    first = losing[0]["tokens"] * args.top_k / args.num_experts if losing else None
    lines.append(
        f"\nKernel first loses at {first:.2f} pairs/expert."
        if first is not None
        else "\nKernel wins at every measured size."
    )
    return "\n".join(lines)


def _metadata(args) -> dict:
    def sh(cmd):
        try:
            return subprocess.check_output(cmd, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    return {
        "chip": sh(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "memory_bytes": sh(["sysctl", "-n", "hw.memsize"]),
        "macos": platform.mac_ver()[0],
        "mlx": mx.__version__,
        "commit": sh(["git", "rev-parse", "--short", "HEAD"]),
        "args": vars(args),
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--qtypes", nargs="+", default=["Q2_K", "Q3_K", "Q6_K"])
    p.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 24, 32, 40, 48, 64, 96, 128, 256],
    )
    p.add_argument("--constants", type=int, nargs="+", default=[1, 2, 4, 8])
    # Qwen3-30B-A3B expert shapes.
    p.add_argument("--num-experts", type=int, default=128)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--hidden", type=int, default=2048)
    p.add_argument("--inter", type=int, default=768)
    p.add_argument("--dtype", choices=list(_DTYPES), default="bfloat16")
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--reps", type=int, default=15)
    p.add_argument("--tolerance", type=float, default=2e-2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", help="also write raw results here")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    random.seed(args.seed)
    mx.random.seed(args.seed)
    meta = _metadata(args)
    print(json.dumps({k: v for k, v in meta.items() if k != "args"}), flush=True)
    results = {}
    for name in args.qtypes:
        glu = _switch_glu(
            GGMLQuantizationType[name], args.num_experts, args.hidden, args.inter
        )
        rows = []
        for tokens in args.tokens:
            r = {"tokens": tokens} | _measure(glu, tokens, args)
            print(
                f"{name} tokens={tokens}: dequant {r['dequant']:.2f} ms, "
                f"kernel {r['kernel']:.2f} ms, rel err {r['err']:.1e}",
                flush=True,
            )
            rows.append(r)
        print(_table(name, rows, args), flush=True)
        results[name] = rows
        del glu
        mx.clear_cache()
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"metadata": meta, "results": results}, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
