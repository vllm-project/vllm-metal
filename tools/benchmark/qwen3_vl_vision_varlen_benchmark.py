# SPDX-License-Identifier: Apache-2.0
"""A/B benchmark: Qwen3-VL vision Attention baseline vs encoder_varlen_attention.

Compares mlx_vlm's per-segment ``ensure_fused_sdpa`` loop against the Metal
``encoder_varlen_attention`` primitive wired through
``qwen3_vl_vision_attention_forward(use_fused_varlen=True)``.

Run:
    .venv-vllm-metal/bin/python tools/benchmark/qwen3_vl_vision_varlen_benchmark.py
    .venv-vllm-metal/bin/python tools/benchmark/qwen3_vl_vision_varlen_benchmark.py --iters 30
    .venv-vllm-metal/bin/python tools/benchmark/qwen3_vl_vision_varlen_benchmark.py --attention-only
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from dataclasses import dataclass

import mlx.core as mx
import numpy as np

try:
    from mlx_vlm.models.qwen3_vl.vision import Attention
except ImportError:  # pragma: no cover
    print("mlx_vlm unavailable; cannot run.")
    sys.exit(0)

from vllm_metal.multimodal.fused_vit_attention import (
    _baseline_segment_attention,
    _fused_rope_varlen_core,
    _qkv_to_shd,
    _shd_to_bhsd,
    qwen3_vl_vision_attention_forward,
)


@dataclass(frozen=True, slots=True)
class BenchConfig:
    label: str
    segment_lens: list[int]
    hidden_size: int = 1024
    num_heads: int = 16
    dtype: mx.Dtype = mx.bfloat16


DEFAULT_CONFIGS: tuple[BenchConfig, ...] = (
    BenchConfig("4x256", [256, 256, 256, 256]),
    BenchConfig("8x1024", [1024] * 8),
    BenchConfig("4x4096", [4096] * 4),
    BenchConfig("varied (Qwen3-VL-ish)", [768, 256, 512, 1024, 384, 896]),
    BenchConfig("16x256", [256] * 16),
    BenchConfig("32x128", [128] * 32),
    BenchConfig("64x64", [64] * 64),
    BenchConfig("4x1024 hd128", [1024] * 4, hidden_size=2048, num_heads=16),
    BenchConfig("8x512 hd64", [512] * 8, hidden_size=512, num_heads=8),
    BenchConfig("8x768 hd96", [768] * 8, hidden_size=1536, num_heads=16),
)


def _build_cu_seqlens(segment_lens: list[int]) -> mx.array:
    total = sum(segment_lens)
    bounds = [0, *np.cumsum(segment_lens).tolist()]
    return mx.array(bounds, dtype=mx.int32), total


def _p50_p95(fn, *, warmup: int, iters: int) -> tuple[float, float]:
    for _ in range(warmup):
        mx.eval(fn())
    times_ms: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        mx.eval(fn())
        times_ms.append((time.perf_counter() - t0) * 1e3)
    times_ms.sort()
    p50 = times_ms[len(times_ms) // 2]
    p95 = times_ms[int(len(times_ms) * 0.95) - 1]
    return p50, p95


def _run_config(
    cfg: BenchConfig, *, warmup: int, iters: int, attention_only: bool
) -> None:
    head_dim = cfg.hidden_size // cfg.num_heads
    cu, total = _build_cu_seqlens(cfg.segment_lens)
    max_seqlen = max(cfg.segment_lens)
    rope_dim = head_dim // 2

    mx.random.seed(0)
    attn = Attention(dim=cfg.hidden_size, num_heads=cfg.num_heads)
    x = mx.random.normal((total, cfg.hidden_size)).astype(cfg.dtype)
    rope = mx.random.normal((total, rope_dim)).astype(cfg.dtype)
    mx.eval(attn.parameters(), x, cu, rope)

    if attention_only:
        from mlx_vlm.models.qwen3_vl.vision import apply_rotary_pos_emb_vision

        qkv = attn.qkv(x).reshape(total, 3, attn.num_heads, -1)
        q, k, v = _qkv_to_shd(qkv)
        mx.eval(q, k, v)

        def baseline() -> mx.array:
            q_r = apply_rotary_pos_emb_vision(mx.expand_dims(q, 0), rope)[0]
            k_r = apply_rotary_pos_emb_vision(mx.expand_dims(k, 0), rope)[0]
            q_b, k_b, v_b = _shd_to_bhsd(q_r, k_r, v)
            return _baseline_segment_attention(q_b, k_b, v_b, cu, attn.scale)

        def fused() -> mx.array:
            return _fused_rope_varlen_core(
                q,
                k,
                v,
                rope,
                cu,
                attn.scale,
                max_seqlen=max_seqlen,
            )
    else:

        def baseline() -> mx.array:
            return qwen3_vl_vision_attention_forward(
                attn,
                x,
                cu,
                rope,
                use_fused_varlen=False,
                max_seqlen=max_seqlen,
            )

        def fused() -> mx.array:
            return qwen3_vl_vision_attention_forward(
                attn,
                x,
                cu,
                rope,
                use_fused_varlen=True,
                use_fused_rope_varlen=True,
                max_seqlen=max_seqlen,
            )

    out_a = baseline()
    out_b = fused()
    mx.eval(out_a, out_b)
    if attention_only:
        out_a = out_a.transpose(0, 2, 1, 3).reshape(total, attn.num_heads, head_dim)
    diff = mx.abs(out_a - out_b)
    max_err = float(mx.max(diff).item())
    mean_err = float(mx.mean(diff).item())

    base_p50, base_p95 = _p50_p95(baseline, warmup=warmup, iters=iters)
    fused_p50, fused_p95 = _p50_p95(fused, warmup=warmup, iters=iters)
    speedup = base_p50 / fused_p50 if fused_p50 > 0 else float("inf")

    print(
        f"{cfg.label:24} tokens={total:6d} "
        f"max_err={max_err:.3e} mean_err={mean_err:.3e} "
        f"baseline_p50={base_p50:8.3f}ms fused_p50={fused_p50:8.3f}ms "
        f"baseline_p95={base_p95:8.3f}ms fused_p95={fused_p95:8.3f}ms "
        f"speedup={speedup:.2f}x"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument(
        "--attention-only",
        action="store_true",
        help="Time RoPE + attention core only (exclude qkv/proj).",
    )
    args = parser.parse_args()

    mode = "attention core only" if args.attention_only else "full forward"
    print(f"Qwen3-VL vision Attention A/B ({mode})")
    print(
        f"host={platform.machine()} mlx={mx.__version__} "
        f"warmup={args.warmup} iters={args.iters}"
    )
    print(
        f"{'config':24} {'tokens':>6} {'max_err':>10} {'mean_err':>10} "
        f"{'base_p50':>10} {'fused_p50':>10} {'base_p95':>10} "
        f"{'fused_p95':>10} {'speedup':>8}"
    )
    print("-" * 120)

    failures = 0
    for cfg in DEFAULT_CONFIGS:
        try:
            _run_config(
                cfg,
                warmup=args.warmup,
                iters=args.iters,
                attention_only=args.attention_only,
            )
        except ValueError as exc:
            failures += 1
            print(f"{cfg.label:24} SKIPPED: {exc}")

    if failures:
        print(f"\n{failures} config(s) skipped due to unsupported head_dim.")


if __name__ == "__main__":
    main()
