"""Manual TurboQuant profiler — since Instruments refuses to show shader names,
we profile the kernel with surgical `mx.synchronize()` fences and reason about
bottlenecks from first principles.

Measures four axes that matter for the 16 → 18 tok/s gap:

  A. tq_encode GPU time sweep: N = 1, 4, 16, 64, 256, 1024, 4096, 16384
     — isolates the kernel at realistic shapes, separating dispatch overhead
       from compute cost.  Also measures effective bandwidth to diagnose
       memory-bound vs dispatch-bound.

  B. Decode-step simulation: 28 × (tq_encode + paged_attention_v2_online)
     back-to-back, comparing TurboQuant vs FP16.  Matches what the model
     actually does per decode step on Qwen3-0.6B.

  C. Python dispatch overhead: how much of each call is nanobind/MLX graph
     construction vs actual GPU work.  If this is >30% at N=1, optimizing
     the kernel is pointless — we need to amortize dispatch.

  D. Grid saturation analysis: theoretical threadgroup count vs M1 GPU
     capacity (8 cores × 2 concurrent TGs ≈ 16 active TGs for saturation).

Run with:
    uv run python tools/tq_bench.py
"""

from __future__ import annotations

import gc
import math
import sys
import time
from dataclasses import dataclass

import mlx.core as mx

from vllm_metal.metal import get_ops
from vllm_metal.metal_kernel_backend.cache import MetalPagedKVCache
from vllm_metal.metal_kernel_backend.turboquant import get_v_centroids

# -----------------------------------------------------------------------------
# Config — Qwen3-0.6B shape (28 layers, 4 KV heads, head_dim=128).
# -----------------------------------------------------------------------------
NUM_LAYERS = 28
NUM_KV_HEADS = 4
NUM_Q_HEADS = 16
HEAD_DIM = 128
BLOCK_SIZE = 16
K_QUANT = "q8_0"  # signed int8, full quality
V_BITS = 3  # Lloyd-Max 3-bit
K_BITS = 8

# GPU capacity heuristics (M1 / M1 Pro / M1 Max all similar):
#   - M1:   8 cores × ~2 concurrent TGs per core   ≈ 16 TGs
#   - M1P: 14 cores × ~2 concurrent TGs per core   ≈ 28 TGs
#   - M1M: 32 cores × ~2 concurrent TGs per core   ≈ 64 TGs
M1_SATURATION_TGS = 16


def _sync() -> None:
    """Block until GPU is idle.  Required for honest wall-clock timing."""
    mx.synchronize()


def _time_gpu(fn, warmup: int = 3, iters: int = 50) -> float:
    """Return GPU-inclusive wall time per call (µs).  Uses `mx.synchronize`
    fences on both sides so the measurement includes *all* GPU work."""
    for _ in range(warmup):
        fn()
    _sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync()
    return (time.perf_counter() - t0) / iters * 1e6


def _time_dispatch_only(fn, warmup: int = 3, iters: int = 200) -> float:
    """Return time spent in Python → nanobind → MLX graph construction per
    call (µs), *without* waiting for GPU.  Compare to `_time_gpu` to infer
    dispatch-bound vs compute-bound."""
    for _ in range(warmup):
        fn()
    _sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    # Note: no sync here — we measure only Python + graph construction.
    return (time.perf_counter() - t0) / iters * 1e6


# -----------------------------------------------------------------------------
# Data helpers
# -----------------------------------------------------------------------------


def _make_tq_cache(num_blocks: int) -> MetalPagedKVCache:
    return MetalPagedKVCache(
        num_layers=1,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        num_blocks=num_blocks,
        block_size=BLOCK_SIZE,
        dtype=mx.float16,
        turboquant=True,
        k_quant=K_QUANT,
    )


def _make_fp_cache(num_blocks: int) -> MetalPagedKVCache:
    return MetalPagedKVCache(
        num_layers=1,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        num_blocks=num_blocks,
        block_size=BLOCK_SIZE,
        dtype=mx.float16,
        turboquant=False,
    )


# -----------------------------------------------------------------------------
# Section A — tq_encode GPU time sweep
# -----------------------------------------------------------------------------


@dataclass
class EncodeSample:
    n: int
    tq_us: float
    tq_dispatch_us: float
    fp_us: float
    fp_dispatch_us: float

    @property
    def tq_per_tok(self) -> float:
        return self.tq_us / self.n

    @property
    def fp_per_tok(self) -> float:
        return self.fp_us / self.n

    @property
    def tq_gpu_us(self) -> float:
        """Estimated GPU-only time (subtracting Python dispatch)."""
        return max(0.0, self.tq_us - self.tq_dispatch_us)

    @property
    def tq_dispatch_frac(self) -> float:
        return self.tq_dispatch_us / self.tq_us if self.tq_us > 0 else 0.0

    @property
    def effective_bw_gbps(self) -> float:
        """Rough bandwidth: total bytes R+W per call / GPU time."""
        # Reads: K/V fp16 (2 bytes × 2 tensors × N × H × D)
        # Writes: K packed (K_BITS/8 B × N × H × D)
        #         V packed (V_BITS/8 B × N × H × D)
        #         K scale (2B), V scale (2B), K zp (2B) × N × H × scale_groups
        scale_groups = HEAD_DIM // 32
        read_b = 2 * 2 * self.n * NUM_KV_HEADS * HEAD_DIM
        write_k = (K_BITS / 8) * self.n * NUM_KV_HEADS * HEAD_DIM
        write_v = (V_BITS / 8) * self.n * NUM_KV_HEADS * HEAD_DIM
        write_scales = 2 * 3 * self.n * NUM_KV_HEADS * scale_groups
        total_b = read_b + write_k + write_v + write_scales
        if self.tq_gpu_us <= 0:
            return float("nan")
        return total_b / (self.tq_gpu_us * 1e-6) / 1e9


def bench_encode_sweep(ns: list[int]) -> list[EncodeSample]:
    ops = get_ops()
    v_centroids = get_v_centroids(V_BITS)
    mx.eval(v_centroids)

    samples: list[EncodeSample] = []

    for n in ns:
        num_blocks = (n + BLOCK_SIZE - 1) // BLOCK_SIZE + 2
        cache_tq = _make_tq_cache(num_blocks)
        cache_fp = _make_fp_cache(num_blocks)

        k = mx.random.normal(shape=(n, NUM_KV_HEADS, HEAD_DIM)).astype(mx.float16)
        v = mx.random.normal(shape=(n, NUM_KV_HEADS, HEAD_DIM)).astype(mx.float16)
        slot = mx.arange(n, dtype=mx.int64)
        mx.eval(k, v, slot)

        # --- TurboQuant fused encode ---
        # Loop-scoped vars (cache_tq, k, v, slot) are bound as defaults so the
        # closures don't late-bind after the end-of-iteration `del` frees them.
        def run_tq(cache_tq=cache_tq, k=k, v=v, slot=slot):
            (nk, nv, nks, nvs, nkz) = ops.tq_encode(
                k,
                v,
                cache_tq.key_caches[0],
                cache_tq.value_caches[0],
                cache_tq.key_scale_caches[0],
                cache_tq.value_scale_caches[0],
                cache_tq.key_zero_caches[0],
                slot,
                v_centroids,
                V_BITS,
                K_BITS,
                True,
            )
            cache_tq.key_caches[0] = nk
            cache_tq.value_caches[0] = nv
            cache_tq.key_scale_caches[0] = nks
            cache_tq.value_scale_caches[0] = nvs
            cache_tq.key_zero_caches[0] = nkz
            return nk, nv, nks, nvs, nkz

        def run_tq_eval(run_tq=run_tq):
            arrs = run_tq()
            mx.eval(*arrs)

        # --- FP16 scatter (MLX-native) ---
        def run_fp(cache_fp=cache_fp, k=k, v=v, slot=slot):
            flat_k = cache_fp.key_caches[0].reshape(-1, NUM_KV_HEADS, HEAD_DIM)
            flat_k[slot] = k
            cache_fp.key_caches[0] = flat_k.reshape(cache_fp.key_caches[0].shape)
            flat_v = cache_fp.value_caches[0].reshape(-1, NUM_KV_HEADS, HEAD_DIM)
            flat_v[slot] = v
            cache_fp.value_caches[0] = flat_v.reshape(cache_fp.value_caches[0].shape)

        def run_fp_eval(run_fp=run_fp, cache_fp=cache_fp):
            run_fp()
            mx.eval(cache_fp.key_caches[0], cache_fp.value_caches[0])

        tq_us = _time_gpu(run_tq_eval)
        fp_us = _time_gpu(run_fp_eval)
        tq_disp_us = _time_dispatch_only(run_tq)
        fp_disp_us = _time_dispatch_only(run_fp)

        samples.append(
            EncodeSample(
                n=n,
                tq_us=tq_us,
                tq_dispatch_us=tq_disp_us,
                fp_us=fp_us,
                fp_dispatch_us=fp_disp_us,
            )
        )
        del k, v, slot, cache_tq, cache_fp
        gc.collect()

    return samples


def print_encode_table(samples: list[EncodeSample]) -> None:
    print()
    print("  Section A — tq_encode vs FP16-scatter sweep")
    print("  " + "─" * 78)
    print(
        f"  {'N':>6}  {'TGs':>4}  "
        f"{'TQ total':>10}  {'TQ/tok':>8}  {'TQ GPU':>8}  {'disp%':>6}  "
        f"{'BW GB/s':>8}  {'FP total':>10}  {'FP/tok':>8}"
    )
    print("  " + "─" * 78)
    for s in samples:
        tgs = s.n * NUM_KV_HEADS
        print(
            f"  {s.n:>6}  {tgs:>4}  "
            f"{s.tq_us:>8.1f}µs  {s.tq_per_tok:>6.2f}µs  "
            f"{s.tq_gpu_us:>6.1f}µs  {s.tq_dispatch_frac * 100:>4.1f}%  "
            f"{s.effective_bw_gbps:>6.1f}   "
            f"{s.fp_us:>8.1f}µs  {s.fp_per_tok:>6.2f}µs"
        )
    print("  " + "─" * 78)
    print(f"  (Dispatch occupancy: TGs / {M1_SATURATION_TGS} ≈ M1 saturation point)")


# -----------------------------------------------------------------------------
# Section B — Decode-step simulation
#
# Mimics the per-step cost of a real decode:
#   - 28 layers × tq_encode (1 token)      (TurboQuant path)
#   - 28 layers × FP16 scatter (1 token)   (baseline path)
#
# We *don't* include paged_attention here because the decode slowdown we see
# (16 vs 18 tok/s) is ~5-7 ms which is roughly in the same order as 28 × encode
# cost.  If this section shows <1 ms for 28 encodes, the bottleneck is elsewhere
# (e.g. paged_attention reading quantized cache is slower than reading fp16).
# -----------------------------------------------------------------------------


def bench_decode_step() -> None:
    ops = get_ops()
    v_centroids = get_v_centroids(V_BITS)
    mx.eval(v_centroids)

    # One cache per layer for realistic graph structure.
    caches_tq = [_make_tq_cache(num_blocks=8) for _ in range(NUM_LAYERS)]
    caches_fp = [_make_fp_cache(num_blocks=8) for _ in range(NUM_LAYERS)]

    k = mx.random.normal(shape=(1, NUM_KV_HEADS, HEAD_DIM)).astype(mx.float16)
    v = mx.random.normal(shape=(1, NUM_KV_HEADS, HEAD_DIM)).astype(mx.float16)
    slot = mx.array([0], dtype=mx.int64)
    mx.eval(k, v, slot)

    def run_tq_step():
        outs = []
        for layer in range(NUM_LAYERS):
            c = caches_tq[layer]
            (nk, nv, nks, nvs, nkz) = ops.tq_encode(
                k,
                v,
                c.key_caches[0],
                c.value_caches[0],
                c.key_scale_caches[0],
                c.value_scale_caches[0],
                c.key_zero_caches[0],
                slot,
                v_centroids,
                V_BITS,
                K_BITS,
                True,
            )
            c.key_caches[0] = nk
            c.value_caches[0] = nv
            c.key_scale_caches[0] = nks
            c.value_scale_caches[0] = nvs
            c.key_zero_caches[0] = nkz
            outs.extend([nk, nv, nks, nvs, nkz])
        mx.eval(*outs)

    def run_fp_step():
        outs = []
        for layer in range(NUM_LAYERS):
            c = caches_fp[layer]
            flat_k = c.key_caches[0].reshape(-1, NUM_KV_HEADS, HEAD_DIM)
            flat_k[slot] = k
            c.key_caches[0] = flat_k.reshape(c.key_caches[0].shape)
            flat_v = c.value_caches[0].reshape(-1, NUM_KV_HEADS, HEAD_DIM)
            flat_v[slot] = v
            c.value_caches[0] = flat_v.reshape(c.value_caches[0].shape)
            outs.extend([c.key_caches[0], c.value_caches[0]])
        mx.eval(*outs)

    tq_step = _time_gpu(run_tq_step, warmup=5, iters=30)
    fp_step = _time_gpu(run_fp_step, warmup=5, iters=30)

    print()
    print("  Section B — Full decode-step simulation (28 layers × 1 tok)")
    print("  " + "─" * 78)
    print(f"    TQ path (28 × tq_encode):      {tq_step:>8.1f} µs")
    print(f"    FP path (28 × fp16 scatter):   {fp_step:>8.1f} µs")
    print(f"    Overhead per step:             {tq_step - fp_step:>+8.1f} µs")
    print(
        f"    Overhead as tok/s hit:         "
        f"{_overhead_tok_hit(tq_step - fp_step):>8.2f} tok/s at 60 tok/s base"
    )


def _overhead_tok_hit(overhead_us: float, base_tok_s: float = 60.0) -> float:
    """Convert a per-step µs overhead into a tok/s delta at a given base rate."""
    base_ms_per_tok = 1000.0 / base_tok_s
    new_ms_per_tok = base_ms_per_tok + overhead_us / 1000.0
    return base_tok_s - (1000.0 / new_ms_per_tok)


# -----------------------------------------------------------------------------
# Section E — Paged-attention READ-side: TQ vs FP16.
#
# This is the hypothesis test for the 16 → 18 tok/s gap.  The encode path is
# already known to be cheap (Section B); if paged_attention on a quantized
# cache costs ~200 µs more per layer than on a FP16 cache, 28 layers ×
# 200 µs ≈ 5.6 ms per token which exactly matches the observed gap.
#
# We populate a cache at a realistic decode context length (ctx=512) and
# time a single-query-token attention dispatch against it, repeated many
# times with proper GPU fencing.
# -----------------------------------------------------------------------------


def bench_paged_attention(
    ctx_lens: list[int],
    num_kv_heads: int = NUM_KV_HEADS,
    num_q_heads: int = NUM_Q_HEADS,
    num_layers: int = NUM_LAYERS,
    label: str = "Qwen3-0.6B (28L × 4KVH)",
    extrapolate_to: int | None = None,
    skip_fp_above: int | None = None,
    iters: int = 40,
) -> list[tuple[int, float, float]]:
    """Returns list of (ctx, tq_us, fp_us) for optional post-processing."""
    ops = get_ops()
    v_centroids = get_v_centroids(V_BITS)
    mx.eval(v_centroids)

    print()
    print(f"  Section E — paged_attention read side  [{label}]")
    print("  " + "─" * 78)
    print(
        f"  {'ctx':>7}  {'TQ µs':>10}  {'FP µs':>10}  "
        f"{'TQ/FP':>6}  {'overhead':>10}  {f'×{num_layers}L step':>12}"
    )
    print("  " + "─" * 78)

    results: list[tuple[int, float, float]] = []

    for ctx in ctx_lens:
        # Allocate enough blocks for ctx tokens (+1 query slot).
        num_blocks = (ctx + BLOCK_SIZE - 1) // BLOCK_SIZE + 4

        cache_tq = MetalPagedKVCache(
            num_layers=1,
            num_kv_heads=num_kv_heads,
            head_dim=HEAD_DIM,
            num_blocks=num_blocks,
            block_size=BLOCK_SIZE,
            dtype=mx.float16,
            turboquant=True,
            k_quant=K_QUANT,
        )
        cache_fp: MetalPagedKVCache | None = None
        measure_fp = skip_fp_above is None or ctx <= skip_fp_above
        if measure_fp:
            cache_fp = MetalPagedKVCache(
                num_layers=1,
                num_kv_heads=num_kv_heads,
                head_dim=HEAD_DIM,
                num_blocks=num_blocks,
                block_size=BLOCK_SIZE,
                dtype=mx.float16,
                turboquant=False,
            )

        k_fill = mx.random.normal(shape=(ctx, num_kv_heads, HEAD_DIM)).astype(
            mx.float16
        )
        v_fill = mx.random.normal(shape=(ctx, num_kv_heads, HEAD_DIM)).astype(
            mx.float16
        )
        slot_fill = mx.arange(ctx, dtype=mx.int64)
        mx.eval(k_fill, v_fill, slot_fill)

        (nk, nv, nks, nvs, nkz) = ops.tq_encode(
            k_fill,
            v_fill,
            cache_tq.key_caches[0],
            cache_tq.value_caches[0],
            cache_tq.key_scale_caches[0],
            cache_tq.value_scale_caches[0],
            cache_tq.key_zero_caches[0],
            slot_fill,
            v_centroids,
            V_BITS,
            K_BITS,
            True,
        )
        cache_tq.key_caches[0] = nk
        cache_tq.value_caches[0] = nv
        cache_tq.key_scale_caches[0] = nks
        cache_tq.value_scale_caches[0] = nvs
        cache_tq.key_zero_caches[0] = nkz
        mx.eval(nk, nv, nks, nvs, nkz)

        if cache_fp is not None:
            flat_k = cache_fp.key_caches[0].reshape(-1, num_kv_heads, HEAD_DIM)
            flat_k[slot_fill] = k_fill
            cache_fp.key_caches[0] = flat_k.reshape(cache_fp.key_caches[0].shape)
            flat_v = cache_fp.value_caches[0].reshape(-1, num_kv_heads, HEAD_DIM)
            flat_v[slot_fill] = v_fill
            cache_fp.value_caches[0] = flat_v.reshape(cache_fp.value_caches[0].shape)
            mx.eval(cache_fp.key_caches[0], cache_fp.value_caches[0])

        q = mx.random.normal(shape=(1, num_q_heads, HEAD_DIM)).astype(mx.float16)
        block_table = mx.arange(num_blocks, dtype=mx.int32).reshape(1, -1)
        seq_lens = mx.array([ctx], dtype=mx.int32)
        cu_seqlens = mx.array([0, 1], dtype=mx.int32)
        mx.eval(q, block_table, seq_lens, cu_seqlens)

        scale = 1.0 / (HEAD_DIM**0.5)

        # Bind all loop-scoped vars as defaults (see Section A note above).
        def run_tq_attn(
            q=q,
            cache_tq=cache_tq,
            block_table=block_table,
            seq_lens=seq_lens,
            cu_seqlens=cu_seqlens,
            ctx=ctx,
            scale=scale,
            num_kv_heads=num_kv_heads,
        ):
            out = mx.array(0)
            ops.paged_attention_primitive(
                q,
                cache_tq.key_caches[0],
                cache_tq.value_caches[0],
                num_kv_heads,
                scale,
                0.0,
                block_table,
                seq_lens,
                cu_seqlens,
                BLOCK_SIZE,
                ctx,
                -1,
                out,
                key_scale_cache=cache_tq.key_scale_caches[0],
                value_scale_cache=cache_tq.value_scale_caches[0],
                key_zero_cache=cache_tq.key_zero_caches[0],
                v_centroids=v_centroids,
                use_turboquant=True,
                quant_type=K_QUANT,
                v_bits=V_BITS,
            )
            mx.eval(out)

        def run_fp_attn(
            q=q,
            cache_fp=cache_fp,
            block_table=block_table,
            seq_lens=seq_lens,
            cu_seqlens=cu_seqlens,
            ctx=ctx,
            scale=scale,
            num_kv_heads=num_kv_heads,
        ):
            out = mx.array(0)
            ops.paged_attention_primitive(
                q,
                cache_fp.key_caches[0],
                cache_fp.value_caches[0],
                num_kv_heads,
                scale,
                0.0,
                block_table,
                seq_lens,
                cu_seqlens,
                BLOCK_SIZE,
                ctx,
                -1,
                out,
                use_turboquant=False,
            )
            mx.eval(out)

        tq_us = _time_gpu(run_tq_attn, warmup=3, iters=iters)
        if cache_fp is not None:
            fp_us = _time_gpu(run_fp_attn, warmup=3, iters=iters)
        else:
            fp_us = float("nan")
        overhead = tq_us - fp_us if not math.isnan(fp_us) else float("nan")
        per_step = overhead * num_layers if not math.isnan(overhead) else float("nan")

        ratio_str = f"{tq_us / fp_us:>5.2f}x" if not math.isnan(fp_us) else "  n/a"
        ov_str = f"{overhead:>+8.1f}µs" if not math.isnan(overhead) else "     n/a"
        step_str = f"{per_step:>+10.1f}µs" if not math.isnan(per_step) else "      n/a"

        print(
            f"  {ctx:>7}  {tq_us:>8.1f}µs  "
            f"{(f'{fp_us:>8.1f}µs' if not math.isnan(fp_us) else '    n/a  ')}  "
            f"{ratio_str}  {ov_str}  {step_str}"
        )
        results.append((ctx, tq_us, fp_us))

        del cache_tq, cache_fp, k_fill, v_fill, q, block_table
        del slot_fill, seq_lens, cu_seqlens, nk, nv, nks, nvs, nkz
        gc.collect()
        # Force MLX to return cached device buffers so the next (larger) ctx
        # iteration actually has headroom on 8 GB machines.
        try:
            mx.metal.clear_cache()
        except (AttributeError, RuntimeError):
            pass

    print("  " + "─" * 78)

    # ---- Optional linear extrapolation ---------------------------------------
    if extrapolate_to is not None and len(results) >= 2:
        # Fit TQ and FP separately as linear: y ≈ a + b*ctx, using the last few
        # points (where per-token cost has plateaued).
        xs = [r[0] for r in results[-3:]]
        tq_ys = [r[1] for r in results[-3:]]
        fp_ys = [r[2] for r in results[-3:] if not math.isnan(r[2])]

        def _fit(xs: list[int], ys: list[float]) -> tuple[float, float]:
            n = len(xs)
            mx_ = sum(xs) / n
            my_ = sum(ys) / n
            num = sum((x - mx_) * (y - my_) for x, y in zip(xs, ys, strict=True))
            den = sum((x - mx_) ** 2 for x in xs)
            b = num / den if den > 0 else 0.0
            a = my_ - b * mx_
            return a, b

        a_tq, b_tq = _fit(xs, tq_ys)
        if len(fp_ys) >= 2:
            a_fp, b_fp = _fit(xs[-len(fp_ys) :], fp_ys)
        else:
            a_fp = b_fp = float("nan")

        print(f"  Linear extrapolation (fit on last {len(xs)} ctx points):")
        print(f"    TQ µs ≈ {a_tq:.1f} + {b_tq:.5f} × ctx")
        if not math.isnan(a_fp):
            print(f"    FP µs ≈ {a_fp:.1f} + {b_fp:.5f} × ctx")
        tq_pred = a_tq + b_tq * extrapolate_to
        print()
        print(f"    Predicted at ctx={extrapolate_to:,}:")
        print(f"      TQ attention:    {tq_pred / 1000:>8.2f} ms per layer per step")
        if not math.isnan(a_fp):
            fp_pred = a_fp + b_fp * extrapolate_to
            ov_pred = tq_pred - fp_pred
            print(
                f"      FP attention:    {fp_pred / 1000:>8.2f} ms per layer per step"
            )
            print(
                f"      TQ overhead:     {ov_pred / 1000:>+8.2f} ms per layer "
                f"({tq_pred / fp_pred:.2f}x FP)"
            )
            print(
                f"      × {num_layers} layers:    {tq_pred * num_layers / 1000:>8.2f} ms TQ "
                f"/ {fp_pred * num_layers / 1000:.2f} ms FP per decode step"
            )
        else:
            print(
                f"      × {num_layers} layers:    {tq_pred * num_layers / 1000:>8.2f} ms TQ "
                f"per decode step"
            )

    return results


# -----------------------------------------------------------------------------
# Section D — Grid-saturation analysis (purely analytical, no GPU).
# -----------------------------------------------------------------------------


def print_saturation_analysis(ns: list[int]) -> None:
    print()
    print("  Section D — Grid saturation analysis")
    print("  " + "─" * 78)
    print("    Each tq_encode dispatch uses (N × num_kv_heads) threadgroups.")
    print(
        f"    Num KV heads: {NUM_KV_HEADS}  |  M1 saturation ≈ {M1_SATURATION_TGS} concurrent TGs"
    )
    print(f"    TG size: {HEAD_DIM} threads ({HEAD_DIM // 32} simdgroups per TG)")
    print("  " + "─" * 78)
    print(f"    {'N':>6}  {'TGs':>6}  {'occupancy':>12}  {'note':s}")
    print("  " + "─" * 78)
    for n in ns:
        tgs = n * NUM_KV_HEADS
        occ = min(100, 100.0 * tgs / M1_SATURATION_TGS)
        note = ""
        if tgs < M1_SATURATION_TGS // 2:
            note = "DISPATCH-BOUND — GPU barely utilised"
        elif tgs < M1_SATURATION_TGS:
            note = "partial occupancy"
        elif tgs < 4 * M1_SATURATION_TGS:
            note = "saturated"
        else:
            note = "saturated, high grid oversubscription"
        print(f"    {n:>6}  {tgs:>6}  {occ:>10.1f}%   {note}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> int:
    print("═" * 80)
    print("  TurboQuant kernel profiler — manual mode")
    print("═" * 80)
    print(f"  Shape: {NUM_LAYERS}L × {NUM_KV_HEADS}KVH × hd{HEAD_DIM}  (Qwen3-0.6B)")
    print(
        f"  Quant: K={K_QUANT} (bits={K_BITS})  V=q{V_BITS}_0 (bits={V_BITS})  "
        f"block_size={BLOCK_SIZE}"
    )

    # Warm up the ops module + JIT.
    ops = get_ops()
    v_centroids = get_v_centroids(V_BITS)
    cache_warm = _make_tq_cache(num_blocks=4)
    k = mx.random.normal(shape=(1, NUM_KV_HEADS, HEAD_DIM)).astype(mx.float16)
    v = mx.random.normal(shape=(1, NUM_KV_HEADS, HEAD_DIM)).astype(mx.float16)
    slot = mx.array([0], dtype=mx.int64)
    mx.eval(k, v, slot, v_centroids)
    for _ in range(3):
        (nk, nv, nks, nvs, nkz) = ops.tq_encode(
            k,
            v,
            cache_warm.key_caches[0],
            cache_warm.value_caches[0],
            cache_warm.key_scale_caches[0],
            cache_warm.value_scale_caches[0],
            cache_warm.key_zero_caches[0],
            slot,
            v_centroids,
            V_BITS,
            K_BITS,
            True,
        )
        mx.eval(nk, nv, nks, nvs, nkz)
    _sync()

    ns = [1, 4, 16, 64, 256, 1024, 4096]

    print_saturation_analysis(ns)

    samples = bench_encode_sweep(ns)
    print_encode_table(samples)

    bench_decode_step()

    bench_paged_attention([128, 512, 2048, 8192])

    # =======================================================================
    # Kimi K2.5 scale-out stress test.
    #
    # Spec from https://huggingface.co/moonshotai/Kimi-K2.5/raw/main/config.json:
    #   num_hidden_layers:    61
    #   num_attention_heads:  64
    #   num_key_value_heads:  64   (MHA config; actual model uses MLA w/ lora_rank=512)
    #   qk_nope_head_dim:    128
    #   max_position:         262144 native, extendable to 1M via YaRN
    #
    # CAVEAT: K2.5 uses Multi-head Latent Attention — the KV cache stores a
    # 512-dim shared latent, not per-head K/V.  Our TQ kernel is MHA-shaped
    # so this is a *pessimistic upper bound*: we treat each of the 64 heads
    # as if it had its own quantized K/V cache.  For real MLA integration
    # you'd quantize the 512-dim latent directly.
    #
    # Memory: at 64 KV heads × 128 HD, FP16 cache is ~16 KB/token; TQ cache
    # is ~7.5 KB/token.  On an M1 8 GB, we cap ctx at 16K with both caches
    # resident, then extrapolate linearly to 1M — our earlier data shows
    # the per-token overhead is dead-linear in ctx once ctx > ~1K, so linear
    # fit is defensible.  If you have more RAM, bump the top ctx up.
    # =======================================================================
    bench_paged_attention(
        [1024, 4096, 16384],
        num_kv_heads=64,
        num_q_heads=64,
        num_layers=61,
        label="Kimi K2.5 MHA-equiv (61L × 64KVH × hd128)",
        extrapolate_to=1_048_576,
        skip_fp_above=16384,
        iters=12,
    )

    print()
    print("═" * 80)
    print("  Done.  Paste output back to Cascade for analysis.")
    print("═" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
