#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""End-to-end throughput benchmark: vllm-metal vs mlx-lm on Qwen3.5-0.8B.

Measures tokens/sec for both prompt processing (prefill) and token generation
(decode) so you can see where each system wins.

Usage:
    # Default: run both backends, compare
    python -m tools.benchmark.qwen35_vs_mlx

    # Single backend only
    python -m tools.benchmark.qwen35_vs_mlx --backend vllm-metal
    python -m tools.benchmark.qwen35_vs_mlx --backend mlx-lm

    # Custom model or prompt length
    python -m tools.benchmark.qwen35_vs_mlx --model Qwen/Qwen3.5-0.8B \\
        --prompt-tokens 512 --output-tokens 200 --batch-size 1

    # Concurrent multi-request (batch) mode — where vllm-metal excels
    python -m tools.benchmark.qwen35_vs_mlx --batch-size 8 --output-tokens 128

    # Write JSON results
    python -m tools.benchmark.qwen35_vs_mlx --output-json /tmp/results.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any

DEFAULT_MODEL = "Qwen/Qwen3.5-0.8B"
DEFAULT_PROMPT_TOKENS = 256
DEFAULT_OUTPUT_TOKENS = 128
DEFAULT_BATCH_SIZE = 1
DEFAULT_WARMUP_RUNS = 2
DEFAULT_BENCH_RUNS = 5


@dataclass
class BenchResult:
    backend: str
    batch_size: int
    prompt_tokens: int
    output_tokens: int
    # Per-run timings
    ttft_ms: list[float]  # time-to-first-token
    decode_ms: list[float]  # total generation time (excl. prefill)
    total_ms: list[float]  # end-to-end
    # Derived
    prefill_tok_per_s: float  # prompt_tokens * batch / mean_ttft_s
    decode_tok_per_s: float  # output_tokens * batch / mean_decode_s
    e2e_tok_per_s: float  # total_tokens / mean_total_s


def _build_prompt(n_tokens: int, tokenizer: Any) -> str:
    """Build a prompt whose tokenization is approximately n_tokens long."""
    # Use a repeated phrase — quick to generate, realistic enough.
    base = "The quick brown fox jumps over the lazy dog. "
    text = base * (n_tokens // 10 + 1)
    # Trim to roughly the right length by encoding and decoding
    ids = tokenizer.encode(text)[:n_tokens]
    return tokenizer.decode(ids)


# ---------------------------------------------------------------------------
# mlx-lm backend
# ---------------------------------------------------------------------------


def bench_mlx_lm(
    model_name: str,
    batch_size: int,
    prompt_tokens: int,
    output_tokens: int,
    warmup: int,
    runs: int,
) -> BenchResult:
    import mlx.core as mx
    from mlx_lm import load

    print(f"  Loading {model_name} via mlx-lm…")
    model, tokenizer = load(model_name)

    prompt = _build_prompt(prompt_tokens, tokenizer)
    # mlx-lm is single-request; for batch_size>1 we run sequentially and
    # divide wall time by batch_size to get per-request throughput.

    def run_once() -> tuple[float, float]:
        """Returns (ttft_ms, total_ms)."""
        mx.synchronize()
        t0 = time.perf_counter()

        first_token_time: list[float] = []

        def _record_first(token_ids: list[int], text: str) -> bool:  # noqa: ARG001
            if not first_token_time:
                first_token_time.append(time.perf_counter())
            return False  # keep going

        # stream_generate yields tokens; we time the first one for TTFT
        from mlx_lm import stream_generate

        tokens_generated = 0
        for _ in stream_generate(
            model, tokenizer, prompt=prompt, max_tokens=output_tokens
        ):
            if not first_token_time:
                first_token_time.append(time.perf_counter())
            tokens_generated += 1

        mx.synchronize()
        t_end = time.perf_counter()

        ttft_ms = (first_token_time[0] - t0) * 1000.0 if first_token_time else 0.0
        total_ms = (t_end - t0) * 1000.0
        return ttft_ms, total_ms

    print(f"  Warming up ({warmup} runs)…")
    for _ in range(warmup):
        run_once()

    print(f"  Benchmarking ({runs} runs × batch_size={batch_size})…")
    ttft_list, decode_list, total_list = [], [], []
    for i in range(runs):
        # For batch > 1, run batch_size sequential requests and aggregate
        run_ttft, run_total = 0.0, 0.0
        for _ in range(batch_size):
            t, tot = run_once()
            run_ttft += t
            run_total += tot
        ttft_list.append(run_ttft / batch_size)
        total_list.append(run_total / batch_size)
        decode_list.append(max(0.0, total_list[-1] - ttft_list[-1]))
        print(
            f"    run {i + 1}/{runs}: ttft={ttft_list[-1]:.1f}ms  total={total_list[-1]:.1f}ms"
        )

    mean_ttft_s = statistics.fmean(ttft_list) / 1000.0
    mean_decode_s = statistics.fmean(decode_list) / 1000.0
    mean_total_s = statistics.fmean(total_list) / 1000.0
    total_tokens = (prompt_tokens + output_tokens) * batch_size

    return BenchResult(
        backend="mlx-lm",
        batch_size=batch_size,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        ttft_ms=ttft_list,
        decode_ms=decode_list,
        total_ms=total_list,
        prefill_tok_per_s=(prompt_tokens * batch_size) / max(mean_ttft_s, 1e-9),
        decode_tok_per_s=(output_tokens * batch_size) / max(mean_decode_s, 1e-9),
        e2e_tok_per_s=total_tokens / max(mean_total_s, 1e-9),
    )


# ---------------------------------------------------------------------------
# vllm-metal backend
# ---------------------------------------------------------------------------


def bench_vllm_metal(
    model_name: str,
    batch_size: int,
    prompt_tokens: int,
    output_tokens: int,
    warmup: int,
    runs: int,
    paged: bool = True,
) -> BenchResult:
    import os

    import mlx.core as mx

    if paged:
        os.environ.setdefault("VLLM_METAL_USE_PAGED_ATTENTION", "1")
    os.environ.setdefault("VLLM_METAL_MEMORY_FRACTION", "auto")
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    from vllm import LLM, SamplingParams

    print(f"  Loading {model_name} via vllm-metal (paged={paged})…")
    llm = LLM(
        model=model_name,
        max_model_len=prompt_tokens + output_tokens + 64,
        max_num_seqs=max(batch_size, 1),
    )

    from mlx_lm import load as mlx_lm_load

    _, tokenizer = mlx_lm_load(model_name)
    prompt = _build_prompt(prompt_tokens, tokenizer)
    prompts = [prompt] * batch_size
    sp = SamplingParams(temperature=0.0, max_tokens=output_tokens)

    def run_once() -> tuple[float, float]:
        mx.synchronize()
        t0 = time.perf_counter()
        _ = llm.generate(prompts, sp)
        mx.synchronize()
        t_end = time.perf_counter()

        # Estimate TTFT from the scheduler's first-token metadata if available,
        # otherwise approximate as prefill fraction of total time.
        # vLLM's LLM.generate() does not expose per-step timing, so we use a
        # rough model: TTFT ≈ total * (prompt_tokens / total_tokens).
        total_ms = (t_end - t0) * 1000.0
        prefill_frac = prompt_tokens / max(prompt_tokens + output_tokens, 1)
        ttft_ms = total_ms * prefill_frac
        return ttft_ms, total_ms

    print(f"  Warming up ({warmup} runs)…")
    for _ in range(warmup):
        run_once()

    print(f"  Benchmarking ({runs} runs × batch_size={batch_size})…")
    ttft_list, decode_list, total_list = [], [], []
    for i in range(runs):
        t, tot = run_once()
        ttft_list.append(t)
        total_list.append(tot)
        decode_list.append(max(0.0, tot - t))
        print(
            f"    run {i + 1}/{runs}: ttft≈{ttft_list[-1]:.1f}ms  total={total_list[-1]:.1f}ms"
        )

    mean_ttft_s = statistics.fmean(ttft_list) / 1000.0
    mean_decode_s = statistics.fmean(decode_list) / 1000.0
    mean_total_s = statistics.fmean(total_list) / 1000.0
    total_tokens = (prompt_tokens + output_tokens) * batch_size

    return BenchResult(
        backend="vllm-metal",
        batch_size=batch_size,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        ttft_ms=ttft_list,
        decode_ms=decode_list,
        total_ms=total_list,
        prefill_tok_per_s=(prompt_tokens * batch_size) / max(mean_ttft_s, 1e-9),
        decode_tok_per_s=(output_tokens * batch_size) / max(mean_decode_s, 1e-9),
        e2e_tok_per_s=total_tokens / max(mean_total_s, 1e-9),
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _fmt(v: float, unit: str = "") -> str:
    return f"{v:>9.1f}{unit}"


def print_comparison(results: list[BenchResult]) -> None:
    if not results:
        return

    print()
    print("=" * 70)
    print(f"  Model: {DEFAULT_MODEL}")
    print(
        f"  Workload: prompt={results[0].prompt_tokens} tokens, "
        f"output={results[0].output_tokens} tokens, "
        f"batch={results[0].batch_size}"
    )
    print("=" * 70)
    header = f"{'Backend':<16}  {'TTFT (ms)':>10}  {'Prefill tok/s':>14}  {'Decode tok/s':>13}  {'E2E tok/s':>10}"
    print(header)
    print("-" * len(header))

    best_decode = max(r.decode_tok_per_s for r in results)
    best_e2e = max(r.e2e_tok_per_s for r in results)

    for r in results:
        ttft = statistics.fmean(r.ttft_ms)
        decode_marker = " ★" if abs(r.decode_tok_per_s - best_decode) < 0.1 else ""
        e2e_marker = " ★" if abs(r.e2e_tok_per_s - best_e2e) < 0.1 else ""
        print(
            f"{r.backend:<16}  "
            f"{_fmt(ttft, 'ms'):>10}  "
            f"{_fmt(r.prefill_tok_per_s, ' t/s'):>14}  "
            f"{_fmt(r.decode_tok_per_s, ' t/s'):>13}{decode_marker}  "
            f"{_fmt(r.e2e_tok_per_s, ' t/s'):>10}{e2e_marker}"
        )

    if len(results) == 2:  # noqa: PLR2004
        a, b = results
        print()
        print("Speedup (vllm-metal ÷ mlx-lm):")
        vm = next((r for r in results if r.backend == "vllm-metal"), None)
        mx_ = next((r for r in results if r.backend == "mlx-lm"), None)
        if vm and mx_:
            print(
                f"  Decode: {vm.decode_tok_per_s / max(mx_.decode_tok_per_s, 1):.2f}x"
            )
            print(f"  E2E:    {vm.e2e_tok_per_s / max(mx_.e2e_tok_per_s, 1):.2f}x")
    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare vllm-metal vs mlx-lm throughput on Qwen3.5-0.8B",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", default=DEFAULT_MODEL, help="HuggingFace model name")
    p.add_argument(
        "--backend",
        choices=["vllm-metal", "mlx-lm", "both"],
        default="both",
        help="Which backend(s) to benchmark",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of concurrent requests (batch size)",
    )
    p.add_argument(
        "--prompt-tokens",
        type=int,
        default=DEFAULT_PROMPT_TOKENS,
        help="Approximate prompt length in tokens",
    )
    p.add_argument(
        "--output-tokens",
        type=int,
        default=DEFAULT_OUTPUT_TOKENS,
        help="Number of tokens to generate",
    )
    p.add_argument(
        "--warmup", type=int, default=DEFAULT_WARMUP_RUNS, help="Warmup runs"
    )
    p.add_argument(
        "--runs", type=int, default=DEFAULT_BENCH_RUNS, help="Benchmark runs"
    )
    p.add_argument(
        "--no-paged",
        action="store_true",
        help="Disable paged attention for vllm-metal (use batched MLX path)",
    )
    p.add_argument("--output-json", help="Write results to JSON file")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    results: list[BenchResult] = []

    if args.backend in ("mlx-lm", "both"):
        print(f"\n[mlx-lm] Benchmarking {args.model}")
        try:
            r = bench_mlx_lm(
                model_name=args.model,
                batch_size=args.batch_size,
                prompt_tokens=args.prompt_tokens,
                output_tokens=args.output_tokens,
                warmup=args.warmup,
                runs=args.runs,
            )
            results.append(r)
        except Exception as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)

    if args.backend in ("vllm-metal", "both"):
        print(f"\n[vllm-metal] Benchmarking {args.model}")
        try:
            r = bench_vllm_metal(
                model_name=args.model,
                batch_size=args.batch_size,
                prompt_tokens=args.prompt_tokens,
                output_tokens=args.output_tokens,
                warmup=args.warmup,
                runs=args.runs,
                paged=not args.no_paged,
            )
            results.append(r)
        except Exception as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)

    if results:
        print_comparison(results)

    if args.output_json and results:
        out = {
            "model": args.model,
            "batch_size": args.batch_size,
            "prompt_tokens": args.prompt_tokens,
            "output_tokens": args.output_tokens,
            "results": [asdict(r) for r in results],
        }
        with open(args.output_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nResults written to {args.output_json}")

    return 0


if __name__ == "__main__":
    if __package__ in (None, ""):
        raise SystemExit("Run as a module: python -m tools.benchmark.qwen35_vs_mlx")
    sys.exit(main())
