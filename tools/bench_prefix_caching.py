#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Benchmark prefix caching TTFT using vllm bench serve with prefix_repetition dataset.

Starts a vllm-metal server, runs the benchmark, and reports TTFT metrics.
Designed to establish a baseline before prefix caching is implemented, and
to measure improvements after.

Usage:
    # Basic (starts server automatically):
    VLLM_METAL_USE_PAGED_ATTENTION=1 VLLM_METAL_MEMORY_FRACTION=0.7 \
        python tools/bench_prefix_caching.py

    # Custom prefix/suffix lengths:
    VLLM_METAL_USE_PAGED_ATTENTION=1 VLLM_METAL_MEMORY_FRACTION=0.7 \
        python tools/bench_prefix_caching.py \
        --prefix-len 512 --suffix-len 128 --num-prefixes 5

    # Against an already running server:
    python tools/bench_prefix_caching.py --server-url http://localhost:8000

    # Save results for comparison:
    python tools/bench_prefix_caching.py --save-result --label baseline
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
import urllib.request


def _wait_for_server(base_url: str, timeout: int) -> bool:
    """Poll server health endpoint until ready or timeout."""
    health_url = f"{base_url}/health"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=5) as resp:
                if resp.status == 200:
                    return True
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(2)
    return False


def _start_server(
    model: str,
    port: int,
    max_model_len: int,
    max_num_seqs: int,
) -> subprocess.Popen:
    """Launch vllm serve as a subprocess."""
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        model,
        "--port",
        str(port),
        "--max-model-len",
        str(max_model_len),
        "--max-num-seqs",
        str(max_num_seqs),
    ]
    print(f"Starting server: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    return proc


def _run_benchmark(
    base_url: str,
    model: str,
    num_prompts: int,
    prefix_len: int,
    suffix_len: int,
    num_prefixes: int,
    output_len: int,
    request_rate: float,
    save_result: bool,
    label: str | None,
    result_dir: str | None,
) -> int:
    """Run vllm bench serve with prefix_repetition dataset."""
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "bench",
        "serve",
        "--backend",
        "openai",
        "--base-url",
        base_url,
        "--model",
        model,
        "--dataset-name",
        "prefix_repetition",
        "--num-prompts",
        str(num_prompts),
        "--prefix-repetition-prefix-len",
        str(prefix_len),
        "--prefix-repetition-suffix-len",
        str(suffix_len),
        "--prefix-repetition-num-prefixes",
        str(num_prefixes),
        "--prefix-repetition-output-len",
        str(output_len),
        "--request-rate",
        str(request_rate),
        "--percentile-metrics",
        "ttft,tpot,e2el",
        "--metric-percentiles",
        "50,99",
    ]
    if save_result:
        cmd.append("--save-result")
    if label:
        cmd.extend(["--label", label])
    if result_dir:
        cmd.extend(["--result-dir", result_dir])

    print(f"\nRunning benchmark: {' '.join(cmd)}\n")
    return subprocess.call(cmd)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # Server options
    parser.add_argument(
        "--server-url",
        default=None,
        help="Base URL of running server. If not set, starts one automatically.",
    )
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument(
        "--server-timeout",
        type=int,
        default=120,
        help="Seconds to wait for server startup (default: 120).",
    )

    # Benchmark options
    parser.add_argument("--num-prompts", type=int, default=100)
    parser.add_argument("--prefix-len", type=int, default=256)
    parser.add_argument("--suffix-len", type=int, default=256)
    parser.add_argument("--num-prefixes", type=int, default=10)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Request rate (req/s). Default: inf (max throughput).",
    )

    # Result options
    parser.add_argument("--save-result", action="store_true")
    parser.add_argument("--label", default=None, help="Label prefix for result file.")
    parser.add_argument("--result-dir", default=None)

    args = parser.parse_args()

    server_proc = None
    base_url = args.server_url

    try:
        if base_url is None:
            # Start server
            server_proc = _start_server(
                args.model,
                args.port,
                args.max_model_len,
                args.max_num_seqs,
            )
            base_url = f"http://localhost:{args.port}"
            print(f"Waiting for server at {base_url} ...")
            if not _wait_for_server(base_url, args.server_timeout):
                print("ERROR: Server failed to start within timeout.", file=sys.stderr)
                sys.exit(1)
            print("Server ready.\n")

        rc = _run_benchmark(
            base_url=base_url,
            model=args.model,
            num_prompts=args.num_prompts,
            prefix_len=args.prefix_len,
            suffix_len=args.suffix_len,
            num_prefixes=args.num_prefixes,
            output_len=args.output_len,
            request_rate=args.request_rate,
            save_result=args.save_result,
            label=args.label,
            result_dir=args.result_dir,
        )
        sys.exit(rc)

    finally:
        if server_proc is not None:
            print("\nShutting down server ...")
            os.killpg(os.getpgid(server_proc.pid), signal.SIGTERM)
            server_proc.wait(timeout=10)


if __name__ == "__main__":
    main()
