#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare MLX and MPS serving with vLLM's benchmark client.

Fresh servers run in MLX/MPS/MPS/MLX order. See docs/tools.md for the contract.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import signal
import statistics
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

if __package__:
    from .check_parity import checkpoint, serving
else:
    from check_parity import checkpoint, serving

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--dataset-path", type=Path, default=Path("sonnet.txt"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--num-prompts", type=int, default=100)
    parser.add_argument("--input-len", type=int, default=1024)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--request-rate", type=float, default=10)
    parser.add_argument("--cache-blocks", type=int, default=2340)
    args = parser.parse_args()
    if (
        min(
            args.num_prompts,
            args.input_len,
            args.output_len,
            args.concurrency,
            args.request_rate,
            args.cache_blocks,
        )
        <= 0
    ):
        parser.error("Workload sizes, request rate and cache blocks must be positive")
    dataset = args.dataset_path.resolve()
    dataset_hash = hashlib.sha256(dataset.read_bytes()).hexdigest()
    model, dtype = checkpoint(args.model)
    output = args.output_dir or Path(tempfile.mkdtemp(prefix="metal-backend-bench-"))
    output.mkdir(parents=True, exist_ok=True)
    output = output.resolve()
    print(f"Artifacts: {output}", flush=True)
    max_len = max(2048, args.input_len + args.output_len)
    server_args = (
        "--dtype",
        dtype,
        "--model-impl",
        "auto",
        "--block-size",
        "16",
        "--max-num-batched-tokens",
        str(max_len),
        "--kv-cache-memory-bytes",
        str(4 << 30),
        "--num-gpu-blocks-override",
        str(args.cache_blocks),
        "--async-scheduling",
        "--seed",
        "0",
    )
    cli = [sys.executable, "-m", "vllm.entrypoints.cli.main"]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(ROOT), env.get("PYTHONPATH")])
    )
    env["VLLM_METAL_MEMORY_FRACTION"] = "0.3"
    manifest = {
        "model": model,
        "dtype": dtype,
        "workload": vars(args),
        "dataset_sha256": dataset_hash,
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "hardware": subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip(),
        "memory_bytes": int(
            subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
        ),
        "platform": platform.platform(),
        "versions": {
            p: importlib.metadata.version(p) for p in ("vllm", "torch", "mlx", "mlx-lm")
        },
        "server_args": server_args,
        "max_model_len": max_len,
        "prefix_caching": False,
        "gpu_memory_utilization": 0.3,
        "env": {
            k: v
            for k, v in env.items()
            if k.startswith(
                (
                    "VLLM_METAL_",
                    "VLLM_MLX_",
                    "VLLM_USE_",
                    "VLLM_ENABLE_V1_",
                    "MLX_",
                    "PYTORCH_MPS_",
                )
            )
        },
        "runs": [],
    }
    results = {"mlx": [], "mps": []}
    for index, backend in enumerate(("mlx", "mps", "mps", "mlx"), 1):
        label = f"{index}-{backend}"
        overrides = {
            "VLLM_METAL_BACKEND": backend,
            "VLLM_USE_V2_MODEL_RUNNER": str(int(backend == "mps")),
            "VLLM_USE_HW_AGNOSTIC": str(int(backend == "mps")),
        }
        with serving(
            model,
            max_len,
            args.concurrency,
            0.3,
            output / f"{label}-server.log",
            {**env, **overrides},
            server_args,
        ) as url:
            command = [
                *cli,
                "bench",
                "serve",
                "--backend",
                "vllm",
                "--base-url",
                url.removesuffix("/v1"),
                "--endpoint",
                "/v1/completions",
                "--model",
                model,
                "--dataset-name",
                "sonnet",
                "--dataset-path",
                str(dataset),
                "--sonnet-input-len",
                str(args.input_len),
                "--sonnet-output-len",
                str(args.output_len),
                "--num-prompts",
                str(args.num_prompts),
                "--request-rate",
                str(args.request_rate),
                "--max-concurrency",
                str(args.concurrency),
                "--temperature",
                "0",
                "--ignore-eos",
                "--seed",
                "0",
                "--num-warmups",
                "4",
                "--save-result",
                "--save-detailed",
                "--result-dir",
                str(output),
                "--result-filename",
                f"{label}.json",
            ]
            manifest["runs"].append(
                {"label": label, "env": overrides, "client": command}
            )
            (output / "config.json").write_text(
                json.dumps(manifest, indent=2, default=str)
            )
            print(f"Running {label}...", flush=True)
            with (output / f"{label}-client.log").open("w") as log:
                subprocess.run(
                    command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True
                )
            result = json.loads((output / f"{label}.json").read_text())
            if (
                result["completed"] != args.num_prompts
                or result["total_output_tokens"] != args.num_prompts * args.output_len
            ):
                raise RuntimeError(f"Incomplete benchmark: inspect {label}-client.log")
            with urllib.request.urlopen(
                url.removesuffix("/v1") + "/metrics", timeout=10
            ) as response:
                metrics = response.read().decode()
            (output / f"{label}-metrics.txt").write_text(metrics)
            preemptions = [
                float(line.split()[-1])
                for line in metrics.splitlines()
                if line.startswith("vllm:num_preemptions_total{")
            ]
            if not preemptions:
                raise RuntimeError(
                    f"Missing preemption counter: inspect {label}-metrics.txt"
                )
            result["preemptions"] = sum(preemptions)
            results[backend].append(result)
    if any(
        r["input_lens"] != results["mlx"][0]["input_lens"]
        for runs in results.values()
        for r in runs
    ):
        raise RuntimeError("Input token lengths differ; inspect the saved results")
    lines = ["| Metric | MLX | MPS | MPS change |", "|---|---:|---:|---:|"]
    for title, key in (
        ("Output tok/s ↑", "output_throughput"),
        ("Mean TTFT (ms) ↓", "mean_ttft_ms"),
        ("Mean TPOT (ms) ↓", "mean_tpot_ms"),
        ("Preemptions/run (including warmup)", "preemptions"),
    ):
        mlx, mps = (statistics.mean(r[key] for r in results[b]) for b in ("mlx", "mps"))
        change = f"{(mps / mlx - 1) * 100:+.1f}%" if mlx else "—"
        lines.append(f"| {title} | {mlx:.1f} | {mps:.1f} | {change} |")
    summary = "\n".join(lines) + "\n"
    (output / "summary.md").write_text(summary)
    print(summary)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(1))
    main()
