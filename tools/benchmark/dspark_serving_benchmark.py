# SPDX-License-Identifier: Apache-2.0
"""Qualify DSpark HTTP serving against target-only and ordinary draft decoding.

Uses vllm bench serve for timings; token-ID checks and worker memory snapshots
run outside that window. Each comparison arm owns a fresh server process group.
Run from the source checkout with local, pinned checkpoint directories.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import platform
import re
import signal
import socket
import statistics
import subprocess
import sys
import time
from itertools import zip_longest
from pathlib import Path
from urllib.request import Request, urlopen

from prometheus_client.parser import text_string_to_metric_families

from tools.attention_bench_utils import package_versions, source_file_hashes
from tools.check_parity import http_generate
from tools.parity_prompts import PROMPTS
from vllm_metal.config import (
    DSPARK_DRAFT_QUANTIZATION_KEY,
    DSPARK_DRAFT_QUANTIZATION_Q4,
)

ROOT = Path(__file__).resolve().parents[2]
ARMS = ("target", "dspark", "draft_model")
COUNTERS = {
    "requests": "vllm:request_success_total",
    "drafts": "vllm:spec_decode_num_drafts_total",
    "draft_tokens": "vllm:spec_decode_num_draft_tokens_total",
    "accepted_tokens": "vllm:spec_decode_num_accepted_tokens_total",
    "preemptions": "vllm:num_preemptions_total",
}


class MemoryProbe:
    """Worker extension used only by this loopback benchmark's idle RPC calls."""

    def benchmark_memory(self, reset: str = "false") -> dict:
        import resource

        import mlx.core as mx
        import psutil

        mx.synchronize()
        runner = self.model_runner
        runtime = runner.paged_attention_runtime
        # Ordinary draft_model still uses count-initialized target/draft pools.
        # Its public page-byte helper includes both models; shared layouts must
        # instead count the one backing allocation, never sum aliasing views.
        mode = runner.scheduler_memory_reporting_mode()
        if mode == "paged_attention_capacity":
            kv_bytes = runtime.num_blocks() * runner.get_cache_block_size_bytes()
        elif mode == "paged_attention_layout_budget":
            kv_bytes = runtime.storage.nbytes
        else:
            raise ValueError(f"Unsupported benchmark cache mode: {mode}")
        result = {
            "mlx_active_bytes": mx.get_active_memory(),
            "mlx_peak_bytes": mx.get_peak_memory(),
            "mlx_cache_bytes": mx.get_cache_memory(),
            # Darwin ru_maxrss is bytes; this benchmark requires macOS.
            "worker_lifetime_peak_rss_bytes": resource.getrusage(
                resource.RUSAGE_SELF
            ).ru_maxrss,
            "worker_rss_bytes": psutil.Process().memory_info().rss,
            "kv_backing_bytes": kv_bytes,
            "kv_allocation_mode": mode,
            "max_model_len": self.model_runner.model_config.max_model_len,
        }
        if reset == "true":
            mx.reset_peak_memory()
        return result


def scheduler_capacity(log: str) -> dict:
    """Read the scheduler's group-aware capacity, including the legacy draft arm.

    The worker cannot infer usable tokens by multiplying physical pool pages:
    groups may share a block-ID pool. Keep vLLM's own logged calculation instead.
    """
    matches = re.findall(
        r"KV cache size: ([\d,]+) tokens, Maximum concurrency for ([\d,]+) "
        r"tokens per request: ([\d.]+)x",
        log,
    )
    if len(matches) != 1:
        raise ValueError("Expected one scheduler cache-capacity report")
    tokens, max_len, concurrency = matches[0]
    return {
        "tokens": int(tokens.replace(",", "")),
        "max_model_len": int(max_len.replace(",", "")),
        "max_concurrency_rounded": float(concurrency),
    }


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def request(base: str, endpoint: str, body: dict | None = None):
    req = Request(
        base + endpoint,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(req, timeout=120) as response:
        return json.load(response)


def memory_snapshot(base: str, *, reset: bool = False) -> dict:
    result = request(
        base,
        "/collective_rpc",
        {"method": "benchmark_memory", "kwargs": {"reset": str(reset).lower()}},
    )["results"]
    if len(result) != 1 or not isinstance(result[0], dict):
        raise ValueError("Expected one Metal worker memory snapshot")
    return result[0]


def metric_counts(text: str, *, arm: str) -> dict[str, float]:
    samples = [
        s for family in text_string_to_metric_families(text) for s in family.samples
    ]
    counts = {}
    for key, name in COUNTERS.items():
        values = [s.value for s in samples if s.name == name]
        # Non-speculative servers do not register speculation counters. Every
        # other missing series is unavailable evidence, not an observed zero.
        if not values and not (
            arm == "target" and key in ("drafts", "draft_tokens", "accepted_tokens")
        ):
            raise ValueError(f"Missing server counter: {name}")
        if any(not math.isfinite(v) or v < 0 or not v.is_integer() for v in values):
            raise ValueError(f"Invalid server counter: {name}")
        counts[key] = sum(values)
    return counts


def settled_metrics(base: str, expected_requests: int, path: Path, *, arm: str) -> dict:
    # Engine counters may lag the final HTTP response. Wait until all finished
    # requests are visible before taking either side of a delta.
    deadline = time.monotonic() + 30
    while True:
        with urlopen(base + "/metrics", timeout=5) as response:
            text = response.read().decode()
        path.write_text(text)
        counts = metric_counts(text, arm=arm)
        if counts["requests"] == expected_requests:
            return counts
        if counts["requests"] > expected_requests or time.monotonic() > deadline:
            raise ValueError(f"Incomplete/unexpected request counters: {counts}")
        time.sleep(0.25)


def validate_measurement(
    result: dict,
    before: dict,
    after: dict,
    *,
    input_lens: list[int],
    output_len: int,
    arm: str,
) -> dict:
    """Reject incomplete, mismatched, non-finite, or non-speculative evidence."""
    count = len(input_lens)
    if (
        result["completed"] != count
        or result["failed"] != 0
        or result["input_lens"] != input_lens
        or result["output_lens"] != [output_len] * count
        or result["total_input_tokens"] != sum(input_lens)
        or result["total_output_tokens"] != count * output_len
        or len(result["errors"]) != count
        or any(result["errors"])
    ):
        raise ValueError("Benchmark did not complete the identical requested workload")
    for key in (
        "duration",
        "output_throughput",
        "mean_ttft_ms",
        "mean_tpot_ms",
        "p95_ttft_ms",
        "p95_tpot_ms",
        "mean_e2el_ms",
        "p95_e2el_ms",
    ):
        if not math.isfinite(result[key]) or result[key] <= 0:
            raise ValueError(f"Invalid benchmark measurement: {key}")
    if not math.isclose(
        result["output_throughput"] * result["duration"],
        result["total_output_tokens"],
        rel_tol=1e-6,
    ):
        raise ValueError("Throughput does not describe the completed workload")
    delta = {key: after[key] - before[key] for key in COUNTERS}
    if any(
        not math.isfinite(v) or v < 0 or not float(v).is_integer()
        for v in delta.values()
    ):
        raise ValueError("Invalid/reset server counters")
    if (
        delta["requests"] != count
        or delta["accepted_tokens"] > delta["draft_tokens"]
        or delta["drafts"] > delta["draft_tokens"]
    ):
        raise ValueError("Inconsistent server counters")
    if arm != "target" and delta["draft_tokens"] == 0:
        raise ValueError("No actual draft verification occurred")
    if arm == "target" and any(
        delta[k] for k in ("drafts", "draft_tokens", "accepted_tokens")
    ):
        raise ValueError("Target-only baseline unexpectedly drafted")
    return delta


def machine_state() -> dict:
    import psutil

    def observe(command):
        # Observations are context, so a stalled or missing system tool must
        # not discard a long benchmark run.
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {
                "returncode": None,
                "stdout": "",
                "stderr": f"{type(exc).__name__}: {exc}",
            }
        return {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    gpu = observe(["ioreg", "-r", "-c", "AGXAccelerator", "-d", "1"])
    values = re.findall(r'"Device Utilization %"=(\d+)', gpu["stdout"])
    gpu_utilization = [int(value) for value in values] if values else None
    return {
        "available_memory_bytes": psutil.virtual_memory().available,
        "swap": psutil.swap_memory()._asdict(),
        "power": observe(["pmset", "-g", "batt"]),
        "thermal": observe(["pmset", "-g", "therm"]),
        "power_settings": observe(["pmset", "-g", "custom"]),
        "gpu_device_utilization_percent": gpu_utilization,
    }


def stop_server(server) -> None:
    # Signal the group even if the API parent has already exited, so no worker
    # from a failed arm can consume memory during the next arm.
    with contextlib.suppress(ProcessLookupError):
        os.killpg(server.pid, signal.SIGTERM)
    try:
        server.wait(timeout=20)
    except subprocess.TimeoutExpired:
        pass
    with contextlib.suppress(ProcessLookupError):
        os.killpg(server.pid, signal.SIGKILL)
    server.wait(timeout=10)


def benchmark_command(args, base: str, directory: Path, concurrency: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "bench",
        "serve",
        "--backend",
        "vllm",
        "--base-url",
        base,
        "--endpoint",
        "/v1/completions",
        "--model",
        str(args.target),
        "--served-model-name",
        "bench",
        "--dataset-name",
        "custom",
        "--dataset-path",
        str(args.output_dir / "prompts.jsonl"),
        "--skip-chat-template",
        "--disable-shuffle",
        "--num-prompts",
        str(args.num_prompts),
        "--custom-output-len",
        str(args.output_len),
        "--max-concurrency",
        str(concurrency),
        "--request-rate",
        "inf",
        "--temperature",
        "0",
        "--ignore-eos",
        "--seed",
        "825",
        "--extra-body",
        '{"add_special_tokens": false}',
        # The orchestrator runs a separate discarded streaming warmup.
        "--num-warmups",
        "0",
        "--ready-check-timeout-sec",
        "0",
        "--disable-tqdm",
        "--percentile-metrics",
        "ttft,tpot,e2el",
        "--metric-percentiles",
        "50,95,99",
        "--save-result",
        "--save-detailed",
        "--result-dir",
        str(directory),
        "--result-filename",
        "benchmark.json",
    ]


def run_arm(
    args, arm: str, repeat: int, reference: list[dict], env: dict
) -> list[dict]:
    directory = args.output_dir / f"r{repeat}-{arm}"
    directory.mkdir()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        str(args.target),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--served-model-name",
        "bench",
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-seqs",
        str(max(args.concurrency)),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--block-size",
        "16",
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--no-enable-prefix-caching",
        "--no-async-scheduling",
        "--generation-config",
        "vllm",
        "--worker-extension-cls",
        "tools.benchmark.dspark_serving_benchmark.MemoryProbe",
    ]
    if arm != "target":
        spec = {
            "method": arm,
            "model": str(args.dspark if arm == "dspark" else args.draft),
            "num_speculative_tokens": args.dspark_width
            if arm == "dspark"
            else args.draft_width,
        }
        if arm == "dspark" and args.dspark_draft_topk is not None:
            spec["dspark_draft_topk"] = args.dspark_draft_topk
        command += [
            "--speculative-config",
            json.dumps(spec),
        ]
    if arm == "dspark" and args.dspark_draft_quantization is not None:
        command += [
            "--additional-config",
            json.dumps({DSPARK_DRAFT_QUANTIZATION_KEY: args.dspark_draft_quantization}),
        ]
    write_json(directory / "server-command.json", command)
    rows = []
    with (directory / "server.log").open("w") as log:
        server = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=ROOT,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + args.timeout
            while True:
                if server.poll() is not None:
                    raise RuntimeError(
                        f"{arm} server exited; see {directory / 'server.log'}"
                    )
                try:
                    with urlopen(base + "/health", timeout=1):
                        break
                except OSError:
                    if time.monotonic() > deadline:
                        raise TimeoutError(f"{arm} server startup timed out") from None
                    time.sleep(0.5)
            capacity = scheduler_capacity((directory / "server.log").read_text())
            write_json(directory / "cache-capacity.json", capacity)
            write_json(directory / "startup-memory.json", memory_snapshot(base))
            completed = 0
            concurrencies = args.concurrency if repeat % 2 else args.concurrency[::-1]
            for concurrency in concurrencies:
                run_dir = directory / f"c{concurrency}"
                run_dir.mkdir()
                # Includes token IDs and validates the server's prompt IDs. No
                # sample logprobs: requesting them would silently disable drafting.
                outputs = http_generate(
                    base + "/v1",
                    "bench",
                    reference,
                    args.output_len,
                    batch_size=concurrency,
                )
                write_json(run_dir / "tokens.json", outputs)
                if any(len(row["tokens"]) != args.output_len for row in outputs):
                    raise ValueError(
                        "Untimed generation returned an incomplete sequence"
                    )
                completed += len(reference)
                for phase in ("warmup", "timed"):
                    phase_dir = run_dir / phase
                    phase_dir.mkdir()
                    before = settled_metrics(
                        base, completed, phase_dir / "metrics-before.txt", arm=arm
                    )
                    write_json(phase_dir / "machine-before.json", machine_state())
                    write_json(
                        phase_dir / "memory-before.json",
                        memory_snapshot(base, reset=True),
                    )
                    bench = benchmark_command(args, base, phase_dir, concurrency)
                    write_json(phase_dir / "benchmark-command.json", bench)
                    with (phase_dir / "benchmark.log").open("w") as bench_log:
                        subprocess.run(
                            bench,
                            env=env,
                            cwd=ROOT,
                            stdout=bench_log,
                            stderr=subprocess.STDOUT,
                            check=True,
                            timeout=args.timeout,
                        )
                    memory = memory_snapshot(base)
                    write_json(phase_dir / "memory-after.json", memory)
                    write_json(phase_dir / "machine-after.json", machine_state())
                    completed += len(reference)
                    after = settled_metrics(
                        base, completed, phase_dir / "metrics-after.txt", arm=arm
                    )
                    result = json.loads((phase_dir / "benchmark.json").read_text())
                    delta = validate_measurement(
                        result,
                        before,
                        after,
                        arm=arm,
                        input_lens=[len(row["input_ids"]) for row in reference],
                        output_len=args.output_len,
                    )
                # Immediate bench scrapes may precede the engine's stats flush;
                # retain them raw and use settled, warmup-excluded counters here.
                row = {
                    "arm": arm,
                    "repeat": repeat,
                    "concurrency": concurrency,
                    "path": str(phase_dir.relative_to(args.output_dir)),
                    "counters": delta,
                    "memory": memory,
                    "scheduler_capacity": capacity,
                    "benchmark": result,
                    "tokens": outputs,
                }
                rows.append(row)
                write_json(
                    phase_dir / "measurement.json",
                    {k: v for k, v in row.items() if k not in ("benchmark", "tokens")},
                )
                print(
                    f"{arm} r{repeat} c{concurrency}: {result['output_throughput']:.2f} output tokens/s; {delta['draft_tokens']:.0f} verified draft tokens",
                    flush=True,
                )
        finally:
            stop_server(server)
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    summary = []
    for concurrency in sorted({r["concurrency"] for r in rows}):
        for arm in ARMS:
            selected = sorted(
                (
                    r
                    for r in rows
                    if r["arm"] == arm and r["concurrency"] == concurrency
                ),
                key=lambda r: r["repeat"],
            )
            rates, ratios, exact, divergences = [], [], [], []
            for row in selected:
                baseline = next(
                    r
                    for r in rows
                    if r["arm"] == "target"
                    and r["repeat"] == row["repeat"]
                    and r["concurrency"] == concurrency
                )
                rates.append(row["benchmark"]["output_throughput"])
                ratios.append(rates[-1] / baseline["benchmark"]["output_throughput"])
                mismatches = []
                for prompt_index, (actual, expected) in enumerate(
                    zip(row["tokens"], baseline["tokens"], strict=True)
                ):
                    if actual["tokens"] == expected["tokens"]:
                        continue
                    token_index, (actual_id, expected_id) = next(
                        (index, pair)
                        for index, pair in enumerate(
                            zip_longest(actual["tokens"], expected["tokens"])
                        )
                        if pair[0] != pair[1]
                    )
                    mismatches.append(
                        {
                            "prompt_index": prompt_index,
                            "token_index": token_index,
                            "target_token_id": expected_id,
                            "arm_token_id": actual_id,
                        }
                    )
                exact.append(len(row["tokens"]) - len(mismatches))
                divergences.append(mismatches)
            summary.append(
                {
                    "arm": arm,
                    "concurrency": concurrency,
                    "repeat_ids": [r["repeat"] for r in selected],
                    "output_tokens_per_s": rates,
                    "median_output_tokens_per_s": statistics.median(rates),
                    "paired_throughput_ratio_vs_target": ratios,
                    "exact_sequences_vs_target": exact,
                    "greedy_parity_passed": not any(divergences),
                    "first_divergences_vs_target": divergences,
                    "sequences_per_repeat": len(selected[0]["tokens"]),
                    "mean_ttft_ms": [r["benchmark"]["mean_ttft_ms"] for r in selected],
                    "mean_tpot_ms": [r["benchmark"]["mean_tpot_ms"] for r in selected],
                    "p95_e2el_ms": [r["benchmark"]["p95_e2el_ms"] for r in selected],
                }
            )
    return summary


def write_summary(output_dir: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("No serving runs to qualify")
    summary = summarize(rows)
    write_json(output_dir / "summary.json", summary)
    failed = [
        f"{row['arm']} c{row['concurrency']}"
        for row in summary
        if not row["greedy_parity_passed"]
    ]
    if failed:
        raise ValueError(
            f"Greedy token parity failed: {', '.join(failed)}. "
            "Timing results and first divergences are retained in summary.json; "
            "throughput ratios do not establish a lossless speedup."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("target", "dspark", "draft", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument(
        "--prompt-file",
        type=Path,
        help="JSONL rows with a plain-text prompt field; default: shared parity corpus",
    )
    parser.add_argument("--num-prompts", type=int, default=40)
    parser.add_argument("--output-len", type=int, default=64)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--dspark-width", type=int, default=7)
    parser.add_argument(
        "--dspark-draft-quantization", choices=[DSPARK_DRAFT_QUANTIZATION_Q4]
    )
    parser.add_argument(
        "--dspark-draft-topk",
        type=int,
        help="Limit DSpark's Markov correction to this many base-logit candidates",
    )
    parser.add_argument("--draft-width", type=int, default=3)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-num-batched-tokens", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.4)
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout in seconds for startup and each benchmark",
    )
    args = parser.parse_args()
    if sys.platform != "darwin":
        parser.error("This benchmark requires macOS and a Metal worker")
    if (
        min(
            args.concurrency
            + [
                args.num_prompts,
                args.dspark_width,
                args.draft_width,
                args.max_model_len,
                args.max_num_batched_tokens,
                args.timeout,
            ]
        )
        < 1
        or args.output_len < 2
        or args.repeats < 2
        or len(set(args.concurrency)) != len(args.concurrency)
        or max(args.concurrency) > args.num_prompts
        or not 0 < args.gpu_memory_utilization < 1
        or (args.dspark_draft_topk is not None and args.dspark_draft_topk < 1)
    ):
        parser.error(
            "Use positive limits, >=2 output tokens/repeats, unique concurrency <= prompts, and memory fraction in (0,1)"
        )
    for name in ("target", "dspark", "draft"):
        path = getattr(args, name).resolve()
        if not (path / "config.json").is_file():
            parser.error(f"--{name} must be a local checkpoint directory")
        setattr(args, name, path)
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists():
        parser.error("Use a new output directory to avoid stale successful evidence")
    from transformers import AutoTokenizer

    prompts = (
        PROMPTS
        if args.prompt_file is None
        else [
            json.loads(line)["prompt"]
            for line in args.prompt_file.read_text().splitlines()
            if line.strip()
        ]
    )
    if len(prompts) < args.num_prompts or any(
        not isinstance(p, str) or not p for p in prompts
    ):
        parser.error("Provide enough nonempty plain-text prompts")
    prompts = prompts[: args.num_prompts]
    tokenizer = AutoTokenizer.from_pretrained(args.target)
    reference = [
        {"input_ids": tokenizer.encode(p, add_special_tokens=False)} for p in prompts
    ]
    if any(
        not r["input_ids"] or len(r["input_ids"]) + args.output_len > args.max_model_len
        for r in reference
    ):
        parser.error("Every prompt plus output must fit max-model-len")
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "prompts.jsonl").write_text(
        "".join(json.dumps({"prompt": p}) + "\n" for p in prompts)
    )
    write_json(args.output_dir / "input-token-ids.json", reference)
    env = os.environ.copy()
    env.update(
        VLLM_ENABLE_V1_MULTIPROCESSING="1",
        VLLM_SERVER_DEV_MODE="1",
        MLX_ENABLE_TF32="0",
    )
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    order = [
        list(ARMS if repeat % 2 else ARMS[::-1])
        for repeat in range(1, args.repeats + 1)
    ]
    import mlx.core as mx
    import psutil

    source_paths = sorted(
        p
        for d in ("tools", "vllm_metal")
        for p in (ROOT / d).rglob("*")
        if p.suffix in (".py", ".metal", ".cpp", ".h")
    )
    hashes = source_file_hashes(ROOT, source_paths)
    write_json(
        args.output_dir / "metadata.json",
        {
            "arguments": {
                k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
            },
            "platform": platform.platform(),
            "hardware": {
                "mlx_device": mx.device_info(),
                "memory_bytes": psutil.virtual_memory().total,
            },
            "checkpoint_configs": {
                name: json.loads((getattr(args, name) / "config.json").read_text())
                for name in ("target", "dspark", "draft")
            },
            "python": platform.python_version(),
            "run_order": order,
            "versions": package_versions(
                "vllm", "vllm-metal", "mlx", "mlx-lm", "transformers"
            ),
            "environment": {
                k: v
                for k, v in env.items()
                if k.startswith("VLLM_METAL_")
                or k
                in (
                    "VLLM_ENABLE_V1_MULTIPROCESSING",
                    "VLLM_SERVER_DEV_MODE",
                    "MLX_ENABLE_TF32",
                )
            },
            "source_sha256": hashes,
        },
    )
    rows = []
    try:
        for repeat, arms in enumerate(order, 1):
            for arm in arms:
                rows.extend(run_arm(args, arm, repeat, reference, env))
        if source_file_hashes(ROOT, source_paths) != hashes:
            raise ValueError(
                "Source changed during the benchmark; repeat from a fixed checkout"
            )
        write_summary(args.output_dir, rows)
    except BaseException as exc:
        write_json(
            args.output_dir / "failure.json",
            {"error": str(exc), "type": type(exc).__name__},
        )
        raise


if __name__ == "__main__":
    main()
