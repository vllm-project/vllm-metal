#!/usr/bin/env python3
"""Benchmark vLLM Metal native Qwen MTP across a curated speed matrix.

The matrix is intentionally curated rather than a full Cartesian product. It
keeps server launches bounded while testing the knobs most likely to affect
Apple Silicon serving throughput: scheduler token budget, paged-attention
block size, speculative verification-window mode, decode pipelining, GDN
kernel mode, request concurrency, and short/long shared-prefix workloads.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import signal
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import aiohttp
import mlx.core as mx
from mlx_lm.generate import mtp_generate_step
from mlx_lm.utils import load
from transformers import AutoTokenizer
from vllm.benchmarks.lib.endpoint_request_func import (
    RequestFuncInput,
    RequestFuncOutput,
    async_request_openai_completions,
)
from vllm.benchmarks.serve import fetch_spec_decode_metrics


@dataclass(frozen=True)
class Profile:
    name: str
    mode: str
    max_num_batched_tokens: int
    block_size: int
    verify_window: bool = False
    decode_pipeline: bool = True
    gdn_lazy: bool = True
    enforce_eager: bool = False


@dataclass(frozen=True)
class Workload:
    name: str
    prompt_tokens: int
    output_tokens: int
    concurrency: int
    requests: int


PROFILES: tuple[Profile, ...] = (
    # Same tuned non-MTP configuration is run first and last to measure drift.
    Profile("baseline_ref", "baseline", 2048, 16),
    # Conservative configuration closest to the first functional benchmark.
    Profile("mtp_safe", "mtp", 512, 16, enforce_eager=True),
    Profile("mtp_batch1024", "mtp", 1024, 16),
    # Matched to baseline_ref except for native MTP itself.
    Profile("mtp_batch2048", "mtp", 2048, 16),
    Profile("mtp_batch4096", "mtp", 4096, 16),
    Profile("mtp_block32", "mtp", 2048, 32),
    Profile("mtp_verify_window", "mtp", 2048, 16, verify_window=True),
    # Negative controls establish whether the default fast paths really win.
    Profile("mtp_no_decode_pipeline", "mtp", 2048, 16, decode_pipeline=False),
    Profile("mtp_gdn_fallback", "mtp", 2048, 16, gdn_lazy=False),
    Profile("baseline_repeat", "baseline", 2048, 16),
)

WORKLOADS: tuple[Workload, ...] = (
    Workload("interactive_c1", 1152, 128, 1, 4),
    Workload("serving_c4", 1152, 128, 4, 8),
    Workload("serving_c8", 1152, 128, 8, 16),
    Workload("serving_c16", 1152, 64, 16, 32),
    Workload("long_prefix_c1", 8192, 64, 1, 2),
    # 8K makes the verification-window experiment meaningful while remaining
    # within the standard hosted Apple Silicon runner's cache capacity.
    Workload("long_prefix_c4", 8192, 64, 4, 4),
)

SPEC_CONFIG = '{"method":"mtp","num_speculative_tokens":1}'
MODEL_MAX_LEN = 8448


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--request-timeout-s", type=float, default=240.0)
    parser.add_argument("--server-ready-timeout-s", type=float, default=720.0)
    parser.add_argument("--native-check-only", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    return args


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def tail(path: Path, lines: int = 250) -> str:
    if not path.exists():
        return "<log file was not created>"
    return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])


def native_mtp_check(model_dir: Path, output_dir: Path) -> dict[str, Any]:
    model, tokenizer = load(str(model_dir))
    cache = model.make_mtp_cache()
    if not bool(getattr(model, "supports_mtp", False)):
        raise RuntimeError("converted model does not advertise supports_mtp")
    if not cache:
        raise RuntimeError("converted model returned an empty MTP cache")

    target_prompt_tokens = 1152
    seed = (
        "Apple Silicon speculative decoding uses a shared cached prefix "
        "and a trained multi-token prediction head. "
    )
    text = seed
    while len(tokenizer.encode(text)) < target_prompt_tokens + 16:
        text += seed
    prompt = mx.array(
        tokenizer.encode(text)[:target_prompt_tokens],
        dtype=mx.uint32,
    )

    accepted = 0
    emitted = 0
    started = time.perf_counter()
    for _token, _logprobs, from_draft in mtp_generate_step(
        prompt,
        model,
        max_tokens=192,
    ):
        emitted += 1
        accepted += int(from_draft)
    elapsed = time.perf_counter() - started

    # mlx-lm exposes whether each emitted token came from the draft, but not
    # the total proposal-attempt count. Do not invent an acceptance denominator.
    result = {
        "supports_mtp": True,
        "mtp_cache_entries": len(cache),
        "output_tokens": emitted,
        "accepted_draft_tokens": accepted,
        "accepted_output_fraction": accepted / emitted if emitted else None,
        "output_throughput_tok_s": emitted / elapsed,
        "elapsed_s": elapsed,
    }
    write_json(output_dir / "native_mtp.json", result)
    print("NATIVE_MTP_RESULT=" + json.dumps(result, sort_keys=True), flush=True)
    return result


def native_mtp_check_isolated(
    model_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--model-dir",
        str(model_dir),
        "--output-dir",
        str(output_dir),
        "--native-check-only",
    ]
    subprocess.run(command, check=True)
    return json.loads((output_dir / "native_mtp.json").read_text())


class Server:
    def __init__(
        self,
        profile: Profile,
        model_dir: Path,
        port: int,
        log_path: Path,
        ready_timeout_s: float,
    ) -> None:
        self.profile = profile
        self.model_dir = model_dir
        self.port = port
        self.log_path = log_path
        self.ready_timeout_s = ready_timeout_s
        self.process: subprocess.Popen[str] | None = None
        self._log_handle: Any = None

    def start(self) -> None:
        executable = shutil.which("vllm")
        if executable is None:
            raise RuntimeError("vllm executable was not found on PATH")

        command = [
            executable,
            "serve",
            str(self.model_dir),
            "--served-model-name",
            "qwen35-mtp-bench",
            "--enable-prefix-caching",
            "--no-async-scheduling",
            "--max-model-len",
            str(MODEL_MAX_LEN),
            "--max-num-batched-tokens",
            str(self.profile.max_num_batched_tokens),
            "--block-size",
            str(self.profile.block_size),
            "--port",
            str(self.port),
        ]
        if self.profile.enforce_eager:
            command.append("--enforce-eager")
        if self.profile.mode == "mtp":
            command.extend(["--speculative-config", SPEC_CONFIG])

        env = os.environ.copy()
        env.update(
            {
                "VLLM_METAL_SPEC_VERIFY_WINDOW": str(int(self.profile.verify_window)),
                "VLLM_METAL_DECODE_PIPELINE": str(int(self.profile.decode_pipeline)),
                "VLLM_METAL_GDN_LAZY_KERNELS": str(int(self.profile.gdn_lazy)),
            }
        )
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = self.log_path.open("w")
        print(
            "SERVER_START="
            + json.dumps(
                {
                    "profile": asdict(self.profile),
                    "command": command,
                    "port": self.port,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        self.process = subprocess.Popen(
            command,
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            start_new_session=True,
        )
        self._wait_ready()

    def _wait_ready(self) -> None:
        assert self.process is not None
        deadline = time.monotonic() + self.ready_timeout_s
        url = f"http://127.0.0.1:{self.port}/v1/models"
        started = time.monotonic()
        last_error = ""
        while time.monotonic() < deadline:
            code = self.process.poll()
            if code is not None:
                raise RuntimeError(
                    f"server {self.profile.name} exited with code {code}\n"
                    + tail(self.log_path)
                )
            try:
                with urllib.request.urlopen(url, timeout=1.0) as response:
                    if response.status == 200:
                        print(
                            f"Server {self.profile.name} ready in "
                            f"{time.monotonic() - started:.1f}s",
                            flush=True,
                        )
                        return
            except (OSError, urllib.error.URLError) as exc:
                last_error = str(exc)
            time.sleep(2)
        raise RuntimeError(
            f"server {self.profile.name} readiness timed out: {last_error}\n"
            + tail(self.log_path)
        )

    def stop(self) -> None:
        process = self.process
        self.process = None
        if process is None:
            return
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=10)
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None


class ServingBenchmarker:
    def __init__(
        self,
        model_dir: Path,
        port: int,
        repeats: int,
        request_timeout_s: float,
    ) -> None:
        self.model_dir = model_dir
        self.port = port
        self.repeats = repeats
        self.request_timeout_s = request_timeout_s
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir),
            trust_remote_code=True,
        )

    def prompt_for(self, prompt_tokens: int) -> tuple[str, int]:
        seed = (
            "Apple Silicon speculative decoding uses a shared cached prefix "
            "and a trained multi-token prediction head. "
        )
        text = seed
        while (
            len(self.tokenizer.encode(text, add_special_tokens=False))
            < prompt_tokens + 16
        ):
            text += seed
        ids = self.tokenizer.encode(text, add_special_tokens=False)[:prompt_tokens]
        prompt = self.tokenizer.decode(ids, skip_special_tokens=False)
        actual = len(self.tokenizer.encode(prompt, add_special_tokens=False))
        return prompt, actual

    async def run(
        self,
        profile: Profile,
        workload: Workload,
    ) -> dict[str, Any]:
        base_url = f"http://127.0.0.1:{self.port}"
        timeout = aiohttp.ClientTimeout(total=self.request_timeout_s)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(base_url + "/v1/models") as response:
                response.raise_for_status()
                served_model = (await response.json())["data"][0]["id"]

            prompt, prompt_len = self.prompt_for(workload.prompt_tokens)
            print(
                f"{profile.name}/{workload.name}: prompt={prompt_len} "
                f"output={workload.output_tokens} "
                f"concurrency={workload.concurrency} repeats={self.repeats}",
                flush=True,
            )

            async def one(tag: str) -> RequestFuncOutput:
                request = RequestFuncInput(
                    prompt=prompt,
                    api_url=base_url + "/v1/completions",
                    prompt_len=prompt_len,
                    output_len=workload.output_tokens,
                    model=served_model,
                    ignore_eos=True,
                    extra_body={"temperature": 0},
                )
                last_error = ""
                for attempt in range(2):
                    started = time.perf_counter()
                    try:
                        output = await asyncio.wait_for(
                            async_request_openai_completions(request, session),
                            timeout=self.request_timeout_s,
                        )
                    except TimeoutError as exc:
                        last_error = f"timed out: {exc}"
                    else:
                        if output.success:
                            print(
                                f"{profile.name}/{workload.name} {tag}: "
                                f"out={output.output_tokens} "
                                f"ttft_ms={output.ttft * 1000:.2f} "
                                f"e2e_s={time.perf_counter() - started:.2f}",
                                flush=True,
                            )
                            return output
                        last_error = output.error
                    if attempt == 0:
                        await asyncio.sleep(2)
                raise RuntimeError(
                    f"{profile.name}/{workload.name} {tag} failed: {last_error}"
                )

            await one("warmup")
            metrics_before = await fetch_spec_decode_metrics(base_url, session)

            repeat_metrics: list[dict[str, Any]] = []
            all_outputs: list[RequestFuncOutput] = []
            for repeat in range(self.repeats):
                outputs: list[RequestFuncOutput] = []
                started = time.perf_counter()
                for offset in range(
                    0,
                    workload.requests,
                    workload.concurrency,
                ):
                    wave_size = min(
                        workload.concurrency,
                        workload.requests - offset,
                    )
                    wave = await asyncio.gather(
                        *(
                            one(
                                f"r{repeat + 1}:"
                                f"{offset + index + 1}/{workload.requests}"
                            )
                            for index in range(wave_size)
                        )
                    )
                    outputs.extend(wave)
                elapsed = time.perf_counter() - started
                all_outputs.extend(outputs)

                total_input = sum(output.prompt_len for output in outputs)
                total_output = sum(output.output_tokens for output in outputs)
                repeat_metrics.append(
                    {
                        "repeat": repeat + 1,
                        "elapsed_s": elapsed,
                        "completed_output_tokens": total_output,
                        "output_throughput_tok_s": total_output / elapsed,
                        "total_token_throughput_tok_s": (total_input + total_output)
                        / elapsed,
                        "mean_ttft_ms": statistics.mean(
                            output.ttft * 1000 for output in outputs
                        ),
                        "median_ttft_ms": statistics.median(
                            output.ttft * 1000 for output in outputs
                        ),
                        "mean_e2e_ms": statistics.mean(
                            output.latency * 1000 for output in outputs
                        ),
                    }
                )

            metrics_after = await fetch_spec_decode_metrics(base_url, session)
            draft_tokens = 0
            accepted_tokens = 0
            if metrics_before is not None and metrics_after is not None:
                draft_tokens = (
                    metrics_after.num_draft_tokens - metrics_before.num_draft_tokens
                )
                accepted_tokens = (
                    metrics_after.num_accepted_tokens
                    - metrics_before.num_accepted_tokens
                )

            output_rates = [
                metric["output_throughput_tok_s"] for metric in repeat_metrics
            ]
            total_rates = [
                metric["total_token_throughput_tok_s"] for metric in repeat_metrics
            ]
            result = {
                "profile": profile.name,
                "mode": profile.mode,
                "profile_config": asdict(profile),
                "workload": workload.name,
                "workload_config": asdict(workload),
                "requests_per_repeat": workload.requests,
                "repeats": self.repeats,
                "concurrency": workload.concurrency,
                "prompt_tokens_per_request": prompt_len,
                "output_tokens_per_request": workload.output_tokens,
                "output_throughput_tok_s": statistics.median(output_rates),
                "output_throughput_min_tok_s": min(output_rates),
                "output_throughput_max_tok_s": max(output_rates),
                "output_throughput_cv": (
                    statistics.pstdev(output_rates) / statistics.mean(output_rates)
                    if len(output_rates) > 1
                    else 0.0
                ),
                "total_token_throughput_tok_s": statistics.median(total_rates),
                "mean_ttft_ms": statistics.median(
                    metric["mean_ttft_ms"] for metric in repeat_metrics
                ),
                "median_ttft_ms": statistics.median(
                    output.ttft * 1000 for output in all_outputs
                ),
                "mean_e2e_ms": statistics.median(
                    metric["mean_e2e_ms"] for metric in repeat_metrics
                ),
                "mtp_draft_tokens": draft_tokens,
                "mtp_accepted_tokens": accepted_tokens,
                "mtp_acceptance_rate": (
                    accepted_tokens / draft_tokens if draft_tokens else None
                ),
                "repeat_metrics": repeat_metrics,
            }
            print(
                "BENCHMARK_RESULT=" + json.dumps(result, sort_keys=True),
                flush=True,
            )
            return result


def aggregate(
    output_dir: Path,
    rows: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    native: dict[str, Any],
) -> tuple[dict[str, Any], str, bool]:
    by_workload: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_workload[row["workload"]].append(row)

    baseline_rates: dict[str, float] = {}
    thermal_drift: dict[str, float] = {}
    for workload, workload_rows in by_workload.items():
        refs = {
            row["profile"]: row["output_throughput_tok_s"]
            for row in workload_rows
            if row["profile"] in {"baseline_ref", "baseline_repeat"}
        }
        if refs:
            baseline_rates[workload] = statistics.median(refs.values())
        if {"baseline_ref", "baseline_repeat"} <= refs.keys():
            thermal_drift[workload] = (
                refs["baseline_repeat"] / refs["baseline_ref"] - 1.0
            )

    for row in rows:
        baseline = baseline_rates.get(row["workload"])
        row["speedup_vs_baseline"] = (
            row["output_throughput_tok_s"] / baseline if baseline else None
        )

    expected_workloads = set(baseline_rates)
    profile_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["mode"] == "mtp" and row["speedup_vs_baseline"] is not None:
            profile_rows[row["profile"]].append(row)

    ranking: list[dict[str, Any]] = []
    incomplete_profiles: list[dict[str, Any]] = []
    for profile, measured_rows in profile_rows.items():
        speedups = [row["speedup_vs_baseline"] for row in measured_rows]
        completed = {row["workload"] for row in measured_rows}
        item = {
            "profile": profile,
            "geomean_speedup": statistics.geometric_mean(speedups),
            "minimum_speedup": min(speedups),
            "maximum_speedup": max(speedups),
            "workloads_completed": len(completed),
            "complete": completed == expected_workloads,
        }
        (ranking if item["complete"] else incomplete_profiles).append(item)

    ranking.sort(key=lambda item: item["geomean_speedup"], reverse=True)
    incomplete_profiles.sort(
        key=lambda item: (
            item["workloads_completed"],
            item["geomean_speedup"],
        ),
        reverse=True,
    )

    best_by_workload: dict[str, dict[str, Any]] = {}
    for workload, workload_rows in by_workload.items():
        mtp_rows = [row for row in workload_rows if row["mode"] == "mtp"]
        if mtp_rows:
            best = max(mtp_rows, key=lambda row: row["output_throughput_tok_s"])
            best_by_workload[workload] = {
                "profile": best["profile"],
                "output_throughput_tok_s": best["output_throughput_tok_s"],
                "speedup_vs_baseline": best["speedup_vs_baseline"],
                "mtp_acceptance_rate": best["mtp_acceptance_rate"],
            }

    summary = {
        "best_overall_profile": ranking[0] if ranking else None,
        "ranking": ranking,
        "incomplete_profiles": incomplete_profiles,
        "best_by_workload": best_by_workload,
        "baseline_output_throughput_tok_s": baseline_rates,
        "thermal_drift": thermal_drift,
        "native_mlx_lm_mtp": native,
        "errors": errors,
        "rows": rows,
    }
    write_json(output_dir / "matrix_summary.json", summary)

    lines = [
        "# Qwen3.5 native MTP speed matrix",
        "",
        "Primary score: median output tokens/s over repeated runs. Speedup uses "
        "the median of the cold and hot tuned baseline for the same workload.",
        "",
        "## Overall MTP profile ranking",
        "",
        "| Rank | Profile | Geomean speedup | Worst | Best | Workloads |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for index, item in enumerate(ranking, 1):
        lines.append(
            f"| {index} | {item['profile']} | "
            f"{item['geomean_speedup']:.3f}x | "
            f"{item['minimum_speedup']:.3f}x | "
            f"{item['maximum_speedup']:.3f}x | "
            f"{item['workloads_completed']} |"
        )
    if not ranking:
        lines.append("| — | No complete MTP profile | — | — | — | — |")

    lines.extend(
        [
            "",
            "## All measurements",
            "",
            "| Workload | Profile | tok/s | Speedup | TTFT ms | E2E ms | "
            "Acceptance | CV |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for workload in sorted(by_workload):
        workload_rows = sorted(
            by_workload[workload],
            key=lambda row: row["output_throughput_tok_s"],
            reverse=True,
        )
        for row in workload_rows:
            acceptance = (
                f"{100 * row['mtp_acceptance_rate']:.1f}%"
                if row["mtp_acceptance_rate"] is not None
                else "n/a"
            )
            speedup = (
                f"{row['speedup_vs_baseline']:.3f}x"
                if row["speedup_vs_baseline"] is not None
                else "n/a"
            )
            lines.append(
                f"| {workload} | {row['profile']} | "
                f"{row['output_throughput_tok_s']:.2f} | {speedup} | "
                f"{row['mean_ttft_ms']:.1f} | {row['mean_e2e_ms']:.1f} | "
                f"{acceptance} | {100 * row['output_throughput_cv']:.1f}% |"
            )

    lines.extend(
        [
            "",
            "## Baseline drift check",
            "",
            "| Workload | Hot-vs-cold drift |",
            "|---|---:|",
        ]
    )
    for workload, drift in sorted(thermal_drift.items()):
        lines.append(f"| {workload} | {100 * drift:+.1f}% |")

    if errors:
        lines.extend(
            [
                "",
                "## Failed cases",
                "",
                "| Profile | Workload | Error |",
                "|---|---|---|",
            ]
        )
        for error in errors:
            text = str(error["error"]).replace("|", "\\|").replace("\n", " ")
            lines.append(
                f"| {error['profile']} | {error.get('workload', 'server')} | "
                f"{text[:500]} |"
            )

    report = "\n".join(lines) + "\n"
    (output_dir / "matrix_summary.md").write_text(report)
    return summary, report, bool(ranking)


async def async_main(args: argparse.Namespace) -> int:
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)
    (output_dir / "results").mkdir(exist_ok=True)
    write_json(output_dir / "profiles.json", [asdict(item) for item in PROFILES])
    write_json(output_dir / "workloads.json", [asdict(item) for item in WORKLOADS])

    native = native_mtp_check_isolated(args.model_dir, output_dir)
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    benchmarker = ServingBenchmarker(
        args.model_dir,
        args.port,
        args.repeats,
        args.request_timeout_s,
    )
    for profile in PROFILES:
        log_path = output_dir / "logs" / f"{profile.name}.log"
        server = Server(
            profile,
            args.model_dir,
            args.port,
            log_path,
            args.server_ready_timeout_s,
        )
        try:
            server.start()
        except Exception as exc:
            errors.append(
                {
                    "profile": profile.name,
                    "workload": "server",
                    "error": str(exc),
                }
            )
            print(f"::warning::{profile.name} server failed: {exc}", flush=True)
            server.stop()
            continue

        try:
            for workload in WORKLOADS:
                try:
                    result = await benchmarker.run(profile, workload)
                except Exception as exc:
                    errors.append(
                        {
                            "profile": profile.name,
                            "workload": workload.name,
                            "error": str(exc),
                        }
                    )
                    print(
                        f"::warning::{profile.name}/{workload.name} failed: {exc}",
                        flush=True,
                    )
                    continue
                rows.append(result)
                write_json(
                    output_dir / "results" / f"{profile.name}__{workload.name}.json",
                    result,
                )
        finally:
            server.stop()
            print(f"===== {profile.name} server log tail =====", flush=True)
            print(tail(log_path), flush=True)

    summary, report, has_complete_profile = aggregate(
        output_dir,
        rows,
        errors,
        native,
    )
    print(report, flush=True)
    print(
        "FINAL_MATRIX_SUMMARY="
        + json.dumps(
            {
                "best_overall_profile": summary["best_overall_profile"],
                "best_by_workload": summary["best_by_workload"],
                "errors": len(errors),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if has_complete_profile else 2


def main() -> int:
    args = parse_args()
    if args.native_check_only:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        native_mtp_check(args.model_dir, args.output_dir)
        return 0
    try:
        return asyncio.run(async_main(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
