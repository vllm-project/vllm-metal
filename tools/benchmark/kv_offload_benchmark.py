# SPDX-License-Identifier: Apache-2.0
"""KV offloading serving benchmark.

Measures what KV offloading is for: serving a prefix that the wired Metal KV
cache no longer holds. Runs `vllm bench serve` against a freshly started
server for each variant and emits structured JSON that can be attached to a
PR.

Four workloads, each run with offloading off and on:

  no-reuse     Distinct prompts. Offloading cannot help here. Reported so a
               regression is visible, not to show a win.
  reuse-fits   Shared prefixes that fit the KV cache. Prefix caching alone
               already handles this, so a wash is the expected result.
  spill_revisit Shared prefixes larger than the KV cache, served twice
               against one server. The first pass is untimed and overflows
               the cache; the second is measured, and by then the early
               prefixes have been evicted by the later ones. A hit on the
               second pass is a genuine restore.

               Two passes are required, not one. vLLM's prefix_repetition
               dataset emits every request for a prefix consecutively, so a
               single pass never revisits a prefix after other traffic has
               evicted it: it measures first-touch plus consecutive hits and
               reports roughly no difference no matter how the tier behaves.
  restart      The same traffic served by a second, fresh process. Without a
               disk tier nothing survives a restart. Both variants are
               restarted identically.

no-reuse and reuse-fits are controls. If either shows offloading ahead by
more than noise the benchmark is measuring something else, most often
admission queueing rather than prefill, and the run is reported as suspect.

One server process per variant, as in the other benchmarks here, so cache
state cannot leak between baseline and treatment.

Example:

    python tools/benchmark/kv_offload_benchmark.py \\
        --model mlx-community/Qwen2.5-32B-Instruct-4bit \\
        --kv-cache-tokens 124464 \\
        --out kv-offload-bench.json

`--kv-cache-tokens` sizes the spilling workload. Take it from the server's
own startup log line (`max_tokens_cached=...`), or leave it unset and the
script starts a server once to read it.

The spilling workload needs a model whose weights leave a KV cache small
enough to overflow. A 1.5B model at a normal memory fraction holds well over
100k tokens of KV on a 128 GiB machine and cannot be overflowed by any
reasonable prompt set; it will report a wash. That is a property of the
machine, not of offloading.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

READY_TIMEOUT = 1800.0
MAX_TOKENS_RE = re.compile(r"max_tokens_cached=(\d+)")
BREAKDOWN_RE = re.compile(
    r"per_block_bytes=(\d+), num_blocks=(\d+), max_tokens_cached=(\d+)"
)

# Leave this much of the filesystem alone. The disk tier writes one file per
# offloaded block and a spilling workload can ask for tens of GB.
DISK_HEADROOM_GB = 15.0

# Leave this much RAM to the rest of the machine. Resident set is weights plus
# the wired KV cache plus the host offload pool, and the host pool is pageable:
# overcommit it and the machine swaps or dies rather than the server failing
# cleanly. Sized so a 128 GB host is warned off --memory-fraction 0.8 with a
# 16 GiB pool (~93 GB projected), which is enough to take the machine down
# while leaving the OS, a browser and an editor running.
RAM_HEADROOM_GB = 40.0

# TTFT only means "time to prefill" while the server is not saturated. Firing
# every request at once measures admission queueing instead, and anything that
# raises throughput then looks like a prefill win.
DEFAULT_CONCURRENCY = 4

# Above this the controls are treated as failing and the run is flagged.
CONTROL_TOLERANCE = 1.25


def _package_version(package_name: str) -> str | None:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment_metadata(args: argparse.Namespace) -> dict[str, Any]:
    """Enough detail to make the JSON auditable by someone who was not here."""
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "vllm": _package_version("vllm"),
        "vllm-metal": _package_version("vllm-metal"),
        "mlx": _package_version("mlx"),
        "cpu_count": os.cpu_count(),
        "load1_at_start": round(os.getloadavg()[0], 2),
        "free_disk_gb": round(shutil.disk_usage(work_root()).free / (1 << 30), 1),
        "memory_fraction": args.memory_fraction,
        "baseline_memory_fraction": args.baseline_memory_fraction or None,
        "offload_size_gib": args.offload_size_gib,
        "disk_tier": not args.no_disk_tier,
        "max_concurrency": args.concurrency,
        "num_prompts": args.num_prompts,
        "repeats": args.repeats,
    }


def total_ram_gb() -> float:
    """Physical RAM, or 0.0 where it cannot be read (non-macOS hosts)."""
    if sys.platform != "darwin":
        return 0.0
    try:
        out = subprocess.run(
            ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=False
        )
    except (FileNotFoundError, OSError):
        return 0.0
    return int(out.stdout.strip()) / (1 << 30) if out.stdout.strip() else 0.0


def preflight_memory(args: argparse.Namespace) -> list[str]:
    """Refuse configurations that would take the machine down with them.

    Reported as warnings rather than enforced: the caller knows what else the
    machine is doing, and this cannot see it.
    """
    warnings: list[str] = []
    ram = total_ram_gb()
    if not ram:
        return warnings
    # The wired cache is sized off the recommended working set, itself about
    # 75% of RAM, so the fraction lands on that rather than on RAM directly.
    projected = ram * 0.75 * args.memory_fraction + args.offload_size_gib
    if projected > ram - RAM_HEADROOM_GB:
        warnings.append(
            f"--memory-fraction {args.memory_fraction} plus a "
            f"{args.offload_size_gib:.0f} GiB host pool projects to "
            f"~{projected:.0f} GB resident on a {ram:.0f} GB machine. The host "
            "pool is pageable, so this swaps or OOMs rather than failing "
            "cleanly. Lower --memory-fraction or --offload-size-gib."
        )
    return warnings


def work_root() -> str:
    """Where block files actually land. Checking free space on the cwd is
    wrong when TMPDIR is on another volume."""
    return tempfile.gettempdir()


def free_port() -> int:
    """OS-assigned. A fixed port collides with a server from an earlier run
    that has not finished releasing it."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class ServerProcess:
    """One `vllm serve` process, torn down on exit."""

    def __init__(
        self,
        args: argparse.Namespace,
        offload: bool,
        store_dir: Path | None,
        log_path: Path,
        fraction: float | None = None,
    ) -> None:
        self.args = args
        self.offload = offload
        self.store_dir = store_dir
        self.log_path = log_path
        # Explicit memory fraction, overriding the arm-based choice in _env.
        self.fraction = fraction
        self.port = free_port()
        self.proc: subprocess.Popen | None = None
        self.max_tokens_cached = 0
        self.bytes_per_token = 0.0

    def _command(self) -> list[str]:
        cmd = [
            "vllm",
            "serve",
            self.args.model,
            "--port",
            str(self.port),
            "--max-model-len",
            str(self.args.max_model_len),
        ]
        if self.args.revision:
            cmd += ["--revision", self.args.revision]
        if self.offload:
            cmd += [
                "--kv-offloading-backend",
                "native",
                "--kv-offloading-size",
                str(self.args.offload_size_gib),
            ]
            if self.store_dir is not None:
                extra = {
                    "secondary_tiers": [{"type": "fs", "root_dir": str(self.store_dir)}]
                }
                cmd += [
                    "--kv-transfer-config",
                    json.dumps({"kv_connector_extra_config": extra}),
                ]
        return cmd + self.args.serve_arg

    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        env["VLLM_METAL_USE_PAGED_ATTENTION"] = "1"
        # The baseline can be given a larger wired cache so both arms hold
        # the same total KV bytes. Without that, an offload run with a host
        # pool on top of its cache simply has more memory, and the result
        # cannot separate "offloading works" from "more memory works".
        fraction = self.args.memory_fraction
        if not self.offload and self.args.baseline_memory_fraction:
            fraction = self.args.baseline_memory_fraction
        if self.fraction is not None:
            fraction = self.fraction
        env["VLLM_METAL_MEMORY_FRACTION"] = str(fraction)
        # Block filenames are content hashes and hash seeding is per process.
        # Without this a restarted server cannot find what the previous one
        # wrote, and the restart workload measures nothing.
        env["PYTHONHASHSEED"] = "0"
        return env

    def __enter__(self) -> ServerProcess:
        with open(self.log_path, "w") as log:
            self.proc = subprocess.Popen(
                self._command(),
                stdout=log,
                stderr=subprocess.STDOUT,
                env=self._env(),
            )
        try:
            self._wait_ready()
        except BaseException:
            # __exit__ does not run when __enter__ raises, and a server left
            # behind holds GPU memory and a port for the rest of the run.
            self.__exit__(None, None, None)
            raise
        return self

    def _wait_ready(self) -> None:
        deadline = time.monotonic() + READY_TIMEOUT
        url = f"http://localhost:{self.port}/health"
        while time.monotonic() < deadline:
            if self.proc is not None and self.proc.poll() is not None:
                raise RuntimeError(
                    f"server exited early (rc={self.proc.returncode})\n"
                    f"{self.log_path.read_text()[-3000:]}"
                )
            with contextlib.suppress(urllib.error.URLError, OSError):
                with urllib.request.urlopen(url, timeout=5) as resp:
                    if resp.status == 200:
                        text = self.log_path.read_text()
                        found = MAX_TOKENS_RE.search(text)
                        if found:
                            self.max_tokens_cached = int(found.group(1))
                        shape = BREAKDOWN_RE.search(text)
                        if shape:
                            block_bytes, blocks, tokens = (
                                int(g) for g in shape.groups()
                            )
                            if tokens:
                                self.bytes_per_token = block_bytes * blocks / tokens
                        return
            time.sleep(2)
        raise RuntimeError(f"server not ready within {READY_TIMEOUT:.0f}s")

    def __exit__(self, *exc: object) -> None:
        if self.proc is None:
            return
        self.proc.terminate()
        try:
            self.proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=30)
        # Do not hand the next server a port still in TIME_WAIT.
        time.sleep(5)


def bench_serve(
    args: argparse.Namespace, port: int, dataset: list[str], out_json: Path
) -> dict[str, Any]:
    cmd = [
        "vllm",
        "bench",
        "serve",
        "--backend",
        "openai",
        "--base-url",
        f"http://localhost:{port}",
        "--model",
        args.model,
        "--percentile-metrics",
        "ttft,tpot,e2el",
        "--metric-percentiles",
        "50,99",
        "--request-rate",
        "inf",
        "--max-concurrency",
        str(args.concurrency),
        "--seed",
        "0",
        "--save-result",
        "--result-filename",
        str(out_json),
        *dataset,
    ]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=args.bench_timeout
    )
    if not out_json.exists():
        raise RuntimeError(
            "vllm bench serve produced no result file\n"
            f"stdout:\n{proc.stdout[-2000:]}\nstderr:\n{proc.stderr[-2000:]}"
        )
    return json.loads(out_json.read_text())


def affordable_spill(
    args: argparse.Namespace, kv_tokens: int, bytes_per_token: float
) -> tuple[float, str | None]:
    """How far past the KV cache the disk can actually take us.

    The disk tier holds whatever the host pool cannot, so a 1.4x spill on a
    large cache can ask for more block files than the filesystem has room
    for. Scale the target down to fit rather than filling the disk mid-run.
    """
    if not bytes_per_token:
        return 1.4, None
    free_gb = shutil.disk_usage(work_root()).free / (1 << 30) - DISK_HEADROOM_GB
    if free_gb <= 0:
        return 1.0, "no disk headroom; the spilling workload cannot run here"
    # Blocks beyond the host pool land on disk.
    budget_bytes = free_gb * (1 << 30) + args.offload_size_gib * (1 << 30)
    max_tokens = budget_bytes / bytes_per_token
    affordable = max_tokens / kv_tokens if kv_tokens else 1.4
    if affordable < 1.4:
        return max(1.0, affordable * 0.9), (
            f"disk allows only ~{affordable:.2f}x the KV cache, not 1.4x; "
            "the spilling workload is scaled down to fit"
        )
    return 1.4, None


def affordable_no_reuse(
    args: argparse.Namespace, bytes_per_token: float
) -> tuple[int, str | None]:
    """How many distinct prompts the disk can take.

    The no-reuse workload writes every evicted block and reads none back, so
    its store is close to the whole working set less the host pool. On a 32B
    model with 96 prompts that was 84 GB. Scale the prompt count down to fit
    rather than filling the disk on a control.
    """
    if not bytes_per_token:
        return args.num_prompts, None
    per_request = args.prefix_len + args.suffix_len
    free_gb = shutil.disk_usage(work_root()).free / (1 << 30) - DISK_HEADROOM_GB
    budget_bytes = max(free_gb, 0.0) * (1 << 30) + args.offload_size_gib * (1 << 30)
    max_prompts = int(budget_bytes / (per_request * bytes_per_token))
    if max_prompts >= args.num_prompts:
        return args.num_prompts, None
    prompts = max(8, max_prompts)
    return prompts, (
        f"disk allows ~{max_prompts} distinct prompts, not {args.num_prompts}; "
        f"the no-reuse workload runs {prompts}"
    )


def workloads(
    args: argparse.Namespace, kv_tokens: int, bytes_per_token: float = 0.0
) -> dict[str, dict]:
    """Prompt sets, with the spilling one sized off the real KV cache."""
    per_request = args.prefix_len + args.suffix_len
    multiple, note = affordable_spill(args, kv_tokens, bytes_per_token)
    if note:
        print(f"NOTE: {note}", file=sys.stderr)
    no_reuse_prompts, note = affordable_no_reuse(args, bytes_per_token)
    if note:
        print(f"NOTE: {note}", file=sys.stderr)
    # Comfortably over the cache, without so much churn that the run is
    # dominated by writing blocks that are never read back.
    spill_prefixes = max(2, round(multiple * kv_tokens / per_request))
    # Every prefix needs at least this many prompts, or there is no
    # repetition to cache. vllm bench serve also rejects num_prefixes >
    # num_requests outright.
    per_prefix = 3
    # Exactly two per prefix, never more. prefix_repetition emits a prefix's
    # requests consecutively, so in the measured pass only the first is a
    # restore or a recompute and the rest are warm hits. More than two per
    # prefix lets warm hits become the median and hides a working tier.
    spill_prompts = spill_prefixes * 2
    # The fits workload only has to stay inside the cache, not fill it. A
    # model with a large KV cache would otherwise derive hundreds of
    # prefixes and need thousands of prompts to serve them.
    fits_prefixes = max(2, min(spill_prefixes // 3, 24))
    fits_prompts = max(args.num_prompts, fits_prefixes * per_prefix)
    prefix_args = [
        "--prefix-repetition-prefix-len",
        str(args.prefix_len),
        "--prefix-repetition-suffix-len",
        str(args.suffix_len),
        "--prefix-repetition-output-len",
        "64",
    ]
    return {
        "no_reuse": {
            "control": True,
            "restart": False,
            "second_pass": False,
            "why": "distinct prompts; offloading cannot help",
            "args": [
                "--dataset-name",
                "random",
                "--num-prompts",
                str(no_reuse_prompts),
                "--random-input-len",
                str(per_request),
                "--random-output-len",
                "64",
                "--random-prefix-len",
                "0",
            ],
        },
        "reuse_fits": {
            "control": True,
            "restart": False,
            "second_pass": False,
            "why": f"{fits_prefixes} shared prefixes over {fits_prompts} "
            "prompts, inside the KV cache",
            "args": [
                "--dataset-name",
                "prefix_repetition",
                "--num-prompts",
                str(args.num_prompts),
                "--prefix-repetition-num-prefixes",
                str(fits_prefixes),
                *prefix_args,
            ],
        },
        "spill_revisit": {
            "control": False,
            "restart": False,
            "second_pass": True,
            "why": f"{spill_prefixes} shared prefixes over {spill_prompts} "
            f"prompts, ~{multiple:.2f}x the KV cache",
            "args": [
                "--dataset-name",
                "prefix_repetition",
                "--num-prompts",
                str(spill_prompts),
                "--prefix-repetition-num-prefixes",
                str(spill_prefixes),
                *prefix_args,
            ],
        },
        "restart": {
            "control": False,
            "restart": True,
            "second_pass": False,
            "why": "same traffic, second process; nothing survives without a disk tier",
            "args": [
                "--dataset-name",
                "prefix_repetition",
                "--num-prompts",
                str(args.num_prompts),
                "--prefix-repetition-num-prefixes",
                str(max(2, fits_prefixes)),
                *prefix_args,
            ],
        },
    }


METRICS = (
    "median_ttft_ms",
    "p99_ttft_ms",
    "mean_ttft_ms",
    "median_tpot_ms",
    "median_e2el_ms",
    "output_throughput",
    "request_throughput",
)


def one_sample(
    args: argparse.Namespace, spec: dict, offload: bool, work_dir: Path, tag: str
) -> dict[str, Any]:
    """One measured run. The restart workload populates, stops, then measures
    against a second process. Both variants are restarted the same way."""
    store = work_dir / f"{tag}-store" if offload and not args.no_disk_tier else None
    if store is not None:
        store.mkdir(parents=True, exist_ok=True)
    try:
        if spec.get("second_pass"):
            # Untimed: fill the cache past capacity so the measured pass has
            # to restore rather than recompute from a warm cache.
            with ServerProcess(
                args, offload, store, work_dir / f"{tag}.pass1.log"
            ) as srv:
                bench_serve(
                    args, srv.port, spec["args"], work_dir / f"{tag}.pass1.json"
                )
                result = bench_serve(
                    args, srv.port, spec["args"], work_dir / f"{tag}.json"
                )
                sample = {k: result[k] for k in METRICS if k in result}
                sample["max_tokens_cached"] = srv.max_tokens_cached
                if store is not None:
                    sample["store_bytes"] = sum(
                        f.stat().st_size for f in store.rglob("*") if f.is_file()
                    )
                return sample
        if spec["restart"]:
            with ServerProcess(
                args, offload, store, work_dir / f"{tag}.warm.log"
            ) as srv:
                bench_serve(args, srv.port, spec["args"], work_dir / f"{tag}.warm.json")
        with ServerProcess(args, offload, store, work_dir / f"{tag}.log") as srv:
            result = bench_serve(args, srv.port, spec["args"], work_dir / f"{tag}.json")
            sample = {k: result[k] for k in METRICS if k in result}
            sample["max_tokens_cached"] = srv.max_tokens_cached
            if store is not None:
                sample["store_bytes"] = sum(
                    f.stat().st_size for f in store.rglob("*") if f.is_file()
                )
            return sample
    finally:
        # Every sample starts from an empty store, so the next one cannot
        # restore what this one wrote. Keeping them also fills the disk: a
        # no-reuse sample writes the whole working set and reads none of it
        # back, which was 84 GB on a 32B model.
        if store is not None:
            shutil.rmtree(store, ignore_errors=True)


def discover_kv_geometry(args: argparse.Namespace, work_dir: Path) -> tuple[int, float]:
    # Probe at the treatment fraction, not the baseline's. With
    # --baseline-memory-fraction the baseline cache is the larger one, and a
    # control sized to it would overflow the treatment cache, which turns a
    # control into a workload offloading can help.
    with ServerProcess(
        args, False, None, work_dir / "probe.log", fraction=args.memory_fraction
    ) as srv:
        if not srv.max_tokens_cached:
            raise RuntimeError(
                "could not read max_tokens_cached from the server log; pass "
                "--kv-cache-tokens explicitly"
            )
        return srv.max_tokens_cached, srv.bytes_per_token


def median(rows: list[dict], key: str) -> float | None:
    vals = [r[key] for r in rows if key in r]
    return statistics.median(vals) if vals else None


def spread_pct(rows: list[dict], key: str) -> float | None:
    """IQR as a percentage of the median. A wide spread means the number is
    noise, and saying so is more useful than quoting it to one decimal."""
    vals = [r[key] for r in rows if key in r]
    if len(vals) < 2:
        return None
    quarts = statistics.quantiles(vals, n=4)
    mid = statistics.median(vals)
    return round(100 * (quarts[2] - quarts[0]) / mid, 1) if mid else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="KV offloading serving benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument(
        "--memory-fraction",
        type=float,
        default=0.8,
        help="VLLM_METAL_MEMORY_FRACTION (default: 0.8, as in docs/CONTRIBUTING.md)",
    )
    parser.add_argument(
        "--offload-size-gib",
        type=float,
        default=16.0,
        help="--kv-offloading-size for the offload variant (default: 16)",
    )
    parser.add_argument(
        "--baseline-memory-fraction",
        type=float,
        default=0.0,
        help="VLLM_METAL_MEMORY_FRACTION for the no-offload arm only. Set it "
        "so the baseline's wired KV cache equals the offload arm's cache "
        "plus its host pool, and the comparison isolates offloading from "
        "simply having more memory. 0 (default) uses --memory-fraction for "
        "both.",
    )
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--prefix-len", type=int, default=3072)
    parser.add_argument("--suffix-len", type=int, default=512)
    parser.add_argument("--num-prompts", type=int, default=96)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help="--max-concurrency for bench serve; keeps TTFT a prefill "
        "measurement rather than a queueing one (default: 4)",
    )
    parser.add_argument(
        "--kv-cache-tokens",
        type=int,
        help="KV cache capacity in tokens, used to size the spilling "
        "workload. Read from a probe server if omitted.",
    )
    parser.add_argument(
        "--workload",
        action="append",
        default=[],
        help="run only these workloads (repeatable)",
    )
    parser.add_argument(
        "--serve-arg",
        action="append",
        default=[],
        help="extra argument passed through to vllm serve (repeatable)",
    )
    parser.add_argument(
        "--bench-timeout",
        type=int,
        default=7200,
        help="seconds allowed per bench serve call (default: 7200). A 70B "
        "serving 96 distinct 3584-token prompts needs more than this.",
    )
    parser.add_argument("--out", default="kv-offload-benchmark.json")
    parser.add_argument(
        "--no-disk-tier",
        action="store_true",
        help="Offload to the host pool only, with no fs secondary tier. "
        "Isolates what the CPU tier is worth on its own: with it set, a "
        "restart has nothing to restore from, and a spill is bounded by the "
        "pool rather than by the disk.",
    )
    parser.add_argument("--keep-logs", action="store_true")
    args = parser.parse_args(argv)

    for warning in preflight_memory(args):
        print(f"WARNING: {warning}", file=sys.stderr)

    load1 = os.getloadavg()[0]
    if load1 > 2.0:
        print(
            f"WARNING: load average is {load1:.1f}. Latencies will be a lower "
            "bound on this hardware. Run on an otherwise idle machine.",
            file=sys.stderr,
        )

    work_dir = Path(tempfile.mkdtemp(prefix="kv-offload-bench-"))
    print(f"logs: {work_dir}", flush=True)

    kv_tokens = args.kv_cache_tokens
    bytes_per_token = 0.0
    if kv_tokens is None:
        print("probing KV cache capacity...", flush=True)
        kv_tokens, bytes_per_token = discover_kv_geometry(args, work_dir)
    print(f"KV cache holds {kv_tokens} tokens", flush=True)
    if not bytes_per_token:
        print(
            "WARNING: --kv-cache-tokens skips the probe, so bytes per token "
            "are unknown and the disk budget cannot be estimated. The "
            "spilling workload runs at 1.4x the cache and the no-reuse "
            "workload at full size; make sure the disk has room.",
            file=sys.stderr,
        )

    if args.offload_size_gib <= 0:
        if not bytes_per_token:
            parser.error(
                "cannot derive --offload-size-gib without a probe; pass it "
                "explicitly alongside --kv-cache-tokens"
            )
        # Above the KV cache, not below it. A pool smaller than the cache
        # loses blocks to its own LRU before the cache would re-request
        # them, so restores mostly miss (see warn_if_pool_undersized).
        args.offload_size_gib = round(1.3 * kv_tokens * bytes_per_token / (1 << 30), 1)
        print(
            f"host pool {args.offload_size_gib} GiB "
            f"(1.3x the {kv_tokens * bytes_per_token / (1 << 30):.1f} GB KV cache)",
            flush=True,
        )
        # The first preflight ran before the pool size was known.
        for warning in preflight_memory(args):
            print(f"WARNING: {warning}", file=sys.stderr)
    specs = workloads(args, kv_tokens, bytes_per_token)
    if args.workload:
        specs = {k: v for k, v in specs.items() if k in args.workload}
        if not specs:
            parser.error(
                f"no such workload; choose from {list(workloads(args, kv_tokens))}"
            )

    record: dict[str, Any] = {
        "benchmark": "kv_offload",
        "model": args.model,
        "kv_cache_tokens": kv_tokens,
        "environment": environment_metadata(args),
        "workloads": {k: v["why"] for k, v in specs.items()},
        "samples": {k: {"off": [], "on": []} for k in specs},
        "failures": [],
    }
    out_path = Path(args.out)

    try:
        for rep in range(args.repeats):
            for name, spec in specs.items():
                # Rotated: running every baseline then every treatment
                # measures thermal drift as much as it measures the change.
                order = ("off", "on") if rep % 2 == 0 else ("on", "off")
                for which in order:
                    tag = f"{name}-{which}-r{rep}"
                    print(f"  {tag}", flush=True)
                    try:
                        sample = one_sample(args, spec, which == "on", work_dir, tag)
                        record["samples"][name][which].append(sample)
                    except Exception as exc:  # keep the matrix going
                        print(f"  FAILED {tag}: {exc}", file=sys.stderr)
                        record["failures"].append(
                            {"tag": tag, "error": str(exc)[:2000]}
                        )
                    # Written every sample: a run that dies late is still
                    # worth what it already measured.
                    out_path.write_text(json.dumps(record, indent=2))
    finally:
        record["load1_at_end"] = round(os.getloadavg()[0], 2)
        out_path.write_text(json.dumps(record, indent=2))
        if not args.keep_logs:
            shutil.rmtree(work_dir, ignore_errors=True)

    summary: dict[str, Any] = {}
    suspect: list[str] = []
    print(f"\n{'workload':16} {'TTFT off':>11} {'TTFT on':>11} {'ratio':>7}  spread")
    for name, spec in specs.items():
        off = record["samples"][name]["off"]
        on = record["samples"][name]["on"]
        off_ttft, on_ttft = median(off, "median_ttft_ms"), median(on, "median_ttft_ms")
        if off_ttft is None or on_ttft is None:
            print(f"{name:16} {'no data':>11}")
            continue
        ratio = off_ttft / on_ttft if on_ttft else float("nan")
        off_spread, on_spread = (
            spread_pct(off, "median_ttft_ms"),
            spread_pct(on, "median_ttft_ms"),
        )
        summary[name] = {
            "median_ttft_ms": {"off": off_ttft, "on": on_ttft},
            "ttft_ratio": round(ratio, 2),
            "iqr_pct": {"off": off_spread, "on": on_spread},
            "output_throughput": {
                "off": median(off, "output_throughput"),
                "on": median(on, "output_throughput"),
            },
            "is_control": spec["control"],
        }
        if spec["control"] and ratio > CONTROL_TOLERANCE:
            suspect.append(name)
        print(
            f"{name:16} {off_ttft:11.1f} {on_ttft:11.1f} {ratio:6.2f}x  "
            f"off {off_spread}% on {on_spread}%"
        )

    record["summary"] = summary
    record["controls_ok"] = not suspect
    out_path.write_text(json.dumps(record, indent=2))

    if suspect:
        print(
            f"\nSUSPECT: control workload(s) {', '.join(suspect)} show offloading "
            f"ahead by more than {CONTROL_TOLERANCE}x. Offloading cannot help "
            "there, so the benchmark is measuring something else. Check that "
            "--max-concurrency is keeping the server unsaturated. Do not quote "
            "these numbers.",
            file=sys.stderr,
        )
    if record["failures"]:
        print(
            f"{len(record['failures'])} sample(s) failed, see {out_path}",
            file=sys.stderr,
        )
    print(f"\nwrote {out_path}")
    # A failed sample leaves a hole in the JSON. Automation must not read
    # that as a clean run.
    return 1 if suspect or record["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
