# SPDX-License-Identifier: Apache-2.0
"""Deterministic offline throughput / TTFT / memory benchmark for vLLM Metal.

Runs a fixed suite of text-generation workloads through the in-process V1
engine (``VLLM_ENABLE_V1_MULTIPROCESSING=0``, ``uni`` executor) on locally
cached MLX checkpoints and reports, per workload:

* decode throughput — output tokens after the first, over the time left after
  the prefill window;
* TTFT — the prefill window, derived from two runs of the same prompt with
  ``max_tokens`` 1 and 2 (``2 * t1 - t2``), so no streaming engine is needed;
* peak RSS and steady-state MLX memory;
* the engine's KV-cache token capacity, parsed from the vLLM log.

Everything is fixed: prompts, sampling parameters (temperature 0, fixed seed,
``ignore_eos``), ``max_model_len``, ``max_num_seqs``, ``gpu_memory_utilization``
and the prefix cache is reset before every timed run. No network access:
checkpoints come from the local Hugging Face cache (``local_files_only=True``).

Parent mode (default) runs one child process per model — Metal memory is only
released on process exit — aggregates the results and prints ``METRIC`` lines::

    python tools/benchmark/metal_suite_benchmark.py --json-out /tmp/bench.json

Child mode (one model, one process; also used for isolation)::

    python tools/benchmark/metal_suite_benchmark.py --run-model qwen3-0.6b-4bit
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MODEL_TAGS: dict[str, str] = {
    "qwen3-0.6b-4bit": "mlx-community/Qwen3-0.6B-4bit",
    "qwen3-1.7b-4bit": "mlx-community/Qwen3-1.7B-4bit",
}

# ---------------------------------------------------------------------------
# Fixed workloads
# ---------------------------------------------------------------------------

_PARAGRAPH = (
    "The harbour town wakes early. Fishing boats return with the tide, and the "
    "market fills with the smell of salt and coffee. Shopkeepers roll up their "
    "shutters while gulls argue over the gutting tables. By mid morning the "
    "square is loud with bargaining, and the ferry has already left for the "
    "island with a full deck of bicycles and crates."
)

_TAILS = (
    " The road out of town climbs past the old lighthouse.",
    " Nobody remembers who painted the mural on the customs house.",
    " The bell in the town hall still rings on the hour, twice.",
    " A cat sleeps on the warm stones beside the fountain.",
    " The librarian keeps a ledger of every book borrowed since the war.",
    " Tourists photograph the pier and then buy ice cream.",
)


def _text(paragraphs: int, tail_index: int) -> str:
    """A deterministic prompt of ``paragraphs`` repeats plus one fixed tail."""
    return " ".join([_PARAGRAPH] * paragraphs) + _TAILS[tail_index % len(_TAILS)]


def _short_prompts() -> tuple[str, ...]:
    """Eight short, distinct prompts (decode-dominant workload)."""
    topics = (
        "Explain in one paragraph why the sky is blue.",
        "Summarise the plot of a novel about a lost expedition.",
        "List the steps to brew a good cup of coffee.",
        "Describe how a bicycle derailleur works.",
        "Write a short dialogue between a pilot and a mechanic.",
        "Explain what a hash table is to a beginner.",
        "Describe the water cycle in simple terms.",
        "Write a product announcement for a folding kayak.",
    )
    return tuple(f"{topic}\n\nAnswer:" for topic in topics)


def _mixed_prompts() -> tuple[str, ...]:
    """Twelve prompts from ~40 to ~700 tokens (ragged batch)."""
    return tuple(_text(i + 1, i) + "\n\nAnswer:" for i in range(12))


def _long_prompts() -> tuple[str, ...]:
    """Two long prompts (~1k tokens) for the prefill/TTFT workload."""
    return (_text(18, 0) + "\n\nAnswer:", _text(18, 3) + "\n\nAnswer:")


def _long_context_prompts() -> tuple[str, ...]:
    """Four ~1.5-1.8k token prompts for KV-read-bound decode."""
    return tuple(_text(18 + index, index) + "\n\nAnswer:" for index in range(4))


def _warmup_prompts(count: int, tag: str) -> tuple[str, ...]:
    """Prompts that share no prefix with any measured prompt."""
    base = (
        f"Warm-up pass {tag}. Read the following note and reply with a single "
        "sentence. The note describes a municipal archive: shelves of ledgers, "
        "a reading room with green lamps, and a catalogue that has been rebuilt "
        "three times since the flood."
    )
    return tuple(f"{base} Entry {index}." for index in range(count))


@dataclass(frozen=True)
class Case:
    """One fixed workload."""

    name: str
    model_tag: str
    prompts: tuple[str, ...]
    max_tokens: int
    max_num_seqs: int
    max_model_len: int = 2048

    @property
    def measures_decode(self) -> bool:
        """Decode throughput counts only cases that generate past the first."""
        return self.max_tokens > 1


CASES: tuple[Case, ...] = (
    Case(
        name="decode_b8",
        model_tag="qwen3-0.6b-4bit",
        prompts=_short_prompts(),
        max_tokens=128,
        max_num_seqs=8,
    ),
    Case(
        name="prefill_long_b2",
        model_tag="qwen3-0.6b-4bit",
        prompts=_long_prompts(),
        max_tokens=1,
        max_num_seqs=2,
    ),
    Case(
        name="mixed_b12",
        model_tag="qwen3-0.6b-4bit",
        prompts=_mixed_prompts(),
        max_tokens=64,
        max_num_seqs=12,
    ),
    Case(
        name="decode_longctx_b4",
        model_tag="qwen3-0.6b-4bit",
        prompts=_long_context_prompts(),
        max_tokens=64,
        max_num_seqs=4,
    ),
    Case(
        name="decode_b4_1p7b",
        model_tag="qwen3-1.7b-4bit",
        prompts=_short_prompts()[:4],
        max_tokens=96,
        max_num_seqs=4,
    ),
)

GPU_MEMORY_UTILIZATION = 0.5
REPS = 3

KV_CACHE_RE = re.compile(r"KV cache size:\s*([\d,]+)\s*tokens")
SHARED_CACHE_RE = re.compile(r"Shared attention cache:\s*([\d,]+)\s*blocks,\s*([\d.]+)\s*GiB")
METAL_MEMORY_RE = re.compile(r"Metal memory:\s*(.+?)\s+total,\s*(.+?)\s+available")


# ---------------------------------------------------------------------------
# Memory sampling
# ---------------------------------------------------------------------------


class RssSampler:
    """Peak RSS of this process over a measurement window (sampled, explicit)."""

    def __init__(self, interval_s: float = 0.02) -> None:
        self._interval_s = interval_s
        self._peak = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        import psutil

        proc = psutil.Process()
        while not self._stop.is_set():
            try:
                self._peak = max(self._peak, proc.memory_info().rss)
            except Exception:  # pragma: no cover - teardown race
                break
            self._stop.wait(self._interval_s)

    def __enter__(self) -> "RssSampler":
        self._peak = 0
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    @property
    def peak_mb(self) -> float:
        return self._peak / (1024 * 1024)


def _maxrss_mb() -> float:
    """Peak RSS of the whole process (macOS reports bytes)."""
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


# ---------------------------------------------------------------------------
# Child: run one model's cases in-process
# ---------------------------------------------------------------------------


@dataclass
class TimedRun:
    """One ``llm.generate`` call."""

    wall_s: float
    prompt_tokens: int
    output_tokens: int
    rss_peak_mb: float = 0.0


@dataclass
class CaseResult:
    """A case's measured runs; the fastest rep is representative."""

    name: str
    model_tag: str
    measures_decode: bool
    num_requests: int
    main: list[TimedRun] = field(default_factory=list)
    one_token: list[float] = field(default_factory=list)
    two_token: list[float] = field(default_factory=list)
    mlx_active_mb: float = 0.0
    maxrss_mb: float = 0.0

    def _representative(self) -> TimedRun:
        return min(self.main, key=lambda run: run.wall_s)

    @property
    def prefill_ms(self) -> float:
        """Prefill window: the two-run estimate ``2 * t1 - t2``, in ms."""
        if not self.one_token or not self.two_token:
            return 0.0
        t1 = min(self.one_token)
        t2 = min(self.two_token)
        return max(0.0, (2.0 * t1 - t2) * 1000.0)

    def summary(self) -> dict[str, float]:
        run = self._representative()
        prefill_s = self.prefill_ms / 1000.0
        # One token per request is produced by the prefill window itself.
        decode_tokens = max(0, run.output_tokens - self.num_requests)
        decode_s = max(1e-6, run.wall_s - prefill_s)
        return {
            "wall_s": run.wall_s,
            "prompt_tokens": float(run.prompt_tokens),
            "output_tokens": float(run.output_tokens),
            "decode_tokens": float(decode_tokens),
            "decode_s": decode_s,
            "decode_tok_s": decode_tokens / decode_s,
            "total_tok_s": (run.prompt_tokens + run.output_tokens) / run.wall_s,
            "ttft_ms": self.prefill_ms,
            "prefill_tok_s": (
                run.prompt_tokens / prefill_s if prefill_s > 0 else 0.0
            ),
            "rss_peak_mb": run.rss_peak_mb,
            "maxrss_mb": self.maxrss_mb,
            "mlx_active_mb": self.mlx_active_mb,
            "measures_decode": 1.0 if self.measures_decode else 0.0,
            "rep_wall_s": ",".join(f"{run.wall_s:.4f}" for run in self.main),
            "rep_one_token_s": ",".join(f"{t:.4f}" for t in self.one_token),
            "rep_two_token_s": ",".join(f"{t:.4f}" for t in self.two_token),
        }


def _generate(llm: Any, prompts: list[str], max_tokens: int) -> TimedRun:
    """One timed, prefix-cache-cold ``generate`` call."""
    import mlx.core as mx
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        temperature=0.0, max_tokens=max_tokens, ignore_eos=True, seed=0
    )
    llm.reset_prefix_cache()
    mx.synchronize()
    with RssSampler() as sampler:
        start = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        mx.synchronize()
        wall_s = time.perf_counter() - start
    return TimedRun(
        wall_s=wall_s,
        prompt_tokens=sum(len(out.prompt_token_ids or ()) for out in outputs),
        output_tokens=sum(len(out.outputs[0].token_ids) for out in outputs),
        rss_peak_mb=sampler.peak_mb,
    )


def _run_model(tag: str, reps: int, only: tuple[str, ...]) -> dict[str, Any]:
    """Load one model and run every case that targets it."""
    import mlx.core as mx
    from vllm import LLM, SamplingParams

    model_path = _resolve_model(tag)
    cases = [case for case in CASES if case.model_tag == tag]
    if only:
        cases = [case for case in cases if case.name in only]
    if not cases:
        return {"model_tag": tag, "cases": {}}

    llm = LLM(
        model=str(model_path),
        max_model_len=max(case.max_model_len for case in cases),
        max_num_seqs=max(case.max_num_seqs for case in cases),
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        seed=0,
        disable_log_stats=True,
    )

    results: dict[str, CaseResult] = {}
    try:
        for case in cases:
            result = CaseResult(
                name=case.name,
                model_tag=tag,
                measures_decode=case.measures_decode,
                num_requests=len(case.prompts),
            )
            warmup = SamplingParams(
                temperature=0.0, max_tokens=case.max_tokens, ignore_eos=True, seed=0
            )
            llm.reset_prefix_cache()
            llm.generate(
                list(_warmup_prompts(len(case.prompts), tag)),
                warmup,
                use_tqdm=False,
            )
            for _ in range(reps):
                result.one_token.append(_generate(llm, list(case.prompts), 1).wall_s)
                result.two_token.append(_generate(llm, list(case.prompts), 2).wall_s)
                if case.max_tokens >= 2:
                    result.main.append(
                        _generate(llm, list(case.prompts), case.max_tokens)
                    )
                else:
                    result.main.append(_generate(llm, list(case.prompts), 1))
            result.mlx_active_mb = mx.get_active_memory() / (1024 * 1024)
            result.maxrss_mb = _maxrss_mb()
            results[case.name] = result
    finally:
        del llm
    return {
        "model_tag": tag,
        "model_path": str(model_path),
        "device": mx.device_info().get("device_name"),
        "cases": {name: result.summary() for name, result in results.items()},
    }


def _resolve_model(tag: str) -> Path:
    """Resolve a checkpoint from the local HF cache; never touches the network."""
    from huggingface_hub import snapshot_download

    if tag not in MODEL_TAGS:
        raise SystemExit(
            f"unknown model tag {tag!r}; known tags: {', '.join(MODEL_TAGS)}"
        )
    repo_id = MODEL_TAGS[tag]
    try:
        return Path(snapshot_download(repo_id, local_files_only=True))
    except Exception as exc:  # pragma: no cover - operator guidance
        raise SystemExit(
            f"{repo_id} is not in the local Hugging Face cache ({exc}). "
            "Fetch it once with: python -c \"from huggingface_hub import "
            f"snapshot_download; snapshot_download('{repo_id}')\""
        ) from exc


# ---------------------------------------------------------------------------
# Parent: one subprocess per model, aggregate, print METRIC lines
# ---------------------------------------------------------------------------


def _run_child(tag: str, reps: int, only: tuple[str, ...]) -> tuple[dict[str, Any], str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--run-model",
        tag,
        "--reps",
        str(reps),
    ]
    if only:
        cmd += ["--only", ",".join(only)]
    env = dict(os.environ)
    env.setdefault("VLLM_METAL_BUILD_FROM_SOURCE", "1")
    env.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    env.setdefault("HF_HUB_OFFLINE", "1")
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"child for {tag} failed with exit code {proc.returncode}")
    lines = proc.stdout.rstrip().splitlines()
    payload = json.loads(lines[-1])
    return payload, "\n".join(lines[:-1]) + "\n" + proc.stderr


def _aggregate(payloads: list[dict[str, Any]], logs: str) -> dict[str, Any]:
    cases: dict[str, dict[str, float]] = {}
    for payload in payloads:
        for name, summary in payload["cases"].items():
            cases[name] = {**summary, "model_tag": payload["model_tag"]}

    decode_cases = [case for case in cases.values() if case["measures_decode"]]
    decode_tokens = sum(case["decode_tokens"] for case in decode_cases)
    decode_s = sum(case["decode_s"] for case in decode_cases)
    prefill_tokens = sum(case["prompt_tokens"] for case in cases.values())
    prefill_s = sum(case["ttft_ms"] / 1000.0 for case in cases.values())

    kv_match = KV_CACHE_RE.search(logs)
    shared_match = SHARED_CACHE_RE.search(logs)
    metal_match = METAL_MEMORY_RE.search(logs)

    return {
        "cases": cases,
        "suite": {
            "decode_tok_s": decode_tokens / decode_s if decode_s else 0.0,
            "prefill_tok_s": prefill_tokens / prefill_s if prefill_s else 0.0,
            # Long-context TTFT (2-way batch of ~1k-token prompts) is the
            # guardrail; the ragged batch is reported per case.
            "ttft_ms": cases.get("prefill_long_b2", {}).get(
                "ttft_ms",
                max((case["ttft_ms"] for case in cases.values()), default=0.0),
            ),
            "ttft_ms_short": cases.get("decode_b8", {}).get("ttft_ms", 0.0),
            "peak_rss_mb": max((case["rss_peak_mb"] for case in cases.values()), default=0.0),
            "maxrss_mb": max((case["maxrss_mb"] for case in cases.values()), default=0.0),
            "mlx_active_mb": max(
                (case["mlx_active_mb"] for case in cases.values()), default=0.0
            ),
            "kv_cache_tokens": (
                float(kv_match.group(1).replace(",", "")) if kv_match else 0.0
            ),
            "kv_cache_blocks": (
                float(shared_match.group(1).replace(",", "")) if shared_match else 0.0
            ),
            "kv_cache_gib": float(shared_match.group(2)) if shared_match else 0.0,
            "metal_memory_total": metal_match.group(1) if metal_match else "",
            "metal_memory_available": metal_match.group(2) if metal_match else "",
        },
    }


def _print_metrics(aggregate: dict[str, Any]) -> None:
    suite = aggregate["suite"]
    print(f"METRIC decode_tok_s={suite['decode_tok_s']:.2f}")
    print(f"METRIC prefill_tok_s={suite['prefill_tok_s']:.2f}")
    print(f"METRIC ttft_ms={suite['ttft_ms']:.2f}")
    print(f"METRIC ttft_ms_short={suite['ttft_ms_short']:.2f}")
    print(f"METRIC peak_rss_mb={suite['peak_rss_mb']:.1f}")
    print(f"METRIC maxrss_mb={suite['maxrss_mb']:.1f}")
    print(f"METRIC mlx_active_mb={suite['mlx_active_mb']:.1f}")
    if suite["kv_cache_tokens"]:
        print(f"METRIC kv_cache_tokens={suite['kv_cache_tokens']:.0f}")
    for name, case in sorted(aggregate["cases"].items()):
        print(f"METRIC case_{name}_decode_tok_s={case['decode_tok_s']:.2f}")
        print(f"METRIC case_{name}_ttft_ms={case['ttft_ms']:.2f}")
        print(f"METRIC case_{name}_wall_s={case['wall_s']:.3f}")


def _environment(reps: int) -> dict[str, Any]:
    import importlib.metadata
    import platform

    def version(name: str) -> str | None:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "reps": reps,
        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
        "versions": {
            name: version(name)
            for name in ("vllm", "vllm-metal", "mlx", "mlx-lm", "transformers")
        },
        "env": {
            name: os.environ.get(name)
            for name in (
                "VLLM_METAL_BUILD_FROM_SOURCE",
                "VLLM_ENABLE_V1_MULTIPROCESSING",
                "MLX_MAX_OPS_PER_BUFFER",
                "VLLM_METAL_COMPILED_MLP",
                "VLLM_METAL_DECODE_PIPELINE",
                "VLLM_METAL_NATIVE_SAMPLING",
                "VLLM_METAL_MLA_KERNEL",
                "VLLM_METAL_SPEC_VERIFY_WINDOW",
            )
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-model", help="child mode: run cases for one model tag")
    parser.add_argument(
        "--models",
        default=",".join(dict.fromkeys(case.model_tag for case in CASES)),
        help="comma-separated model tags for parent mode",
    )
    parser.add_argument("--only", default="", help="comma-separated case names to run")
    parser.add_argument("--reps", type=int, default=REPS)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)

    only = tuple(name for name in args.only.split(",") if name) if args.only else ()

    if args.run_model:
        payload = _run_model(args.run_model, args.reps, only)
        print(json.dumps(payload), flush=True)
        return 0

    payloads: list[dict[str, Any]] = []
    logs = ""
    for tag in [item for item in args.models.split(",") if item]:
        payload, child_logs = _run_child(tag, args.reps, only)
        payloads.append(payload)
        logs += child_logs

    aggregate = _aggregate(payloads, logs)
    aggregate["environment"] = _environment(args.reps)
    _print_metrics(aggregate)
    if args.json_out:
        args.json_out.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
