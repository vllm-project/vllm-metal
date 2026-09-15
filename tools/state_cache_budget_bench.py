#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run one state-cache configuration, or compare two independent run reports.

Only ``run`` imports vLLM/MLX. ``compare`` and the report helpers are engine-free.
Run baseline and budget in separate processes, sequentially, on the same machine.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 2
SAMPLING_IDENTITY = {"temperature": 0, "logprobs": None, "ignore_eos": True, "seed": 0}
CORPUS = (
    "A library stores numbered documents and keeps a separate index of reading "
    "checkpoints. Readers can resume an earlier document, while unused records "
    "are retired before their storage is assigned to another document. "
)
ROOT = Path(__file__).resolve().parent.parent


def digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def verify_loaded_source(module_file: str, root: Path = ROOT) -> dict:
    """Reject another editable checkout/wheel before recording this tree's SHA."""
    source_file = Path(module_file).resolve()
    source_root = root.resolve()
    if not source_file.is_relative_to(source_root):
        raise RuntimeError(
            f"loaded vllm_metal from {source_file}, outside benchmark checkout {source_root}"
        )
    return {
        "loaded_source_root": str(source_file.parent.parent),
        "loaded_source_file": str(source_file),
    }


def local_weights_manifest(model: str | Path) -> list[dict]:
    """Hash every local safetensors shard without decoding/loading tensors."""
    root = Path(model).expanduser()
    if not root.is_dir():
        raise ValueError(
            "a local model directory is required for weight identity; "
            "download an immutable snapshot before running this benchmark"
        )
    manifest = []
    for shard in sorted(root.rglob("*.safetensors")):
        if not shard.is_file():
            continue
        before = shard.stat()
        hasher = hashlib.sha256()
        size = 0
        with shard.open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                hasher.update(chunk)
                size += len(chunk)
        after = shard.stat()
        if (before.st_size, before.st_mtime_ns) != (
            after.st_size,
            after.st_mtime_ns,
        ) or size != after.st_size:
            raise RuntimeError(f"model shard changed while hashing: {shard}")
        manifest.append(
            {
                "name": shard.relative_to(root).as_posix(),
                "bytes": size,
                "sha256": hasher.hexdigest(),
            }
        )
    if not manifest:
        raise ValueError(f"no .safetensors model shards found in {root}")
    return manifest


def write_report(path: Path, report: dict) -> None:
    """Checkpoint atomically so a later failure preserves completed cases."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def positive_ints(value: str) -> list[int]:
    try:
        values = [int(part) for part in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not values or any(n <= 0 for n in values) or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("expected unique positive integers")
    return values


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run one configuration in this process")
    run.add_argument(
        "--model",
        required=True,
        help="local model snapshot directory containing safetensors shards",
    )
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--state-budget-mib", type=int)
    run.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    run.add_argument("--max-model-len", type=int, default=8192)
    run.add_argument("--max-num-batched-tokens", type=int, default=2048)
    run.add_argument("--max-num-seqs", type=int, default=4)
    run.add_argument("--prompt-lengths", type=positive_ints, default=[1024, 4096])
    run.add_argument("--concurrency", type=positive_ints, default=[1, 4])
    run.add_argument("--repeats", type=int, default=3)
    run.add_argument("--max-new-tokens", type=int, default=32)
    run.add_argument(
        "--async-scheduling",
        choices=("auto", "on", "off"),
        default="auto",
        help="auto leaves the engine default unchanged",
    )
    compare = commands.add_parser("compare", help="compare complete token outputs")
    compare.add_argument("--baseline", type=Path, required=True)
    compare.add_argument("--budget", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    return parser


def engine_kwargs(args: argparse.Namespace) -> dict:
    values = {
        name: getattr(args, name)
        for name in (
            "model",
            "gpu_memory_utilization",
            "max_model_len",
            "max_num_batched_tokens",
            "max_num_seqs",
        )
    }
    values.update(enable_prefix_caching=True, seed=SAMPLING_IDENTITY["seed"])
    if args.state_budget_mib is not None:
        values["additional_config"] = {"state_cache_budget_mib": args.state_budget_mib}
    if args.async_scheduling != "auto":
        values["async_scheduling"] = args.async_scheduling == "on"
    return values


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.command != "run":
        return
    for name in (
        "max_model_len",
        "max_num_batched_tokens",
        "max_num_seqs",
        "repeats",
        "max_new_tokens",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.state_budget_mib is not None and args.state_budget_mib <= 0:
        parser.error("--state-budget-mib must be positive; omit it for baseline")
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("--gpu-memory-utilization must be in (0, 1]")
    if max(args.prompt_lengths) + args.max_new_tokens > args.max_model_len:
        parser.error(
            "each prompt length plus --max-new-tokens must fit --max-model-len"
        )


def build_prompt_tokens(tokenizer: Any, length: int, request_index: int) -> list[int]:
    """Construct exactly N tokens, with deterministic independent batch prefixes."""
    prefix = list(
        tokenizer.encode(f"Library request {request_index}. ", add_special_tokens=False)
    )
    corpus = list(tokenizer.encode(CORPUS, add_special_tokens=False))
    if not corpus:
        raise ValueError("tokenizer produced an empty benchmark corpus")
    needed = max(0, length - len(prefix))
    return (prefix + corpus * ((needed + len(corpus) - 1) // len(corpus)))[:length]


def case_plan(args: argparse.Namespace) -> list[dict]:
    cases = []
    for length in args.prompt_lengths:
        for concurrency in args.concurrency:
            for repeat in range(args.repeats):
                salt = f"state-budget-bench-v1/{length}/{concurrency}/{repeat}"
                for phase in ("cold", "hot"):
                    cases.append(
                        {
                            "key": f"len{length}/c{concurrency}/r{repeat}/{phase}",
                            "prompt_length": length,
                            "concurrency": concurrency,
                            "repeat": repeat,
                            "phase": phase,
                            "cache_salt": salt,
                        }
                    )
    return cases


def source_metadata() -> dict:
    def git(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    packages = {}
    for dist in importlib.metadata.distributions():
        package = dist.metadata["Name"].lower().replace("_", "-")
        direct = dist.read_text("direct_url.json")
        packages[package] = {
            "version": dist.version,
            "direct_url": json.loads(direct) if direct else None,
        }
    diff = git("diff", "HEAD", "--")
    untracked = {}
    for name in (git("ls-files", "--others", "--exclude-standard") or "").splitlines():
        path = ROOT / name
        if name.startswith(("vllm_metal/", "tools/")) and path.suffix == ".py":
            untracked[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "host_identity_sha256": digest(
            {"hostname": platform.node(), "machine": platform.machine()}
        ),
        "source": {
            "commit": git("rev-parse", "HEAD"),
            "tracked_diff_sha256": digest(diff),
            "dirty": bool(diff),
            "untracked_python_sha256": untracked,
        },
        "packages": packages,
        "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": {
            name: os.environ[name]
            for name in (
                "VLLM_ENABLE_V1_MULTIPROCESSING",
                "VLLM_METAL_MEMORY_FRACTION",
                "VLLM_METAL_USE_PAGED_ATTENTION",
                "VLLM_METAL_TURBOQUANT",
                "HF_HUB_OFFLINE",
                "MLX_METAL_FAST_SYNCH",
            )
            if name in os.environ
        },
    }


def _get(obj: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(obj, name, default)
    except (AttributeError, RuntimeError):
        return default


def optional_telemetry(obj: Any) -> dict:
    """Read optional budget payloads without requiring one scheduler revision."""
    result = {}
    if obj is None:
        return result
    step = _get(obj, "metal_state_cache")
    if step is not None:
        result["metal_state_cache"] = {
            "sequence": _get(step, "sequence"),
            "resident_blocks": len(_get(step, "resident_blocks", ())),
        }
    for name in dir(obj):
        if not (
            "state" in name
            and any(
                part in name
                for part in ("budget", "evict", "retention", "retained", "slot")
            )
        ):
            continue
        value = _get(obj, name)
        if value is None or isinstance(value, (str, int, float, bool)):
            result[name] = value
        elif isinstance(value, dict):
            result[name] = {
                str(k): v
                for k, v in value.items()
                if v is None or isinstance(v, (str, int, float, bool))
            }
        elif isinstance(value, (list, tuple, set)):
            result[name] = {"count": len(value)}
    return result


def runtime_snapshot(runner: Any) -> dict:
    runtime = _get(runner, "paged_attention_runtime")
    if runtime is None:
        return {}
    cache = _get(runtime, "state_cache")
    manager = _get(runtime, "state_manager")
    arrays = {}
    for field in ("conv_states", "recurrent_states"):
        for array in _get(cache, field, []) or []:
            arrays[id(array)] = array
    return {
        "runtime_type": type(runtime).__name__,
        "state_manager_type": type(manager).__name__ if manager is not None else None,
        "num_blocks": runtime.num_blocks(),
        "block_size": _get(runtime, "_block_size"),
        "slot_cap": _get(cache, "max_seqs"),
        "explicit_state_slot_capacity": _get(runtime, "state_slot_capacity"),
        "allocated_slots": _get(cache, "allocated_seqs"),
        "occupied_slots": _get(manager, "occupied_slots"),
        "state_cache_telemetry": runtime.state_cache_telemetry()
        if callable(_get(runtime, "state_cache_telemetry"))
        else None,
        "num_state_pools": _get(cache, "num_state_pools"),
        # Array capacity is neither process RSS nor total allocator residency.
        "state_tensor_capacity_bytes": sum(array.nbytes for array in arrays.values())
        if arrays
        else None,
        "optional": {
            "runtime": optional_telemetry(runtime),
            "manager": optional_telemetry(manager),
        },
    }


def scheduler_snapshot(scheduler: Any) -> dict:
    quota = _get(scheduler, "_state_quota")
    return {
        "type": type(scheduler).__name__ if scheduler is not None else None,
        "max_num_running_reqs": _get(scheduler, "max_num_running_reqs"),
        "retention_capacity_rows": _get(quota, "capacity"),
        "resident_blocks": len(quota.resident_blocks) if quota is not None else None,
        "evicted_checkpoints": _get(quota, "evicted_checkpoints"),
        "retired_uncached": _get(quota, "retired_uncached"),
        "quota_stalls": _get(quota, "quota_stalls"),
        "optional": optional_telemetry(scheduler),
    }


class RunnerObserver:
    """Observe admission and optional scheduler metadata; never sync per step."""

    def __init__(self) -> None:
        self.runner = None
        self.events: list[dict] = []
        self.case_key = "startup"

    def install(self, runner_class: Any) -> None:
        self.runner_class = runner_class
        self.original_install = runner_class.install_paged_attention_runtime
        self.original_admit = runner_class._handle_new_requests
        observer = self

        def install(runner, *args, **kwargs):
            result = observer.original_install(runner, *args, **kwargs)
            observer.runner = runner
            return result

        def admit(runner, batch, new_reqs, scheduler_output):
            observer.runner = runner
            observer.events.append(
                {
                    "case": observer.case_key,
                    "admissions": [
                        {
                            "request_id": req.req_id,
                            "num_computed_tokens": req.num_computed_tokens,
                        }
                        for req in new_reqs
                    ],
                    "scheduler": optional_telemetry(scheduler_output),
                    "runtime": runtime_snapshot(runner),
                }
            )
            return observer.original_admit(runner, batch, new_reqs, scheduler_output)

        runner_class.install_paged_attention_runtime = install
        runner_class._handle_new_requests = admit

    def restore(self) -> None:
        if hasattr(self, "runner_class"):
            self.runner_class.install_paged_attention_runtime = self.original_install
            self.runner_class._handle_new_requests = self.original_admit


def memory_snapshot(mx: Any, process: Any) -> dict:
    return {
        "mlx_active_bytes": mx.get_active_memory(),
        "mlx_cache_bytes": mx.get_cache_memory(),
        "mlx_peak_active_bytes": mx.get_peak_memory(),
        "process_rss_bytes": process.memory_info().rss,
    }


def summarize_run(report: dict) -> dict:
    records = report.get("records", [])
    complete = (
        report.get("status") == "completed"
        and len(records) == report.get("expected_cases")
        and bool(records)
    )
    result = {
        "completed_cases": len(records),
        "expected_cases": report.get("expected_cases"),
        "complete": complete,
        "phases": {},
    }
    for phase in ("cold", "hot"):
        group = [record for record in records if record["phase"] == phase]
        seconds = sum(record["elapsed_seconds"] for record in group)
        tokens = sum(
            len(tokens) for record in group for tokens in record["output_token_ids"]
        )
        restores = sum(
            n > 0 for record in group for n in record["admitted_num_computed_tokens"]
        )
        result["phases"][phase] = {
            "cases": len(group),
            "elapsed_seconds": seconds,
            "output_tokens": tokens,
            "output_tokens_per_second": tokens / seconds if seconds else None,
            "restored_admissions": restores,
        }
    memories = [
        report[key]
        for key in ("memory_before_startup", "memory_after_startup")
        if key in report
    ]
    memories += [
        point
        for record in records
        for point in (record["memory_before"], record["memory_after"])
    ]
    result["max_observed_memory"] = (
        {name: max(point[name] for point in memories) for name in memories[0]}
        if memories
        else {}
    )
    result["pool_saturation_proven"] = False
    result["pool_saturation_note"] = (
        "Successful startup and this workload do not establish full KV/state-pool residency or maximum supported concurrency."
    )
    return result


def run_benchmark(args: argparse.Namespace) -> dict:
    config = {
        key: value
        for key, value in vars(args).items()
        if key not in ("command", "output")
    }
    plan = case_plan(args)
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": "run",
        "status": "running",
        "stage": "imports",
        "config": config,
        "sampling": {**SAMPLING_IDENTITY, "max_tokens": args.max_new_tokens},
        "metadata": source_metadata(),
        "expected_cases": len(plan),
        "records": [],
        "events": [],
    }
    observer = RunnerObserver()
    write_report(args.output, report)
    try:
        report["stage"] = "weights_identity"
        write_report(args.output, report)
        report["metadata"]["weights_manifest"] = local_weights_manifest(args.model)
        report["stage"] = "imports"
        # The observer needs the in-process worker, and the production decode
        # pipeline remains eligible: no per-step synchronization or logprobs.
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        sys.path.insert(0, str(ROOT.resolve()))
        import vllm_metal

        report["metadata"].update(verify_loaded_source(vllm_metal.__file__))
        import mlx.core as mx
        import psutil
        from vllm import LLM, SamplingParams

        from vllm_metal.v1.model_runner import MetalModelRunner

        process = psutil.Process()
        observer.install(MetalModelRunner)
        report["metadata"]["system_memory_bytes"] = psutil.virtual_memory().total
        report["metadata"]["device_info"] = dict(mx.device_info())
        report["metadata"]["effective_multiprocessing"] = "0"
        report["stage"] = "engine_startup"
        mx.synchronize()
        mx.reset_peak_memory()
        report["memory_before_startup"] = memory_snapshot(mx, process)
        kwargs = engine_kwargs(args)
        report["engine_kwargs"] = kwargs
        write_report(args.output, report)
        started = time.perf_counter()
        llm = LLM(**kwargs)
        mx.synchronize()
        report["startup_seconds"] = time.perf_counter() - started
        report["memory_after_startup"] = memory_snapshot(mx, process)
        report["runtime_after_startup"] = runtime_snapshot(observer.runner)
        cache_config = llm.llm_engine.vllm_config.cache_config
        report["resolved_cache_config"] = {
            name: _get(cache_config, name)
            for name in (
                "block_size",
                "mamba_block_size",
                "mamba_cache_mode",
                "num_gpu_blocks",
                "prefix_cache_retention_interval",
                "hash_block_size",
            )
        }
        core = _get(llm.llm_engine.engine_core, "engine_core")
        scheduler = _get(core, "scheduler")
        report["resolved_cache_config"]["scheduler_hash_block_size"] = _get(
            _get(_get(scheduler, "kv_cache_manager"), "block_pool"), "hash_block_size"
        )
        report["resolved_scheduler"] = scheduler_snapshot(scheduler)
        report["resolved_scheduler"]["async_scheduling"] = _get(
            llm.llm_engine.vllm_config.scheduler_config, "async_scheduling"
        )
        report["budget_applied"] = (
            report["runtime_after_startup"].get("explicit_state_slot_capacity")
            is not None
            and report["resolved_scheduler"].get("retention_capacity_rows") is not None
        )
        if args.state_budget_mib is not None and not report["budget_applied"]:
            raise RuntimeError(
                "state budget requested but bounded runtime/scheduler was not observed"
            )
        if cache_config.mamba_cache_mode != "align":
            raise RuntimeError(
                f"expected align mode, got {cache_config.mamba_cache_mode}"
            )
        if observer.runner is None:
            raise RuntimeError(
                "runner observer did not see the in-process Metal worker"
            )
        tokenizer = llm.get_tokenizer()
        model_config = llm.llm_engine.vllm_config.model_config.hf_config.to_dict()
        report["metadata"]["model_config"] = model_config
        report["metadata"]["model_config_sha256"] = digest(model_config)
        report["metadata"]["chat_template_sha256"] = digest(
            _get(tokenizer, "chat_template")
        )
        report["metadata"]["chat_template_applied"] = False
        prompts = {
            (case["prompt_length"], case["concurrency"]): [
                build_prompt_tokens(tokenizer, case["prompt_length"], index)
                for index in range(case["concurrency"])
            ]
            for case in plan
        }
        params = SamplingParams(**report["sampling"])
        for case in plan:
            report["stage"] = case["key"]
            observer.case_key = case["key"]
            ids = prompts[(case["prompt_length"], case["concurrency"])]
            # Fresh salt for each cold batch; exactly the same salt for its hot
            # repeat. Later repeats churn checkpoints without clearing the pool.
            inputs = [
                {"prompt_token_ids": tokens, "cache_salt": case["cache_salt"]}
                for tokens in ids
            ]
            first_event = len(observer.events)
            mx.synchronize()
            mx.reset_peak_memory()
            before = memory_snapshot(mx, process)
            started = time.perf_counter()
            outputs = llm.generate(inputs, params, use_tqdm=False)
            mx.synchronize()
            elapsed = time.perf_counter() - started
            events = observer.events[first_event:]
            record = {
                **case,
                "prompt_token_ids": ids,
                "output_token_ids": [
                    list(output.outputs[0].token_ids) for output in outputs
                ],
                "finish_reasons": [
                    output.outputs[0].finish_reason for output in outputs
                ],
                "elapsed_seconds": elapsed,
                "memory_before": before,
                "memory_after": memory_snapshot(mx, process),
                "runtime_after": runtime_snapshot(observer.runner),
                "scheduler_after": scheduler_snapshot(scheduler),
                "admitted_num_computed_tokens": [
                    admit["num_computed_tokens"]
                    for event in events
                    for admit in event["admissions"]
                ],
            }
            if len(outputs) != len(inputs):
                raise RuntimeError(
                    f"expected {len(inputs)} outputs, got {len(outputs)}"
                )
            if any(
                len(tokens) != args.max_new_tokens
                for tokens in record["output_token_ids"]
            ):
                raise RuntimeError(
                    "generation ended before the requested output length despite ignore_eos=True"
                )
            report["records"].append(record)
            report["events"] = observer.events
            write_report(args.output, report)
            print(
                f"{case['key']}: {elapsed:.3f}s; restored={record['admitted_num_computed_tokens']}; runtime={record['runtime_after']}",
                flush=True,
            )
        report["status"] = "completed"
        report["stage"] = "completed"
    except (Exception, KeyboardInterrupt) as exc:
        report["status"] = "failed"
        report["failure"] = {
            "stage": report["stage"],
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
    finally:
        observer.restore()
        report["events"] = observer.events
        report["summary"] = summarize_run(report)
        write_report(args.output, report)
    return report


def _evidence_problems(report: dict, side: str) -> list[str]:
    """Validate recorded evidence independently of its self-reported status.

    This only runs during comparison. Keep acquisition unchanged so older raw
    reports can be reassessed without changing their workload or tool identity.
    """
    problems = []

    def reject(reason: str) -> None:
        problems.append(f"{side}: {reason}")

    def contains_diagnostic(value: Any) -> bool:
        if isinstance(value, dict):
            return value.get("qualification_eligible") is False or any(
                contains_diagnostic(item) for item in value.values()
            )
        if isinstance(value, list):
            return any(contains_diagnostic(item) for item in value)
        return False

    if report.get("qualification_eligible") is False or contains_diagnostic(
        report.get("metadata", {})
    ):
        reject("diagnostic evidence is not eligible for qualification")
    if report.get("schema_version") != SCHEMA_VERSION or report.get("kind") != "run":
        reject("expected a supported run report")
    if report.get("failure"):
        reject("run report contains a failure")
    config = report.get("config", {})
    if not isinstance(config, dict):
        reject("workload configuration is missing or invalid")
        return problems

    def positive(value: Any) -> bool:
        return type(value) is int and value > 0

    valid_config = True
    for name in ("prompt_lengths", "concurrency"):
        values = config.get(name)
        if (
            not isinstance(values, list)
            or not values
            or not all(positive(value) for value in values)
            or len(set(values)) != len(values)
        ):
            reject(f"invalid workload configuration: {name}")
            valid_config = False
    for name in (
        "repeats",
        "max_new_tokens",
        "max_model_len",
        "max_num_batched_tokens",
        "max_num_seqs",
    ):
        if not positive(config.get(name)):
            reject(f"invalid workload configuration: {name}")
            valid_config = False
    if config.get("async_scheduling") not in ("auto", "on", "off"):
        reject("invalid workload configuration: async_scheduling")
    utilization = config.get("gpu_memory_utilization")
    if (
        isinstance(utilization, bool)
        or not isinstance(utilization, (int, float))
        or not 0 < utilization <= 1
    ):
        reject("invalid workload configuration: gpu_memory_utilization")
    if not valid_config:
        return problems
    if (
        max(config["prompt_lengths"]) + config["max_new_tokens"]
        > config["max_model_len"]
    ):
        reject("requested prompt and output exceed max_model_len")
    plan = case_plan(argparse.Namespace(**config))
    if type(report.get("expected_cases")) is not int or report["expected_cases"] != len(
        plan
    ):
        reject("expected_cases differs from the matrix rebuilt from configuration")
    records = report.get("records")
    if not isinstance(records, list) or any(not isinstance(r, dict) for r in records):
        reject("records must be a list of case objects")
        return problems
    if [r.get("key") for r in records] != [case["key"] for case in plan]:
        reject("recorded case order/keys differ from the configured matrix")
    expected = {case["key"]: case for case in plan}

    def token_rows_valid(value: Any, rows: int, width: int) -> bool:
        return (
            isinstance(value, list)
            and len(value) == rows
            and all(
                isinstance(tokens, list)
                and len(tokens) == width
                and all(type(token) is int and token >= 0 for token in tokens)
                for tokens in value
            )
        )

    for record in records:
        key = record.get("key")
        case = expected.get(key) if isinstance(key, str) else None
        if case is None:
            continue
        if any(record.get(name) != value for name, value in case.items()):
            reject(f"case metadata differs from configured matrix: {key}")
        count = case["concurrency"]
        if not token_rows_valid(
            record.get("prompt_token_ids"), count, case["prompt_length"]
        ):
            reject(f"prompt rows/token counts are incomplete or invalid: {key}")
        if not token_rows_valid(
            record.get("output_token_ids"), count, config["max_new_tokens"]
        ):
            reject(f"output rows/token counts are incomplete or invalid: {key}")
        if record.get("finish_reasons") != ["length"] * count:
            reject(f"finish reasons must cover every fixed-length output: {key}")
    return problems


def compare_reports(baseline: dict, budget: dict) -> dict:
    problems = _evidence_problems(baseline, "baseline") + _evidence_problems(
        budget, "budget"
    )
    baseline_config, budget_config = (
        baseline.get("config", {}),
        budget.get("config", {}),
    )
    if (
        "state_budget_mib" not in baseline_config
        or baseline_config["state_budget_mib"] is not None
        or baseline.get("budget_applied") is not False
    ):
        problems.append(
            "baseline must omit the state budget and confirm budget_applied=false"
        )
    requested_budget = budget_config.get("state_budget_mib")
    if (
        isinstance(requested_budget, bool)
        or not isinstance(requested_budget, int)
        or requested_budget <= 0
        or budget.get("budget_applied") is not True
    ):
        problems.append(
            "budget arm must request a positive state budget and confirm budget_applied=true"
        )
    for name in (
        "block_size",
        "mamba_block_size",
        "mamba_cache_mode",
        "prefix_cache_retention_interval",
        "hash_block_size",
        "scheduler_hash_block_size",
    ):
        a, b = (
            baseline.get("resolved_cache_config", {}),
            budget.get("resolved_cache_config", {}),
        )
        if name not in a or name not in b or a[name] != b[name]:
            problems.append(f"missing or different cache policy: {name}")
    if not summarize_run(baseline)["complete"] or not summarize_run(budget)["complete"]:
        problems.append(
            "both input reports must contain every expected case and have completed status"
        )
    for key in (
        "prompt_lengths",
        "concurrency",
        "repeats",
        "max_new_tokens",
        "max_model_len",
        "max_num_batched_tokens",
        "max_num_seqs",
        "gpu_memory_utilization",
        "async_scheduling",
    ):
        if baseline.get("config", {}).get(key) != budget.get("config", {}).get(key):
            problems.append(f"configuration differs: {key}")
    for key in ("model_config_sha256", "chat_template_sha256", "weights_manifest"):
        left, right = (
            baseline.get("metadata", {}).get(key),
            budget.get("metadata", {}).get(key),
        )
        if not left or left != right:
            problems.append(f"missing or different model identity: {key}")
    for key in (
        "host_identity_sha256",
        "platform",
        "machine",
        "system_memory_bytes",
        "device_info",
        "python",
        "tool_sha256",
    ):
        a = baseline.get("metadata", {}).get(key)
        b = budget.get("metadata", {}).get(key)
        if a is None or a != b:
            problems.append(f"missing or different execution identity: {key}")
    dependencies = []
    for report in (baseline, budget):
        # The editable project is the source under test; its path/version may
        # change. Every other installed distribution and direct URL must match.
        dependencies.append(
            {
                name: value
                for name, value in report.get("metadata", {})
                .get("packages", {})
                .items()
                if name != "vllm-metal"
            }
        )
    if not dependencies[0] or dependencies[0] != dependencies[1]:
        problems.append(
            "installed dependency versions/direct URLs differ or are missing"
        )
    for side, report in (("baseline", baseline), ("budget", budget)):
        sampling = report.get("sampling", {})
        if any(
            name not in sampling or sampling[name] != value
            for name, value in SAMPLING_IDENTITY.items()
        ):
            problems.append(
                f"{side} sampling does not match the strict benchmark policy"
            )
        if sampling.get("max_tokens") != report.get("config", {}).get("max_new_tokens"):
            problems.append(
                f"{side} sampling output length differs from its workload configuration"
            )
        if not report.get("metadata", {}).get("source", {}).get("commit"):
            problems.append(f"{side} source commit is missing")
    if baseline.get("sampling") != budget.get("sampling"):
        problems.append("sampling configuration differs")
    for section, key in (
        ("resolved_scheduler", "async_scheduling"),
        ("metadata", "environment"),
    ):
        a, b = baseline.get(section, {}), budget.get(section, {})
        if key not in a or key not in b or a[key] != b[key]:
            problems.append(f"missing or different execution policy: {key}")
    left = {record["key"]: record for record in baseline.get("records", [])}
    right = {record["key"]: record for record in budget.get("records", [])}
    if (
        set(left) != set(right)
        or len(left) != len(baseline.get("records", []))
        or len(right) != len(budget.get("records", []))
    ):
        problems.append("case keys differ or are duplicated")
    comparisons = []
    for key in sorted(set(left) & set(right)):
        a, b = left[key], right[key]
        inputs_equal = (
            a["prompt_token_ids"] == b["prompt_token_ids"]
            and a["cache_salt"] == b["cache_salt"]
        )
        outputs_equal = a["output_token_ids"] == b["output_token_ids"]
        comparisons.append(
            {
                "key": key,
                "inputs_equal": inputs_equal,
                "outputs_equal": outputs_equal,
                "baseline_seconds": a["elapsed_seconds"],
                "budget_seconds": b["elapsed_seconds"],
            }
        )
        if not inputs_equal or not outputs_equal:
            problems.append(
                f"{'input' if not inputs_equal else 'output'} token mismatch: {key}"
            )
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "comparison",
        "comparator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "status": "pass" if not problems and comparisons else "fail",
        "problems": problems,
        "comparisons": comparisons,
        "baseline_summary": summarize_run(baseline),
        "budget_summary": summarize_run(budget),
        "baseline_capacity": {
            key: baseline.get(key)
            for key in (
                "budget_applied",
                "resolved_cache_config",
                "resolved_scheduler",
                "runtime_after_startup",
            )
        },
        "budget_capacity": {
            key: budget.get(key)
            for key in (
                "budget_applied",
                "resolved_cache_config",
                "resolved_scheduler",
                "runtime_after_startup",
            )
        },
        "baseline_metadata": baseline.get("metadata"),
        "budget_metadata": budget.get("metadata"),
        "source_comparison": {
            "same_source": baseline.get("metadata", {}).get("source")
            == budget.get("metadata", {}).get("source"),
            "baseline": baseline.get("metadata", {}).get("source"),
            "budget": budget.get("metadata", {}).get("source"),
            "policy": "Source revisions may differ; dependencies, device, benchmark tool, sampling, and execution policy must match.",
        },
        "note": "Strict whole-output comparison; no tie waivers. Timing and memory are workload observations, not a maximum-capacity claim.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    validate_args(args, parser)
    if args.command == "run":
        report = run_benchmark(args)
        print(json.dumps(report["summary"], indent=2))
        if "failure" in report:
            print(json.dumps(report["failure"], indent=2), file=sys.stderr)
        return 0 if report["status"] == "completed" else 1
    try:
        report = compare_reports(
            json.loads(args.baseline.read_text()), json.loads(args.budget.read_text())
        )
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        report = {
            "schema_version": SCHEMA_VERSION,
            "kind": "comparison",
            "status": "fail",
            "problems": [str(exc)],
            "comparisons": [],
            "failure": {
                "stage": "compare",
                "exception_type": type(exc).__name__,
                "message": str(exc),
            },
        }
    report["input_files"] = {"baseline": str(args.baseline), "budget": str(args.budget)}
    write_report(args.output, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "problems": report["problems"],
                "cases": len(report["comparisons"]),
            },
            indent=2,
        )
    )
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
