# SPDX-License-Identifier: Apache-2.0
"""Continuous-arrival, real-scheduler GDN correctness and delivery-time gate.

Run baseline/budget arms separately. Observers read CPU metadata only: they do
not reorder requests, manufacture preemption, evaluate state arrays or alter
sampling. A small num_gpu_blocks_override is an explicit pressure experiment.
The in-process frontend retains the real async scheduler/decode pipeline; this
is not an HTTP/network or multiprocessing-lifecycle benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import state_cache_budget_bench as bench  # noqa: E402


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def workload(
    tokenizer,
    block: int,
    scenario: str,
    waves: int = 1,
    wave_interval: float = 2.0,
    ignore_eos: bool = True,
) -> list[dict]:
    if scenario == "mixed":
        shapes = [
            (8 * block + 1, 256, 0.0, None),
            (128, block + 128, 0.0, None),
            (2 * block + 1, 32, 0.03, None),
            (4 * block + 17, 128, 0.08, None),
            (12 * block + 1, 96, 0.12, "prefill"),
            (block + 1, 256, 0.18, "decode"),
            (96, 16, 0.25, None),
            (6 * block + 1, block + 64, 0.40, None),
            (8 * block + 1, 64, 0.70, None),
            (256, 48, 1.00, None),
            (2 * block + 17, 128, 1.30, None),
            (128, 16, 1.60, None),
        ]
    else:
        shapes = [
            (8 * block + 1, 256, 0.0, None),
            (6 * block + 17, 384, 0.0, None),
            (10 * block + 1, 256, 0.04, None),
            (4 * block + 1, 128, 0.08, None),
            (128, 32, 0.20, None),
            (7 * block + 1, 192, 0.40, None),
            (2 * block + 17, 64, 0.80, None),
            (96, 16, 1.20, None),
        ]
    result = []
    for wave in range(waves):
        for offset, (length, output, arrival, cancel) in enumerate(shapes):
            index = wave * len(shapes) + offset
            result.append(
                {
                    "request_id": f"{scenario}-{index:02d}",
                    "prompt_token_ids": bench.build_prompt_tokens(
                        tokenizer, length, index
                    ),
                    "max_tokens": output,
                    "ignore_eos": ignore_eos,
                    "sampling": {"temperature": 0, "logprobs": None, "seed": 0},
                    "arrival_offset_s": wave * wave_interval + arrival,
                    "cancel_stage": cancel,
                }
            )
    return result


def apply_chat_template_to_plan(tokenizer, plan: list[dict]) -> None:
    """Rebuild each planned prompt through the tokenizer chat template.

    Planned token counts stay exact so block-size coverage does not change.
    The template plus generation prompt must fit the shortest planned prompt.
    """
    template = getattr(tokenizer, "chat_template", None)
    if not template:
        raise ValueError("apply_chat_template requires a tokenizer chat_template")
    for item in plan:
        length = len(item["prompt_token_ids"])
        wrapped = tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": f"Library request {item['request_id']}. Continue.",
                }
            ],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
        )
        if hasattr(wrapped, "input_ids"):
            wrapped = wrapped["input_ids"]
        if hasattr(wrapped, "tolist"):
            wrapped = wrapped.tolist()
        if (
            wrapped
            and isinstance(wrapped, list)
            and wrapped
            and isinstance(wrapped[0], list)
        ):
            wrapped = wrapped[0]
        if isinstance(wrapped, str):
            wrapped = tokenizer.encode(wrapped, add_special_tokens=False)
        try:
            wrapped = [int(token) for token in wrapped]
        except (TypeError, ValueError) as exc:
            raise ValueError("chat template produced no prompt tokens") from exc
        if not wrapped:
            raise ValueError("chat template produced no prompt tokens")
        if len(wrapped) > length:
            raise ValueError(
                f"{item['request_id']}: chat template has {len(wrapped)} tokens, "
                f"planned prompt is {length}"
            )
        suffix = item["request_id"].rsplit("-", 1)[-1]
        pad_index = int(suffix) if suffix.isdigit() else 0
        pad = bench.build_prompt_tokens(tokenizer, length - len(wrapped), pad_index)
        item["prompt_token_ids"] = wrapped + pad


class Observer:
    """Instance-local transparent wrappers around real scheduler/runner calls."""

    def __init__(self, core, runner, report: dict, records: dict) -> None:
        self.core, self.runner, self.report, self.records = (
            core,
            runner,
            report,
            records,
        )
        self.scheduler = core.scheduler
        self.runtime = runner._paged_attention_runtime
        self.ids: dict[str, str] = {}
        self.internal_ids: dict[str, str] = {}
        self.started = 0.0
        self.by_output: dict[int, dict] = {}
        self.current = None
        self.pending_cancels: list[dict] = []
        self.last_phases: dict[str, str] = {}
        self.originals = []

    def now(self) -> float:
        return time.perf_counter() - self.started

    def external(self, request_id: str) -> str:
        return self.ids.get(request_id, request_id)

    def wrap(self, obj, name, function) -> None:
        self.originals.append((obj, name, getattr(obj, name)))
        setattr(obj, name, function)

    def install(self) -> None:
        schedule = self.scheduler.schedule

        def observed_schedule(*args, **kwargs):
            output = schedule(*args, **kwargs)
            cached = output.scheduled_cached_reqs
            step = {
                "sequence": len(self.report["steps"]) + 1,
                "at_s": self.now(),
                "batch_request_ids": [],
                "phases": {},
                "admitted_req_ids": [
                    self.external(r.req_id) for r in output.scheduled_new_reqs
                ],
                "preempted_req_ids": sorted(
                    self.external(r) for r in (output.preempted_req_ids or ())
                ),
                "resumed_req_ids": sorted(
                    self.external(r) for r in cached.resumed_req_ids
                ),
                "preempted_internal_req_ids": sorted(output.preempted_req_ids or ()),
                "resumed_internal_req_ids": sorted(cached.resumed_req_ids),
                "cancelled_requests": self.pending_cancels,
                "scheduled_tokens": {
                    self.external(r): n for r, n in output.num_scheduled_tokens.items()
                },
                "scheduler_running_ids": [
                    self.external(r.request_id) for r in self.scheduler.running
                ],
                "finished_req_ids": sorted(
                    self.external(r) for r in output.finished_req_ids
                ),
                "budget": getattr(output, "state_cache_budget_stats", None),
                "processed_step_seq": self.scheduler.processed_step_seq,
                "sched_step_seq": self.scheduler.sched_step_seq,
            }
            self.pending_cancels = []
            self.report["steps"].append(step)
            self.by_output[id(output)] = step
            if step["sequence"] % 100 == 0:
                print(
                    f"STEP {step['sequence']}: running={len(step['scheduler_running_ids'])}",
                    flush=True,
                )
            return output

        self.wrap(self.scheduler, "schedule", observed_schedule)
        forward = self.runner._start_paged_forward

        def observed_forward(batch, prefill_reqs, decode_reqs, scheduler_output):
            step = self.by_output[id(scheduler_output)]
            decode = [self.external(r) for r, _ in decode_reqs]
            prefill = [self.external(r.req_id) for r in prefill_reqs]
            step["batch_request_ids"] = decode + prefill
            step["batch_internal_req_ids"] = [r for r, _ in decode_reqs] + [
                r.req_id for r in prefill_reqs
            ]
            step["phases"] = {
                **dict.fromkeys(decode, "decode"),
                **dict.fromkeys(prefill, "prefill"),
            }
            self.last_phases.update(step["phases"])
            self.current = step
            try:
                result = forward(batch, prefill_reqs, decode_reqs, scheduler_output)
                step["forward_submitted"] = True
                return result
            finally:
                self.current = None

        self.wrap(self.runner, "_start_paged_forward", observed_forward)
        populate = self.runtime.populate_step_context

        def observed_populate(*args, **kwargs):
            result = populate(*args, **kwargs)
            if self.current is not None:
                context = kwargs["ctx"]
                self.current["state_request_ids"] = [
                    self.external(r) for r in kwargs["req_ids"]
                ]
                self.current["state_slot_mappings"] = [
                    list(row) for row in context.state_group_slot_mappings
                ]
                self.current["step_positions"] = [
                    list(p) for p in kwargs["step_positions"]
                ]
            return result

        self.wrap(self.runtime, "populate_step_context", observed_populate)
        finish = self.scheduler.finish_requests

        def observed_finish(request_ids, finished_status):
            before = {
                rid: {
                    "phase": self.last_phases.get(self.external(rid)),
                    "computed": r.num_computed_tokens,
                    "in_flight": r.num_in_flight_tokens,
                    "prompt_tokens": r.num_prompt_tokens,
                }
                for rid, r in self.scheduler.requests.items()
            }
            result = finish(request_ids, finished_status)
            if finished_status.name == "FINISHED_ABORTED":
                for request in result:
                    rid = self.external(request.request_id)
                    event = {
                        "request_id": rid,
                        "internal_request_id": request.request_id,
                        "at_s": self.now(),
                        **before[request.request_id],
                    }
                    self.pending_cancels.append(event)
                    self.report["abort_events"].append(event)
                    if rid in self.records:
                        self.records[rid].update(
                            finish_reason="abort", finished_at_s=event["at_s"]
                        )
            return result

        self.wrap(self.scheduler, "finish_requests", observed_finish)

    def restore(self) -> None:
        for obj, name, method in reversed(self.originals):
            setattr(obj, name, method)


def run(args) -> dict:
    from tools.continuous_batch_gate_checks import audit_run

    config = {
        name: getattr(args, name)
        for name in (
            "model",
            "scenario",
            "block_size",
            "max_num_seqs",
            "gpu_memory_utilization",
            "state_budget_mib",
            "num_gpu_blocks_override",
            "timeout_seconds",
            "max_steps",
            "async_scheduling",
            "waves",
            "wave_interval_seconds",
            "ignore_eos",
            "apply_chat_template",
        )
    }
    config.update(
        max_model_len=16 * args.block_size,
        max_num_batched_tokens=4 * args.block_size,
        long_prefill_token_threshold=args.block_size,
        enable_prefix_caching=True,
        transport="in_process_step_driven",
    )
    report = {
        "schema_version": 1,
        "kind": "continuous_batch_run",
        "status": "running",
        "config": config,
        "metadata": bench.source_metadata(),
        "requirements": {
            "continuous_admission": args.max_num_seqs > 1,
            "mixed_lengths": args.max_num_seqs > 1,
            "reorder": args.scenario == "mixed" and args.max_num_seqs > 1,
            "preempt_resume": args.scenario == "pressure",
            "cancel": args.scenario == "mixed",
            "min_peak_concurrency": min(args.max_num_seqs, 2)
            if args.scenario == "pressure"
            else args.max_num_seqs,
        },
        "planned_requests": [],
        "expected_request_ids": [],
        "requests": [],
        "steps": [],
        "abort_events": [],
        "client_abort_calls": [],
        "limitations": [
            "Real engine scheduling, in-process frontend; no HTTP or MP-lifecycle claim.",
            "Arrivals are submitted at engine-step boundaries; actual lateness is recorded.",
            "Token times are frontend delivery times; a multi-token callback has equal timestamps.",
            "CPU metadata observers do not evaluate GPU state or change request ordering.",
        ],
    }
    report["metadata"]["tool_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    report["metadata"]["checks_sha256"] = hashlib.sha256(
        (ROOT / "tools/continuous_batch_gate_checks.py").read_bytes()
    ).hexdigest()
    observer = None
    try:
        report["metadata"]["weights_manifest"] = bench.local_weights_manifest(
            args.model
        )
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        import mlx.core as mx
        import psutil
        from vllm import LLM, SamplingParams, TokensPrompt
        from vllm.sampling_params import RequestOutputKind

        import vllm_metal

        report["metadata"].update(bench.verify_loaded_source(vllm_metal.__file__))
        report["metadata"].update(
            device_info=dict(mx.device_info()),
            system_memory_bytes=psutil.virtual_memory().total,
            effective_multiprocessing="0",
        )
        kwargs = {
            key: config[key]
            for key in (
                "model",
                "max_model_len",
                "max_num_batched_tokens",
                "max_num_seqs",
                "gpu_memory_utilization",
                "long_prefill_token_threshold",
            )
        }
        kwargs.update(
            enable_prefix_caching=True,
            enable_chunked_prefill=True,
            seed=0,
        )
        if args.state_budget_mib is not None:
            kwargs["additional_config"] = {
                "state_cache_budget_mib": args.state_budget_mib
            }
        if args.num_gpu_blocks_override is not None:
            kwargs["num_gpu_blocks_override"] = args.num_gpu_blocks_override
        if args.async_scheduling != "auto":
            kwargs["async_scheduling"] = args.async_scheduling == "on"
        report["engine_kwargs"] = kwargs
        write_json(args.output, report)
        start = time.perf_counter()
        llm = LLM(**kwargs)
        report["startup_seconds"] = time.perf_counter() - start
        engine = llm.llm_engine
        core = engine.engine_core.engine_core
        runner = core.model_executor.driver_worker.worker.model_runner
        assert runner._paged_block_size == args.block_size, (
            "model block geometry mismatch"
        )
        tokenizer = llm.get_tokenizer()
        hf_config = engine.model_config.hf_config
        report["metadata"]["model_config_sha256"] = bench.digest(hf_config.to_dict())
        report["metadata"]["chat_template_sha256"] = bench.digest(
            tokenizer.chat_template
        )
        report["metadata"]["chat_template_applied"] = bool(args.apply_chat_template)
        plan = workload(
            tokenizer,
            args.block_size,
            args.scenario,
            args.waves,
            args.wave_interval_seconds,
            ignore_eos=args.ignore_eos,
        )
        if args.apply_chat_template:
            apply_chat_template_to_plan(tokenizer, plan)
        records = {
            item["request_id"]: {
                **item,
                "output_token_ids": [],
                "token_timestamps_s": [],
                "submitted_at_s": None,
                "first_token_at_s": None,
                "finished_at_s": None,
                "finish_reason": None,
            }
            for item in plan
        }
        report.update(
            planned_requests=plan,
            expected_request_ids=list(records),
            requests=list(records.values()),
        )
        report["runtime_before"] = bench.runtime_snapshot(runner)
        state_capacity = runner._paged_attention_runtime.state_slot_capacity
        quota_capacity = getattr(core.scheduler.kv_cache_manager, "capacity", None)
        report["budget_applied"] = state_capacity is not None
        if args.state_budget_mib is None:
            if state_capacity is not None or quota_capacity is not None:
                raise RuntimeError("baseline unexpectedly enabled a state quota")
        elif (
            type(state_capacity) is not int
            or state_capacity <= 0
            or quota_capacity != state_capacity
            or report["runtime_before"]["allocated_slots"] != state_capacity
        ):
            raise RuntimeError(
                "requested state budget was not materialized and enforced"
            )
        report["resolved"] = {
            "async_scheduling": engine.vllm_config.scheduler_config.async_scheduling,
            "num_gpu_blocks": engine.vllm_config.cache_config.num_gpu_blocks,
            "max_num_running_reqs": core.scheduler.max_num_running_reqs,
            "state_groups": len(runner._paged_state_group_indices),
            "max_concurrent_batches": engine.vllm_config.max_concurrent_batches,
            "scheduler_class": type(core.scheduler).__name__,
            "state_slot_capacity": state_capacity,
            "state_quota_capacity": quota_capacity,
        }
        process = psutil.Process()
        mx.synchronize()
        mx.reset_peak_memory()
        report["memory_before"] = bench.memory_snapshot(mx, process)
        observer = Observer(core, runner, report, records)
        observer.started = time.perf_counter()
        observer.install()
        submitted = set()
        cancel_called = set()
        iterations = 0
        while (
            len(submitted) < len(plan)
            or any(r["finished_at_s"] is None for r in records.values())
            or core.scheduler.has_requests()
            or bool(core.batch_queue)
        ):
            now = observer.now()
            if now > args.timeout_seconds or iterations >= args.max_steps:
                raise TimeoutError(
                    "continuous workload exceeded its explicit time/step limit"
                )
            for item in plan:
                rid = item["request_id"]
                if rid not in submitted and now >= item["arrival_offset_s"]:
                    record = records[rid]
                    record["submitted_at_s"] = observer.now()
                    record["arrival_lateness_s"] = (
                        record["submitted_at_s"] - item["arrival_offset_s"]
                    )
                    params = SamplingParams(
                        **item["sampling"],
                        max_tokens=item["max_tokens"],
                        ignore_eos=item["ignore_eos"],
                        output_kind=RequestOutputKind.DELTA,
                    )
                    internal = engine.add_request(
                        rid,
                        TokensPrompt(prompt_token_ids=item["prompt_token_ids"]),
                        params,
                    )
                    observer.ids[internal] = rid
                    observer.internal_ids[rid] = internal
                    record["internal_request_id"] = internal
                    submitted.add(rid)
            for item in plan:
                rid, stage = item["request_id"], item["cancel_stage"]
                if (
                    stage is None
                    or rid not in submitted
                    or rid in cancel_called
                    or records[rid]["finished_at_s"] is not None
                ):
                    continue
                internal = observer.internal_ids[rid]
                request = core.scheduler.requests.get(internal)
                phase = observer.last_phases.get(rid)
                ready = request is not None and phase == stage
                if stage == "prefill":
                    ready = (
                        ready
                        and 0 < request.num_computed_tokens < request.num_prompt_tokens
                    )
                else:
                    ready = ready and len(records[rid]["output_token_ids"]) >= 2
                if ready:
                    report["client_abort_calls"].append(
                        {"request_id": rid, "stage": stage, "at_s": observer.now()}
                    )
                    cancel_called.add(rid)
                    engine.abort_request([rid])
            if not core.scheduler.has_requests() and not core.batch_queue:
                if len(submitted) == len(plan):
                    break
                time.sleep(0.001)
                continue
            outputs = engine.step()
            iterations += 1
            arrived = observer.now()
            for output in outputs:
                rid = observer.external(output.request_id)
                record = records[rid]
                value = output.outputs[0]
                tokens = list(value.token_ids)
                if tokens and record["first_token_at_s"] is None:
                    record["first_token_at_s"] = arrived
                record["output_token_ids"].extend(tokens)
                record["token_timestamps_s"].extend([arrived] * len(tokens))
                if output.finished:
                    record.update(
                        finished_at_s=arrived, finish_reason=value.finish_reason
                    )
        if observer.pending_cancels:
            raise RuntimeError("engine ended before cancelled-request metadata drained")
        mx.synchronize()
        report["elapsed_seconds"] = observer.now()
        report["memory_after"] = bench.memory_snapshot(mx, process)
        report["runtime_after"] = bench.runtime_snapshot(runner)
        report["decode_pipeline_submissions"] = runner._decode_pipeline._pending_serial
        report["remaining_scheduler_requests"] = list(core.scheduler.requests)
        for record in records.values():
            record["decoded_text"] = tokenizer.decode(record["output_token_ids"])
        report["status"] = "completed"
        report["audit"] = audit_run(report)
    except Exception as error:
        report.update(
            status="failed",
            failure={
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        )
    finally:
        if observer is not None:
            observer.restore()
        write_json(args.output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run")
    p.add_argument("--model", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--scenario", choices=("mixed", "pressure"), default="mixed")
    p.add_argument("--block-size", type=int, required=True)
    p.add_argument("--max-num-seqs", type=int, default=4)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.25)
    p.add_argument("--state-budget-mib", type=int)
    p.add_argument("--num-gpu-blocks-override", type=int)
    p.add_argument("--async-scheduling", choices=("auto", "on", "off"), default="auto")
    p.add_argument("--timeout-seconds", type=float, default=600)
    p.add_argument("--max-steps", type=int, default=10000)
    p.add_argument("--waves", type=int, default=3)
    p.add_argument("--wave-interval-seconds", type=float, default=2.0)
    p.add_argument(
        "--ignore-eos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="default True keeps the fixed-length gate; --no-ignore-eos allows configured EOS",
    )
    p.add_argument(
        "--apply-chat-template",
        action="store_true",
        help="rebuild planned prompts through the tokenizer chat template",
    )
    p = sub.add_parser("compare")
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--budget", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "compare":
        from tools.continuous_batch_gate_checks import compare_runs

        result = compare_runs(
            json.loads(args.baseline.read_text()), json.loads(args.budget.read_text())
        )
        write_json(args.output, result)
        print(json.dumps(result, indent=2))
        return 0 if str(result.get("status")).lower() == "pass" else 1
    if (
        args.block_size <= 0
        or args.max_num_seqs <= 0
        or not 0 < args.gpu_memory_utilization <= 1
    ):
        parser.error("positive block/concurrency and utilization in (0,1] are required")
    if args.state_budget_mib is not None and args.state_budget_mib <= 0:
        parser.error("state budget must be positive; omit it for the baseline")
    if args.num_gpu_blocks_override is not None and args.num_gpu_blocks_override <= 0:
        parser.error("KV block override must be positive")
    if args.scenario == "pressure" and args.num_gpu_blocks_override is None:
        parser.error("pressure runs require an explicit KV block override")
    if (
        args.waves <= 0
        or not math.isfinite(args.wave_interval_seconds)
        or args.wave_interval_seconds < 0
    ):
        parser.error("waves must be positive and the wave interval nonnegative")
    if (
        args.max_steps <= 0
        or not math.isfinite(args.timeout_seconds)
        or args.timeout_seconds <= 0
    ):
        parser.error("positive finite time and step limits are required")
    result = run(args)
    print(
        json.dumps({k: result.get(k) for k in ("status", "failure", "audit")}, indent=2)
    )
    return (
        0
        if result["status"] == "completed"
        and str(result.get("audit", {}).get("status")).lower() == "pass"
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
