# SPDX-License-Identifier: Apache-2.0
"""Engine-free audit of continuous-batch observations and exact output parity.

Schema 1 uses a fixed ``planned_requests`` / ``expected_request_ids`` plan,
complete ``requests``, and
chronological ``steps``. All times are seconds from the same run-local monotonic
origin. ``batch_request_ids`` is the worker's actual model-row order, not the
submission list. Admissions/preemptions/resumptions/cancellations must come from
observed scheduler lifecycle events. Client abort calls alone prove nothing.

Each request records prompt/output token IDs, planned arrival_offset_s and
cancel_stage (None/prefill/decode), finish_reason (length/stop/abort), submitted,
first-token and finished timestamps, and one token_timestamps_s entry per token.
Steps record sequence, at_s, batch_request_ids, phases, admitted_req_ids,
preempted_req_ids, resumed_req_ids, and cancelled_requests
({request_id, phase, at_s}). Cancellation at_s is the observed engine lifecycle
event time; the next scheduler step may carry it after the request has ended.
Plan and observed records both include max_tokens, ignore_eos and sampling
({temperature, logprobs, seed}); successful length completion must be complete.
Requirements explicitly select continuous_admission, mixed_lengths, reorder,
preempt_resume, cancel, and min_peak_concurrency. Neither absent events nor an
incomplete trace can earn a selected coverage PASS.
"""

from __future__ import annotations

import math
from itertools import combinations, pairwise
from typing import Any

SCHEMA_VERSION = 1
TARGETS = (
    "continuous_admission",
    "mixed_lengths",
    "reorder",
    "preempt_resume",
    "cancel",
)
# These are the only configuration differences permitted in a code/budget A/B.
VARIABLE_CONFIG = {"state_budget_mib"}
REQUEST_IDENTITY = (
    "prompt_token_ids",
    "arrival_offset_s",
    "cancel_stage",
    "max_tokens",
    "ignore_eos",
    "sampling",
)
METADATA_IDENTITY = (
    "source",
    "tool_sha256",
    "checks_sha256",
    "weights_manifest",
    "packages",
    "host_identity_sha256",
    "device_info",
    "system_memory_bytes",
    "python",
    "model_config_sha256",
    "chat_template_sha256",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def _sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


def _ids(value: Any, label: str, known: set[str] | None = None) -> list[str]:
    _require(isinstance(value, list), f"{label} must be an explicit list")
    _require(all(isinstance(v, str) and v for v in value), f"{label}: invalid ID")
    _require(len(value) == len(set(value)), f"{label}: duplicate ID")
    if known is not None:
        _require(set(value) <= known, f"{label}: unknown request ID")
    return value


def _tokens(value: Any, label: str) -> None:
    _require(isinstance(value, list), f"{label} must be a list")
    _require(
        all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in value),
        f"{label}: invalid token ID",
    )


def _planned_request(request: dict) -> None:
    req_id = request["request_id"]
    _tokens(request.get("prompt_token_ids"), f"{req_id} prompt")
    _require(bool(request["prompt_token_ids"]), f"{req_id}: empty prompt")
    _require(
        _number(request.get("arrival_offset_s")), f"{req_id}: invalid planned arrival"
    )
    _require("cancel_stage" in request, f"{req_id}: missing cancel_stage")
    _require(
        request["cancel_stage"] in (None, "prefill", "decode"),
        f"{req_id}: invalid cancel stage",
    )
    _require(
        type(request.get("max_tokens")) is int and request["max_tokens"] > 0,
        f"{req_id}: invalid max_tokens",
    )
    _require(type(request.get("ignore_eos")) is bool, f"{req_id}: missing ignore_eos")
    sampling = request.get("sampling", {})
    _require(isinstance(sampling, dict), f"{req_id}: missing sampling identity")
    _require(_number(sampling.get("temperature")), f"{req_id}: invalid temperature")
    _require(
        "logprobs" in sampling and sampling["logprobs"] is None,
        f"{req_id}: gate requires production path without logprobs",
    )
    _require(type(sampling.get("seed")) is int, f"{req_id}: missing sampling seed")


def _validate_budget_activation(report: dict) -> None:
    config = report["config"]
    _require("state_budget_mib" in config, "missing explicit budget configuration")
    requested = config["state_budget_mib"]
    _require(
        requested is None or (type(requested) is int and requested > 0),
        "state_budget_mib must be None or a positive integer",
    )
    enabled = requested is not None
    _require(
        type(report.get("budget_applied")) is bool
        and report["budget_applied"] == enabled,
        "observed budget_applied does not match requested budget",
    )
    resolved = report.get("resolved", {})
    _require(isinstance(resolved, dict), "missing resolved scheduler state")
    for key in (
        "scheduler_class",
        "state_slot_capacity",
        "state_quota_capacity",
        "max_num_running_reqs",
        "async_scheduling",
    ):
        _require(key in resolved, f"missing observed scheduler field: {key}")
    scheduler = resolved["scheduler_class"]
    _require(
        scheduler
        in (
            ("StateBudgetScheduler", "StateBudgetAsyncScheduler")
            if enabled
            else ("Scheduler", "AsyncScheduler")
        ),
        "observed scheduler class does not match budget role",
    )
    _require(
        type(resolved["async_scheduling"]) is bool,
        "observed async_scheduling must be a boolean",
    )
    _require(
        resolved["async_scheduling"]
        == (scheduler in ("AsyncScheduler", "StateBudgetAsyncScheduler")),
        "observed async mode disagrees with scheduler class",
    )
    if "max_concurrent_batches" in resolved:
        _require(
            type(resolved["max_concurrent_batches"]) is int
            and resolved["max_concurrent_batches"] > 0,
            "invalid observed max_concurrent_batches",
        )
    _require(
        type(resolved["max_num_running_reqs"]) is int
        and resolved["max_num_running_reqs"] > 0,
        "invalid observed running-request limit",
    )
    capacity = resolved["state_slot_capacity"]
    if enabled:
        _require(
            type(capacity) is int and capacity > 0,
            "budget has no observed positive state capacity",
        )
        _require(
            type(resolved["state_quota_capacity"]) is int
            and resolved["state_quota_capacity"] == capacity,
            "scheduler quota differs from physical state capacity",
        )
    else:
        _require(
            capacity is None and resolved["state_quota_capacity"] is None,
            "baseline unexpectedly has a state budget or quota",
        )
    for key in ("runtime_before", "runtime_after"):
        runtime = report.get(key, {})
        _require(isinstance(runtime, dict), f"{key}: missing runtime observations")
        for field in ("explicit_state_slot_capacity", "slot_cap", "allocated_slots"):
            _require(field in runtime, f"{key}: missing {field}")
        _require(
            type(runtime["slot_cap"]) is int and runtime["slot_cap"] > 0,
            f"{key}: invalid slot cap",
        )
        _require(
            type(runtime["allocated_slots"]) is int
            and 0 <= runtime["allocated_slots"] <= runtime["slot_cap"],
            f"{key}: allocated slots exceed observed cap",
        )
        if enabled:
            _require(
                type(runtime["explicit_state_slot_capacity"]) is int
                and runtime["explicit_state_slot_capacity"] == capacity
                and runtime["slot_cap"] == capacity,
                f"{key}: physical state budget is not active or changed capacity",
            )
        else:
            _require(
                runtime["explicit_state_slot_capacity"] is None,
                f"{key}: baseline unexpectedly has explicit state capacity",
            )


def _validate(report: dict) -> dict[str, dict]:
    _require(isinstance(report, dict), "report must be an object")
    _require(report.get("schema_version") == SCHEMA_VERSION, "unsupported schema")
    _require(report.get("kind") == "continuous_batch_run", "wrong report kind")
    _require(report.get("status") == "completed", "run is not completed")
    expected = _ids(report.get("expected_request_ids"), "expected_request_ids")
    _require(bool(expected), "request plan is empty")
    planned = report.get("planned_requests", [])
    _require(
        isinstance(planned, list) and bool(planned),
        "fixed planned_requests are missing",
    )
    _require(all(isinstance(r, dict) for r in planned), "invalid planned request")
    planned_ids = _ids([r.get("request_id") for r in planned], "planned_requests")
    _require(planned_ids == expected, "expected_request_ids differ from fixed plan")
    plan = {r["request_id"]: r for r in planned}
    for request in planned:
        _planned_request(request)
    _require(
        any(r["cancel_stage"] is None for r in planned),
        "gate requires at least one noncancelled completion",
    )
    config = report.get("config")
    _require(isinstance(config, dict) and bool(config), "config is missing")
    _validate_budget_activation(report)
    metadata = report.get("metadata", {})
    _require(isinstance(metadata, dict), "metadata is missing")
    for key in METADATA_IDENTITY:
        _require(bool(metadata.get(key)), f"metadata identity is missing: {key}")
    _require(
        isinstance(metadata["source"], dict) and bool(metadata["source"].get("commit")),
        "source commit is missing",
    )
    _require(
        _sha256(metadata["source"].get("tracked_diff_sha256")),
        "tracked source identity is missing",
    )
    for key in (
        "tool_sha256",
        "checks_sha256",
        "host_identity_sha256",
        "model_config_sha256",
        "chat_template_sha256",
    ):
        _require(_sha256(metadata[key]), f"invalid SHA-256 identity: {key}")
    _require(
        isinstance(metadata["device_info"], dict), "device identity must be an object"
    )
    _require(
        type(metadata["system_memory_bytes"]) is int
        and metadata["system_memory_bytes"] > 0,
        "invalid system memory identity",
    )
    _require(isinstance(metadata["python"], str), "invalid Python identity")
    _require(
        isinstance(metadata["packages"], dict), "dependency identity must be an object"
    )
    _require(
        isinstance(metadata["weights_manifest"], list), "weight identity must be a list"
    )
    weight_names = []
    for shard in metadata["weights_manifest"]:
        _require(isinstance(shard, dict), "invalid weight shard")
        _require(
            isinstance(shard.get("name"), str) and bool(shard["name"]),
            "missing weight shard name",
        )
        _require(
            type(shard.get("bytes")) is int and shard["bytes"] > 0,
            "invalid weight shard size",
        )
        _require(_sha256(shard.get("sha256")), "invalid weight shard SHA-256")
        weight_names.append(shard["name"])
    _require(len(weight_names) == len(set(weight_names)), "duplicate weight shard")
    requirements = report.get("requirements", {})
    _require(isinstance(requirements, dict), "requirements are missing")
    for target in TARGETS:
        _require(
            type(requirements.get(target)) is bool, f"missing requirement {target}"
        )
    peak = requirements.get("min_peak_concurrency")
    _require(type(peak) is int and peak >= 1, "invalid min_peak_concurrency")
    records = report.get("requests", [])
    _require(isinstance(records, list), "requests must be a list")
    _require(all(isinstance(r, dict) for r in records), "invalid request record")
    actual = _ids([r.get("request_id") for r in records], "requests")
    _require(set(actual) == set(expected), "observed requests do not match fixed plan")
    requests = {r["request_id"]: r for r in records}
    for req_id, request in requests.items():
        _require(
            all(key in request for key in REQUEST_IDENTITY),
            f"{req_id}: incomplete request identity",
        )
        _require(
            all(request[key] == plan[req_id][key] for key in REQUEST_IDENTITY),
            f"{req_id}: observed identity differs from fixed plan",
        )
        _tokens(request.get("output_token_ids"), f"{req_id} output")
        stage = request["cancel_stage"]
        reason = request.get("finish_reason")
        _require(
            reason in ("length", "stop", "abort"), f"{req_id}: invalid finish reason"
        )
        _require(
            (reason == "abort") == (stage is not None), f"{req_id}: unexpected abort"
        )
        size = len(request["output_token_ids"])
        _require(size <= request["max_tokens"], f"{req_id}: output exceeds max_tokens")
        if stage is None:
            if reason == "length":
                _require(
                    size == request["max_tokens"],
                    f"{req_id}: truncated length completion",
                )
            _require(
                not request["ignore_eos"] or reason == "length",
                f"{req_id}: unexpected early stop with ignore_eos",
            )
        for key in ("arrival_offset_s", "submitted_at_s", "finished_at_s"):
            _require(_number(request.get(key)), f"{req_id}: invalid {key}")
        start, end = request["submitted_at_s"], request["finished_at_s"]
        _require(
            start >= request["arrival_offset_s"],
            f"{req_id}: submitted before planned arrival",
        )
        _require(start <= end, f"{req_id}: finish precedes submission")
        stamps = request.get("token_timestamps_s")
        _require(isinstance(stamps, list), f"{req_id}: missing token timestamps")
        _require(
            len(stamps) == len(request["output_token_ids"]),
            f"{req_id}: token timestamp count",
        )
        _require(all(_number(v) for v in stamps), f"{req_id}: invalid token timestamp")
        _require(stamps == sorted(stamps), f"{req_id}: token timestamps are unordered")
        _require(
            all(start <= v <= end for v in stamps),
            f"{req_id}: token outside request lifetime",
        )
        _require(
            "first_token_at_s" in request, f"{req_id}: missing first-token timestamp"
        )
        _require(
            request["first_token_at_s"] == (stamps[0] if stamps else None),
            f"{req_id}: first-token timestamp does not match token observations",
        )
        _require(
            bool(stamps) or stage == "prefill", f"{req_id}: completed without output"
        )
    steps = report.get("steps", [])
    _require(isinstance(steps, list) and bool(steps), "step trace is missing")
    previous_sequence, previous_time = -1, -1.0
    known = set(expected)
    for step in steps:
        _require(isinstance(step, dict), "invalid step record")
        sequence, at = step.get("sequence"), step.get("at_s")
        _require(
            type(sequence) is int and sequence > previous_sequence,
            "step sequence is not increasing",
        )
        _require(
            _number(at) and at >= previous_time, "step timestamps are not increasing"
        )
        previous_sequence, previous_time = sequence, at
        for key in (
            "batch_request_ids",
            "admitted_req_ids",
            "preempted_req_ids",
            "resumed_req_ids",
        ):
            _ids(step.get(key), f"step {sequence} {key}", known)
        phases = step.get("phases")
        _require(isinstance(phases, dict), f"step {sequence}: missing phases")
        _require(
            set(phases) == set(step["batch_request_ids"]),
            f"step {sequence}: phases do not match batch",
        )
        _require(
            all(v in ("prefill", "decode") for v in phases.values()),
            f"step {sequence}: invalid phase",
        )
        _require(
            set(step["admitted_req_ids"]) <= set(step["batch_request_ids"]),
            f"step {sequence}: admission absent from actual batch",
        )
        cancellations = step.get("cancelled_requests")
        _require(
            isinstance(cancellations, list),
            f"step {sequence}: missing cancellation observations",
        )
        for event in cancellations:
            _require(isinstance(event, dict), "invalid cancellation event")
            _require(event.get("request_id") in known, "unknown cancellation request")
            _require(
                event.get("phase") in ("prefill", "decode"),
                "invalid cancellation phase",
            )
            _require(_number(event.get("at_s")), "invalid cancellation timestamp")
            _require(
                event["at_s"] <= at,
                "cancellation event is later than its carrying step",
            )
    return requests


def percentiles(values: list[float]) -> dict:
    """Linear interpolation at (N-1)*p, with no fabricated zero for no samples."""
    values = sorted(values)
    result: dict[str, Any] = {"count": len(values), "samples_s": values}
    for label, p in (("p50_s", 0.50), ("p95_s", 0.95), ("p99_s", 0.99)):
        if not values:
            result[label] = None
            continue
        index = (len(values) - 1) * p
        low, high = math.floor(index), math.ceil(index)
        result[label] = values[low] + (values[high] - values[low]) * (index - low)
    return result


def performance_metrics(requests: dict[str, dict]) -> dict:
    complete = [r for r in requests.values() if r["cancel_stage"] is None]
    cancelled = [r for r in requests.values() if r["cancel_stage"] is not None]
    first = min(r["submitted_at_s"] for r in requests.values())
    last = max(r["finished_at_s"] for r in requests.values())
    duration = last - first
    output_tokens = sum(len(r["output_token_ids"]) for r in complete)
    cancelled_tokens = sum(len(r["output_token_ids"]) for r in cancelled)
    return {
        "timestamp_source_note": "Client-observed timestamps; co-delivered tokens may have equal timestamps. Not a measurement of kernel decode latency.",
        "percentile_method": "linear interpolation at (N-1)*p; complete noncancelled requests only",
        "ttft": percentiles(
            [r["first_token_at_s"] - r["submitted_at_s"] for r in complete]
        ),
        "inter_token_delivery_interval": percentiles(
            [b - a for r in complete for a, b in pairwise(r["token_timestamps_s"])]
        ),
        "request_completion_latency": percentiles(
            [r["finished_at_s"] - r["submitted_at_s"] for r in complete]
        ),
        "throughput_window": {
            "start_s": first,
            "end_s": last,
            "duration_s": duration,
            "definition": "first request submission through last request termination; includes arrival gaps, prefill, queueing and cancellation work; excludes engine startup",
            "completed_noncancelled_requests": len(complete),
            "completed_noncancelled_output_tokens": output_tokens,
            "observed_cancelled_prefix_tokens": cancelled_tokens,
            "noncancelled_output_tokens_per_second": output_tokens / duration
            if duration
            else None,
            "all_observed_output_tokens_per_second": (output_tokens + cancelled_tokens)
            / duration
            if duration
            else None,
            "completed_noncancelled_requests_per_second": len(complete) / duration
            if duration
            else None,
        },
    }


def audit_run(report: dict) -> dict:
    """Return fail, rather than silently waive missing/invalid target evidence."""
    try:
        requests = _validate(report)
        return _audit_validated(report, requests)
    except (ValueError, KeyError, TypeError) as error:
        return {
            "status": "fail",
            "problems": [str(error)],
            "coverage": {},
            "performance": None,
        }


def _audit_validated(report: dict, requests: dict[str, dict]) -> dict:
    admissions: dict[str, int] = {}
    admission_times: dict[str, float] = {}
    pending_preemptions: dict[str, int] = {}
    cancelled_ids: set[str] = set()
    cancelled_at_by_id = {
        event["request_id"]: event["at_s"]
        for step in report["steps"]
        for event in step["cancelled_requests"]
    }
    phase_history: dict[str, list[tuple[float, str, int]]] = {}
    pair_order: dict[tuple[str, str], tuple[bool, int]] = {}
    continuous, mixed, reorder, recovered, cancellations = [], [], [], [], []
    prefill_decode_overlap = []
    peak = 0
    problems = []
    for step in report["steps"]:
        sequence, at, batch = step["sequence"], step["at_s"], step["batch_request_ids"]
        _require(
            not set(step["preempted_req_ids"]).intersection(batch),
            f"step {sequence}: preempted request remains in actual batch",
        )
        peak = max(peak, len(batch))
        for req_id in step["admitted_req_ids"]:
            _require(req_id not in admissions, f"{req_id}: duplicate initial admission")
            request = requests[req_id]
            _require(
                request["submitted_at_s"] <= at <= request["finished_at_s"],
                f"{req_id}: admission outside lifetime",
            )
            active = [
                other
                for other, admitted_sequence in admissions.items()
                if admitted_sequence < sequence
                and requests[other]["finished_at_s"] > at
                and other not in cancelled_ids
                and other not in pending_preemptions
            ]
            if request["arrival_offset_s"] > 0 and active:
                continuous.append(
                    {
                        "request_id": req_id,
                        "sequence": sequence,
                        "overlapping_request_ids": active,
                    }
                )
            admissions[req_id] = sequence
            admission_times[req_id] = at
            _require(
                all(stamp >= at for stamp in request["token_timestamps_s"]),
                f"{req_id}: output delivered before actual engine admission",
            )
        for req_id in batch:
            _require(req_id in admissions, f"{req_id}: batch row before admission")
            _require(
                req_id not in cancelled_ids, f"{req_id}: batch row after cancellation"
            )
            _require(
                requests[req_id]["submitted_at_s"]
                <= at
                <= requests[req_id]["finished_at_s"],
                f"{req_id}: batch row outside lifetime",
            )
            _require(
                req_id not in pending_preemptions or req_id in step["resumed_req_ids"],
                f"{req_id}: preempted request runs without resume",
            )
            _require(
                at < cancelled_at_by_id.get(req_id, math.inf),
                f"{req_id}: batch row after engine cancellation",
            )
            phase_history.setdefault(req_id, []).append(
                (at, step["phases"][req_id], sequence)
            )
        if len({len(requests[req_id]["prompt_token_ids"]) for req_id in batch}) > 1:
            mixed.append(sequence)
        if set(step["phases"].values()) == {"prefill", "decode"}:
            prefill_decode_overlap.append(sequence)
        positions = {req_id: index for index, req_id in enumerate(batch)}
        for pair in combinations(sorted(batch), 2):
            orientation = positions[pair[0]] < positions[pair[1]]
            old = pair_order.get(pair)
            if old is not None and old[0] != orientation:
                reorder.append(
                    {
                        "request_ids": list(pair),
                        "previous_sequence": old[1],
                        "sequence": sequence,
                    }
                )
            pair_order[pair] = (orientation, sequence)
        for req_id in step["preempted_req_ids"]:
            _require(req_id in admissions, f"{req_id}: preempted before admission")
            _require(
                at < cancelled_at_by_id.get(req_id, math.inf),
                f"{req_id}: preemption after engine cancellation",
            )
            _require(
                req_id not in pending_preemptions,
                f"{req_id}: repeated preemption without resume",
            )
            _require(
                requests[req_id]["submitted_at_s"]
                <= at
                <= requests[req_id]["finished_at_s"],
                f"{req_id}: preemption outside lifetime",
            )
            pending_preemptions[req_id] = sequence
        for req_id in step["resumed_req_ids"]:
            _require(
                at < cancelled_at_by_id.get(req_id, math.inf)
                and at <= requests[req_id]["finished_at_s"],
                f"{req_id}: resume after termination",
            )
            prior = pending_preemptions.pop(req_id, None)
            _require(
                prior is not None and prior < sequence,
                f"{req_id}: resume without an earlier preemption",
            )
            _require(req_id in batch, f"{req_id}: resumed request absent from batch")
            if requests[req_id]["cancel_stage"] is None:
                recovered.append(
                    {
                        "request_id": req_id,
                        "preempted_sequence": prior,
                        "resumed_sequence": sequence,
                    }
                )
        for event in step["cancelled_requests"]:
            req_id = event["request_id"]
            request = requests[req_id]
            cancelled_at = event["at_s"]
            _require(
                req_id not in cancelled_ids,
                f"{req_id}: duplicate cancellation observation",
            )
            _require(
                req_id in admissions, f"{req_id}: cancellation before engine admission"
            )
            _require(
                request["cancel_stage"] == event["phase"],
                f"{req_id}: cancellation stage mismatch",
            )
            _require(
                request["submitted_at_s"] <= cancelled_at <= request["finished_at_s"],
                f"{req_id}: cancellation outside lifetime",
            )
            _require(
                cancelled_at >= admission_times[req_id],
                f"{req_id}: cancellation event precedes actual admission",
            )
            observed_phase = next(
                (
                    (phase, phase_sequence)
                    for phase_at, phase, phase_sequence in reversed(
                        phase_history.get(req_id, [])
                    )
                    if phase_at <= cancelled_at
                ),
                None,
            )
            _require(
                observed_phase is not None and observed_phase[0] == event["phase"],
                f"{req_id}: cancellation phase differs from last observed worker phase",
            )
            if event["phase"] == "decode":
                _require(
                    request["first_token_at_s"] is not None
                    and request["first_token_at_s"] <= cancelled_at,
                    f"{req_id}: decode cancellation has no observed token",
                )
            else:
                _require(
                    not request["output_token_ids"],
                    f"{req_id}: prefill cancellation emitted output",
                )
            cancelled_ids.add(req_id)
            cancellations.append({**event, "sequence": sequence})
    _require(
        set(admissions) == set(requests), "some planned requests were never admitted"
    )
    planned_cancelled = {
        req_id for req_id, r in requests.items() if r["cancel_stage"] is not None
    }
    _require(
        cancelled_ids == planned_cancelled,
        "planned cancellation was not observed by engine",
    )
    _require(
        set(pending_preemptions) <= cancelled_ids, "preempted request never resumed"
    )
    counts = {
        "continuous_admission": len(continuous),
        "mixed_lengths": len(mixed),
        "reorder": len(reorder),
        "preempt_resume": len(recovered),
        "cancel": len(cancellations),
    }
    for name, count in counts.items():
        if report["requirements"][name] and count == 0:
            problems.append(f"required event was not observed: {name}")
    if peak < report["requirements"]["min_peak_concurrency"]:
        problems.append("actual batch peak concurrency is below required minimum")
    return {
        "status": "fail" if problems else "pass",
        "problems": problems,
        "coverage": {
            "requests": len(requests),
            "fully_completed_requests": len(requests) - len(cancelled_ids),
            "cancelled_requests": len(cancelled_ids),
            "peak_actual_batch_concurrency": peak,
            "event_counts": counts,
            "continuous_admissions": continuous,
            "mixed_length_step_sequences": mixed,
            "prefill_decode_overlap_steps": len(prefill_decode_overlap),
            "prefill_decode_overlap_step_sequences": prefill_decode_overlap,
            "reorders": reorder,
            "preempted_and_resumed_requests": recovered,
            "cancellations": cancellations,
        },
        "performance": performance_metrics(requests),
    }


def compare_runs(baseline: dict, candidate: dict) -> dict:
    """Compare every planned noncancelled request, without truncation or waiver."""
    audits = {"baseline": audit_run(baseline), "candidate": audit_run(candidate)}
    problems = [
        f"{arm}: {problem}"
        for arm, audit in audits.items()
        for problem in audit["problems"]
    ]
    comparisons = []
    # Missing requested coverage must keep the gate failed, while valid output
    # records can still explain whether the uncovered run had a parity change.
    if all(audit["performance"] is not None for audit in audits.values()):
        if baseline["config"]["state_budget_mib"] is not None:
            problems.append("baseline must have state_budget_mib=None")
        if candidate["config"]["state_budget_mib"] is None:
            problems.append("candidate must enable a positive state budget")
        for key in ("expected_request_ids", "planned_requests", "requirements"):
            if baseline[key] != candidate[key]:
                problems.append(f"{key} differ")
        for key in sorted(set(baseline["metadata"]) | set(candidate["metadata"])):
            if (
                key not in baseline["metadata"]
                or key not in candidate["metadata"]
                or baseline["metadata"][key] != candidate["metadata"][key]
            ):
                problems.append(f"metadata identity differs: {key}")
        configs = [
            {k: v for k, v in report["config"].items() if k not in VARIABLE_CONFIG}
            for report in (baseline, candidate)
        ]
        if configs[0] != configs[1]:
            problems.append("workload/sampling configuration differs")
        for key in ("async_scheduling", "max_concurrent_batches"):
            if (key in baseline["resolved"] or key in candidate["resolved"]) and (
                key not in baseline["resolved"]
                or key not in candidate["resolved"]
                or baseline["resolved"][key] != candidate["resolved"][key]
            ):
                problems.append(f"observed execution mode differs: {key}")
        left = {r["request_id"]: r for r in baseline["requests"]}
        right = {r["request_id"]: r for r in candidate["requests"]}
        for req_id in baseline["expected_request_ids"]:
            if req_id not in right:
                problems.append(f"{req_id}: missing candidate request")
                continue
            a, b = left[req_id], right[req_id]
            inputs_equal = all(a[key] == b[key] for key in REQUEST_IDENTITY)
            cancelled = a["cancel_stage"] is not None or b["cancel_stage"] is not None
            outputs_equal = a["output_token_ids"] == b["output_token_ids"]
            finish_equal = a["finish_reason"] == b["finish_reason"]
            item = {
                "request_id": req_id,
                "inputs_equal": inputs_equal,
                "finish_reason_equal": finish_equal,
                "cancelled": cancelled,
                "complete_output_tokens_equal": None if cancelled else outputs_equal,
                "baseline_output_tokens": len(a["output_token_ids"]),
                "candidate_output_tokens": len(b["output_token_ids"]),
            }
            if cancelled:
                size = min(len(a["output_token_ids"]), len(b["output_token_ids"]))
                item["cancelled_prefix_audit"] = {
                    "baseline_token_ids": a["output_token_ids"],
                    "candidate_token_ids": b["output_token_ids"],
                    "common_prefix_length_compared": size,
                    "common_prefix_equal": a["output_token_ids"][:size]
                    == b["output_token_ids"][:size],
                    "note": "observational only; cancelled requests do not qualify for complete-output parity",
                }
            if (
                not inputs_equal
                or not finish_equal
                or (not cancelled and not outputs_equal)
            ):
                problems.append(
                    f"{req_id}: input, finish reason or complete output differs"
                )
            comparisons.append(item)
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "continuous_batch_comparison",
        "status": "fail" if problems else "pass",
        "problems": problems,
        "audits": audits,
        "comparisons": comparisons,
    }
