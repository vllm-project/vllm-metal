# SPDX-License-Identifier: Apache-2.0
"""Engine-free adversarial tests of continuous batching evidence qualification."""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parent.parent / "tools/continuous_batch_gate_checks.py"
spec = importlib.util.spec_from_file_location("continuous_batch_gate_checks", TOOL)
checks = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = checks
spec.loader.exec_module(checks)


def _step(sequence, at, batch, *, admitted=(), preempted=(), resumed=(), cancelled=()):
    return {
        "sequence": sequence,
        "at_s": at,
        "batch_request_ids": list(batch),
        "phases": dict.fromkeys(batch, "decode" if sequence >= 2 else "prefill"),
        "admitted_req_ids": list(admitted),
        "preempted_req_ids": list(preempted),
        "resumed_req_ids": list(resumed),
        "cancelled_requests": list(cancelled),
    }


def _budget_version(value):
    result = copy.deepcopy(value)
    result["config"]["state_budget_mib"] = 512
    result["budget_applied"] = True
    result["resolved"].update(
        scheduler_class="StateBudgetAsyncScheduler",
        state_slot_capacity=12,
        state_quota_capacity=12,
    )
    for key in ("runtime_before", "runtime_after"):
        result[key].update(
            explicit_state_slot_capacity=12, slot_cap=12, allocated_slots=12
        )
    return result


def report(*, budget=False):
    plan = [
        {
            "request_id": req_id,
            "prompt_token_ids": list(range(length)),
            "max_tokens": 3,
            "ignore_eos": True,
            "sampling": {"temperature": 0, "logprobs": None, "seed": 0},
            "arrival_offset_s": arrival,
            "cancel_stage": stage,
        }
        for req_id, length, arrival, stage in (
            ("a", 16, 0.0, None),
            ("b", 4, 1.0, None),
            ("c", 8, 1.2, "prefill"),
            ("d", 6, 1.3, "decode"),
        )
    ]
    records = []
    for request, times, finish in zip(
        plan, ([2, 3, 4], [2, 3, 4], [], [2.8]), (5, 4.5, 2.2, 3.8), strict=True
    ):
        records.append(
            {
                **copy.deepcopy(request),
                "output_token_ids": [10 + i for i in range(len(times))],
                "finish_reason": "abort" if request["cancel_stage"] else "length",
                "submitted_at_s": request["arrival_offset_s"],
                "first_token_at_s": times[0] if times else None,
                "finished_at_s": finish,
                "token_timestamps_s": times,
            }
        )
    result = {
        "schema_version": 1,
        "kind": "continuous_batch_run",
        "status": "completed",
        "expected_request_ids": ["a", "b", "c", "d"],
        "planned_requests": plan,
        "config": {"workload_id": "mixed-v1", "state_budget_mib": None},
        "requirements": {
            "continuous_admission": True,
            "mixed_lengths": True,
            "reorder": True,
            "preempt_resume": True,
            "cancel": True,
            "min_peak_concurrency": 4,
        },
        "metadata": {
            "source": {
                "commit": "a" * 40,
                "dirty": False,
                "tracked_diff_sha256": "b" * 64,
            },
            "tool_sha256": "c" * 64,
            "checks_sha256": "f" * 64,
            "host_identity_sha256": "d" * 64,
            "weights_manifest": [{"name": "weights", "bytes": 8, "sha256": "e" * 64}],
            "packages": {"vllm": "same-version"},
            "device_info": {"device_name": "test"},
            "system_memory_bytes": 1024,
            "python": "3.12-test",
            "model_config_sha256": "1" * 64,
            "chat_template_sha256": "2" * 64,
        },
        "budget_applied": False,
        "resolved": {
            "scheduler_class": "AsyncScheduler",
            "async_scheduling": True,
            "max_concurrent_batches": 2,
            "state_slot_capacity": None,
            "state_quota_capacity": None,
            "max_num_running_reqs": 4,
        },
        "runtime_before": {
            "explicit_state_slot_capacity": None,
            "slot_cap": 128,
            "allocated_slots": 0,
        },
        "runtime_after": {
            "explicit_state_slot_capacity": None,
            "slot_cap": 128,
            "allocated_slots": 96,
        },
        "requests": records,
        "steps": [
            _step(0, 0.1, ["a"], admitted=["a"]),
            _step(1, 1.5, ["a", "b", "c", "d"], admitted=["b", "c", "d"]),
            _step(
                2,
                2.0,
                ["b", "d"],
                preempted=["a"],
                cancelled=[{"request_id": "c", "phase": "prefill", "at_s": 2.0}],
            ),
            _step(3, 2.6, ["b", "a", "d"], resumed=["a"]),
            _step(
                4,
                3.4,
                ["a", "b"],
                cancelled=[{"request_id": "d", "phase": "decode", "at_s": 3.4}],
            ),
        ],
    }
    return _budget_version(result) if budget else result


def test_complete_observed_lifecycle_and_exact_parity_pass():
    baseline, candidate = report(), report(budget=True)
    candidate["config"]["state_budget_mib"] = 512
    result = checks.compare_runs(baseline, candidate)
    assert result["status"] == "pass", result["problems"]
    audit = result["audits"]["candidate"]
    assert audit["coverage"]["peak_actual_batch_concurrency"] == 4
    assert audit["coverage"]["event_counts"]["continuous_admission"] == 3
    assert audit["coverage"]["preempted_and_resumed_requests"] == [
        {"request_id": "a", "preempted_sequence": 2, "resumed_sequence": 3}
    ]
    assert len(result["comparisons"]) == 4
    assert [r["complete_output_tokens_equal"] for r in result["comparisons"]] == [
        True,
        True,
        None,
        None,
    ]


@pytest.mark.parametrize("arm", ["baseline", "candidate"])
def test_missing_request_cannot_pass_even_when_other_requests_match(arm):
    reports = {"baseline": report(), "candidate": report(budget=True)}
    reports[arm]["requests"].pop()
    assert checks.compare_runs(**reports)["status"] == "fail"


def test_both_arms_losing_same_request_cannot_pass():
    invalid = report()
    invalid["requests"].pop()
    invalid["expected_request_ids"].pop()
    assert checks.compare_runs(invalid, invalid)["status"] == "fail"


@pytest.mark.parametrize(
    "change",
    [
        "output_token",
        "output_length",
        "prompt",
        "finish",
        "max_tokens",
        "ignore_eos",
        "arrival",
        "sampling",
    ],
)
def test_full_request_identity_and_output_are_compared(change):
    candidate = report(budget=True)
    r = candidate["requests"][0]
    if change == "output_token":
        r["output_token_ids"][-1] = 999
    elif change == "output_length":
        r["output_token_ids"].pop()
        r["token_timestamps_s"].pop()
    elif change == "prompt":
        r["prompt_token_ids"][-1] = 999
    elif change == "finish":
        r["finish_reason"] = "stop"
    elif change == "max_tokens":
        r["max_tokens"] = 2
    elif change == "ignore_eos":
        r["ignore_eos"] = False
    elif change == "arrival":
        r["arrival_offset_s"] = 0.1
    else:
        r["sampling"]["seed"] = 1
    assert checks.compare_runs(report(), candidate)["status"] == "fail"


def test_short_stop_is_allowed_only_when_eos_is_enabled_and_nonempty():
    r = report()
    r["planned_requests"][0]["ignore_eos"] = False
    r["requests"][0].update(
        ignore_eos=False,
        finish_reason="stop",
        output_token_ids=[10],
        token_timestamps_s=[2],
    )
    assert checks.audit_run(r)["status"] == "pass"
    r["requests"][0].update(
        output_token_ids=[], token_timestamps_s=[], first_token_at_s=None
    )
    assert checks.audit_run(r)["status"] == "fail"


def test_reorder_requires_inversion_not_only_shrinking_membership():
    r = report()
    r["steps"][3]["batch_request_ids"] = ["a", "b", "d"]
    # (a,b,c,d)->(b,d)->(a,b,d)->(a,b) preserves every surviving pair.
    result = checks.audit_run(r)
    assert result["status"] == "fail"
    assert result["coverage"]["event_counts"]["reorder"] == 0


def test_resume_without_actual_preemption_cannot_qualify():
    r = report()
    r["steps"][2]["preempted_req_ids"] = []
    result = checks.audit_run(r)
    assert result["status"] == "fail"
    assert "earlier preemption" in result["problems"][0]


def test_preemption_without_resume_cannot_qualify():
    r = report()
    r["steps"][3]["resumed_req_ids"] = []
    assert checks.audit_run(r)["status"] == "fail"


def test_no_preemption_and_no_resume_is_not_coverage():
    r = report()
    r["steps"][2]["preempted_req_ids"] = []
    r["steps"][3]["resumed_req_ids"] = []
    result = checks.audit_run(r)
    assert result["status"] == "fail"
    assert "required event was not observed: preempt_resume" in result["problems"]
    comparison = checks.compare_runs(r, _budget_version(r))
    assert comparison["status"] == "fail"
    assert len(comparison["comparisons"]) == 4
    assert comparison["comparisons"][0]["complete_output_tokens_equal"] is True


def test_client_abort_call_without_engine_observation_cannot_pass():
    r = report()
    r["steps"][4]["cancelled_requests"] = []
    r["requests"][3]["abort_requested_at_s"] = 3.0
    result = checks.audit_run(r)
    assert result["status"] == "fail"
    assert "not observed by engine" in result["problems"][0]


def test_wrong_cancellation_phase_cannot_pass():
    r = report()
    r["steps"][4]["cancelled_requests"][0]["phase"] = "prefill"
    assert checks.audit_run(r)["status"] == "fail"


def test_between_step_engine_cancellation_can_be_carried_after_completion():
    r = report()
    # finish_requests returns between schedule calls. Its event time also
    # becomes finished_at_s; only the next step can flush the pending event.
    r["requests"][2]["finished_at_s"] = 1.9
    r["steps"][2]["cancelled_requests"][0]["at_s"] = 1.9
    r["requests"][3]["finished_at_s"] = 3.1
    r["steps"][4]["cancelled_requests"][0]["at_s"] = 3.1
    result = checks.audit_run(r)
    assert result["status"] == "pass", result["problems"]
    assert [event["at_s"] for event in result["coverage"]["cancellations"]] == [
        1.9,
        3.1,
    ]
    assert checks.compare_runs(r, _budget_version(r))["status"] == "pass"


def test_cancellation_event_cannot_come_from_a_future_step():
    r = report()
    r["steps"][2]["cancelled_requests"][0]["at_s"] = 2.1
    result = checks.audit_run(r)
    assert result["status"] == "fail"
    assert "later than its carrying step" in result["problems"][0]


def test_cancellation_event_after_request_completion_is_rejected():
    r = report()
    r["requests"][2]["finished_at_s"] = 1.8
    r["steps"][2]["cancelled_requests"][0]["at_s"] = 1.9
    result = checks.audit_run(r)
    assert result["status"] == "fail"
    assert "cancellation outside lifetime" in result["problems"][0]


def test_decode_cancellation_uses_event_time_for_first_token_boundary():
    r = report()
    r["steps"][4]["cancelled_requests"][0]["at_s"] = 2.7
    result = checks.audit_run(r)
    assert result["status"] == "fail"
    assert "decode cancellation has no observed token" in result["problems"][0]


def test_missing_engine_cancellation_timestamp_cannot_fall_back_to_client_time():
    r = report()
    del r["steps"][2]["cancelled_requests"][0]["at_s"]
    r["requests"][2]["abort_requested_at_s"] = 1.9
    result = checks.audit_run(r)
    assert result["status"] == "fail"
    assert "invalid cancellation timestamp" in result["problems"][0]


def test_preempted_request_cannot_remain_in_the_same_worker_batch():
    r = report()
    r["steps"][2]["batch_request_ids"].insert(0, "a")
    r["steps"][2]["phases"]["a"] = "decode"
    result = checks.audit_run(r)
    assert result["status"] == "fail"
    assert "preempted request remains" in result["problems"][0]


def test_engine_cancelled_request_cannot_be_preempted_before_client_finish():
    r = report()
    r["steps"][3]["at_s"] = 2.1
    r["steps"][3]["preempted_req_ids"] = ["c"]
    result = checks.audit_run(r)
    assert result["status"] == "fail"
    assert "preemption after engine cancellation" in result["problems"][0]


def test_prefill_cancellation_must_match_last_real_worker_phase():
    r = report()
    r["steps"][1]["phases"]["c"] = "decode"
    result = checks.audit_run(r)
    assert result["status"] == "fail"
    assert "last observed worker phase" in result["problems"][0]


def test_decode_cancellation_needs_an_observed_decode_batch():
    r = report()
    for step in r["steps"]:
        if "d" in step["phases"]:
            step["phases"]["d"] = "prefill"
    result = checks.audit_run(r)
    assert result["status"] == "fail"
    assert "last observed worker phase" in result["problems"][0]


def test_tokens_cannot_be_delivered_before_first_actual_admission():
    r = report()
    r["requests"][0]["token_timestamps_s"] = [0.01, 0.02, 0.03]
    r["requests"][0]["first_token_at_s"] = 0.01
    result = checks.audit_run(r)
    assert result["status"] == "fail"
    assert "before actual engine admission" in result["problems"][0]


def test_cancelled_prefixes_are_preserved_but_not_full_output_parity():
    candidate = report(budget=True)
    candidate["requests"][3]["output_token_ids"] = [999, 888]
    candidate["requests"][3]["token_timestamps_s"] = [2.8, 3.0]
    result = checks.compare_runs(report(), candidate)
    assert result["status"] == "pass"
    prefix = result["comparisons"][3]["cancelled_prefix_audit"]
    assert prefix["baseline_token_ids"] == [10]
    assert prefix["candidate_token_ids"] == [999, 888]
    assert prefix["common_prefix_equal"] is False
    assert result["comparisons"][3]["complete_output_tokens_equal"] is None


def test_simultaneous_first_admissions_do_not_prove_continuous_arrivals():
    r = report()
    r["steps"] = r["steps"][1:]
    r["steps"][0]["admitted_req_ids"] = ["a", "b", "c", "d"]
    result = checks.audit_run(r)
    assert result["status"] == "fail"
    assert result["coverage"]["event_counts"]["continuous_admission"] == 0


def test_different_lengths_in_separate_runs_are_not_mixed_batch_coverage():
    r = report()
    for request in r["planned_requests"] + r["requests"]:
        request["prompt_token_ids"] = [1, 2]
    result = checks.audit_run(r)
    assert result["status"] == "fail"
    assert result["coverage"]["event_counts"]["mixed_lengths"] == 0


def test_mixed_prompt_lengths_do_not_imply_mixed_execution_phases():
    coverage = checks.audit_run(report())["coverage"]
    assert coverage["event_counts"]["mixed_lengths"] > 0
    assert coverage["prefill_decode_overlap_steps"] == 0
    assert coverage["prefill_decode_overlap_step_sequences"] == []


def test_prefill_decode_overlap_counts_actual_worker_phases():
    r = report()
    r["steps"][1]["phases"]["b"] = "decode"
    result = checks.audit_run(r)
    assert result["status"] == "pass"
    assert result["coverage"]["prefill_decode_overlap_steps"] == 1
    assert result["coverage"]["prefill_decode_overlap_step_sequences"] == [1]


def test_submitted_count_cannot_replace_actual_batch_peak():
    r = report()
    r["requirements"]["min_peak_concurrency"] = 5
    assert checks.audit_run(r)["status"] == "fail"


@pytest.mark.parametrize("key", checks.METADATA_IDENTITY)
def test_identity_difference_and_missing_identity_fail(key):
    r = report(budget=True)
    r["metadata"][key] = {"changed": True}
    assert checks.compare_runs(report(), r)["status"] == "fail"
    del r["metadata"][key]
    assert checks.compare_runs(report(), r)["status"] == "fail"


def test_requirements_cannot_be_silently_downgraded_in_one_arm():
    r = report(budget=True)
    r["requirements"]["reorder"] = False
    assert checks.compare_runs(report(), r)["status"] == "fail"
    del r["requirements"]["reorder"]
    assert checks.audit_run(r)["status"] == "fail"


@pytest.mark.parametrize(
    "change",
    [
        "empty",
        "duplicate_request",
        "duplicate_step",
        "missing_phase",
        "nan_time",
        "token_timestamp_count",
        "wrong_first_token",
        "wrong_schema",
        "not_completed",
    ],
)
def test_malformed_or_incomplete_evidence_fails_closed(change):
    r = report()
    if change == "empty":
        r["steps"] = []
    elif change == "duplicate_request":
        r["requests"].append(copy.deepcopy(r["requests"][0]))
    elif change == "duplicate_step":
        r["steps"][1]["sequence"] = 0
    elif change == "missing_phase":
        r["steps"][1]["phases"].pop("b")
    elif change == "nan_time":
        r["requests"][0]["finished_at_s"] = float("nan")
    elif change == "token_timestamp_count":
        r["requests"][0]["token_timestamps_s"].pop()
    elif change == "wrong_first_token":
        r["requests"][0]["first_token_at_s"] = 1
    elif change == "wrong_schema":
        r["schema_version"] = 2
    else:
        r["status"] = "failed"
    assert checks.audit_run(r)["status"] == "fail"


def test_delivery_metrics_preserve_window_counts_and_percentile_definition():
    result = checks.audit_run(report())
    metrics = result["performance"]
    assert metrics["ttft"]["samples_s"] == [1.0, 2.0]
    assert metrics["ttft"]["p50_s"] == 1.5
    assert metrics["ttft"]["p95_s"] == pytest.approx(1.95)
    assert metrics["ttft"]["p99_s"] == pytest.approx(1.99)
    assert metrics["inter_token_delivery_interval"]["samples_s"] == [1, 1, 1, 1]
    window = metrics["throughput_window"]
    assert (window["start_s"], window["end_s"], window["duration_s"]) == (0, 5, 5)
    assert window["completed_noncancelled_output_tokens"] == 6
    assert window["observed_cancelled_prefix_tokens"] == 1
    assert window["noncancelled_output_tokens_per_second"] == 1.2
    assert window["all_observed_output_tokens_per_second"] == 1.4


def test_codelivered_tokens_allow_zero_delivery_intervals():
    r = report()
    r["requests"][0]["token_timestamps_s"] = [2, 2, 4]
    result = checks.audit_run(r)
    assert result["status"] == "pass"
    assert result["performance"]["inter_token_delivery_interval"]["samples_s"] == [
        0,
        1,
        1,
        2,
    ]


def test_no_percentile_samples_are_unavailable_not_zero():
    result = checks.percentiles([])
    assert result == {
        "count": 0,
        "samples_s": [],
        "p50_s": None,
        "p95_s": None,
        "p99_s": None,
    }


def test_config_differences_beyond_budget_fail():
    r = report(budget=True)
    r["config"]["max_num_seqs"] = 8
    assert checks.compare_runs(report(), r)["status"] == "fail"


def test_first_step_with_multiple_delayed_requests_still_needs_overlap():
    r = report()
    for request in r["planned_requests"] + r["requests"]:
        request["arrival_offset_s"] = 0.0
    result = checks.audit_run(r)
    assert result["status"] == "fail"
    assert result["coverage"]["event_counts"]["continuous_admission"] == 0


def test_repeated_preemption_cannot_be_counted_as_two_recoveries():
    r = report()
    r["steps"][3]["preempted_req_ids"] = ["a"]
    assert checks.audit_run(r)["status"] == "fail"


@pytest.mark.parametrize(
    "missing", ["planned_requests", "expected_request_ids", "requirements", "metadata"]
)
def test_missing_fixed_plan_or_audit_contract_fails(missing):
    r = report()
    del r[missing]
    assert checks.compare_runs(r, r)["status"] == "fail"


def test_duplicate_or_unfingerprinted_weights_fail():
    r = report()
    r["metadata"]["weights_manifest"] *= 2
    assert checks.audit_run(r)["status"] == "fail"
    r = report()
    del r["metadata"]["weights_manifest"][0]["sha256"]
    assert checks.audit_run(r)["status"] == "fail"


def test_engine_is_not_imported_for_comparison(monkeypatch):
    import builtins

    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"vllm", "mlx", "torch", "vllm_metal"}:
            raise AssertionError(f"engine dependency imported: {name}")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    assert checks.compare_runs(report(), report(budget=True))["status"] == "pass"


@pytest.mark.parametrize("left,right", [(False, False), (True, True), (True, False)])
def test_baseline_and_candidate_roles_cannot_be_swapped_or_duplicated(left, right):
    result = checks.compare_runs(report(budget=left), report(budget=right))
    assert result["status"] == "fail"
    assert any(
        "baseline must" in p or "candidate must" in p for p in result["problems"]
    )


@pytest.mark.parametrize("value", [0, -1, True, 512.0, "512"])
def test_candidate_budget_must_be_a_positive_integer(value):
    r = report(budget=True)
    r["config"]["state_budget_mib"] = value
    assert checks.audit_run(r)["status"] == "fail"


@pytest.mark.parametrize(
    "change",
    [
        "not_applied",
        "before_unbounded",
        "after_unbounded",
        "cap_changed",
        "allocated_over_cap",
        "resolved_capacity_changed",
        "quota_missing",
        "plain_scheduler",
        "running_limit_zero",
        "runtime_missing",
    ],
)
def test_requested_budget_requires_matching_observed_activation(change):
    r = report(budget=True)
    if change == "not_applied":
        r["budget_applied"] = False
    elif change == "before_unbounded":
        r["runtime_before"]["explicit_state_slot_capacity"] = None
    elif change == "after_unbounded":
        r["runtime_after"]["explicit_state_slot_capacity"] = None
    elif change == "cap_changed":
        r["runtime_after"]["slot_cap"] = 13
    elif change == "allocated_over_cap":
        r["runtime_after"]["allocated_slots"] = 13
    elif change == "resolved_capacity_changed":
        r["resolved"]["state_slot_capacity"] = 11
    elif change == "quota_missing":
        r["resolved"]["state_quota_capacity"] = None
    elif change == "plain_scheduler":
        r["resolved"]["scheduler_class"] = "AsyncScheduler"
    elif change == "running_limit_zero":
        r["resolved"]["max_num_running_reqs"] = 0
    else:
        del r["runtime_before"]
    result = checks.compare_runs(report(), r)
    assert result["status"] == "fail"


@pytest.mark.parametrize("change", ["explicit", "quota", "scheduler", "applied"])
def test_baseline_must_have_no_observed_explicit_budget(change):
    r = report()
    if change == "explicit":
        r["runtime_before"]["explicit_state_slot_capacity"] = 12
    elif change == "quota":
        r["resolved"]["state_quota_capacity"] = 12
    elif change == "scheduler":
        r["resolved"]["scheduler_class"] = "StateBudgetAsyncScheduler"
    else:
        r["budget_applied"] = True
    assert checks.compare_runs(r, report(budget=True))["status"] == "fail"


def test_unknown_metadata_fields_cannot_silently_differ():
    base, candidate = report(), report(budget=True)
    base["metadata"]["additional_identity"] = {"version": 1}
    candidate["metadata"]["additional_identity"] = {"version": 2}
    result = checks.compare_runs(base, candidate)
    assert result["status"] == "fail"
    assert "metadata identity differs: additional_identity" in result["problems"]


def test_metadata_field_cannot_be_omitted_in_one_arm():
    base, candidate = report(), report(budget=True)
    base["metadata"]["additional_identity"] = "recorded"
    result = checks.compare_runs(base, candidate)
    assert result["status"] == "fail"
    assert "metadata identity differs: additional_identity" in result["problems"]


def test_auto_config_does_not_waive_different_observed_async_modes():
    base, candidate = report(), report(budget=True)
    base["config"]["async_scheduling"] = "auto"
    candidate["config"]["async_scheduling"] = "auto"
    candidate["resolved"].update(
        async_scheduling=False, scheduler_class="StateBudgetScheduler"
    )
    result = checks.compare_runs(base, candidate)
    assert result["status"] == "fail"
    assert "observed execution mode differs: async_scheduling" in result["problems"]


def test_observed_async_flag_must_match_scheduler_class():
    r = report(budget=True)
    r["resolved"]["async_scheduling"] = False
    result = checks.audit_run(r)
    assert result["status"] == "fail"
    assert "disagrees with scheduler class" in result["problems"][0]


@pytest.mark.parametrize("value", [None, 1])
def test_observed_async_flag_is_a_required_boolean(value):
    r = report(budget=True)
    if value is None:
        del r["resolved"]["async_scheduling"]
    else:
        r["resolved"]["async_scheduling"] = value
    assert checks.audit_run(r)["status"] == "fail"


@pytest.mark.parametrize("missing", [False, True])
def test_observed_batch_queue_depth_must_match_when_recorded(missing):
    base, candidate = report(), report(budget=True)
    if missing:
        del candidate["resolved"]["max_concurrent_batches"]
    else:
        candidate["resolved"]["max_concurrent_batches"] = 1
    result = checks.compare_runs(base, candidate)
    assert result["status"] == "fail"
    assert (
        "observed execution mode differs: max_concurrent_batches" in result["problems"]
    )


def test_matching_synchronous_execution_remains_valid():
    base, candidate = report(), report(budget=True)
    for r, scheduler in ((base, "Scheduler"), (candidate, "StateBudgetScheduler")):
        r["resolved"].update(
            async_scheduling=False, scheduler_class=scheduler, max_concurrent_batches=1
        )
    assert checks.compare_runs(base, candidate)["status"] == "pass"
