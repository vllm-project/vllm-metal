# SPDX-License-Identifier: Apache-2.0
"""Reject misleading serving evidence and isolate consecutive server arms."""

import contextlib
import json
import os
import signal
import subprocess
import sys
from copy import deepcopy
from types import SimpleNamespace

import pytest

from tools.benchmark.dspark_serving_benchmark import (
    COUNTERS,
    machine_state,
    metric_counts,
    position_acceptance,
    run_arm,
    scheduler_capacity,
    stop_server,
    summarize,
    validate_measurement,
    write_summary,
)


@pytest.mark.parametrize("arm", ["target", "dspark", "draft_model"])
@pytest.mark.parametrize("quantization", [None, "q4"])
def test_q4_server_option_only_applies_to_dspark(
    monkeypatch, tmp_path, arm, quantization
):
    args = SimpleNamespace(
        output_dir=tmp_path,
        target="target",
        dspark="dspark",
        draft="draft",
        max_model_len=128,
        concurrency=[1, 4],
        max_num_batched_tokens=32,
        gpu_memory_utilization=0.3,
        dspark_width=7,
        draft_width=3,
        dspark_draft_topk=64,
        dspark_draft_quantization=quantization,
    )

    class CommandCapturedError(Exception):
        pass

    def popen(command, **kwargs):
        if arm == "dspark" and quantization is not None:
            assert json.loads(command[command.index("--additional-config") + 1]) == {
                "dspark_draft_quantization": "q4"
            }
        else:
            assert "--additional-config" not in command
        raise CommandCapturedError

    monkeypatch.setattr(subprocess, "Popen", popen)
    with pytest.raises(CommandCapturedError):
        run_arm(args, arm, 1, [], {})


def measurement():
    result = {
        "completed": 2,
        "failed": 0,
        "input_lens": [3, 5],
        "output_lens": [8, 8],
        "total_input_tokens": 8,
        "total_output_tokens": 16,
        "errors": ["", ""],
        "duration": 2.0,
        "output_throughput": 8.0,
        "mean_ttft_ms": 10.0,
        "mean_tpot_ms": 20.0,
        "p95_ttft_ms": 12.0,
        "p95_tpot_ms": 25.0,
        "mean_e2el_ms": 200.0,
        "p95_e2el_ms": 220.0,
    }
    before = dict.fromkeys(COUNTERS, 0.0)
    before.update(requests=2.0, drafts=3.0, draft_tokens=21.0, accepted_tokens=9.0)
    after = {
        "requests": 4.0,
        "drafts": 10.0,
        "draft_tokens": 40.0,
        "accepted_tokens": 15.0,
        "preemptions": 1.0,
    }
    return result, before, after


def test_counts_exclude_created_series_and_per_position_acceptance():
    text = """# TYPE vllm:spec_decode_num_accepted_tokens counter
vllm:spec_decode_num_accepted_tokens_total{model_name="some model"} 10
vllm:spec_decode_num_accepted_tokens_created{model_name="some model"} 123
vllm:spec_decode_num_accepted_tokens_per_pos_total{position="0"} 10
vllm:request_success_total{finished_reason="length"} 2
vllm:request_success_total{finished_reason="stop"} 3
vllm:spec_decode_num_drafts_total 4
vllm:spec_decode_num_draft_tokens_total 12
vllm:num_preemptions_total 0
"""
    counts = metric_counts(text, arm="dspark")
    assert counts["requests"] == 5
    assert counts["accepted_tokens"] == 10
    assert counts["draft_tokens"] == 12
    assert counts["preemptions"] == 0


def test_target_only_can_omit_speculation_counters():
    text = "vllm:request_success_total 2\nvllm:num_preemptions_total 0\n"
    assert metric_counts(text, arm="target") == dict.fromkeys(COUNTERS, 0) | {
        "requests": 2
    }


@pytest.mark.parametrize("missing", COUNTERS)
@pytest.mark.parametrize("arm", ("dspark", "draft_model"))
def test_missing_counter_cannot_be_reported_as_zero(missing, arm):
    text = "\n".join(f"{name} 0" for key, name in COUNTERS.items() if key != missing)
    with pytest.raises(ValueError, match="Missing server counter"):
        metric_counts(text, arm=arm)


@pytest.mark.parametrize("missing", ("requests", "preemptions"))
def test_target_only_still_requires_workload_counters(missing):
    text = "\n".join(f"{name} 0" for key, name in COUNTERS.items() if key != missing)
    with pytest.raises(ValueError, match="Missing server counter"):
        metric_counts(text, arm="target")


@pytest.mark.parametrize("value", ("NaN", "+Inf", "-1", "0.5"))
def test_invalid_cumulative_counter_samples_are_rejected(value):
    text = "\n".join(f"{name} 0" for name in COUNTERS.values())
    text += f'\nvllm:num_preemptions_total{{engine="1"}} {value}\n'
    with pytest.raises(ValueError, match="Invalid server counter"):
        metric_counts(text, arm="dspark")


def test_measurement_uses_only_completed_timed_requests():
    result, before, after = measurement()
    assert validate_measurement(
        result, before, after, input_lens=[3, 5], output_len=8, arm="dspark"
    ) == {
        "requests": 2.0,
        "drafts": 7.0,
        "draft_tokens": 19.0,
        "accepted_tokens": 6.0,
        "preemptions": 1.0,
    }


def position_metrics(values, *, engine="0"):
    return "\n".join(
        "vllm:spec_decode_num_accepted_tokens_per_pos_total"
        f'{{engine="{engine}",position="{i}"}} {v}'
        for i, v in enumerate(values)
    )


def test_position_yields_exclude_warmup_and_aggregate_engines():
    result, before, after = measurement()
    counters = validate_measurement(
        result, before, after, input_lens=[3, 5], output_len=8, arm="dspark"
    )
    start = position_metrics([5, 3, 1]) + "\n" + position_metrics([0, 0, 0], engine="1")
    end = position_metrics([7, 4, 1]) + "\n" + position_metrics([1, 1, 1], engine="1")
    # A counter's creation timestamp is not an acceptance observation.
    end += '\nvllm:spec_decode_num_accepted_tokens_per_pos_created{position="0"} 123\n'
    assert position_acceptance(start, end, draft_width=3, counters=counters) == {
        "accepted_tokens_by_position": [3, 2, 1],
        "accepted_tokens_per_draft_by_position": [3 / 7, 2 / 7, 1 / 7],
        "mean_accepted_draft_tokens": 6 / 7,
    }


def test_absent_target_positions_and_zero_draft_yields_are_not_probabilities():
    counters = {"drafts": 0, "accepted_tokens": 0}
    assert position_acceptance("", "", draft_width=0, counters=counters) is None
    text = position_metrics([0, 0, 0])
    result = position_acceptance(text, text, draft_width=3, counters=counters)
    assert result == {
        "accepted_tokens_by_position": [0, 0, 0],
        "accepted_tokens_per_draft_by_position": None,
        "mean_accepted_draft_tokens": None,
    }


@pytest.mark.parametrize(
    "bad_snapshot",
    [
        "",  # missing data must not become a zero
        position_metrics([3, 2]),  # incomplete width
        position_metrics([3, 2, 1, 0]),  # unexpected position
        position_metrics([3, 2, 1]).replace('position="1"', 'position="bad"'),
        position_metrics([3, 2, 1]).replace('position="1"', 'position="01"'),
        position_metrics([3, 2, 1]) + "\n" + position_metrics([3]),
        *[position_metrics([3, 2, value]) for value in ("NaN", "+Inf", -1, 0.5)],
    ],
)
def test_invalid_position_evidence_is_rejected(bad_snapshot):
    with pytest.raises(ValueError):
        position_acceptance(
            position_metrics([0, 0, 0]),
            bad_snapshot,
            draft_width=3,
            counters={"drafts": 7, "accepted_tokens": 6},
        )


@pytest.mark.parametrize(
    "start,end,drafts,accepted",
    [
        (position_metrics([1, 0, 0]), position_metrics([0, 1, 0]), 7, 0),
        (position_metrics([0, 0, 0]), position_metrics([3, 2, 1], engine="1"), 7, 6),
        (position_metrics([0, 0, 0]), position_metrics([1, 2, 0]), 7, 3),
        (position_metrics([0, 0, 0]), position_metrics([3, 2, 1]), 2, 6),
        (position_metrics([0, 0, 0]), position_metrics([3, 2, 1]), 7, 5),
    ],
)
def test_position_deltas_must_describe_the_same_verified_prefixes(
    start, end, drafts, accepted
):
    with pytest.raises(ValueError):
        position_acceptance(
            start,
            end,
            draft_width=3,
            counters={"drafts": drafts, "accepted_tokens": accepted},
        )


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("completed", 1),
        ("failed", 1),
        ("input_lens", [5, 3]),
        ("output_lens", [8, 7]),
        ("total_output_tokens", 15),
        ("errors", ["", "truncated"]),
        ("errors", []),
        ("output_throughput", float("nan")),
        ("mean_tpot_ms", float("inf")),
        ("duration", 0.0),
    ],
)
def test_rejects_partial_or_mismatched_benchmark(key, value):
    result, before, after = measurement()
    result[key] = value
    with pytest.raises(ValueError):
        validate_measurement(
            result, before, after, input_lens=[3, 5], output_len=8, arm="dspark"
        )


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("requests", 5.0),  # an extra warmup/test request contaminated the window
        ("draft_tokens", 21.0),  # spec configured, but only the warmup drafted
        ("drafts", -1.0),
        ("accepted_tokens", 50.0),
        ("preemptions", float("nan")),
        ("requests", 4.5),
    ],
)
def test_rejects_invalid_or_non_drafting_server_evidence(key, value):
    result, before, after = measurement()
    after[key] = value
    with pytest.raises(ValueError):
        validate_measurement(
            result, before, after, input_lens=[3, 5], output_len=8, arm="dspark"
        )


def test_baseline_cannot_include_drafting():
    result, before, after = measurement()
    with pytest.raises(ValueError, match="baseline unexpectedly drafted"):
        validate_measurement(
            result, before, after, input_lens=[3, 5], output_len=8, arm="target"
        )


@pytest.fixture
def serving_rows():
    result, _, _ = measurement()
    rows = []
    for concurrency in (1, 4):
        for repeat in (1, 2):
            for arm, rate in (
                ("target", 8 * repeat),
                ("dspark", 6 * repeat),
                ("draft_model", 10 * repeat),
            ):
                bench = deepcopy(result)
                bench["output_throughput"] = rate
                rows.append(
                    {
                        "repeat": repeat,
                        "arm": arm,
                        "concurrency": concurrency,
                        "benchmark": bench,
                    }
                )
    return rows


def test_summary_pairs_repeats_in_execution_independent_order(serving_rows):
    for row in serving_rows:
        row["acceptance"] = (
            {"accepted_tokens_by_position": [3 * row["repeat"], row["repeat"]]}
            if row["arm"] != "target"
            else None
        )
    summary = summarize(serving_rows[::-1])
    dspark = next(row for row in summary if row["arm"] == "dspark")
    assert dspark["repeat_ids"] == [1, 2]
    assert dspark["paired_throughput_ratio_vs_target"] == [0.75, 0.75]
    assert dspark["median_paired_throughput_ratio_vs_target"] == 0.75
    assert dspark["median_output_tokens_per_s"] == 9
    assert dspark["acceptance_per_repeat"] == [
        {"accepted_tokens_by_position": [3, 1]},
        {"accepted_tokens_by_position": [6, 2]},
    ]


def test_summary_preserves_timings_without_claiming_parity(tmp_path, serving_rows):
    write_summary(tmp_path, serving_rows)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert len(summary) == 6
    assert all(row["sequences_per_repeat"] == 2 for row in summary)
    assert all("greedy_parity_passed" not in row for row in summary)
    assert all(len(row["output_tokens_per_s"]) == 2 for row in summary)


def test_ordinary_draft_comparison_is_optional(serving_rows):
    rows = [r for r in serving_rows if r["arm"] != "draft_model"]
    summary = summarize(rows)
    assert len(summary) == 4
    assert {r["arm"] for r in summary} == {"target", "dspark"}


@pytest.mark.parametrize(
    "problem", ["empty", "duplicate", "missing", "no_target", "no_spec"]
)
def test_incomplete_comparisons_cannot_produce_summary(tmp_path, serving_rows, problem):
    rows = serving_rows
    if problem == "empty":
        rows = []
    elif problem == "duplicate":
        rows.append(deepcopy(rows[0]))
    elif problem == "missing":
        rows.pop()
    elif problem == "no_target":
        rows = [r for r in rows if r["arm"] != "target"]
    else:
        rows = [r for r in rows if r["arm"] == "target"]
    with pytest.raises(ValueError):
        write_summary(tmp_path, rows)
    assert not (tmp_path / "summary.json").exists()


@pytest.mark.parametrize("key", ["input_lens", "output_lens"])
def test_summary_rejects_comparison_of_different_workloads(serving_rows, key):
    serving_rows[1]["benchmark"][key][0] += 1
    with pytest.raises(ValueError, match="different workloads"):
        summarize(serving_rows)


@pytest.mark.parametrize("rate", [0, -1, float("nan"), float("inf")])
def test_summary_rejects_invalid_throughput(serving_rows, rate):
    serving_rows[0]["benchmark"]["output_throughput"] = rate
    with pytest.raises(ValueError, match="finite and positive"):
        summarize(serving_rows)


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS process group lifecycle")
def test_stop_server_reaps_workers_even_after_parent_exits():
    # Model loading/startup failure can exit the API parent while a worker lives.
    program = """
import signal, subprocess, sys
child = subprocess.Popen([sys.executable, '-c',
    'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print("ready",flush=True); time.sleep(60)'],
    stdout=subprocess.PIPE, text=True)
assert child.stdout.readline().strip() == 'ready'
print(child.pid, flush=True)
"""
    server = subprocess.Popen(
        [sys.executable, "-c", program],
        start_new_session=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        child_pid = json.loads(server.stdout.readline())
        server.wait(timeout=10)
        stop_server(server)
        import psutil

        try:
            child = psutil.Process(child_pid)
            child.wait(timeout=10)
        except psutil.NoSuchProcess:
            pass
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(server.pid, signal.SIGKILL)
        server.wait(timeout=10)


def test_capacity_uses_scheduler_report_instead_of_physical_page_count():
    log = "(EngineCore pid=123) INFO CPU KV cache size: 30,400 tokens, Maximum concurrency for 2,048 tokens per request: 14.84x"
    assert scheduler_capacity(log) == {
        "tokens": 30400,
        "max_model_len": 2048,
        "max_concurrency_rounded": 14.84,
    }
    for invalid in ("", log + "\n" + log):
        with pytest.raises(ValueError, match="cache-capacity report"):
            scheduler_capacity(invalid)


def test_machine_state_survives_stalled_system_tools(monkeypatch):
    def stalled(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", stalled)
    state = machine_state()
    assert state["power"]["returncode"] is None
    assert "TimeoutExpired" in state["thermal"]["stderr"]
    assert state["gpu_device_utilization_percent"] is None
