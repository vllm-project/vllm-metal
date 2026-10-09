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
                        "tokens": [{"tokens": [0, 1, 2]}, {"tokens": [3, 4, 5]}],
                    }
                )
    return rows


def test_summary_pairs_repeats_and_preserves_token_divergence(serving_rows):
    for row in serving_rows:
        if row["arm"] == "dspark":
            row["tokens"][1]["tokens"][2] = 6
    # A reversed execution order must still compare each repeat to its own baseline.
    summary = summarize(serving_rows[::-1])
    dspark = next(row for row in summary if row["arm"] == "dspark")
    assert dspark["paired_throughput_ratio_vs_target"] == [0.75, 0.75]
    assert dspark["exact_sequences_vs_target"] == [1, 1]
    assert dspark["median_output_tokens_per_s"] == 9
    assert not dspark["greedy_parity_passed"]
    assert (
        dspark["first_divergences_vs_target"]
        == [
            [
                {
                    "prompt_index": 1,
                    "token_index": 2,
                    "target_token_id": 5,
                    "arm_token_id": 6,
                }
            ]
        ]
        * 2
    )


def test_matching_sequences_pass_qualification(tmp_path, serving_rows):
    write_summary(tmp_path, serving_rows)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert len(summary) == 6
    assert all(row["greedy_parity_passed"] for row in summary)
    assert all(row["exact_sequences_vs_target"] == [2, 2] for row in summary)
    assert all(row["first_divergences_vs_target"] == [[], []] for row in summary)


def test_empty_run_cannot_pass_qualification(tmp_path):
    with pytest.raises(ValueError, match="No serving runs to qualify"):
        write_summary(tmp_path, [])
    assert not (tmp_path / "summary.json").exists()


@pytest.mark.parametrize("arm", ("dspark", "draft_model"))
@pytest.mark.parametrize(
    ("tokens", "index", "actual_id", "target_id"),
    [([6, 1, 2], 0, 6, 0), ([0, 1, 6], 2, 6, 2), ([0, 1], 2, None, 2)],
)
def test_divergence_fails_after_preserving_all_results(
    tmp_path, serving_rows, arm, tokens, index, actual_id, target_id
):
    row = next(
        r
        for r in serving_rows
        if r["arm"] == arm and r["repeat"] == 2 and r["concurrency"] == 4
    )
    row["tokens"][0]["tokens"] = tokens
    with pytest.raises(ValueError, match=f"Greedy token parity failed: {arm} c4"):
        write_summary(tmp_path, serving_rows[::-1])
    summary = json.loads((tmp_path / "summary.json").read_text())
    failed = [r for r in summary if not r["greedy_parity_passed"]]
    assert len(failed) == 1
    assert failed[0]["exact_sequences_vs_target"] == [2, 1]
    assert failed[0]["first_divergences_vs_target"] == [
        [],
        [
            {
                "prompt_index": 0,
                "token_index": index,
                "target_token_id": target_id,
                "arm_token_id": actual_id,
            }
        ],
    ]
    assert len(summary) == 6
    for result in summary:
        assert len(result["output_tokens_per_s"]) == 2
        assert len(result["paired_throughput_ratio_vs_target"]) == 2


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
