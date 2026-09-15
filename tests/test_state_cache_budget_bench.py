# SPDX-License-Identifier: Apache-2.0
"""Engine-free checks for benchmark evidence completeness and strict comparison."""

from __future__ import annotations

import builtins
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

TOOL = Path(__file__).resolve().parent.parent / "tools/state_cache_budget_bench.py"
spec = importlib.util.spec_from_file_location("state_cache_budget_bench", TOOL)
bench = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = bench
spec.loader.exec_module(bench)


def args(tmp_path, *extra):
    parser = bench.make_parser()
    result = parser.parse_args(
        ["run", "--model", "fake", "--output", str(tmp_path / "run.json"), *extra]
    )
    bench.validate_args(result, parser)
    return result


def report(budget=False):
    result = {
        "schema_version": 2,
        "kind": "run",
        "status": "completed",
        "expected_cases": 2,
        "config": {
            "async_scheduling": "auto",
            "max_new_tokens": 2,
            "prompt_lengths": [2],
            "concurrency": [1],
            "repeats": 1,
            "max_model_len": 8,
            "max_num_batched_tokens": 4,
            "max_num_seqs": 1,
            "gpu_memory_utilization": 0.25,
            "state_budget_mib": 256 if budget else None,
        },
        "budget_applied": budget,
        "resolved_cache_config": {
            "block_size": 544,
            "mamba_block_size": 544,
            "mamba_cache_mode": "align",
            "prefix_cache_retention_interval": 0,
            "hash_block_size": None,
            "scheduler_hash_block_size": 544,
        },
        "sampling": {**bench.SAMPLING_IDENTITY, "max_tokens": 2},
        "resolved_scheduler": {"async_scheduling": True},
        "metadata": {
            "model_config_sha256": "config",
            "chat_template_sha256": "template",
            "weights_manifest": [
                {"name": "model.safetensors", "bytes": 8, "sha256": "a" * 64}
            ],
            "host_identity_sha256": "host",
            "platform": "test-platform",
            "machine": "arm64",
            "system_memory_bytes": 64 * 1024**3,
            "device_info": {"device_name": "Test M5 Pro"},
            "python": "test-python",
            "tool_sha256": "tool",
            "source": {"commit": "source-a"},
            "packages": {"mlx": {"version": "0.32.1", "direct_url": None}},
            "environment": {},
        },
        "records": [
            {
                "key": "one/hot",
                "phase": "hot",
                "cache_salt": "one",
                "prompt_token_ids": [[1, 2]],
                "output_token_ids": [[3, 4]],
                "elapsed_seconds": 0.5,
                "admitted_num_computed_tokens": [512],
                "memory_before": {"process_rss_bytes": 100},
                "memory_after": {"process_rss_bytes": 200},
            }
        ],
    }

    template = result["records"][0]
    result["records"] = [
        {
            **template,
            **case,
            "prompt_token_ids": [[1, 2]],
            "output_token_ids": [[3, 4]],
            "finish_reasons": ["length"],
            "admitted_num_computed_tokens": [512 if case["phase"] == "hot" else 0],
        }
        for case in bench.case_plan(SimpleNamespace(**result["config"]))
    ]
    return result


def test_cli_baseline_and_budget_are_explicit(tmp_path):
    assert args(tmp_path).state_budget_mib is None
    assert args(tmp_path, "--state-budget-mib", "4096").state_budget_mib == 4096
    assert "async_scheduling" not in bench.engine_kwargs(args(tmp_path))
    assert (
        bench.engine_kwargs(args(tmp_path, "--async-scheduling", "on"))[
            "async_scheduling"
        ]
        is True
    )
    assert (
        bench.engine_kwargs(args(tmp_path, "--async-scheduling", "off"))[
            "async_scheduling"
        ]
        is False
    )
    assert "additional_config" not in bench.engine_kwargs(args(tmp_path))


@pytest.mark.parametrize(
    "extra",
    [
        ("--prompt-lengths", "8192"),
        ("--state-budget-mib", "0"),
        ("--repeats", "0"),
        ("--gpu-memory-utilization", "1.1"),
        ("--concurrency", "1,1"),
    ],
)
def test_invalid_workload_is_rejected_before_engine_import(tmp_path, extra):
    with pytest.raises(SystemExit):
        args(tmp_path, *extra)


def test_cold_hot_and_churn_plan_has_stable_distinct_salts(tmp_path):
    plan = bench.case_plan(
        args(
            tmp_path, "--prompt-lengths", "100", "--concurrency", "1", "--repeats", "2"
        )
    )
    assert [case["phase"] for case in plan] == ["cold", "hot", "cold", "hot"]
    assert plan[0]["cache_salt"] == plan[1]["cache_salt"]
    assert plan[2]["cache_salt"] == plan[3]["cache_salt"]
    assert plan[0]["cache_salt"] != plan[2]["cache_salt"]


def test_prompts_are_exact_length_and_deterministic():
    tokenizer = SimpleNamespace(encode=lambda text, **kw: list(text.encode()))
    first = bench.build_prompt_tokens(tokenizer, 2000, 0)
    assert len(first) == 2000
    assert first == bench.build_prompt_tokens(tokenizer, 2000, 0)
    assert first != bench.build_prompt_tokens(tokenizer, 2000, 1)
    assert len(bench.build_prompt_tokens(tokenizer, 3, 0)) == 3


def test_whole_output_comparison_includes_length_and_prompt_identity():
    baseline = report()
    assert bench.compare_reports(baseline, report(True))["status"] == "pass"
    changed = report(True)
    changed["records"][0]["output_token_ids"][0].append(5)
    assert bench.compare_reports(baseline, changed)["status"] == "fail"
    changed = report(True)
    changed["records"][0]["prompt_token_ids"][0][0] = 9
    assert bench.compare_reports(baseline, changed)["status"] == "fail"


def test_incomplete_duplicate_and_empty_evidence_cannot_pass():
    baseline = report()
    budget = report(True)
    for invalid in (
        {**budget, "status": "failed"},
        {**budget, "expected_cases": 3},
        {**budget, "records": []},
        {**budget, "expected_cases": 2, "records": budget["records"] * 2},
    ):
        assert bench.compare_reports(baseline, invalid)["status"] == "fail"
    assert bench.compare_reports({"records": []}, {"records": []})["status"] == "fail"


def test_model_identity_and_salt_must_match():
    baseline = report()
    changed = report(True)
    changed["metadata"]["model_config_sha256"] = "other"
    assert bench.compare_reports(baseline, changed)["status"] == "fail"
    changed = report(True)
    changed["records"][0]["cache_salt"] = "other"
    assert bench.compare_reports(baseline, changed)["status"] == "fail"


def test_summary_reports_observed_memory_and_restores_without_saturation_claim():
    observed = report()
    observed["memory_after_startup"] = {"process_rss_bytes": 900}
    summary = bench.summarize_run(observed)
    assert summary["max_observed_memory"]["process_rss_bytes"] == 900
    assert summary["phases"]["hot"]["restored_admissions"] == 1
    assert summary["phases"]["hot"]["output_tokens_per_second"] == 4
    assert summary["pool_saturation_proven"] is False


def test_failure_is_saved_with_stage_and_exception_without_engine(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setattr(
        bench,
        "local_weights_manifest",
        lambda model: [{"name": "fake.safetensors", "bytes": 8, "sha256": "a" * 64}],
    )
    original = builtins.__import__

    def guarded(name, *a, **kw):
        if name == "mlx.core":
            raise ImportError("deliberate test import failure")
        return original(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", guarded)
    options = args(tmp_path)
    result = bench.run_benchmark(options)
    stored = json.loads(options.output.read_text())
    assert stored == result
    assert stored["status"] == "failed"
    assert stored["failure"]["stage"] == "imports"
    assert stored["failure"]["exception_type"] == "ImportError"
    assert stored["summary"]["complete"] is False


def test_optional_scheduler_payload_preserves_sequence_and_counters():
    output = SimpleNamespace(
        metal_state_cache=SimpleNamespace(sequence=8, resident_blocks=((1, 2), (3, 4))),
        state_cache_budget_stats={"evictions": 5, "stalls": 2},
    )
    telemetry = bench.optional_telemetry(output)
    assert telemetry["metal_state_cache"] == {"sequence": 8, "resident_blocks": 2}
    assert telemetry["state_cache_budget_stats"] == {"evictions": 5, "stalls": 2}
    assert bench.scheduler_snapshot(None)["max_num_running_reqs"] is None


def test_compare_cli_records_invalid_input_as_failure(tmp_path):
    source = tmp_path / "bad.json"
    source.write_text("not json")
    target = tmp_path / "comparison.json"
    assert (
        bench.main(
            [
                "compare",
                "--baseline",
                str(source),
                "--budget",
                str(source),
                "--output",
                str(target),
            ]
        )
        == 1
    )
    assert json.loads(target.read_text())["failure"]["stage"] == "compare"


@pytest.mark.parametrize(
    "field,value",
    [
        ("host_identity_sha256", "other-host"),
        ("device_info", {"device_name": "M5 Max"}),
        ("system_memory_bytes", 48 * 1024**3),
        ("packages", {"mlx": {"version": "0.32.0", "direct_url": None}}),
        (
            "packages",
            {"mlx": {"version": "0.32.1", "direct_url": {"url": "other-source"}}},
        ),
        ("packages", {}),
    ],
)
def test_cross_device_or_dependency_evidence_is_not_before_after(field, value):
    baseline, changed = report(), report(True)
    changed["metadata"][field] = value
    assert bench.compare_reports(baseline, changed)["status"] == "fail"


@pytest.mark.parametrize(
    "name,value",
    [("seed", 1), ("temperature", 1), ("logprobs", 2), ("ignore_eos", False)],
)
def test_sampling_identity_is_strict_even_if_both_reports_changed(name, value):
    baseline, changed = report(), report(True)
    baseline["sampling"][name] = changed["sampling"][name] = value
    assert bench.compare_reports(baseline, changed)["status"] == "fail"


def test_requested_and_resolved_async_policy_must_match():
    baseline, changed = report(), report(True)
    changed["config"]["async_scheduling"] = "off"
    assert bench.compare_reports(baseline, changed)["status"] == "fail"
    changed = report(True)
    changed["resolved_scheduler"]["async_scheduling"] = False
    assert bench.compare_reports(baseline, changed)["status"] == "fail"


def test_source_revision_and_editable_path_changes_are_explicit_but_allowed():
    baseline, changed = report(), report(True)
    baseline["metadata"]["packages"]["vllm-metal"] = {
        "version": "0.28",
        "direct_url": {"url": "file:///baseline"},
    }
    changed["metadata"]["packages"]["vllm-metal"] = {
        "version": "0.29",
        "direct_url": {"url": "file:///budget"},
    }
    changed["metadata"]["source"]["commit"] = "source-b"
    compared = bench.compare_reports(baseline, changed)
    assert compared["status"] == "pass"
    assert compared["source_comparison"]["same_source"] is False


def test_optional_zero_policy_is_not_collapsed_to_none():
    baseline, changed = report(), report(True)
    baseline["resolved_cache_config"]["prefix_cache_retention_interval"] = None
    compared = bench.compare_reports(baseline, changed)
    assert compared["status"] == "fail"
    assert (
        compared["baseline_capacity"]["resolved_cache_config"][
            "prefix_cache_retention_interval"
        ]
        is None
    )
    assert (
        compared["budget_capacity"]["resolved_cache_config"][
            "prefix_cache_retention_interval"
        ]
        == 0
    )


def test_loaded_source_must_belong_to_the_reported_checkout(tmp_path):
    checkout = tmp_path / "checkout"
    source = checkout / "vllm_metal" / "__init__.py"
    observed = bench.verify_loaded_source(str(source), checkout)
    assert observed["loaded_source_root"] == str(checkout)
    assert observed["loaded_source_file"] == str(source)
    with pytest.raises(RuntimeError, match="outside benchmark checkout"):
        bench.verify_loaded_source(
            str(tmp_path / "wheel/vllm_metal/__init__.py"), checkout
        )


def test_loaded_source_symlink_cannot_escape_checkout(tmp_path):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (checkout / "vllm_metal").symlink_to(foreign, target_is_directory=True)
    with pytest.raises(RuntimeError, match="outside benchmark checkout"):
        bench.verify_loaded_source(str(checkout / "vllm_metal/__init__.py"), checkout)


def test_comparing_baseline_to_itself_is_not_budget_evidence():
    baseline = report()
    compared = bench.compare_reports(baseline, baseline)
    assert compared["status"] == "fail"
    assert any("budget arm" in problem for problem in compared["problems"])


@pytest.mark.parametrize(
    "side,field,value",
    [
        ("baseline", "budget_applied", True),
        ("budget", "budget_applied", False),
        ("budget", "state_budget_mib", None),
        ("budget", "state_budget_mib", 0),
        ("budget", "state_budget_mib", True),
        ("baseline", "state_budget_mib", 256),
    ],
)
def test_arm_role_and_actual_activation_are_required(side, field, value):
    baseline, budget = report(), report(True)
    changed = baseline if side == "baseline" else budget
    target = changed if field == "budget_applied" else changed["config"]
    target[field] = value
    assert bench.compare_reports(baseline, budget)["status"] == "fail"


@pytest.mark.parametrize(
    "field,value",
    [
        ("block_size", 784),
        ("mamba_block_size", 784),
        ("mamba_cache_mode", "none"),
        ("hash_block_size", 16),
        ("scheduler_hash_block_size", 16),
        ("prefix_cache_retention_interval", None),
    ],
)
def test_resolved_cache_policy_must_match(field, value):
    baseline, budget = report(), report(True)
    budget["resolved_cache_config"][field] = value
    assert bench.compare_reports(baseline, budget)["status"] == "fail"


def test_configured_capacity_difference_is_allowed_under_same_policy():
    baseline, budget = report(), report(True)
    baseline["resolved_cache_config"]["num_gpu_blocks"] = 870
    budget["resolved_cache_config"]["num_gpu_blocks"] = 1710
    assert bench.compare_reports(baseline, budget)["status"] == "pass"


def test_manifest_hashes_all_local_shards_with_relative_names(tmp_path):
    first = tmp_path / "one.safetensors"
    first.write_bytes(b"first shard")
    nested = tmp_path / "weights"
    nested.mkdir()
    (nested / "two.safetensors").write_bytes(b"second shard")
    (tmp_path / "config.json").write_text("{}")
    manifest = bench.local_weights_manifest(tmp_path)
    assert [entry["name"] for entry in manifest] == [
        "one.safetensors",
        "weights/two.safetensors",
    ]
    assert manifest[0]["bytes"] == len(b"first shard")
    assert manifest[0]["sha256"] == bench.hashlib.sha256(b"first shard").hexdigest()
    first.write_bytes(b"same size!!")
    assert bench.local_weights_manifest(tmp_path) != manifest


def test_manifest_follows_huggingface_snapshot_file_symlinks(tmp_path):
    blob = tmp_path / "blob"
    blob.write_bytes(b"weight bytes")
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "model.safetensors").symlink_to(blob)
    assert bench.local_weights_manifest(snapshot) == [
        {
            "name": "model.safetensors",
            "bytes": len(b"weight bytes"),
            "sha256": bench.hashlib.sha256(b"weight bytes").hexdigest(),
        }
    ]


def test_missing_local_weights_fail_at_identity_stage(tmp_path):
    options = args(tmp_path, "--model", str(tmp_path))
    result = bench.run_benchmark(options)
    assert result["status"] == "failed"
    assert result["failure"]["stage"] == "weights_identity"
    assert result["failure"]["exception_type"] == "ValueError"
    with pytest.raises(ValueError, match="local model directory"):
        bench.local_weights_manifest("remote-org/not-downloaded-model")


@pytest.mark.parametrize(
    "manifest",
    [
        [],
        None,
        [{"name": "model.safetensors", "bytes": 8, "sha256": "b" * 64}],
        [{"name": "model.safetensors", "bytes": 9, "sha256": "a" * 64}],
        [{"name": "another.safetensors", "bytes": 8, "sha256": "a" * 64}],
    ],
)
def test_weight_fingerprint_is_required_and_must_match(manifest):
    baseline, budget = report(), report(True)
    budget["metadata"]["weights_manifest"] = manifest
    assert bench.compare_reports(baseline, budget)["status"] == "fail"


@pytest.mark.parametrize("field", ["observer_trace", "working_tensor_trace", "nested"])
def test_diagnostic_reports_cannot_qualify_even_if_both_outputs_match(field):
    baseline, budget = report(), report(True)
    for observed in (baseline, budget):
        observed["metadata"][field] = {
            "variant": "diagnostic_fallback_sync",
            "qualification_eligible": False,
        }
    compared = bench.compare_reports(baseline, budget)
    assert compared["status"] == "fail"
    assert any("diagnostic evidence" in p for p in compared["problems"])


def test_self_declared_smaller_case_count_cannot_hide_missing_matrix():
    baseline, budget = report(), report(True)
    for observed in (baseline, budget):
        observed["records"] = observed["records"][:1]
        observed["expected_cases"] = 1
    compared = bench.compare_reports(baseline, budget)
    assert compared["status"] == "fail"
    assert any("rebuilt from configuration" in p for p in compared["problems"])


@pytest.mark.parametrize(
    "field,value",
    [
        ("output_token_ids", [[]]),
        ("output_token_ids", []),
        ("output_token_ids", [[3]]),
        ("output_token_ids", [[3, 4], [3, 4]]),
        ("output_token_ids", [[True, 4]]),
        ("prompt_token_ids", [[1]]),
        ("finish_reasons", []),
        ("finish_reasons", [None]),
        ("finish_reasons", ["abort"]),
        ("finish_reasons", ["stop"]),
    ],
)
def test_matching_incomplete_requests_or_finish_reasons_cannot_pass(field, value):
    baseline, budget = report(), report(True)
    for observed in (baseline, budget):
        for record in observed["records"]:
            record[field] = value
    assert bench.compare_reports(baseline, budget)["status"] == "fail"


@pytest.mark.parametrize(
    "change", ["reorder", "missing_config", "wrong_phase", "wrong_schema"]
)
def test_matching_malformed_matrix_evidence_cannot_pass(change):
    baseline, budget = report(), report(True)
    for observed in (baseline, budget):
        if change == "reorder":
            observed["records"].reverse()
        elif change == "missing_config":
            del observed["config"]["prompt_lengths"]
        elif change == "wrong_phase":
            observed["records"][0]["phase"] = "hot"
        else:
            observed["schema_version"] = 1
    assert bench.compare_reports(baseline, budget)["status"] == "fail"


def test_comparison_identifies_its_own_code_separately_from_acquisition():
    baseline, budget = report(), report(True)
    compared = bench.compare_reports(baseline, budget)
    assert compared["status"] == "pass"
    assert (
        compared["comparator_sha256"]
        == bench.hashlib.sha256(TOOL.read_bytes()).hexdigest()
    )
    assert compared["baseline_metadata"]["tool_sha256"] == "tool"
