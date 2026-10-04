# SPDX-License-Identifier: Apache-2.0
"""Ensure crossover measurements compare distinct, valid production paths."""

import json
from types import SimpleNamespace

import mlx.core as mx
import pytest

from tools.benchmark import tq_lane_verify as bench
from tools.benchmark.tq_e2e_arm import main as e2e_main
from tools.benchmark.tq_e2e_arm import prefix_probe_layout, validate_prefix_reuse
from vllm_metal.attention.impls import turboquant_prefill as tq_prefill


@pytest.mark.parametrize("kv_heads", [2, 8])
def test_crossover_materializes_below_policy_threshold(
    monkeypatch, capsys, force_tiled_prefill, kv_heads
):
    monkeypatch.setenv("VLLM_METAL_TQ_PREFILL", "1")
    monkeypatch.setenv("VLLM_METAL_TQ_PREFILL_MAX_MIB", "64")
    threshold = tq_prefill.min_prefill_tokens(8, kv_heads, 128)
    bench.measure(
        "below-policy",
        {"qlens": (32,), "context_lens": (257,), "n_kv_heads": kv_heads},
        reps=1,
        warmup=0,
        force_materialization=True,
    )
    row = json.loads(capsys.readouterr().out.strip())
    assert row["production_lane_selected"] is False
    assert row["lane_selected"] is True
    assert row["forced_materialization"] is True
    assert row["workspace_estimate_bytes"] > 0
    assert all(value > 0 for value in row["median_ms"].values())
    assert tq_prefill.min_prefill_tokens(8, kv_heads, 128) == threshold


def test_crossover_does_not_bypass_workspace_limit(monkeypatch, force_tiled_prefill):
    monkeypatch.setenv("VLLM_METAL_TQ_PREFILL", "1")
    monkeypatch.setenv("VLLM_METAL_TQ_PREFILL_MAX_MIB", "1")
    with pytest.raises(RuntimeError, match="materialization rejected"):
        bench.measure(
            "over-budget",
            {"qlens": (32,), "context_lens": (8192,)},
            reps=1,
            warmup=0,
            force_materialization=True,
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_benchmark_rejects_matching_nonfinite_outputs(monkeypatch, value):
    """Agreement between invalid arms must not produce a timing result."""
    case = SimpleNamespace(
        ctx=SimpleNamespace(kernel_metadata_cache={}),
        forward=lambda: mx.array([value]),
    )
    monkeypatch.setattr(bench, "build_case", lambda **_: case)
    with pytest.raises(RuntimeError, match="nonfinite attention output"):
        bench.measure("nonfinite", {}, reps=1, warmup=0)


def test_prefix_probe_uses_requested_history_and_exact_query_lengths():
    assert prefix_probe_layout(8192, [32, 64, 96, 128], 16, 8320) == (8192, 8320)
    assert prefix_probe_layout(None, [1, 9, 257], 16, 1024) == (32, 289)
    with pytest.raises(ValueError, match="positive multiple"):
        prefix_probe_layout(8193, [64], 16, 9000)
    with pytest.raises(ValueError, match="needs --prompt-tokens"):
        prefix_probe_layout(8192, [128], 16, 8192)


def test_prefix_probe_rejects_query_splitting_before_loading_model(monkeypatch, capsys):
    monkeypatch.setattr(
        "sys.argv",
        [
            "tq_e2e_arm.py",
            "--model",
            "unused",
            "--prefix-probe",
            "--query-tokens",
            "129",
            "--batch-tokens",
            "128",
        ],
    )
    with pytest.raises(SystemExit) as error:
        e2e_main()
    assert error.value.code == 2
    assert "one --batch-tokens step" in capsys.readouterr().err


@pytest.mark.parametrize(
    "query,eligible,selected", [(1, 0, 0), (64, 2, 2), (256, 2, 2)]
)
def test_prefix_probe_accepts_observed_policy_instead_of_fixed_128(
    query, eligible, selected
):
    row = {
        "arm": "tq",
        "prompt_tokens": 8192 + query,
        "num_cached_tokens": 8192,
        "dispatch": {
            "threshold_eligible_layer_calls": eligible,
            "lane_layer_calls": selected,
        },
    }
    validate_prefix_reuse({"num_cached_tokens": 0}, row, 8192, query)


@pytest.mark.parametrize(
    "cached,query,eligible,selected,error",
    [
        (8176, 64, 0, 0, "exact requested prefix"),
        (8192, 65, 0, 0, "different query length"),
        (8192, 64, 2, 0, "dispatch disagrees"),
        (8192, 64, 0, 2, "dispatch disagrees"),
    ],
)
def test_prefix_probe_rejects_wrong_hits_or_inactive_eligible_lane(
    cached, query, eligible, selected, error
):
    row = {
        "arm": "tq",
        "prompt_tokens": 8192 + query,
        "num_cached_tokens": cached,
        "dispatch": {
            "threshold_eligible_layer_calls": eligible,
            "lane_layer_calls": selected,
        },
    }
    with pytest.raises(RuntimeError, match=error):
        validate_prefix_reuse({"num_cached_tokens": 0}, row, 8192, 64)
