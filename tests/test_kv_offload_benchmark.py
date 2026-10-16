# SPDX-License-Identifier: Apache-2.0
"""Memory diagnostics for the KV offload serving benchmark."""

from argparse import Namespace

import pytest

from tools.benchmark import kv_offload_benchmark as bench


@pytest.mark.parametrize("ram", [16, 24, 32, 64, 128, 256])
def test_default_fraction_does_not_warn_solely_because_of_machine_size(
    monkeypatch, ram
):
    monkeypatch.setattr(bench, "total_ram_gb", lambda: ram)
    args = Namespace(
        memory_fraction=0.8, baseline_memory_fraction=0.0, ram_headroom_gib=None
    )

    assert bench.preflight_memory(args) == []


@pytest.mark.parametrize("ram", [24, 128])
def test_default_reserve_still_warns_about_high_memory_fractions(monkeypatch, ram):
    monkeypatch.setattr(bench, "total_ram_gb", lambda: ram)
    args = Namespace(
        memory_fraction=1.0, baseline_memory_fraction=0.0, ram_headroom_gib=None
    )

    warnings = bench.preflight_memory(args)

    assert len(warnings) == 1
    assert "--memory-fraction 1.0" in warnings[0]
    assert "estimated" in warnings[0]


@pytest.mark.parametrize(
    "headroom,expected_warning", [(0.0, False), (6.0, False), (8.0, True)]
)
def test_explicit_reserve_controls_the_warning(monkeypatch, headroom, expected_warning):
    monkeypatch.setattr(bench, "total_ram_gb", lambda: 24)
    args = Namespace(
        memory_fraction=1.0,
        baseline_memory_fraction=0.0,
        ram_headroom_gib=headroom,
    )

    assert bool(bench.preflight_memory(args)) is expected_warning


def test_baseline_override_is_checked_separately(monkeypatch):
    monkeypatch.setattr(bench, "total_ram_gb", lambda: 24)
    args = Namespace(
        memory_fraction=0.5, baseline_memory_fraction=1.0, ram_headroom_gib=None
    )

    warnings = bench.preflight_memory(args)

    assert len(warnings) == 1
    assert "--baseline-memory-fraction 1.0" in warnings[0]


def test_equal_baseline_override_does_not_duplicate_the_warning(monkeypatch):
    monkeypatch.setattr(bench, "total_ram_gb", lambda: 128)
    args = Namespace(
        memory_fraction=1.0, baseline_memory_fraction=1.0, ram_headroom_gib=None
    )

    assert len(bench.preflight_memory(args)) == 1


@pytest.mark.parametrize("headroom", ["-1", "nan", "inf"])
def test_invalid_reserve_is_rejected_before_starting_a_server(headroom, capsys):
    with pytest.raises(SystemExit) as exc:
        bench.main(["--model", "unused", "--ram-headroom-gib", headroom])

    assert exc.value.code == 2
    assert (
        "--ram-headroom-gib must be finite and non-negative" in capsys.readouterr().err
    )
