# SPDX-License-Identifier: Apache-2.0
"""Memory preflight for the KV offload serving benchmark."""

from argparse import Namespace

import pytest

from tools.benchmark import kv_offload_benchmark as bench


@pytest.mark.parametrize("ram", [16, 24, 32, 64, 128, 256])
def test_default_fraction_does_not_warn_on_any_mac_size(monkeypatch, ram):
    monkeypatch.setattr(bench, "total_ram_gb", lambda: ram)

    assert bench.preflight_memory(Namespace(memory_fraction=0.8)) == []


@pytest.mark.parametrize("ram", [16, 24, 32, 64, 128])
def test_full_fraction_still_warns(monkeypatch, ram):
    monkeypatch.setattr(bench, "total_ram_gb", lambda: ram)

    warnings = bench.preflight_memory(Namespace(memory_fraction=1.0))

    assert len(warnings) == 1
    assert warnings[0].startswith("--memory-fraction 1.0 projects to")
