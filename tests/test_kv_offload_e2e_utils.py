# SPDX-License-Identifier: Apache-2.0
"""Unit checks for the KV offload serving-test metrics and verdicts."""

from __future__ import annotations

import pytest

from tests.e2e.kv_offload_utils import (
    FS_TIER,
    MAX_TOKENS,
    _parse_counters,
    _restore_result,
)

SAMPLE_METRICS = f"""\
# TYPE vllm:prompt_tokens_by_source counter
vllm:prompt_tokens_by_source_total{{model_name="m",source="local_compute"}} 11.0
vllm:prompt_tokens_by_source_created{{model_name="m",source="local_compute"}} 1e9
vllm:prompt_tokens_by_source_total{{model_name="m",source="external_kv_transfer"}} 992.0
# TYPE vllm:kv_offload_load_bytes counter
vllm:kv_offload_load_bytes_total{{model_name="m"}} 100.0
# TYPE vllm:kv_offload_tiering_read_bytes counter
vllm:kv_offload_tiering_read_bytes_total{{model_name="m",tier="{FS_TIER}"}} 100.0
vllm:kv_offload_tiering_read_bytes_total{{model_name="m",tier="0:primary"}} 7.0
# TYPE vllm:kv_offload_tiering_write_bytes counter
vllm:kv_offload_tiering_write_bytes_total{{model_name="m",tier="{FS_TIER}"}} 300.0
"""


def test_counters_read_totals_for_the_disk_tier_only() -> None:
    counters = _parse_counters(SAMPLE_METRICS)
    assert counters["local_compute"] == 11
    assert counters["external_kv_transfer"] == 992
    assert counters["load_bytes"] == 100
    assert counters["disk_read"] == 100
    assert counters["disk_write"] == 300
    assert counters["local_cache_hit"] == 0


@pytest.mark.parametrize(
    "disk,changes,passed",
    [
        (False, {}, True),
        (False, {"disk_read": 1}, False),  # pool restore served from disk
        (True, {"disk_read": 100}, True),
        (True, {"disk_read": 40}, False),  # part of the restore from the pool
        (False, {"load_bytes": 0}, False),  # tokens counted, nothing loaded
        (False, {"external_kv_transfer": 900, "local_compute": 103}, False),
        (False, {"local_cache_hit": 992}, False),
        (False, {"output": [9]}, False),
    ],
)
def test_restore_row_requires_tier_coverage_and_exact_output(
    disk, changes, passed
) -> None:
    def row(tokens):
        return {
            "tokens": tokens,
            "text": "",
            "top_logprobs": [
                [{"id": t, "text": "", "logprob": 0.0, "rank": 1}] for t in tokens
            ],
        }

    changes = dict(changes)  # parametrize shares the dict across reruns
    expected = row(list(range(MAX_TOKENS)))
    output = row([*changes.pop("output", []), *range(MAX_TOKENS)][:MAX_TOKENS])
    delta = {
        "external_kv_transfer": 992.0,
        "local_compute": 11.0,
        "local_cache_hit": 0.0,
        "load_bytes": 100.0,
        "disk_read": 0.0,
    } | changes
    assert _restore_result("r", expected, output, delta, disk=disk)["passed"] is passed
