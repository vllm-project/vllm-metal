# SPDX-License-Identifier: Apache-2.0
"""Metrics and result checks for the KV offload serving test."""

from __future__ import annotations

from prometheus_client.parser import text_string_to_metric_families

from tools.parity_utils import parity_report

BLOCK_SIZE = 16
PROMPT_TOKENS = 1003  # not block aligned, so the last partial block recomputes
FULL_TOKENS = PROMPT_TOKENS // BLOCK_SIZE * BLOCK_SIZE  # 992, the restorable part
MAX_TOKENS = 20
TOP_K = 5
FS_TIER = "1:fs"  # the only secondary tier configured by the serving test


def _parse_counters(text: str) -> dict[str, float]:
    """Prompt tokens by source, bytes loaded to the device, disk tier traffic."""
    totals = dict.fromkeys(
        (
            "local_compute",
            "local_cache_hit",
            "external_kv_transfer",
            "load_bytes",
            "disk_read",
            "disk_write",
        ),
        0.0,
    )
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            name, labels = sample.name, sample.labels
            if name == "vllm:prompt_tokens_by_source_total":
                key = labels.get("source")
            elif name == "vllm:kv_offload_load_bytes_total":
                key = "load_bytes"
            elif labels.get("tier") != FS_TIER:
                continue
            elif name == "vllm:kv_offload_tiering_read_bytes_total":
                key = "disk_read"
            elif name == "vllm:kv_offload_tiering_write_bytes_total":
                key = "disk_write"
            else:
                continue
            if key in totals:
                totals[key] += sample.value
    return totals


def _restore_result(name: str, expected: dict, output: dict, delta: dict, disk: bool):
    """One summary row. A restore loads all of the probe but its last partial
    block, through the connector, from the expected tier, with exact output."""
    status = parity_report(
        [expected],
        [output],
        max_tokens=MAX_TOKENS,
        top_k=TOP_K,
        labels=("prefix cache", "restore"),
    )["results"][0]["status"]
    if disk:
        tier_ok = delta["load_bytes"] > 0 and delta["disk_read"] >= delta["load_bytes"]
    else:
        tier_ok = delta["load_bytes"] > 0 and delta["disk_read"] == 0
    loaded = (
        delta["external_kv_transfer"] >= FULL_TOKENS
        and delta["local_cache_hit"] == 0
        and tier_ok
    )
    return {
        "restore": name,
        "restored_tokens": delta["external_kv_transfer"],
        "recomputed_tokens": delta["local_compute"],
        "disk_read_bytes": delta["disk_read"],
        "output": status,
        "passed": loaded and status == "EXACT",
    }
