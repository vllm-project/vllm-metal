# SPDX-License-Identifier: Apache-2.0
"""Restore a prompt's KV from the host pool, then from disk after a restart.

`--num-gpu-blocks-override` keeps the device cache at 192 blocks, so four
filler prompts evict the probe. Each restore must come from the expected tier
and match, token for token, the probe served from the device prefix cache.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import tempfile
import time
import urllib.request
from pathlib import Path

import pytest

from tests.e2e.kv_offload_utils import (
    BLOCK_SIZE,
    FS_TIER,
    FULL_TOKENS,
    MAX_TOKENS,
    PROMPT_TOKENS,
    TOP_K,
    _parse_counters,
    _restore_result,
)
from tools.check_parity import http_generate, serving

MODEL = os.environ.get("QWEN3_MODEL_PATH")
DEVICE_BLOCKS = 192  # 3,072 tokens; the four fillers need 252 blocks


def _prompt(seed: int) -> dict:
    """Random token IDs, so another prompt's blocks give different output."""
    rng = random.Random(seed)
    ids = [rng.randrange(1_000, 100_000) for _ in range(PROMPT_TOKENS)]
    return {"prompt": f"seed {seed}", "input_ids": ids}


PROBE = _prompt(0)
FILLERS = [_prompt(seed) for seed in range(1, 5)]


def _counters(base_url: str) -> dict[str, float]:
    root = base_url.removesuffix("/v1")
    with urllib.request.urlopen(f"{root}/metrics", timeout=30) as response:
        return _parse_counters(response.read().decode())


def _generate(base_url: str, rows: list[dict]) -> list[dict]:
    return http_generate(base_url, MODEL, rows, MAX_TOKENS, top_k=TOP_K)


def _probe(base_url: str) -> tuple[dict, dict[str, float]]:
    """Send the probe and return its output and the counter changes."""
    before = _counters(base_url)
    [output] = _generate(base_url, [PROBE])
    after = _counters(base_url)
    return output, {key: after[key] - before[key] for key in after}


def _wait_for_disk(base_url: str, nbytes: float, timeout: float = 120) -> None:
    """Shutdown drops queued disk stores, so wait until they have landed."""
    deadline = time.monotonic() + timeout
    while (written := _counters(base_url)["disk_write"]) < nbytes:
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"disk tier {FS_TIER} wrote {written:g} of {nbytes:g} bytes"
            )
        time.sleep(0.5)


def _server(log_dir: Path, name: str, *args: str):
    return serving(
        MODEL,
        max_model_len=2048,
        max_num_seqs=4,
        gpu_memory_utilization=float(
            os.environ.get("PARITY_GPU_MEMORY_UTILIZATION", "0.65")
        ),
        log_path=log_dir / f"{name}.log",
        env=os.environ.copy(),
        prefix_caching=True,
        extra_args=[
            "--block-size",
            str(BLOCK_SIZE),
            "--num-gpu-blocks-override",
            str(DEVICE_BLOCKS),
            *args,
        ],
    )


@pytest.mark.slow
@pytest.mark.skipif(
    not MODEL and not os.getenv("CI"), reason="set QWEN3_MODEL_PATH to Qwen3-0.6B"
)
def test_restores_from_the_host_pool_and_from_disk(tmp_path) -> None:
    assert MODEL, "set QWEN3_MODEL_PATH to Qwen3-0.6B"  # CI must not skip

    # CI uploads tmp_path, and block files hold the prompts' KV. RUNNER_TEMP is
    # emptied after every job, so a store left by a killed run does not pile up.
    store = Path(tempfile.mkdtemp(prefix="kv-offload-", dir=os.getenv("RUNNER_TEMP")))
    tier = {"type": "fs", "root_dir": str(store), "max_size_gib": 2}
    args = (
        "--kv-offloading-size",
        "1",
        "--kv-transfer-config",
        json.dumps(
            {
                "kv_connector_extra_config": {
                    # Each prompt stores its 62 full blocks, as _wait_for_disk expects.
                    "offload_prompt_only": True,
                    "secondary_tiers": [tier],
                }
            }
        ),
    )
    result = {
        "probe_tokens": PROMPT_TOKENS,
        "device_blocks": DEVICE_BLOCKS,
        "restores": [],
    }

    def record(row: dict) -> None:
        result["restores"].append(row)
        partial = tmp_path / "restores.json.tmp"
        partial.write_text(json.dumps(result, indent=2))
        partial.replace(tmp_path / "restores.json")

    restored = False
    try:
        with _server(tmp_path, "offload", *args) as base_url:
            _probe(base_url)
            reference, delta = _probe(base_url)
            assert delta["local_cache_hit"] >= FULL_TOKENS, delta
            assert delta["external_kv_transfer"] == 0, delta
            _generate(base_url, FILLERS)
            # Proves writes are counted under FS_TIER. A changed read label
            # still fails the disk row.
            _wait_for_disk(base_url, 1)
            pool, pool_delta = _probe(base_url)
            record(
                _restore_result("host pool", reference, pool, pool_delta, disk=False)
            )
            restored = pool_delta["external_kv_transfer"] > 0
            if restored:
                # Every full block of the five prompts goes to disk, written
                # through. The restore gives the bytes per block.
                per_block = pool_delta["load_bytes"] * BLOCK_SIZE
                per_block /= pool_delta["external_kv_transfer"]
                blocks = (1 + len(FILLERS)) * (PROMPT_TOKENS // BLOCK_SIZE)
                _wait_for_disk(base_url, blocks * per_block)
        # Without a pool restore there is no block size to wait for, and a
        # restart would race the store queue. The pool row already fails.
        if restored:
            with _server(tmp_path, "restart", *args) as base_url:
                disk, disk_delta = _probe(base_url)
                record(
                    _restore_result(
                        "disk after restart", reference, disk, disk_delta, disk=True
                    )
                )
    finally:
        shutil.rmtree(store, ignore_errors=True)

    assert all(row["passed"] for row in result["restores"]), json.dumps(result)
