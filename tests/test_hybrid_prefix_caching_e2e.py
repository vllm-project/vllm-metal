# SPDX-License-Identifier: Apache-2.0
"""End-to-end correctness of align-mode prefix caching on hybrid GDN models.

Thin wrapper around ``tools/hybrid_apc_parity_matrix.py --quick`` — the
single cache-on/off parity harness.  The tool spawns cache-off and cache-on
children (Metal is not fork-safe), asserts config reach (the cached run
really resolved to align with ``mamba_block_size == block_size``), hit reach
(an admission with a block-aligned ``num_computed_tokens > 0`` — a genuine
restore), and token identity across a block-edge/divergent-suffix/repeat/
batch matrix.  Run the tool directly for the full two-arm gate.

Not in CI — requires local Qwen3.5-0.8B weights.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parent.parent / "tools" / "hybrid_apc_parity_matrix.py"


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("VLLM_METAL_E2E", "1") != "1",
    reason="e2e disabled",
)
def test_hybrid_align_prefix_caching_is_lossless_and_hits() -> None:
    result = subprocess.run(
        [sys.executable, str(TOOL), "--quick"],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert result.returncode == 0 and "PARITY PASS" in result.stdout, (
        f"parity gate failed (exit {result.returncode}):\n"
        f"{result.stdout[-4000:]}\n{result.stderr[-4000:]}"
    )
