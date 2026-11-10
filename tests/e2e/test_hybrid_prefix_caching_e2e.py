# SPDX-License-Identifier: Apache-2.0
"""End-to-end correctness of align-mode prefix caching on hybrid GDN models.

``tools/hybrid_apc_parity_matrix.py --quick`` compares cache-off/on greedy
tokens across block edges, divergent suffixes, repeats, and a batch.
It asserts align mode, matching block sizes, and a real block-aligned prefix hit.
Both modes use a 544-token prefill cap to control shape-dependent rounding;
run the tool directly for the full diagnostic matrix.

Not in CI — requires local Qwen3.5-0.8B weights.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[2] / "tools" / "hybrid_apc_parity_matrix.py"


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
