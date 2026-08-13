# SPDX-License-Identifier: Apache-2.0
"""End-to-end parity of the body-only intermediate prefill path.

Thin wrapper around ``tools/intermediate_prefill_parity.py --quick`` — the
on-vs-off greedy parity harness.  The tool spawns two children on the same
chunked-prefill schedule: one runs the projection-free intermediate forward
(and asserts it really engaged), the other disables the capability so the
identical chunks run full forwards, then asserts token-identical outputs.
Run the tool directly for the full prompt matrix.

Not in CI — requires local hybrid GDN model weights (Qwen3.5-0.8B).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

TOOL = (
    Path(__file__).resolve().parent.parent / "tools" / "intermediate_prefill_parity.py"
)


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("VLLM_METAL_E2E", "1") != "1",
    reason="e2e disabled",
)
def test_intermediate_body_only_prefill_is_lossless_and_engages() -> None:
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
