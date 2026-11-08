# SPDX-License-Identifier: Apache-2.0
"""Runs the Gemma 4 vision smoke tool (tiny checkpoint, real vLLM engine)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[2] / "tools" / "gemma4_vision_smoke.py"


@pytest.mark.slow
@pytest.mark.network
@pytest.mark.skipif(os.environ.get("VLLM_METAL_E2E", "1") != "1", reason="e2e disabled")
def test_gemma4_vision_smoke(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(TOOL), "--workdir", str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert result.returncode == 0 and "SMOKE PASS" in result.stdout, (
        f"smoke failed (exit {result.returncode}):\n"
        f"{result.stdout[-4000:]}\n{result.stderr[-4000:]}"
    )
