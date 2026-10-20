# SPDX-License-Identifier: Apache-2.0
"""Tests for multimodal helper import boundaries."""

from __future__ import annotations

import subprocess
import sys
import textwrap


def test_multimodal_helpers_do_not_import_v1_worker() -> None:
    code = textwrap.dedent(
        """
        import sys

        import vllm_metal.multimodal.embeddings
        import vllm_metal.multimodal.gemma4
        import vllm_metal.multimodal.gemma4.adapter
        import vllm_metal.multimodal.gemma4.geometry
        import vllm_metal.multimodal.gemma4.sidecar
        import vllm_metal.multimodal.nemotron_nano_vl
        import vllm_metal.multimodal.nemotron_nano_vl.adapter
        import vllm_metal.multimodal.qwen3_vl
        import vllm_metal.multimodal.qwen3_vl.adapter
        import vllm_metal.v1.mm.encoder_cache

        if "vllm_metal.v1.worker" in sys.modules:
            raise SystemExit("vllm_metal.v1.worker was imported")
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
