# SPDX-License-Identifier: Apache-2.0
"""Tests for attention backend dispatch.

Verifies that Qwen/Qwen3.5-0.8B with paged attention raises on
linear attention layers until the Metal kernel is implemented.
"""

from __future__ import annotations

import pytest


@pytest.mark.slow
@pytest.mark.xfail(
    raises=(NotImplementedError, ValueError),
    reason=(
        "Qwen3.5 paged attention is blocked by two issues: "
        "(1) mlx_vlm does not support qwen3_5 model type yet (ValueError), "
        "(2) linear attention (GatedDeltaNet) Metal kernel not implemented (NotImplementedError). "
        "Once model loading is fixed, this test should hit NotImplementedError."
    ),
    strict=True,
)
def test_qwen35_paged_attention_raises_on_linear_layers():
    """Loading Qwen/Qwen3.5-0.8B with paged attention must eventually raise
    NotImplementedError on the linear attention layers.

    Currently fails earlier at model loading (mlx_vlm does not support
    qwen3_5).  Once that is resolved, the failure will move to the
    linear attention dispatch.
    """
    from vllm import LLM, SamplingParams

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        mp.setenv("VLLM_METAL_USE_PAGED_ATTENTION", "1")
        mp.setenv("VLLM_METAL_MEMORY_FRACTION", "0.2")

        llm = LLM(model="Qwen/Qwen3.5-0.8B", max_model_len=512, max_num_seqs=1)
        sp = SamplingParams(temperature=0, max_tokens=5)
        llm.generate(["Hello"], sp)
