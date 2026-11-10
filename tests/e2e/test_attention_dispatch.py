# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5 paged attention through a real engine."""

from __future__ import annotations

import pytest


@pytest.mark.slow
def test_qwen35_paged_attention_hybrid():
    """Qwen3.5 hybrid model loads and generates with paged attention."""
    from vllm import LLM, SamplingParams

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

        llm = LLM(
            model="Qwen/Qwen3.5-0.8B",
            max_model_len=512,
            max_num_seqs=1,
            gpu_memory_utilization=0.3,
        )
        sp = SamplingParams(temperature=0, max_tokens=5)
        outputs = llm.generate(["The capital of France is"], sp)
        assert len(outputs) == 1
        assert len(outputs[0].outputs[0].token_ids) > 0
