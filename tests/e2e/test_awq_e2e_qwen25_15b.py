# SPDX-License-Identifier: Apache-2.0
"""AWQ generation through the Metal paged runner."""

from __future__ import annotations

import gc

import mlx.core as mx
import pytest

_AWQ_REPO = "Qwen/Qwen2.5-1.5B-Instruct-AWQ"


@pytest.mark.slow
def test_awq_e2e_paged_runner_smoke():
    """Full vLLM stack: load 1.5B-AWQ, greedy 16 tokens, check 'Paris'
    appears. Guards against AWQ load regressions that produce token
    garbage even when individual unit tests pass.
    """
    from vllm import LLM, SamplingParams

    llm = None
    monkeypatch_ctx = pytest.MonkeyPatch.context()
    try:
        with monkeypatch_ctx as mp:
            mp.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

            llm = LLM(
                model=_AWQ_REPO,
                max_model_len=512,
                max_num_seqs=1,
                gpu_memory_utilization=0.3,
            )
            sp = SamplingParams(temperature=0, max_tokens=16)
            outputs = llm.generate(["The capital of France is"], sp)

            assert len(outputs) == 1
            text = outputs[0].outputs[0].text
            assert text, "AWQ load produced empty output"
            # Greedy first-token should be " Paris" (verified by the standalone
            # parity scripts; reproduced here as a regression guard).
            assert "Paris" in text, (
                f"expected 'Paris' in AWQ greedy continuation, got: {text!r}"
            )
    finally:
        # Drop Python references before returning Metal allocations to the pool.
        del llm
        gc.collect()
        mx.clear_cache()
