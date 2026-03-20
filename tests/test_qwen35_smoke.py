# SPDX-License-Identifier: Apache-2.0
"""Smoke test for Qwen3.5-0.8B: proves transformers 5.x model works end-to-end.

Qwen3.5 uses the `qwen3_5` architecture which requires transformers>=5.0.0.
This test verifies that the upgraded dependency stack (mlx-lm>=0.31.0,
mlx-vlm>=0.4.0, transformers>=5.0.0) works correctly with vLLM on Metal.

Golden token IDs were generated with greedy decoding (temperature=0) on
Qwen/Qwen3.5-0.8B, one sequence at a time (max_num_seqs=1).

Run:
    VLLM_ENABLE_V1_MULTIPROCESSING=0 \
        python -m pytest tests/test_qwen35_smoke.py -v -s
"""

from __future__ import annotations

import os

import pytest
from vllm import LLM, SamplingParams

MODEL_NAME = "Qwen/Qwen3.5-0.8B"
MAX_TOKENS = 10

PROMPTS = [
    "The capital of France is",
    "The weather today is not",
    "One plus one equals",
    "The largest planet in our solar system is",
    "Water boils at a temperature of",
]

# fmt: off
# Golden token IDs from MLX inline cache, greedy decoding (Qwen3.5-0.8B).
GOLDEN_MLX = {
    "The capital of France is":                     [11751, 13, 198, 760, 6511, 314, 9338, 369, 11751, 13],
    "The weather today is not":                     [1603, 13, 198, 760, 8831, 3242, 369, 524, 1603, 13],
    "One plus one equals":                          [1330, 13, 198, 3833, 5346, 799, 16327, 1330, 13, 198],
    "The largest planet in our solar system is":    [279, 7806, 13, 271, 248068, 271, 248069, 271, 332, 2665],
    "Water boils at a temperature of":              [220, 16, 15, 15, 29922, 13, 1368, 264, 1637, 369],
}
# fmt: on


def _setenv_default(mp: pytest.MonkeyPatch, key: str, default: str) -> str:
    """Set an env var only when absent and return the effective value."""
    value = os.environ.get(key)
    if value is None:
        mp.setenv(key, default)
        return default
    return value


@pytest.fixture(autouse=True, scope="module")
def _set_env():
    """Set default env vars for this test."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        _setenv_default(mp, "VLLM_METAL_MEMORY_FRACTION", "auto")
        yield


@pytest.fixture(scope="module")
def vllm_outputs():
    """Run vLLM offline inference once for all prompts."""
    llm = LLM(model=MODEL_NAME, max_model_len=512, max_num_seqs=1)
    sp = SamplingParams(temperature=0, max_tokens=MAX_TOKENS)
    outputs = llm.generate(PROMPTS, sp)
    return {o.prompt: o for o in outputs}


class TestQwen35Smoke:
    @pytest.mark.slow
    @pytest.mark.parametrize("prompt", PROMPTS)
    def test_generate_matches_golden(self, vllm_outputs, prompt):
        output = vllm_outputs[prompt]
        token_ids = list(output.outputs[0].token_ids)
        text = output.outputs[0].text

        expected = GOLDEN_MLX[prompt]
        matched = token_ids == expected

        print(f"\n  prompt: {prompt!r}")
        print(f"  output: {text!r}")
        print(f"  ids:    {token_ids}")
        if matched:
            print("  result: MATCHED golden")
        else:
            print("  result: NO MATCH")
            print(f"  expected: {expected}")

        assert matched, (
            f"Output for {prompt!r} did not match golden set.\n"
            f"Got:      {token_ids}\n"
            f"Expected: {expected}"
        )
