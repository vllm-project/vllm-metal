# SPDX-License-Identifier: Apache-2.0
"""Deterministic golden-token test for Gemma4 on the paged attention path.

Verifies that vllm-metal's paged attention implementation produces the same
greedy-decoded token IDs as running the same Gemma4 checkpoint through
mlx_lm directly.  This catches regressions in YOCO sharing, K-eq-V fallback,
v_norm, and variable head_dim padding — the paths exercised by Gemma4 that
are not stressed by other models.

Golden token IDs are generated once offline via ``tools/gen_gemma4_golden.py``
using ``mlx_lm.stream_generate(..., sampler=temp=0)`` with the tokenizer's
EOS set cleared so every sequence is exactly ``MAX_TOKENS`` long.

Run:
    GEMMA4_MODEL_PATH=/path/to/gemma-4-E2B-it \
        pytest tests/test_gemma4_golden.py -v -s -m slow
"""

from __future__ import annotations

import os

import pytest
from vllm import LLM, SamplingParams

MODEL_ENV = "GEMMA4_MODEL_PATH"
MAX_TOKENS = 10

PROMPTS = [
    "The capital of France is",
    "The weather today is not",
    "One plus one equals",
    "The largest planet in our solar system is",
    "Water boils at a temperature of",
]

# fmt: off
# Golden token IDs from mlx_lm stream_generate (greedy, EOS disabled).
# Regenerate with tools/gen_gemma4_golden.py.
GOLDEN_MLX_LM = {
    "The capital of France is":                   [7001, 563, 7001, 563, 7001, 563, 7001, 563, 7001, 563],
    "The weather today is not":                   [711, 711, 711, 711, 711, 108, 106, 108, 106, 108],
    "One plus one equals":                        [2915, 886, 14339, 2915, 886, 107, 106, 107, 1, 107],
    "The largest planet in our solar system is":  [10321, 1458, 563, 10321, 1458, 10321, 1458, 10321, 1458, 10321],
    "Water boils at a temperature of":            [104264, 657, 104264, 657, 104264, 106, 106, 106, 106, 106],
}
# fmt: on


@pytest.fixture(autouse=True, scope="module")
def _set_env():
    """Run the paged attention path in single-process mode for determinism."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        yield


@pytest.fixture(scope="module")
def vllm_outputs():
    """Run Gemma4 inference once through vllm-metal's paged attention path."""
    model_path = os.environ.get(MODEL_ENV)
    if not model_path:
        pytest.skip(f"{MODEL_ENV} not set — skipping Gemma4 deterministic test")
    if not os.path.isdir(model_path):
        pytest.skip(f"{MODEL_ENV}={model_path} is not a directory")

    llm = LLM(model=model_path, max_model_len=512, max_num_seqs=1)
    sp = SamplingParams(temperature=0, max_tokens=MAX_TOKENS, ignore_eos=True)
    outputs = llm.generate(PROMPTS, sp)
    return {o.prompt: o for o in outputs}


class TestGemma4Golden:
    @pytest.mark.slow
    @pytest.mark.parametrize("prompt", PROMPTS)
    def test_matches_mlx_lm_groundtruth(self, vllm_outputs, prompt):
        output = vllm_outputs[prompt]
        token_ids = list(output.outputs[0].token_ids)
        expected = GOLDEN_MLX_LM[prompt]

        print(f"\n  prompt:   {prompt!r}")
        print(f"  got:      {token_ids}")
        print(f"  expected: {expected}")

        assert token_ids == expected, (
            f"Paged attention output for {prompt!r} diverged from mlx_lm "
            f"groundtruth.\nGot:      {token_ids}\nExpected: {expected}"
        )
