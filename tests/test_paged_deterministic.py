# SPDX-License-Identifier: Apache-2.0
"""Deterministic smoke test: vLLM offline inference with golden token comparison.

Golden token IDs were generated on the main branch using vLLM offline inference
with temperature=0 (greedy decoding) on Qwen/Qwen3-0.6B, running one sequence
at a time (max_num_seqs=1) to avoid batch-invariance issues on Metal.

Findings from golden generation (main branch, HF paged-attention kernel):
- The HF kernel paged KV path produces correct, coherent output.
- 4/5 prompts are identical to the MLX inline cache path.
- 1/5 ("The capital of France is") diverges at token 5 — both continuations
  are valid English ("France is also the capital" vs "Italy is Rome. The").
  Likely caused by floating-point non-determinism in the attention kernel
  where top-2 logits are very close.

The assert accepts EITHER golden set (mlx-cache or paged-cache) and prints
which path matched.

Run (paged KV path, the default):
    python -m pytest tests/test_paged_deterministic.py -v -s

To test the MLX inline cache path instead, pass env vars explicitly:
    VLLM_METAL_USE_PAGED_ATTENTION=0 VLLM_METAL_MEMORY_FRACTION=auto \
        python -m pytest tests/test_paged_deterministic.py -v -s

Note: MLX requires VLLM_METAL_MEMORY_FRACTION=auto (numeric fractions are
only valid for the paged attention path).
"""

from __future__ import annotations

import os

import pytest
from vllm import LLM, SamplingParams

MODEL_NAME = "Qwen/Qwen3-0.6B"
MAX_TOKENS = 10

PROMPTS = [
    "The capital of France is",
    "The weather today is not",
    "One plus one equals",
    "The largest planet in our solar system is",
    "Water boils at a temperature of",
]

# fmt: off
# Golden token IDs from MLX inline cache (default path), greedy decoding.
# Generated on main branch via: VLLM_ENABLE_V1_MULTIPROCESSING=0 python tools/gen_golden_token_ids_for_deterministics.py
GOLDEN_MLX = {
    "The capital of France is":                   [12095, 13, 576, 6722, 315, 9625, 374, 1083, 279, 6722],
    "The weather today is not":                   [1661, 13, 576, 9315, 374, 220, 17, 15, 12348, 13],
    "One plus one equals":                        [825, 11, 825, 5519, 825, 16819, 1378, 13, 2055, 11],
    "The largest planet in our solar system is":  [1112, 30, 362, 13, 43562, 425, 13, 48976, 356, 13],
    "Water boils at a temperature of":            [220, 16, 15, 15, 30937, 13, 3555, 374, 279, 9315],
}

# Golden token IDs from paged KV cache (HF kernel on main branch), greedy decoding.
# Generated on main branch via: VLLM_METAL_USE_PAGED_ATTENTION=1 VLLM_METAL_MEMORY_FRACTION=0.3 \
#                                VLLM_ENABLE_V1_MULTIPROCESSING=0 python tools/gen_golden_token_ids_for_deterministics.py
GOLDEN_PAGED = {
    "The capital of France is":                   [12095, 13, 576, 6722, 315, 15344, 374, 21718, 13, 576],
    "The weather today is not":                   [1661, 13, 576, 9315, 374, 220, 17, 15, 12348, 13],
    "One plus one equals":                        [825, 11, 825, 5519, 825, 16819, 1378, 13, 2055, 11],
    "The largest planet in our solar system is":  [1112, 30, 362, 13, 43562, 425, 13, 48976, 356, 13],
    "Water boils at a temperature of":            [220, 16, 15, 15, 30937, 13, 3555, 374, 279, 9315],
}
# fmt: on


@pytest.fixture(autouse=True, scope="module")
def _set_env():
    """Set default env vars for this test.

    Uses MonkeyPatch.context() so env changes are automatically reverted
    after the module, avoiding side effects on other tests.

    Defaults to the paged KV cache path to ensure the test actually exercises
    the paged attention kernel, but respects any env vars already set by the
    user (e.g. to run the MLX path).
    """
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

        # Default to paged attention, but allow the caller to override.
        use_paged = os.environ.get("VLLM_METAL_USE_PAGED_ATTENTION")
        if use_paged is None:
            mp.setenv("VLLM_METAL_USE_PAGED_ATTENTION", "1")
            use_paged = "1"

        # Set a sensible default memory setting for the selected path, unless
        # the caller has already specified one.
        if os.environ.get("VLLM_METAL_MEMORY_FRACTION") is None:
            if use_paged == "1":
                mp.setenv("VLLM_METAL_MEMORY_FRACTION", "0.2")
            else:
                mp.setenv("VLLM_METAL_MEMORY_FRACTION", "auto")
        yield


@pytest.fixture(scope="module")
def vllm_outputs():
    """Run vLLM offline inference once for all prompts.

    Uses max_num_seqs=1 to avoid batch-invariance non-determinism on Metal.
    """
    llm = LLM(model=MODEL_NAME, max_model_len=512, max_num_seqs=1)
    sp = SamplingParams(temperature=0, max_tokens=MAX_TOKENS)
    outputs = llm.generate(PROMPTS, sp)
    return {o.prompt: o for o in outputs}


class TestPagedDeterministic:
    @pytest.mark.slow
    @pytest.mark.parametrize("prompt", PROMPTS)
    def test_generate_matches_golden(self, vllm_outputs, prompt):
        output = vllm_outputs[prompt]
        token_ids = list(output.outputs[0].token_ids)
        text = output.outputs[0].text

        mlx_expected = GOLDEN_MLX[prompt]
        paged_expected = GOLDEN_PAGED[prompt]

        mlx_match = token_ids == mlx_expected
        paged_match = token_ids == paged_expected

        print(f"\n  prompt: {prompt!r}")
        print(f"  output: {text!r}")
        print(f"  ids:    {token_ids}")
        if mlx_match:
            print("  result: MATCHED mlx-cache golden")
        elif paged_match:
            print("  result: MATCHED paged-cache golden")
        else:
            print("  result: NO MATCH")
            print(f"  expected (mlx):   {mlx_expected}")
            print(f"  expected (paged): {paged_expected}")

        assert mlx_match or paged_match, (
            f"Output for {prompt!r} matched neither golden set.\n"
            f"Got:            {token_ids}\n"
            f"Expected (mlx): {mlx_expected}\n"
            f"Expected (pgd): {paged_expected}"
        )
