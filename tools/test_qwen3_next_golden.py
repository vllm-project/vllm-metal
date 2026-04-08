#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Qwen3-Next golden token deterministic test: paged vs mlx_lm ground truth.

Verifies that the paged attention path produces the same tokens as the
MLX inline cache path for Qwen3-Next hybrid models (GDN + SDPA).

Not in CI — requires local model weights (~80GB).

Usage:
    VLLM_ENABLE_V1_MULTIPROCESSING=0 python tools/test_qwen3_next_golden.py
"""

from __future__ import annotations

import os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_METAL_USE_PAGED_ATTENTION", "1")
os.environ.setdefault("VLLM_METAL_MEMORY_FRACTION", "0.9")

from vllm import LLM, SamplingParams

MODEL_NAME = "mlx-community/Qwen3-Next-80B-A3B-Instruct-8bit"
MAX_TOKENS = 10

PROMPTS = [
    "The capital of France is",
    "The weather today is not",
    "One plus one equals",
    "The largest planet in our solar system is",
    "Water boils at a temperature of",
    "Machine learning is",
]

# fmt: off
# Golden token IDs from mlx_lm greedy decoding (argmax sampler).
# Model: mlx-community/Qwen3-Next-80B-A3B-Instruct-8bit
# Environment: mlx 0.31.1, mlx-lm 0.31.1
GOLDEN_MLX = {
    "The capital of France is": [59604, 13, 576, 6722, 315, 9856, 374, 19846, 13, 576],
    "The weather today is not": [1661, 438, 432, 572, 13671, 13, 42344, 40916, 64559],
    "One plus one equals": [267, 1126, 13, 9043, 5519, 1378, 16819, 3040, 13, 13322],
    "The largest planet in our solar system is": [41, 19519, 11, 448, 264, 23033, 315, 220, 23, 23],
    "Water boils at a temperature of": [16, 15, 15, 30937, 13, 1913, 279, 68723, 5452],
    "Machine learning is": [25993, 315, 20443, 11229, 429, 23497, 389, 11220, 25185],
}
# fmt: on


def main():
    llm = LLM(model=MODEL_NAME, max_model_len=512, max_num_seqs=1)
    sp = SamplingParams(temperature=0, max_tokens=MAX_TOKENS)
    outputs = llm.generate(PROMPTS, sp)

    passed = 0
    failed = 0
    for output in outputs:
        prompt = output.prompt
        token_ids = list(output.outputs[0].token_ids)
        text = output.outputs[0].text
        expected = GOLDEN_MLX[prompt]
        matched = token_ids[: len(expected)] == expected

        print(f"\n  prompt: {prompt!r}")
        print(f"  output: {text!r}")
        print(f"  ids:    {token_ids}")
        if matched:
            print("  result: MATCHED golden")
            passed += 1
        else:
            print("  result: NO MATCH")
            print(f"  expected: {expected}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(PROMPTS)}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
