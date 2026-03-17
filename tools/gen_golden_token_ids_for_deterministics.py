#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate golden token IDs for the deterministic smoke test.

Runs vLLM offline inference (greedy, max_num_seqs=1) and prints golden
token-ID dicts to paste into test_paged_deterministic.py.

Usage:
    # MLX inline cache (default):
    VLLM_ENABLE_V1_MULTIPROCESSING=0 python tools/gen_golden_token_ids_for_deterministics.py

    # Paged KV cache:
    VLLM_METAL_USE_PAGED_ATTENTION=1 VLLM_METAL_MEMORY_FRACTION=0.3 \
        VLLM_ENABLE_V1_MULTIPROCESSING=0 python tools/gen_golden_token_ids_for_deterministics.py

Note: MLX path requires VLLM_METAL_MEMORY_FRACTION=auto (the default).
      Numeric fractions are only valid for the paged attention path.
"""

import os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen3-0.6B"
MAX_TOKENS = 10

PROMPTS = [
    "The capital of France is",
    "The weather today is not",
    "One plus one equals",
    "The largest planet in our solar system is",
    "Water boils at a temperature of",
    "Machine learning is",
]

if __name__ == "__main__":
    paged = os.environ.get("VLLM_METAL_USE_PAGED_ATTENTION", "0") == "1"
    label = "PAGED" if paged else "MLX"
    print(f"\n--- Generating golden values for {label} path ---\n")

    llm = LLM(model=MODEL, max_model_len=512, max_num_seqs=1)
    sp = SamplingParams(temperature=0, max_tokens=MAX_TOKENS)
    outputs = llm.generate(PROMPTS, sp)

    print(f"\nGOLDEN_{label} = {{")
    for o in outputs:
        prompt = o.prompt
        ids = list(o.outputs[0].token_ids)
        text = o.outputs[0].text
        pad = 45 - len(prompt)
        print(f"    {prompt!r}:{' ' * pad}{ids},")
        print(f"        # → {text!r}")
    print("}")
