#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate golden token IDs for the e2e smoke test via vLLM offline inference.

Usage:
    # MLX inline cache (default):
    VLLM_ENABLE_V1_MULTIPROCESSING=0 python tools/gen_golden.py

    # Paged KV cache:
    VLLM_METAL_USE_PAGED_ATTENTION=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
        python tools/gen_golden.py
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
]

if __name__ == "__main__":
    paged = os.environ.get("VLLM_METAL_USE_PAGED_ATTENTION", "0") == "1"
    label = "PAGED" if paged else "MLX"
    print(f"\n--- Generating golden values for {label} path ---\n")

    llm = LLM(model=MODEL, max_model_len=512)
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
