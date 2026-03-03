#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Reproduce KV cache block exhaustion with vLLM offline inference.

Usage:
    VLLM_METAL_MEMORY_FRACTION=0.1 \
    VLLM_METAL_USE_PAGED_ATTENTION=1 \
    python tools/repro_block_exhaustion.py
"""

import os

os.environ.setdefault("VLLM_METAL_MEMORY_FRACTION", "0.12")
os.environ.setdefault("VLLM_METAL_USE_PAGED_ATTENTION", "1")
os.environ.setdefault("VLLM_METAL_DEBUG", "1")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "DEBUG")

from vllm import LLM, SamplingParams

if __name__ == "__main__":
    llm = LLM(model="Qwen/Qwen3-0.6B", max_model_len=2048, disable_log_stats=False)

    prompts = [
        "Explain the theory of relativity.",
        "Write a quicksort implementation in Python.",
        "List all countries in Europe and their capitals.",
        "Describe photosynthesis step by step.",
    ] * 40

    out = llm.generate(prompts, SamplingParams(max_tokens=400))
    for o in out:
        print(o.outputs[0].text[:80], "…")
        break
