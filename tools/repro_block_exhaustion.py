#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Reproduce KV cache block exhaustion with vLLM offline inference.

Usage:
    python tools/repro_block_exhaustion.py --gpu-memory-utilization 0.1
"""

import argparse
import os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "DEBUG")

from vllm import LLM, SamplingParams


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.12)
    args = parser.parse_args()

    llm = LLM(
        model="Qwen/Qwen3-0.6B",
        max_model_len=2048,
        disable_log_stats=False,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

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


if __name__ == "__main__":
    main()
