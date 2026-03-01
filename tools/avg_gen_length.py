#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Measure average response length (tokens) via offline vLLM inference on ShareGPT.

Usage:
    # Download ShareGPT dataset (~642 MB):
    huggingface-cli download anon8231489123/ShareGPT_Vicuna_unfiltered \
        --repo-type dataset --local-dir . ShareGPT_V3_unfiltered_cleaned_split.json

    # Batch size 1 (sequential):
    VLLM_METAL_USE_PAGED_ATTENTION=1 VLLM_METAL_MEMORY_FRACTION=0.7 \
        python tools/avg_gen_length.py --max-num-seqs 1

    # Batch size 8:
    VLLM_METAL_USE_PAGED_ATTENTION=1 VLLM_METAL_MEMORY_FRACTION=0.7 \
        python tools/avg_gen_length.py --max-num-seqs 8

    # Compare both in one run (reloads model for each):
    VLLM_METAL_USE_PAGED_ATTENTION=1 VLLM_METAL_MEMORY_FRACTION=0.7 \
        python tools/avg_gen_length.py --max-num-seqs 1 8
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import mlx.core as mx
from vllm import LLM, SamplingParams


def load_sharegpt_prompts(path: Path, num_prompts: int, seed: int) -> list[str]:
    """Extract the first human turn from each ShareGPT conversation."""
    with open(path) as f:
        data = json.load(f)

    prompts: list[str] = []
    for entry in data:
        for turn in entry.get("conversations", []):
            if turn["from"] == "human" and turn["value"].strip():
                prompts.append(turn["value"].strip())
                break

    rng = random.Random(seed)
    rng.shuffle(prompts)
    return prompts[:num_prompts]


def run_offline(
    model: str,
    prompts: list[str],
    max_tokens: int,
    max_model_len: int,
    max_num_seqs: int,
) -> list[int]:
    """Run offline inference, return list of completion token counts."""
    llm = LLM(
        model=model,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
    )
    sampling_params = SamplingParams(temperature=0.8, max_tokens=max_tokens)

    conversations = [[{"role": "user", "content": p}] for p in prompts]

    t0 = time.monotonic()
    outputs = llm.chat(conversations, sampling_params)
    elapsed = time.monotonic() - t0

    token_counts = [len(o.outputs[0].token_ids) for o in outputs]
    print(f"  max_num_seqs={max_num_seqs}  |  wall time: {elapsed:.1f}s")
    return token_counts


def print_summary(results: dict[int, list[int]]) -> None:
    print("\n" + "=" * 60)
    print(f"{'max_num_seqs':>14} {'N':>6} {'Mean tokens':>13} {'Std':>10}")
    print("-" * 60)
    for mns, counts in results.items():
        arr = mx.array(counts, dtype=mx.float32)
        print(
            f"{mns:>14} {len(counts):>6} {arr.mean().item():>13.1f} {arr.std().item():>10.1f}"
        )
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("ShareGPT_V3_unfiltered_cleaned_split.json"),
        help="Path to ShareGPT JSON file",
    )
    parser.add_argument("--num-prompts", type=int, default=100)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        nargs="+",
        default=[1, 8],
        help="Max concurrent sequences (batch sizes) to test",
    )
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prompts = load_sharegpt_prompts(args.dataset, args.num_prompts, args.seed)
    print(f"Loaded {len(prompts)} prompts from {args.dataset}")
    print(f"Model: {args.model}  |  Max tokens: {args.max_tokens}")

    results: dict[int, list[int]] = {}
    for mns in args.max_num_seqs:
        print(f"\nRunning with max_num_seqs={mns} ...")
        counts = run_offline(
            args.model, prompts, args.max_tokens, args.max_model_len, mns
        )
        results[mns] = counts

    print_summary(results)


if __name__ == "__main__":
    main()
