#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate golden token IDs for deterministic smoke tests.

Runs vLLM offline inference (greedy, max_num_seqs=1) and prints golden
token-ID dicts to paste into test files or smoke scripts.

Usage:
    # Qwen3 (default, MLX inline cache):
    VLLM_ENABLE_V1_MULTIPROCESSING=0 python tools/gen_golden_token_ids_for_deterministics.py

    # GPT-OSS (requires chat template):
    VLLM_ENABLE_V1_MULTIPROCESSING=0 python tools/gen_golden_token_ids_for_deterministics.py \
        --model openai/gpt-oss-20b --max-tokens 100 --chat-template

    # Paged KV cache:
    VLLM_METAL_USE_PAGED_ATTENTION=1 VLLM_METAL_MEMORY_FRACTION=0.3 \
        VLLM_ENABLE_V1_MULTIPROCESSING=0 python tools/gen_golden_token_ids_for_deterministics.py

Note: MLX path requires VLLM_METAL_MEMORY_FRACTION=auto (the default).
      Numeric fractions are only valid for the paged attention path.
"""

import argparse
import os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

from vllm import LLM, SamplingParams  # noqa: E402

import vllm_metal.envs as envs  # noqa: E402

PROMPTS = [
    "The capital of France is",
    "The weather today is not",
    "One plus one equals",
    "The largest planet in our solar system is",
    "Water boils at a temperature of",
    "Machine learning is",
]


def _apply_chat_template(model_name, prompts):
    """Apply chat template and return (formatted_prompts, reverse_map)."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    formatted = []
    reverse_map = {}
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        fmt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        formatted.append(fmt)
        reverse_map[fmt] = prompt
    return formatted, reverse_map


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--max-tokens", type=int, default=10)
    parser.add_argument(
        "--chat-template",
        action="store_true",
        help="Apply chat template before inference (required for GPT-OSS)",
    )
    args = parser.parse_args()

    paged = envs.VLLM_METAL_USE_PAGED_ATTENTION
    label = "PAGED" if paged else "MLX"
    print(f"\n--- Generating golden values for {label} path ({args.model}) ---\n")

    prompts = PROMPTS
    reverse_map = None
    if args.chat_template:
        prompts, reverse_map = _apply_chat_template(args.model, PROMPTS)

    llm = LLM(model=args.model, max_model_len=512, max_num_seqs=1)
    sp = SamplingParams(temperature=0, max_tokens=args.max_tokens)
    outputs = llm.generate(prompts, sp)

    print(f"\nGOLDEN_{label} = {{")
    for o in outputs:
        display = reverse_map[o.prompt] if reverse_map else o.prompt
        ids = list(o.outputs[0].token_ids)
        text = o.outputs[0].text
        pad = 50 - len(display)
        print(f"    {display!r}:{' ' * max(pad, 1)}{ids},")
        print(f"        # → {text!r}")
    print("}")
