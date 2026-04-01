#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""GPT-OSS 20B smoke test: mlx_lm ground truth for sink attention work (#148).

Loads openai/gpt-oss-20b, generates with greedy decoding, and compares
output against golden token IDs.  Not in CI since it requires ~21.5 GB model.

Run:
    VLLM_ENABLE_V1_MULTIPROCESSING=0 python tools/test_gpt_oss_smoke.py
"""

import os
import sys

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

from transformers import AutoTokenizer  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

MODEL_NAME = "openai/gpt-oss-20b"
MAX_TOKENS = 100

PROMPTS = [
    "The capital of France is",
    "The weather today is not",
    "One plus one equals",
    "The largest planet in our solar system is",
    "Water boils at a temperature of",
]

# fmt: off
# Golden token IDs from MLX inline cache, greedy decoding (openai/gpt-oss-20b).
# Generated via:
#   VLLM_ENABLE_V1_MULTIPROCESSING=0 python tools/gen_golden_token_ids_for_deterministics.py \
#       --model openai/gpt-oss-20b --max-tokens 100 --chat-template
#
# Note: FP non-determinism at longer sequences may cause 2-3 prompts to diverge
# after ~25 tokens across runs.  Regenerate with the command above if needed.
GOLDEN_MLX = {
    "The capital of France is": [200005, 35644, 200008, 976, 1825, 5003, 25, 392, 976, 9029, 328, 10128, 382, 4050, 3164, 6960, 1682, 290, 6052, 25, 392, 72782, 4050, 2632, 9570, 483, 392, 72782, 4050, 63659, 1327, 6052, 13, 200007, 200006, 173781, 200005, 17196, 200008, 72782, 200002],
    "The weather today is not": [200005, 35644, 200008, 976, 1825, 5003, 25, 392, 976, 11122, 4044, 382, 625, 4050, 4569, 7890, 60592, 13, 3164, 3572, 413, 8601, 261, 21872, 25, 392, 976, 11122, 4044, 382, 625, 723, 64493, 49706, 889, 1023, 9289, 9115, 13, 3164, 3572, 413, 16054, 395, 3543, 30, 2604, 10112, 1023, 1682, 316, 1761, 290, 11122, 30, 623, 1825, 5003, 392, 976, 11122, 4044, 382, 625, 4050, 4569, 382, 60592, 13, 1416, 1309, 316, 9570, 54286, 13, 1416, 2023, 3810, 395, 108041, 25, 392, 4827, 1481, 481, 1299, 316, 1761, 1078, 290, 11122, 16842, 2604, 581, 2023, 18135, 484, 1023, 1682, 316],
    "One plus one equals": [200005, 35644, 200008, 976, 1825, 5003, 25, 392, 5045, 2932, 1001, 29702, 4050, 3164, 6960, 1682, 290, 6052, 25, 220, 17, 13, 3072, 10112, 1023, 1682, 261, 945, 65742, 6052, 30, 623, 1825, 3572, 413, 11493, 13, 623, 63122, 6052, 25, 220, 17, 13, 3072, 10112, 1023, 1682, 261, 15681, 30, 623, 21179, 25, 392, 3575, 553, 17554, 162016, 11, 261, 4410, 6439, 2359, 22203, 656, 7788, 17527, 3692, 32711, 860, 3582, 21179, 13, 2632, 6052, 25, 220, 17, 13, 200007, 200006, 173781, 200005, 17196, 200008, 5045, 2932, 1001, 29702, 6240, 17, 410, 13, 200002],
    "The largest planet in our solar system is": [200005, 35644, 200008, 976, 1825, 31064, 25, 392, 976, 10574, 17921, 306, 1039, 17624, 2420, 382, 4050, 3164, 6960, 1682, 290, 6052, 25, 79575, 13, 3164, 3572, 1682, 261, 18128, 13, 2632, 6052, 25, 79575, 13, 138743, 8633, 4275, 290, 10574, 13, 2632, 9570, 25, 79575, 13, 200007, 200006, 173781, 200005, 17196, 200008, 976, 10574, 17921, 306, 1039, 17624, 2420, 382, 6240, 41, 26451, 410, 13, 200002],
    "Water boils at a temperature of": [200005, 35644, 200008, 976, 1825, 5003, 25, 392, 27874, 165683, 540, 261, 12088, 328, 4050, 3164, 6960, 1682, 290, 79667, 2438, 328, 3411, 13, 3072, 290, 4928, 382, 60592, 25, 392, 27874, 165683, 540, 261, 12088, 328, 4050, 3164, 3572, 1682, 290, 6052, 25, 220, 1353, 26557, 540, 220, 16, 83327, 11, 503, 220, 19584, 68854, 13, 3072, 10112, 1023, 1682, 290, 12088, 306, 181775, 25, 220, 33797, 13, 1055, 658, 13, 623, 1825, 3572, 413, 35885, 261, 52077, 6052, 13, 623, 4928, 382, 60592, 889, 6960, 1023, 1682, 290, 79667, 2438, 13, 2632, 6052, 25, 220, 1353, 26557, 350],
}
# fmt: on


def _apply_chat_template(model_name, prompts):
    """Apply chat template and return (formatted_prompts, reverse_map)."""
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
    formatted_prompts, reverse_map = _apply_chat_template(MODEL_NAME, PROMPTS)

    llm = LLM(model=MODEL_NAME, max_model_len=512, max_num_seqs=1)
    sp = SamplingParams(temperature=0, max_tokens=MAX_TOKENS)
    outputs = llm.generate(formatted_prompts, sp)

    passed = 0
    failed = 0
    for o in outputs:
        prompt = reverse_map[o.prompt]
        token_ids = list(o.outputs[0].token_ids)
        text = o.outputs[0].text
        expected = GOLDEN_MLX[prompt]
        matched = token_ids == expected

        status = "PASS" if matched else "FAIL"
        print(f"  [{status}] {prompt!r}")
        print(f"         output: {text!r}")
        if not matched:
            print(f"         got:      {token_ids}")
            print(f"         expected: {expected}")
            failed += 1
        else:
            passed += 1

    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
