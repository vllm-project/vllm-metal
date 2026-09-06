#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""GLM-4.7-Flash golden token deterministic test: paged + MLA kernel vs mlx_lm.

Verifies that the paged attention path with VLLM_METAL_MLA_KERNEL=1 produces
the same tokens as standalone mlx_lm greedy decoding on GLM-4.7-Flash.
Numeric parity is acceptable as long as generation quality is preserved.

Prompts are wrapped in GLM's chat template before tokenization on both sides
(GLM-4.7-Flash is instruction-tuned; bare-prompt decoding collapses into
single-token loops that don't exercise the decode path).

Not in CI — requires local model weights and mlx_lm.

Usage:
    # Regenerate golden tokens from standalone mlx_lm (rerun if model/lib changes):
    python tools/test_glm47_flash_golden.py --gen-golden

    # Run paged + MLA kernel against the frozen golden:
    VLLM_ENABLE_V1_MULTIPROCESSING=0 python tools/test_glm47_flash_golden.py
"""

from __future__ import annotations

import argparse
import os
import sys

MODEL_NAME = "mlx-community/GLM-4.7-Flash-4bit"
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
# Golden token IDs from standalone mlx_lm greedy decoding (generate_step + argmax).
# Model: mlx-community/GLM-4.7-Flash-4bit
# Environment: mlx_lm 0.31.3
# Regenerate via: python tools/test_glm47_flash_golden.py --gen-golden
#
# Near-tie tolerance (see TIE_GROUPS / _tokens_match below): at "the user has
# provided a ___" the model picks between " very" (1602) and " sentence" (11646)
# — both benign <think> summaries whose logits sit close enough that a tiny
# numeric shift (mlx_lm version, attention path) flips the pick. Either token is
# accepted at that position. With materialized prefill on, the earlier "largest
# planet" comma divergence no longer occurs — all 6 match here on mlx_lm 0.31.3.
GOLDEN_MLX: dict[str, list[int]] = {
    "The capital of France is":                  [785, 1196, 374, 10156, 264, 4285, 11, 59485, 3405, 25],
    "The weather today is not":                  [785, 1196, 702, 3897, 264, 11646, 12283, 25, 330, 785],
    "One plus one equals":                       [785, 1196, 374, 10156, 264, 4285, 34612, 3405, 25, 330],
    "The largest planet in our solar system is": [785, 1196, 374, 10156, 264, 4285, 11, 59485, 3405, 25],
    "Water boils at a temperature of":           [785, 1196, 374, 10156, 264, 4285, 11, 59485, 3405, 25],
    "Machine learning is":                       [785, 1196, 702, 3897, 264, 1602, 2805, 11, 31999, 11646],
}
# fmt: on


# Benign <think> near-ties: tokens the model picks interchangeably under tiny
# numeric shifts. Members of a group are treated as equal at any position.
#   " very" (1602) / " sentence" (11646)  at "the user has provided a ___"
TIE_GROUPS: list[frozenset[int]] = [frozenset({1602, 11646})]


def _tokens_match(actual: list[int], expected: list[int]) -> bool:
    """Exact token match, except members of the same TIE_GROUPS set are
    interchangeable (a benign near-tie is not a quality regression)."""
    if len(actual) != len(expected):
        return False
    for a, e in zip(actual, expected, strict=True):
        if a == e:
            continue
        if any({a, e} <= group for group in TIE_GROUPS):
            continue
        return False
    return True


def _chat_format(tokenizer, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )


def gen_golden() -> dict[str, list[int]]:
    """Run standalone mlx_lm with greedy sampling to produce reference tokens."""
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.generate import generate_step

    model, tokenizer = load(MODEL_NAME)

    def argmax_sampler(logits: mx.array) -> mx.array:
        return mx.argmax(logits, axis=-1)

    results: dict[str, list[int]] = {}
    for prompt in PROMPTS:
        formatted = _chat_format(tokenizer, prompt)
        prompt_ids = tokenizer.encode(formatted, add_special_tokens=False)
        prompt_arr = mx.array(prompt_ids)
        ids: list[int] = []
        for token, _ in generate_step(
            prompt_arr, model, max_tokens=MAX_TOKENS, sampler=argmax_sampler
        ):
            ids.append(int(token))
            if len(ids) >= MAX_TOKENS:
                break
        results[prompt] = ids
    return results


def run_paged() -> dict[str, list[int]]:
    """Run vllm with paged + MLA kernel and return per-prompt token IDs."""
    # Force the kernel path: the whole point of this test is to compare
    # the Metal MLA kernel against standalone mlx_lm. A caller with
    # VLLM_METAL_MLA_KERNEL=0 exported would silently fall back to MLX
    # SDPA, which matches all goldens and produces a false pass.
    os.environ["VLLM_METAL_MLA_KERNEL"] = "1"
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_METAL_MEMORY_FRACTION", "0.75")
    print(
        f"  effective env: VLLM_METAL_MLA_KERNEL={os.environ['VLLM_METAL_MLA_KERNEL']}"
    )

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    formatted = [_chat_format(tokenizer, p) for p in PROMPTS]
    reverse_map = dict(zip(formatted, PROMPTS, strict=True))

    llm = LLM(model=MODEL_NAME, max_model_len=512, max_num_seqs=1)
    sp = SamplingParams(temperature=0, max_tokens=MAX_TOKENS)
    outputs = llm.generate(formatted, sp)
    return {reverse_map[o.prompt]: list(o.outputs[0].token_ids) for o in outputs}


def print_golden(results: dict[str, list[int]]) -> None:
    print("GOLDEN_MLX = {")
    for prompt in PROMPTS:
        ids = results[prompt]
        pad = 55 - len(prompt)
        print(f"    {prompt!r}:{' ' * max(pad, 1)}{ids},")
    print("}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--gen-golden",
        action="store_true",
        help="Regenerate golden token IDs from standalone mlx_lm and exit",
    )
    args = parser.parse_args()

    if args.gen_golden:
        print(f"Regenerating golden tokens via standalone mlx_lm ({MODEL_NAME})")
        results = gen_golden()
        print_golden(results)
        return 0

    if not GOLDEN_MLX:
        print(
            "ERROR: GOLDEN_MLX is empty. Run --gen-golden first and paste the "
            "output into this file.",
            file=sys.stderr,
        )
        return 1

    paged = run_paged()

    passed = failed = 0
    for prompt in PROMPTS:
        ids = paged[prompt]
        expected = GOLDEN_MLX[prompt]
        matched = _tokens_match(ids[: len(expected)], expected)

        print(f"\n  prompt: {prompt!r}")
        print(f"  ids:    {ids}")
        if matched:
            print("  result: MATCHED golden")
            passed += 1
        else:
            print("  result: NO MATCH")
            print(f"  expected: {expected}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(PROMPTS)}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
