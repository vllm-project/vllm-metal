#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""EXAONE 4.0 parity smoke tool: mlx_lm vs vllm-metal paged.

Runs greedy decoding (temperature=0) on the same prompts through both paths,
prints both token sequences, and exits non-zero on any divergence.

Usage:
    python tools/gen_exaone4_golden.py [--model REPO] [--max-tokens N]
"""

from __future__ import annotations

import argparse
import os

PROMPTS = [
    "One plus one equals",
    "Two plus two equals",
    "Monday, Tuesday, Wednesday,",
    "서울은 대한민국의",
    "인공지능은",
]


def _mlx_tokens(model: str, max_tokens: int) -> dict[str, list[int]]:
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    from vllm_metal.compat import apply_compat_patches

    apply_compat_patches()  # EXAONE needs the Exaone4Config shim to load.
    m, tokenizer = load(model)
    tokenizer._eos_token_ids = set()  # decode a fixed length, ignore EOS
    sampler = make_sampler(temp=0.0)
    return {
        prompt: [
            r.token
            for r in stream_generate(
                m, tokenizer, prompt, max_tokens=max_tokens, sampler=sampler
            )
        ]
        for prompt in PROMPTS
    }


def _paged_tokens(model: str, max_tokens: int) -> dict[str, list[int]]:
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_METAL_USE_PAGED_ATTENTION", "1")
    os.environ.setdefault("VLLM_METAL_MEMORY_FRACTION", "0.5")
    from vllm import LLM, SamplingParams

    llm = LLM(model=model, max_model_len=512, max_num_seqs=1, disable_log_stats=True)
    sp = SamplingParams(temperature=0, max_tokens=max_tokens, ignore_eos=True)
    return {o.prompt: list(o.outputs[0].token_ids) for o in llm.generate(PROMPTS, sp)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", default="mlx-community/exaone-4.0-1.2b-4bit")
    parser.add_argument("--max-tokens", type=int, default=10)
    args = parser.parse_args()

    mlx = _mlx_tokens(args.model, args.max_tokens)
    paged = _paged_tokens(args.model, args.max_tokens)

    parity = True
    for prompt in PROMPTS:
        match = mlx[prompt] == paged[prompt]
        parity &= match
        print(f"[{'MATCH' if match else 'DIFF '}] {prompt!r}")
        print(f"    mlx_lm: {mlx[prompt]}")
        print(f"    paged:  {paged[prompt]}")
    print("\nPARITY OK" if parity else "\nPARITY FAILED")
    raise SystemExit(0 if parity else 1)


if __name__ == "__main__":
    main()
