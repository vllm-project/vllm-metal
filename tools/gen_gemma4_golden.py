#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate Gemma4 golden token IDs via mlx_lm (independent reference).

Runs greedy decoding directly through mlx_lm (bypassing vllm-metal) so the
resulting token IDs can serve as groundtruth for
``tests/test_gemma4_golden.py``.

Usage:
    python tools/gen_gemma4_golden.py /path/to/gemma-4-E2B-it
"""

from __future__ import annotations

import sys

from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

_PROMPTS = [
    "The capital of France is",
    "The weather today is not",
    "One plus one equals",
    "The largest planet in our solar system is",
    "Water boils at a temperature of",
]
_MAX_TOKENS = 10


def _greedy_tokens(model, tokenizer, prompt: str, max_tokens: int) -> list[int]:
    # Force full-length generation: disable EOS so the stream doesn't stop
    # early and produces a deterministic fixed-size reference.
    tokenizer._eos_token_ids = set()
    sampler = make_sampler(temp=0.0)
    return [
        resp.token
        for resp in stream_generate(
            model, tokenizer, prompt, max_tokens=max_tokens, sampler=sampler
        )
    ]


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python tools/gen_gemma4_golden.py <model-path>")
        sys.exit(1)
    model_path = sys.argv[1]

    print(f"Loading {model_path} via mlx_lm...")
    model, tokenizer = load(model_path)
    print("Loaded.\n")

    print("GOLDEN_MLX_LM = {")
    for prompt in _PROMPTS:
        ids = _greedy_tokens(model, tokenizer, prompt, _MAX_TOKENS)
        text = tokenizer.decode(ids)
        pad = 50 - len(prompt)
        print(f"    {prompt!r}:{' ' * max(pad, 1)}{ids},")
        print(f"        # → {text!r}")
    print("}")


if __name__ == "__main__":
    main()
