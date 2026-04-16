#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate Gemma4 golden token IDs.

Two modes:

- ``mlx_lm`` (default): greedy decoding via ``mlx_lm.stream_generate``.
  Produces the independent reference (``GOLDEN_MLX_LM``).
- ``--paged``: greedy decoding via vllm-metal's paged attention path.
  Produces the in-tree baseline (``GOLDEN_PAGED``) so small floating-point
  tie-break drifts between the two paths don't cause spurious failures.

Usage:
    python tools/gen_gemma4_golden.py <model-path>
    python tools/gen_gemma4_golden.py --paged <model-path>
"""

from __future__ import annotations

import os
import sys

_PROMPTS = [
    "The capital of France is",
    "The weather today is not",
    "One plus one equals",
    "The largest planet in our solar system is",
    "Water boils at a temperature of",
]
_MAX_TOKENS = 10


def _mlx_lm_golden(model_path: str) -> dict[str, list[int]]:
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    print(f"Loading {model_path} via mlx_lm...")
    model, tokenizer = load(model_path)
    print("Loaded.")

    # Disable EOS so every sequence is exactly MAX_TOKENS long.
    tokenizer._eos_token_ids = set()
    sampler = make_sampler(temp=0.0)

    results: dict[str, list[int]] = {}
    for prompt in _PROMPTS:
        results[prompt] = [
            r.token
            for r in stream_generate(
                model, tokenizer, prompt, max_tokens=_MAX_TOKENS, sampler=sampler
            )
        ]
    return results


def _paged_golden(model_path: str) -> dict[str, list[int]]:
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_METAL_MEMORY_FRACTION", "0.35")

    from vllm import LLM, SamplingParams  # noqa: WPS433 — deferred import

    print(f"Loading {model_path} via vllm-metal paged path...")
    llm = LLM(
        model=model_path,
        max_model_len=512,
        max_num_seqs=1,
        disable_log_stats=True,
    )
    sp = SamplingParams(temperature=0, max_tokens=_MAX_TOKENS, ignore_eos=True)
    return {o.prompt: list(o.outputs[0].token_ids) for o in llm.generate(_PROMPTS, sp)}


def _print_dict(name: str, outputs: dict[str, list[int]]) -> None:
    print()
    print(f"{name} = {{")
    for prompt in _PROMPTS:
        pad = 50 - len(prompt)
        print(f"    {prompt!r}:{' ' * max(pad, 1)}{outputs[prompt]},")
    print("}")


def main() -> None:
    args = sys.argv[1:]
    paged = False
    if args and args[0] == "--paged":
        paged = True
        args = args[1:]
    if len(args) != 1:
        print("Usage: python tools/gen_gemma4_golden.py [--paged] <model-path>")
        sys.exit(1)

    model_path = args[0]
    if paged:
        _print_dict("GOLDEN_PAGED", _paged_golden(model_path))
    else:
        _print_dict("GOLDEN_MLX_LM", _mlx_lm_golden(model_path))


if __name__ == "__main__":
    main()
