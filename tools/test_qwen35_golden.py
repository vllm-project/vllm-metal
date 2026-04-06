#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5 golden token deterministic test: paged vs mlx_lm ground truth.

Verifies that the hybrid paged attention path (SDPA + GDN) produces the
same tokens as the MLX inline cache path for Qwen3.5.

Not in CI — requires local model weights.

Usage:
    # Generate golden tokens (MLX inline cache, greedy):
    VLLM_ENABLE_V1_MULTIPROCESSING=0 python tools/test_qwen35_golden.py --gen-golden

    # Run deterministic test (paged path vs golden):
    VLLM_ENABLE_V1_MULTIPROCESSING=0 python tools/test_qwen35_golden.py

    # Custom model path:
    VLLM_ENABLE_V1_MULTIPROCESSING=0 python tools/test_qwen35_golden.py \
        --model /path/to/Qwen3.5-0.8B
"""

import argparse
import os
import sys

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

from vllm import LLM, SamplingParams  # noqa: E402

MODEL_DEFAULT = os.environ.get("QWEN35_MODEL_PATH", "Qwen/Qwen3.5-4B")
MAX_TOKENS = 20

PROMPTS = [
    "The capital of France is",
    "One plus one equals",
    "The largest planet in our solar system is",
    "Machine learning is a branch of",
]


def generate(model: str, max_tokens: int) -> dict[str, list[int]]:
    """Run greedy generation and return {prompt: token_ids}."""
    llm = LLM(model=model, max_model_len=512, max_num_seqs=1)
    sp = SamplingParams(temperature=0, max_tokens=max_tokens)
    outputs = llm.generate(PROMPTS, sp)
    result = {}
    for o in outputs:
        result[o.prompt] = list(o.outputs[0].token_ids)
    return result


def print_golden(results: dict[str, list[int]], label: str) -> None:
    """Print golden token dict for copy-paste."""
    print(f"\nGOLDEN_{label} = {{")
    for prompt, ids in results.items():
        pad = 55 - len(prompt)
        print(f"    {prompt!r}:{' ' * max(pad, 1)}{ids},")
    print("}")


def _run_in_subprocess(
    model: str, max_tokens: int, paged: bool
) -> dict[str, list[int]]:
    """Run generation in a subprocess to avoid memory interference."""
    import json
    import subprocess

    env = os.environ.copy()
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    if paged:
        env["VLLM_METAL_USE_PAGED_ATTENTION"] = "1"
        env.setdefault("VLLM_METAL_MEMORY_FRACTION", "0.5")

    script = f"""
import os, json
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
if {paged!r}:
    os.environ["VLLM_METAL_USE_PAGED_ATTENTION"] = "1"
    os.environ.setdefault("VLLM_METAL_MEMORY_FRACTION", "0.5")
from vllm import LLM, SamplingParams
llm = LLM(model={model!r}, max_model_len=512, max_num_seqs=1)
sp = SamplingParams(temperature=0, max_tokens={max_tokens})
prompts = {PROMPTS!r}
outputs = llm.generate(prompts, sp)
result = {{o.prompt: list(o.outputs[0].token_ids) for o in outputs}}
print("GOLDEN_JSON:" + json.dumps(result))
"""
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
    )
    if proc.returncode != 0:
        print(proc.stderr[-2000:] if len(proc.stderr) > 2000 else proc.stderr)
        raise RuntimeError(f"Subprocess failed (paged={paged})")

    for line in proc.stdout.splitlines():
        if line.startswith("GOLDEN_JSON:"):
            return json.loads(line[len("GOLDEN_JSON:") :])
    raise RuntimeError("No GOLDEN_JSON output found")


def run_test(model: str, max_tokens: int) -> bool:
    """Compare paged path output against MLX inline cache path."""
    print("=== Step 1: MLX inline cache (ground truth) ===")
    mlx_results = _run_in_subprocess(model, max_tokens, paged=False)

    print("=== Step 2: Paged attention path ===")
    paged_results = _run_in_subprocess(model, max_tokens, paged=True)

    # Compare
    print("\n=== Results ===")
    all_match = True
    for prompt in PROMPTS:
        mlx_ids = mlx_results[prompt]
        paged_ids = paged_results[prompt]
        match = mlx_ids == paged_ids
        status = "MATCH" if match else "MISMATCH"
        if not match:
            all_match = False
            # Find first divergence
            for i, (a, b) in enumerate(zip(mlx_ids, paged_ids, strict=False)):
                if a != b:
                    print(
                        f"  [{status}] {prompt!r} — diverges at token {i}: "
                        f"mlx={a} vs paged={b}"
                    )
                    break
            else:
                print(
                    f"  [{status}] {prompt!r} — length differs: "
                    f"mlx={len(mlx_ids)} vs paged={len(paged_ids)}"
                )
        else:
            print(f"  [{status}] {prompt!r}")

    print(f"\n{'ALL PASSED' if all_match else 'SOME MISMATCHES'}")
    return all_match


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", default=MODEL_DEFAULT)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument(
        "--gen-golden", action="store_true", help="Just print golden token IDs and exit"
    )
    args = parser.parse_args()

    if args.gen_golden:
        paged = os.environ.get("VLLM_METAL_USE_PAGED_ATTENTION", "0") == "1"
        label = "PAGED" if paged else "MLX"
        print(f"Generating golden tokens ({label} path, {args.model})")
        results = _run_in_subprocess(args.model, args.max_tokens, paged=paged)
        print_golden(results, label)
    else:
        ok = run_test(args.model, args.max_tokens)
        sys.exit(0 if ok else 1)
