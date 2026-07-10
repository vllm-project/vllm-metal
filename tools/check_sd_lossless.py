# SPDX-License-Identifier: Apache-2.0
"""Verify speculative decoding is lossless under greedy sampling.

Greedy SD must produce token-identical output to greedy non-SD: the verifier
accepts a draft token iff it equals the target argmax, so any divergence means
the verification forward (or the draft/KV plumbing behind it) is wrong. This
exercises the whole spec-decode stack end to end — including the attention
kernel's multi-token verify windows — with no golden files to maintain.

Each engine runs in its own subprocess (clean Metal state between configs).

Usage:
  python tools/check_sd_lossless.py                          # Qwen3-0.6B self-draft K=3
  python tools/check_sd_lossless.py --method mtp \
      --model mlx-community/gemma-4-e2b-it-bf16 \
      --draft mlx-community/gemma-4-E2B-it-assistant-bf16 -k 1
  python tools/check_sd_lossless.py --method dflash \
      --model Qwen/Qwen3-8B --draft RedHatAI/Qwen3-8B-speculator.dflash \
      -k 7 --memory-fraction 0.62 --max-model-len 2048
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

PROMPTS = [
    "The capital of France is",
    "One plus one equals",
    "Water boils at a temperature of",
    "The largest planet in our solar system is",
    "In Python, a list comprehension is",
    "The theory of relativity states that",
    "A binary search tree is a data structure where",
    "Photosynthesis is the process by which",
    "The French Revolution began in",
    "To reverse a linked list, you",
    "The speed of light in a vacuum is",
    "Machine learning models are trained by",
]

_WORKER = r"""
import json, sys, time
from vllm import LLM, SamplingParams

cfg = json.load(open(sys.argv[1]))
llm = LLM(
    model=cfg["model"],
    max_model_len=cfg["max_model_len"],
    enable_prefix_caching=False,
    speculative_config=cfg["speculative_config"],
    # Both engines run synchronous scheduling: SD on Metal requires it, and
    # keeping the base engine identical isolates the SD stack as the only
    # variable in the token-identity comparison.
    async_scheduling=False,
    disable_log_stats=False,  # keep spec-decode acceptance counters available
)
params = SamplingParams(temperature=0.0, max_tokens=cfg["max_tokens"], ignore_eos=True)
start = time.perf_counter()
outputs = llm.generate(cfg["prompts"], params)
elapsed = time.perf_counter() - start
result = {o.prompt: list(o.outputs[0].token_ids) for o in outputs}
gen_tokens = sum(len(v) for v in result.values())
stats = {"gen_tokens": gen_tokens, "elapsed_s": round(elapsed, 2),
         "tok_per_s": round(gen_tokens / elapsed, 2)}
try:
    for metric in llm.get_metrics():
        if "spec_decode" in metric.name:
            key = metric.name.split(":")[-1]
            value = getattr(metric, "value", None)
            if value is None:
                value = getattr(metric, "values", None)
            stats[key] = value
except Exception as exc:  # pragma: no cover - metrics API drift
    stats["metrics_error"] = repr(exc)
json.dump({"tokens": result, "stats": stats}, open(cfg["out"], "w"))
"""


def run_engine(args, spec_config: dict | None, out_path: str) -> dict[str, list[int]]:
    cfg = {
        "model": args.model,
        "max_model_len": args.max_model_len,
        "max_tokens": args.max_tokens,
        "prompts": PROMPTS[: args.num_prompts],
        "speculative_config": spec_config,
        "out": out_path,
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(cfg, f)
        cfg_path = f.name
    env = os.environ.copy()
    env.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    env.setdefault("VLLM_METAL_USE_PAGED_ATTENTION", "1")
    env.setdefault("VLLM_METAL_MEMORY_FRACTION", args.memory_fraction)
    label = json.dumps(spec_config) if spec_config else "no-SD"
    print(f"[check_sd_lossless] running engine: {label}", flush=True)
    subprocess.run(
        [sys.executable, "-c", _WORKER, cfg_path],
        env=env,
        check=True,
    )
    return json.load(open(out_path))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--draft", default="Qwen/Qwen3-0.6B")
    p.add_argument(
        "--method", default="draft_model", choices=["draft_model", "mtp", "dflash"]
    )
    p.add_argument("-k", "--num-speculative-tokens", type=int, default=3)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--num-prompts", type=int, default=len(PROMPTS))
    p.add_argument("--max-model-len", type=int, default=512)
    p.add_argument("--memory-fraction", default="0.35")
    args = p.parse_args()

    spec = {
        "method": args.method,
        "model": args.draft,
        "num_speculative_tokens": args.num_speculative_tokens,
    }
    with tempfile.TemporaryDirectory() as td:
        base_result = run_engine(args, None, os.path.join(td, "base.json"))
        sd_result = run_engine(args, spec, os.path.join(td, "sd.json"))
    base, sd = base_result["tokens"], sd_result["tokens"]
    print(f"[stats] base: {json.dumps(base_result['stats'])}")
    print(f"[stats] sd:   {json.dumps(sd_result['stats'])}")

    mismatches = 0
    for prompt in base:
        if base[prompt] != sd.get(prompt):
            mismatches += 1
            b, s = base[prompt], sd.get(prompt) or []
            div = next(
                (i for i, (x, y) in enumerate(zip(b, s, strict=False)) if x != y),
                min(len(b), len(s)),
            )
            print(f"MISMATCH @ token {div}: {prompt!r}")
            print(f"  base[{div}:{div + 5}] = {b[div : div + 5]}")
            print(f"  sd  [{div}:{div + 5}] = {s[div : div + 5]}")
    if mismatches:
        print(f"FAIL: {mismatches}/{len(base)} prompts diverged — SD is NOT lossless")
        return 1
    print(
        f"PASS: {len(base)} prompts x {args.max_tokens} tokens are token-identical "
        f"(greedy, method={args.method}, K={args.num_speculative_tokens})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
