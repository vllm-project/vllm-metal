#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""On-vs-off parity for the compiled stateless-MLP dispatch.

Runs the same deterministic greedy prompt set in two spawned children
(Metal is not fork-safe): one with ``VLLM_METAL_COMPILED_MLP=1`` and a
reach spy counting compiled-branch dispatches (at least one per prompt
required), one with the kill switch off (the spy must count zero).
Quantized serving checkpoints are bitwise-stable under the compiled
trace, so the gate is STRICT token identity across arms.

Not in CI — requires local weights for a Qwen3-Next-family checkpoint
(default Qwen3.5-0.8B).

Usage:
    python tools/compiled_mlp_parity.py
    python tools/compiled_mlp_parity.py --model <path-or-repo> --quick
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys

MODEL_DEFAULT = os.environ.get("QWEN35_MODEL_PATH", "Qwen/Qwen3.5-0.8B")

PROMPTS = (
    "Write a short essay about the history of computing.",
    "Explain how a B-tree works and why databases use it.",
    "Describe the CAP theorem and its practical consequences.",
)


def _child_env(enabled: bool) -> None:
    os.environ["VLLM_METAL_COMPILED_MLP"] = "1" if enabled else "0"
    for key, val in (
        ("VLLM_ENABLE_V1_MULTIPROCESSING", "0"),
        ("VLLM_METAL_USE_PAGED_ATTENTION", "1"),
        ("VLLM_METAL_MEMORY_FRACTION", "0.5"),
    ):
        os.environ.setdefault(key, val)


def run_child(model: str, enabled: bool, quick: bool, queue) -> None:
    _child_env(enabled)
    from vllm import LLM, SamplingParams

    import vllm_metal.compiled_mlp as cm

    # Reach spy on dispatch_compiled — the wrapper's single choke point
    # into the trace — so the count cannot drift from what actually ran
    # compiled.
    calls = {"n": 0}
    orig_dispatch = cm.CompiledMLPBlock.dispatch_compiled

    def spy(self, x):
        calls["n"] += 1
        return orig_dispatch(self, x)

    cm.CompiledMLPBlock.dispatch_compiled = spy
    try:
        llm = LLM(model=model, max_model_len=2048, max_num_seqs=4)
        prompts = PROMPTS[:1] if quick else PROMPTS
        sp = SamplingParams(temperature=0, max_tokens=64)
        results = {}
        for prompt in prompts:
            out = llm.generate([prompt], sp)[0]
            results[prompt] = list(out.outputs[0].token_ids)
    finally:
        cm.CompiledMLPBlock.dispatch_compiled = orig_dispatch

    if enabled:
        assert calls["n"] >= len(prompts), (
            f"enabled arm routed only {calls['n']} compiled dispatches for "
            f"{len(prompts)} prompts — the compiled path did not engage"
        )
    else:
        assert calls["n"] == 0, (
            f"kill-switch arm unexpectedly routed {calls['n']} compiled dispatches"
        )
    queue.put(results)


def run_pair(model: str, quick: bool) -> tuple[int, dict]:
    ctx = mp.get_context("spawn")
    per_arm: dict[bool, dict] = {}
    for enabled in (False, True):
        queue = ctx.Queue()
        proc = ctx.Process(target=run_child, args=(model, enabled, quick, queue))
        proc.start()
        try:
            per_arm[enabled] = queue.get(timeout=1800)
        finally:
            proc.join(timeout=60)
            if proc.is_alive():
                proc.terminate()
        if proc.exitcode != 0:
            raise RuntimeError(f"child (enabled={enabled}) exited with {proc.exitcode}")
    reference, compiled = per_arm[False], per_arm[True]
    mismatches = {
        key: {"ref": reference[key], "compiled": compiled[key]}
        for key in reference
        if reference[key] != compiled[key]
    }
    return len(reference), mismatches


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_DEFAULT)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    total, mismatches = run_pair(args.model, args.quick)
    if mismatches:
        print(f"PARITY FAIL: {len(mismatches)}/{total} prompts diverged")
        for key, pair in mismatches.items():
            print(f"  {key}: ref={pair['ref']} compiled={pair['compiled']}")
        return 1
    print(f"PARITY PASS: {total}/{total} prompts token-identical")
    return 0


if __name__ == "__main__":
    sys.exit(main())
