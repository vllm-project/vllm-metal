#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""On-vs-off parity for the body-only intermediate prefill path.

Runs the same deterministic greedy prompt set in two spawned children
(Metal is not fork-safe), both with ``max_num_batched_tokens`` far below
the prompt lengths so every prompt prefills through intermediate chunks.
One child runs normally (the projection-free intermediate forward engages);
the other disables the capability probe, so the same chunk schedule runs
full forwards.  Chunk boundaries are identical in both arms, so the only
difference is the skipped projection — token-identical outputs prove the
intermediate KV and GDN cache writes feed the final chunk exactly like the
full forward does.

Both children assert reach so the gate cannot silently test the wrong
thing: the engaged child counts ``DefaultModelAdapter.intermediate_forward``
calls and requires at least one per prompt; the disabled child requires
zero.

Not in CI — requires local model weights (hybrid GDN by default so the
recurrent-state writes are exercised alongside the paged KV).

Usage:
    python tools/intermediate_prefill_parity.py
    python tools/intermediate_prefill_parity.py --model /path/to/Qwen3.5-0.8B
    python tools/intermediate_prefill_parity.py --quick
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys

MODEL_DEFAULT = os.environ.get("QWEN35_MODEL_PATH", "Qwen/Qwen3.5-0.8B")
CHUNK_MNBT = 256

CORPUS = (
    "The vLLM Metal backend executes transformer inference on Apple "
    "silicon GPUs using MLX. A chunked prefill splits a long prompt into "
    "scheduler-sized pieces, and every piece except the last writes its "
    "keys, values and recurrent state without sampling a token. "
) * 40

SUFFIXES = (
    " Summarize the key points now.",
    " List three risks in bullets.",
    " What is the main topic? Answer briefly.",
)


def _child_env() -> None:
    for key, val in (
        ("VLLM_ENABLE_V1_MULTIPROCESSING", "0"),
        ("VLLM_METAL_USE_PAGED_ATTENTION", "1"),
        ("VLLM_METAL_MEMORY_FRACTION", "0.5"),
    ):
        os.environ.setdefault(key, val)


def build_prompts(tok, quick: bool) -> list[tuple[str, list[int]]]:
    ids = tok(CORPUS)["input_ids"]
    lengths = (700,) if quick else (700, 1100, 1500)
    prompts: list[tuple[str, list[int]]] = []
    for length in lengths:
        assert len(ids) > length
        for i, suffix in enumerate(SUFFIXES):
            sids = tok(suffix, add_special_tokens=False)["input_ids"]
            prompts.append((f"len{length}_s{i}", ids[:length] + sids))
    return prompts


def run_child(model: str, engaged: bool, quick: bool, queue) -> None:
    _child_env()
    from vllm import LLM, SamplingParams

    import vllm_metal.v1.model_adapter as ma_mod

    # Reach spy: count projection-free intermediate forwards so the engaged
    # arm can prove the body-only path really ran (and the disabled arm
    # that it never does).
    calls = {"n": 0}
    orig_forward = ma_mod.DefaultModelAdapter.intermediate_forward
    orig_supports = ma_mod.DefaultModelAdapter.supports_intermediate_forward

    def spy(self, model_obj, input_ids, *, cache=None):
        calls["n"] += 1
        return orig_forward(self, model_obj, input_ids, cache=cache)

    ma_mod.DefaultModelAdapter.intermediate_forward = spy
    if not engaged:
        ma_mod.DefaultModelAdapter.supports_intermediate_forward = (
            lambda self, model_obj: False
        )
    try:
        llm = LLM(
            model=model,
            max_model_len=2048,
            max_num_seqs=4,
            max_num_batched_tokens=CHUNK_MNBT,
            enable_chunked_prefill=True,
            enable_prefix_caching=False,
        )
        tok = llm.get_tokenizer()
        prompts = build_prompts(tok, quick)
        sp = SamplingParams(temperature=0, max_tokens=32)
        results: dict[str, list[int]] = {}
        for tag, token_ids in prompts:
            out = llm.generate([{"prompt_token_ids": token_ids}], sp)[0]
            results[tag] = list(out.outputs[0].token_ids)
    finally:
        ma_mod.DefaultModelAdapter.intermediate_forward = orig_forward
        ma_mod.DefaultModelAdapter.supports_intermediate_forward = orig_supports

    if engaged:
        assert calls["n"] >= len(prompts), (
            f"engaged arm ran only {calls['n']} intermediate forwards for "
            f"{len(prompts)} prompts — the body-only path did not engage"
        )
    else:
        assert calls["n"] == 0, (
            f"disabled arm unexpectedly ran {calls['n']} intermediate forwards"
        )
    queue.put(results)


def run_pair(model: str, quick: bool) -> tuple[int, dict]:
    ctx = mp.get_context("spawn")
    per_arm: dict[bool, dict[str, list[int]]] = {}
    for engaged in (False, True):
        queue = ctx.Queue()
        proc = ctx.Process(target=run_child, args=(model, engaged, quick, queue))
        proc.start()
        try:
            per_arm[engaged] = queue.get(timeout=1500)
        finally:
            proc.join(timeout=60)
            if proc.is_alive():
                proc.terminate()
        if proc.exitcode != 0:
            raise RuntimeError(f"child (engaged={engaged}) exited with {proc.exitcode}")
    reference, engaged_results = per_arm[False], per_arm[True]
    mismatches = {
        key: {"ref": reference[key], "engaged": engaged_results[key]}
        for key in reference
        if reference[key] != engaged_results[key]
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
            print(f"  {key}: ref={pair['ref']} engaged={pair['engaged']}")
        return 1
    print(f"PARITY PASS: {total}/{total} prompts token-identical")
    return 0


if __name__ == "__main__":
    sys.exit(main())
