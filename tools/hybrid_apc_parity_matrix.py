#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Hybrid GDN prefix-caching parity matrix: cache-on vs cache-off, greedy.

Runs the same deterministic prompt matrix in two spawned children (Metal is
not fork-safe) — one with ``enable_prefix_caching=False`` (mamba_cache_mode
"none") and one with ``True`` ("align") — and asserts token-identical
outputs.  The matrix is shaped like PR #547's qualification gate:

- token-exact prompts sliced at mamba block edges (k*544 - 1 / +0 / +1),
- divergent suffixes grafted at 1-block and 2-block shared depths,
- a short-after-long prefix-of-prefix case,
- two rounds (round 2 = exact-repeat hits on round-1 blocks),
- a concurrent same-prefix batch,
- two config arms: engine defaults, and max_num_batched_tokens=1088 to
  force chunked-prefill splits at aligned boundaries.

Edge cases and hit cases use two different corpora so that every restored
checkpoint is prefill-produced; the default run is a strict gate (exit 0 iff
all comparisons match).  With ``--shared-corpus`` everything shares one
corpus, which reproduces a documented tie-level property rather than a bug:
a block completed by a *decode* step can content-hash-match a later prompt's
prefix block, and its state — computed via prefill+decode instead of pure
prefill — is mathematically but not bitwise equal, so downstream greedy
near-ties may flip (observed: identical top-2 logprobs at the flip).  This
is the same fp-path caveat as upstream's decode-KV block reuse.

Both children assert reach so the gate cannot silently test the wrong
thing: the cached child must resolve to align mode with
``mamba_block_size == block_size`` and must admit at least one request at a
block-aligned ``num_computed_tokens > 0`` (a genuine restore).

This is the single cache-on/off parity harness;
``tests/test_hybrid_prefix_caching_e2e.py`` is a thin slow-pytest wrapper
around ``--quick``.

Not in CI — requires local model weights.

Usage:
    python tools/hybrid_apc_parity_matrix.py
    python tools/hybrid_apc_parity_matrix.py --model /path/to/Qwen3.5-0.8B
    python tools/hybrid_apc_parity_matrix.py --quick           # pytest gate
    python tools/hybrid_apc_parity_matrix.py --shared-corpus   # tie-level demo
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys

MODEL_DEFAULT = os.environ.get("QWEN35_MODEL_PATH", "Qwen/Qwen3.5-0.8B")

CORPUS_A = (
    "The vLLM Metal backend executes transformer inference on Apple "
    "silicon GPUs using MLX. Paged attention divides the key value "
    "cache into fixed size blocks that the scheduler allocates on "
    "demand, and prefix caching lets a later request reuse blocks "
    "whose content hashes match. "
) * 60
CORPUS_B = (
    "Recurrent state space layers summarize an arbitrarily long history "
    "into a fixed size matrix that is updated once per token, trading "
    "recall precision for constant memory and linear compute on long "
    "sequences of text. "
) * 60

SUFFIXES = (
    " Summarize the key points now.",
    " List three risks in bullets.",
    " Translate the previous sentence into French.",
    " What is the main topic? Answer briefly.",
)


def _child_env() -> None:
    for key, val in (
        ("VLLM_ENABLE_V1_MULTIPROCESSING", "0"),
        ("VLLM_METAL_USE_PAGED_ATTENTION", "1"),
        ("VLLM_METAL_MEMORY_FRACTION", "0.5"),
    ):
        os.environ.setdefault(key, val)


def build_cases(tok, block_size: int, shared_corpus: bool, quick: bool = False):
    ids_edge = tok(CORPUS_A)["input_ids"]
    ids_hit = ids_edge if shared_corpus else tok(CORPUS_B)["input_ids"]
    assert min(len(ids_edge), len(ids_hit)) > 3 * block_size + 64

    singles: list[tuple[str, list[int]]] = []
    for k in (1,) if quick else (1, 2, 3):
        for d in (-1, 0, 1):
            singles.append((f"edge_k{k}{d:+d}", ids_edge[: k * block_size + d]))
    depths = (
        (("1B", block_size),)
        if quick
        else (
            ("1B", block_size),
            ("2B", 2 * block_size),
        )
    )
    for depth_name, depth in depths:
        prefix = ids_hit[:depth]
        for i, suffix in enumerate(SUFFIXES):
            sids = tok(suffix, add_special_tokens=False)["input_ids"]
            singles.append((f"div_{depth_name}_{i}", prefix + sids))
    singles.append(("short_after_long", ids_hit[: block_size + 100]))

    batch = [
        (
            f"batch_{i}",
            ids_hit[: 2 * block_size]
            + tok(f" Task {i}: name one advantage.", add_special_tokens=False)[
                "input_ids"
            ],
        )
        for i in range(4)
    ]
    return singles, batch


def run_child(model, enable_prefix_caching, mnbt, shared_corpus, quick, queue):
    _child_env()
    from vllm import LLM, SamplingParams

    import vllm_metal.v1.model_runner as mr_mod

    # Reach spy: record every admission's num_computed_tokens so the cached
    # arm can prove a genuine block-aligned restore happened (not just a
    # config that silently fell back to the none path).
    admissions: list[int] = []
    orig = mr_mod.MetalModelRunner._handle_new_requests

    def spy(self, batch, new_reqs, scheduler_output):
        admissions.extend(req.num_computed_tokens for req in new_reqs)
        return orig(self, batch, new_reqs, scheduler_output)

    mr_mod.MetalModelRunner._handle_new_requests = spy
    try:
        kwargs = {
            "model": model,
            "max_model_len": 2048,
            "max_num_seqs": 4,
            "enable_prefix_caching": enable_prefix_caching,
        }
        if mnbt is not None:
            kwargs["max_num_batched_tokens"] = mnbt
            kwargs["enable_chunked_prefill"] = True
        llm = LLM(**kwargs)
        cache_config = llm.llm_engine.vllm_config.cache_config
        expected_mode = "align" if enable_prefix_caching else "none"
        assert cache_config.mamba_cache_mode == expected_mode
        if enable_prefix_caching:
            assert cache_config.mamba_block_size == cache_config.block_size
        block_size = cache_config.block_size

        tok = llm.get_tokenizer()
        singles, batch = build_cases(tok, block_size, shared_corpus, quick)
        sp = SamplingParams(temperature=0, max_tokens=24)

        results: dict[str, list[int]] = {}
        for rnd in (1, 2):
            for tag, token_ids in singles:
                out = llm.generate([{"prompt_token_ids": token_ids}], sp)[0]
                results[f"{tag}/r{rnd}"] = list(out.outputs[0].token_ids)
        outs = llm.generate([{"prompt_token_ids": t} for _, t in batch], sp)
        for (tag, _), out in zip(batch, outs, strict=True):
            results[f"{tag}/batched"] = list(out.outputs[0].token_ids)
    finally:
        mr_mod.MetalModelRunner._handle_new_requests = orig

    if enable_prefix_caching:
        hits = [n for n in admissions if n > 0]
        assert hits and any(n % block_size == 0 for n in hits), (
            f"cached arm admitted no block-aligned restore: {admissions}"
        )
    queue.put(results)


def run_pair(model, mnbt, shared_corpus, quick=False):
    ctx = mp.get_context("spawn")
    per_mode: dict[bool, dict[str, list[int]]] = {}
    for enable in (False, True):
        queue = ctx.Queue()
        proc = ctx.Process(
            target=run_child, args=(model, enable, mnbt, shared_corpus, quick, queue)
        )
        proc.start()
        try:
            per_mode[enable] = queue.get(timeout=1500)
        finally:
            proc.join(timeout=60)
            if proc.is_alive():
                proc.terminate()
        if proc.exitcode != 0:
            raise RuntimeError(f"child exited with {proc.exitcode}")
    reference, cached = per_mode[False], per_mode[True]
    mismatches = {
        key: {"ref": reference[key], "cached": cached[key]}
        for key in reference
        if reference[key] != cached[key]
    }
    return len(reference), mismatches


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_DEFAULT)
    parser.add_argument(
        "--shared-corpus",
        action="store_true",
        help="use one corpus for edge and hit cases (reproduces the "
        "documented decode-completed-checkpoint tie-level divergence)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="reduced gate for the slow pytest wrapper: defaults arm only, "
        "1-block edges and 1-block-depth hits",
    )
    args = parser.parse_args()

    arms = (("defaults", None),)
    if not args.quick:
        arms = (("defaults", None), ("mnbt1088", 1088))
    failed = False
    for arm_name, mnbt in arms:
        n_cases, mismatches = run_pair(args.model, mnbt, args.shared_corpus, args.quick)
        failed = failed or bool(mismatches)
        print(
            f"[{arm_name}] {n_cases} comparisons, {len(mismatches)} mismatches",
            flush=True,
        )
        if mismatches:
            print(json.dumps(mismatches, indent=2), flush=True)
    print("PARITY FAIL" if failed else "PARITY PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
