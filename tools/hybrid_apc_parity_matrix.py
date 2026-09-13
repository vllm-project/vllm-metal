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

Since the tie-level mechanism is inherent to any restore-vs-recompute
pair on fp hardware (not just the decode-completed-checkpoint case above),
the gate waives a mismatch when either arm's top-2 logprobs are bitwise
equal at the flip position (an exact tie): neither choice is wrong.
Mismatches without an exact tie stay hard failures — that class is what
the gate exists to catch (real state corruption).

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
import queue as queue_mod
import sys
import time

MODEL_DEFAULT = os.environ.get("QWEN35_MODEL_PATH", "Qwen/Qwen3.5-0.8B")

CORPUS_A = (
    "The vLLM Metal backend executes transformer inference on Apple "
    "silicon GPUs using MLX. Paged attention divides the key value "
    "cache into fixed size blocks that the scheduler allocates on "
    "demand, and prefix caching lets a later request reuse blocks "
    "whose content hashes match. "
)
CORPUS_B = (
    "Recurrent state space layers summarize an arbitrarily long history "
    "into a fixed size matrix that is updated once per token, trading "
    "recall precision for constant memory and linear compute on long "
    "sequences of text. "
)

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


def _tokens_at_least(tok, corpus: str, min_tokens: int) -> list[int]:
    """Tokenize ``corpus`` repeated to exceed ``min_tokens``.

    TurboQuant hybrid alignment grows ``block_size`` with the model's
    state page (544 tokens for Qwen3.5-0.8B, ~3x that for the 27B), so a
    fixed repetition count cannot cover every qualification target.
    """
    ids = tok(corpus)["input_ids"]
    while len(ids) <= min_tokens:
        ids = ids + ids
    return ids


def build_cases(
    tok, block_size: int, shared_corpus: bool, quick: bool = False, max_len: int = 2048
):
    """Build the case matrix, reduced to fit ``max_len`` when necessary.

    The full matrix's k=3 edges need 3*block_size+1 prompt tokens; large
    aligned block sizes (784 for Qwen3.8-27B) do not fit the harness's
    2048 cap, so k and the shared depths shrink with a notice instead of
    producing prompts the engine must reject. (Growing max_model_len with
    the block size needs a two-phase build — documented as follow-up.)
    """
    k_fit = max(1, (max_len - 64) // block_size)
    ks = (1,) if quick else tuple(range(1, min(3, k_fit) + 1))
    if not quick and k_fit < 3:
        print(
            f"  [cases] k reduced to {ks[-1]}: 3 edges need "
            f"{3 * block_size + 1} tokens > max_model_len {max_len}",
            flush=True,
        )
    need = max(ks[-1] * block_size, 2 * block_size) + 64
    ids_edge = _tokens_at_least(tok, CORPUS_A, need)
    ids_hit = ids_edge if shared_corpus else _tokens_at_least(tok, CORPUS_B, need)
    assert min(len(ids_edge), len(ids_hit)) > need

    singles: list[tuple[str, list[int]]] = []
    for k in ks:
        for d in (-1, 0, 1):
            singles.append((f"edge_k{k}{d:+d}", ids_edge[: k * block_size + d]))
    depths = (("1B", block_size),)
    if not quick and 2 * block_size + 64 <= max_len:
        depths = (("1B", block_size), ("2B", 2 * block_size))
    for depth_name, depth in depths:
        prefix = ids_hit[:depth]
        for i, suffix in enumerate(SUFFIXES):
            sids = tok(suffix, add_special_tokens=False)["input_ids"]
            singles.append((f"div_{depth_name}_{i}", prefix + sids))
    singles.append(("short_after_long", ids_hit[: block_size + 100]))

    batch_blocks = 2 if 2 * block_size + 64 <= max_len else 1
    batch = [
        (
            f"batch_{i}",
            ids_hit[: batch_blocks * block_size]
            + tok(f" Task {i}: name one advantage.", add_special_tokens=False)[
                "input_ids"
            ],
        )
        for i in range(4)
    ]
    return singles, batch


def _top2_gaps(output) -> list[list]:
    """Per generated token: ``[top1_logprob - top2_logprob, top1, top2]``.

    An exact 0.0 gap means the greedy choice sat on a bitwise tie — the
    documented tie-level property (see module docstring) — but it only
    excuses a divergence when the tied pair *is* the two tokens the arms
    respectively chose; the caller checks that identity.
    """
    rows = []
    for lp_dict in output.logprobs:
        ranked = sorted(lp_dict.items(), key=lambda kv: -kv[1].logprob)
        if len(ranked) > 1:
            (t1, lp1), (t2, lp2) = ranked[0], ranked[1]
            rows.append([lp1.logprob - lp2.logprob, t1, t2])
        else:
            rows.append([float("inf"), ranked[0][0], -1])
    return rows


def run_child(
    model, enable_prefix_caching, mnbt, shared_corpus, quick, queue, collect_gaps=False
):
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
            # profile_run builds a single (1, max_num_batched_tokens) dummy
            # sequence. Without a cap the LLM-class default (8192) makes that
            # dummy four times longer than max_model_len, and its full-vocab
            # logits alone can push the measured overhead past the KV budget
            # on 16 GB hosts (negative kv_budget -> startup rejection). Keep
            # the profile within the shapes this harness actually runs.
            "max_num_batched_tokens": 2048 if mnbt is None else mnbt,
        }
        if mnbt is not None:
            kwargs["enable_chunked_prefill"] = True
        llm = LLM(**kwargs)
        cache_config = llm.llm_engine.vllm_config.cache_config
        expected_mode = "align" if enable_prefix_caching else "none"
        assert cache_config.mamba_cache_mode == expected_mode
        if enable_prefix_caching:
            assert cache_config.mamba_block_size == cache_config.block_size
        block_size = cache_config.block_size

        if mnbt is not None and mnbt % block_size != 0:
            queue.put(
                (
                    {
                        "__skipped__": (
                            f"chunked arm mnbt={mnbt} is not block-aligned with "
                            f"block_size={block_size}; pass an mnbt that is a "
                            "multiple of the model's block size once the harness "
                            "supports adaptive chunk sizing"
                        )
                    },
                    None,
                )
            )
            return

        tok = llm.get_tokenizer()
        singles, batch = build_cases(
            tok,
            block_size,
            shared_corpus,
            quick,
            max_len=llm.llm_engine.vllm_config.model_config.max_model_len,
        )
        # The strict pass must not request logprobs: num_logprobs disables
        # native greedy (SamplingBatch.params_allow_native_greedy) and with
        # it the decode pipeline, so a logprobs-instrumented run exercises a
        # different production path. Gaps are collected only in the
        # diagnostic rerun of already-mismatched cases.
        sp = SamplingParams(
            temperature=0, max_tokens=24, logprobs=2 if collect_gaps else None
        )

        results: dict[str, list[int]] = {}
        gaps: dict[str, list[list]] | None = {} if collect_gaps else None
        for rnd in (1, 2):
            for tag, token_ids in singles:
                out = llm.generate([{"prompt_token_ids": token_ids}], sp)[0]
                key = f"{tag}/r{rnd}"
                results[key] = list(out.outputs[0].token_ids)
                if gaps is not None:
                    gaps[key] = _top2_gaps(out.outputs[0])
        outs = llm.generate([{"prompt_token_ids": t} for _, t in batch], sp)
        for (tag, _), out in zip(batch, outs, strict=True):
            key = f"{tag}/batched"
            results[key] = list(out.outputs[0].token_ids)
            if gaps is not None:
                gaps[key] = _top2_gaps(out.outputs[0])
    finally:
        mr_mod.MetalModelRunner._handle_new_requests = orig

    if enable_prefix_caching:
        hits = [n for n in admissions if n > 0]
        assert hits and any(n % block_size == 0 for n in hits), (
            f"cached arm admitted no block-aligned restore: {admissions}"
        )
    queue.put((results, gaps))


def _run_child_pair(ctx, model, mnbt, shared_corpus, quick, collect_gaps=False):
    per_mode: dict[bool, tuple] = {}
    for enable in (False, True):
        queue = ctx.Queue()
        proc = ctx.Process(
            target=run_child,
            args=(model, enable, mnbt, shared_corpus, quick, queue, collect_gaps),
        )
        proc.start()
        try:
            # Poll instead of a single long get: if the child dies during
            # engine startup (e.g. a capacity rejection) it never reaches
            # queue.put, and a bare get(timeout=1500) would sit for 25
            # minutes before surfacing an unrelated queue.Empty. Fail fast
            # with the child's exit code instead.
            deadline = time.monotonic() + 1500
            while True:
                try:
                    per_mode[enable] = queue.get(timeout=5)
                    break
                except queue_mod.Empty:
                    if proc.exitcode is not None:
                        raise RuntimeError(
                            f"child (prefix_caching={enable}) exited with "
                            f"{proc.exitcode} before reporting results; "
                            "see its traceback above"
                        ) from None
                    if time.monotonic() >= deadline:
                        raise
        finally:
            proc.join(timeout=60)
            if proc.is_alive():
                proc.terminate()
        if proc.exitcode != 0:
            raise RuntimeError(f"child exited with {proc.exitcode}")
    return per_mode


def run_pair(model, mnbt, shared_corpus, quick=False):
    ctx = mp.get_context("spawn")

    def first_flip(ref_toks, cached_toks):
        return next(
            i
            for i, (a, b) in enumerate(zip(ref_toks, cached_toks, strict=False))
            if a != b
        )

    # Pass 1 — strict, production sampling path (no logprobs requested).
    strict = _run_child_pair(ctx, model, mnbt, shared_corpus, quick)
    skipped = [
        arm[0]["__skipped__"] for arm in strict.values() if "__skipped__" in arm[0]
    ]
    if skipped:
        for reason in skipped:
            print(f"  [arm] SKIPPED: {reason}", flush=True)
        return 0, {}, {}
    reference, cached = strict[False][0], strict[True][0]
    mismatched_keys = [k for k in reference if reference[k] != cached[k]]
    tie_level: dict[str, dict] = {}
    if mismatched_keys:
        # Pass 2 — diagnostic rerun with logprobs to classify the flips.
        # The rerun itself takes the non-native path, so its tokens may
        # differ from the strict pass; classification therefore uses the
        # rerun's own flip, and strict-only flips stay hard failures.
        diag = _run_child_pair(
            ctx, model, mnbt, shared_corpus, quick, collect_gaps=True
        )
        d_ref, d_cached = diag[False][0], diag[True][0]
        d_ref_gaps, d_cached_gaps = diag[False][1], diag[True][1]
        still_bad = []
        for key in mismatched_keys:
            if key not in d_ref or d_ref[key] == d_cached[key]:
                print(
                    f"  [diag] {key}: rerun converged — path-sensitive, "
                    "kept as failure",
                    flush=True,
                )
                continue
            flip = first_flip(d_ref[key], d_cached[key])
            chosen = {d_ref[key][flip], d_cached[key][flip]}

            def tied_pair(arm_gaps, _flip=flip, _chosen=chosen):
                if arm_gaps is None or _flip >= len(arm_gaps):
                    return False
                gap, t1, t2 = arm_gaps[_flip]
                return gap == 0.0 and {t1, t2} == _chosen

            if tied_pair(d_ref_gaps.get(key)) or tied_pair(d_cached_gaps.get(key)):
                tie_level[key] = {"ref": reference[key], "cached": cached[key]}
            else:
                still_bad.append(key)
        mismatched_keys = still_bad
    mismatches = {
        key: {"ref": reference[key], "cached": cached[key]} for key in mismatched_keys
    }
    return len(reference), mismatches, tie_level


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
        n_cases, mismatches, tie_level = run_pair(
            args.model, mnbt, args.shared_corpus, args.quick
        )
        failed = failed or bool(mismatches) or n_cases == 0
        print(
            f"[{arm_name}] {n_cases} comparisons, {len(mismatches)} divergent, "
            f"{len(tie_level)} tie-level (waived)",
            flush=True,
        )
        if tie_level:
            print(
                f"  tie-level (exact top-2 logprob ties at the flip): "
                f"{sorted(tie_level)}",
                flush=True,
            )
        if mismatches:
            print(json.dumps(mismatches, indent=2), flush=True)
    print("PARITY FAIL" if failed else "PARITY PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
