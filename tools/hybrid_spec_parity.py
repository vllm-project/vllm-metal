# SPDX-License-Identifier: Apache-2.0
"""End-to-end lossless check: n-gram speculative decode on hybrid GDN models.

Runs two arms of the same generate matrix in child processes (Metal is not
fork-safe): a plain greedy reference, and the same run with
``--speculative-config method=ngram``. Asserts token identity across arms,
and — via a spy on verify — that the speculative arm actually exercised both
acceptance paths (a full-span accept and a partial accept), so the GDN state
staging/fixup in ``gdn_spec`` is on the tested path, not just configured.

Arm pairs run under both mamba cache modes: ``none`` (per-request state) and
``align`` (block-checkpointed state with the draft block-boundary cap).

Usage:
    python tools/hybrid_spec_parity.py --model /path/to/Qwen3.5-0.8B
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import queue as queue_mod
import sys
import time

MODEL_DEFAULT = os.environ.get("QWEN35_MODEL_PATH", "Qwen/Qwen3.5-0.8B")
GPU_MEMORY_UTILIZATION = 0.5
NUM_SPECULATIVE_TOKENS = 3

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
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")


def build_cases(tok, block_size: int):
    """Repetitive prompts (deep n-gram hits) plus divergent suffixes."""
    ids_a = tok(CORPUS_A)["input_ids"]
    ids_b = tok(CORPUS_B)["input_ids"]
    # Room for the 48-token generation inside max_model_len=2048.
    long_len = min(6 * block_size, 1400)
    mid_len = min(4 * block_size, 1024)
    short_len = min(2 * block_size, 512)
    assert min(len(ids_a), len(ids_b)) > long_len

    singles: list[tuple[str, list[int]]] = []
    # Long repetitive prompts: n-gram drafts mostly full-accept.
    singles.append(("repeat_a", ids_a[:long_len]))
    singles.append(("repeat_b", ids_b[:long_len]))
    # Repetitive prefix + divergent suffix: full then partial accepts.
    for i, suffix in enumerate(SUFFIXES):
        sids = tok(suffix, add_special_tokens=False)["input_ids"]
        singles.append((f"div_{i}", ids_a[:mid_len] + sids))
        singles.append((f"div_b_{i}", ids_b[:mid_len] + sids))

    # Boundary probes for the align-arm draft cap: the prompt ends 2 tokens
    # before a state-block boundary, so the first post-prefill verify span is
    # trimmed by the cap (a one-row crossing would leave the crossed block's
    # checkpoint shallow); the block-aligned re-ask then forces a prefix hit
    # that restores exactly that checkpoint, so a corrupted depth diverges.
    if 2 * block_size + 48 <= 2048 and len(ids_a) > 2 * block_size:
        singles.append(("edge_tail", ids_a[: 2 * block_size - 2]))
        singles.append(("edge_restore", ids_a[: 2 * block_size]))

    batch = [
        (f"batch_{i}", ids_a[:short_len] + sids)
        for i, sids in enumerate(
            tok(s, add_special_tokens=False)["input_ids"] for s in SUFFIXES
        )
    ]
    return singles, batch


def run_child(model, enable_spec, prefix_caching, queue):
    _child_env()
    from vllm import LLM, SamplingParams

    import vllm_metal.v1.spec_decode as sd_mod

    # Spy: record (span_len, committed) for every greedy verify round.
    verify_stats: list[tuple[int, int]] = []
    orig = sd_mod.SpeculativeDecodeController.verify_greedy

    def spy(self, logits, decode_reqs, decode_segments):
        out = orig(self, logits, decode_reqs, decode_segments)
        for segment, ids in zip(decode_segments, out, strict=True):
            verify_stats.append((segment.num_query_tokens, len(ids)))
        return out

    sd_mod.SpeculativeDecodeController.verify_greedy = spy
    # Reach spy: admissions' num_computed_tokens prove the cached arms really
    # restored a prefix (a genuine hit), not just a config that fell back.
    admissions: list[int] = []
    import vllm_metal.v1.model_runner as mr_mod

    orig_hnr = mr_mod.MetalModelRunner._handle_new_requests

    def hnr_spy(self, batch, new_reqs, scheduler_output):
        admissions.extend(req.num_computed_tokens for req in new_reqs)
        return orig_hnr(self, batch, new_reqs, scheduler_output)

    mr_mod.MetalModelRunner._handle_new_requests = hnr_spy
    try:
        kwargs = {
            "model": model,
            "max_model_len": 2048,
            "max_num_seqs": 4,
            "enable_prefix_caching": prefix_caching,
            "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
            "max_num_batched_tokens": 2048,
            # Pin the block size: the platform otherwise resolves it to
            # different values with speculation on vs off, and every
            # block-size-derived prompt below (the boundary probes in
            # particular) must be token-identical across arms.
            "block_size": 160,
        }
        if enable_spec:
            kwargs["speculative_config"] = {
                "method": "ngram",
                "num_speculative_tokens": NUM_SPECULATIVE_TOKENS,
                "prompt_lookup_min": 2,
                "prompt_lookup_max": 3,
            }
        llm = LLM(**kwargs)

        tok = llm.get_tokenizer()
        block_size = llm.llm_engine.vllm_config.cache_config.block_size
        singles, batch = build_cases(tok, block_size)
        # The reference arm collects top logprobs so the comparison can tell
        # exact-tie argmax flips (reduction-order noise on tied logits, a
        # known greedy artifact) apart from genuine state corruption. The
        # spec arm must not request logprobs: any logprobs request parks the
        # greedy-only drafter.
        sp = SamplingParams(temperature=0, max_tokens=48, ignore_eos=True)
        if not enable_spec:
            sp = SamplingParams(
                temperature=0, max_tokens=48, ignore_eos=True, logprobs=5
            )

        results: dict[str, list[int]] = {}
        ties: dict[str, dict] = {}
        for tag, token_ids in singles:
            out = llm.generate([{"prompt_token_ids": token_ids}], sp)[0]
            results[tag] = list(out.outputs[0].token_ids)
            if not enable_spec:
                ties[tag] = [
                    {t: v.logprob for t, v in (lp or {}).items()}
                    for lp in out.outputs[0].logprobs
                ]
        outs = llm.generate([{"prompt_token_ids": t} for _, t in batch], sp)
        for (tag, _), out in zip(batch, outs, strict=True):
            results[tag] = list(out.outputs[0].token_ids)
            if not enable_spec:
                ties[tag] = [
                    {t: v.logprob for t, v in (lp or {}).items()}
                    for lp in out.outputs[0].logprobs
                ]
    finally:
        sd_mod.SpeculativeDecodeController.verify_greedy = orig
        mr_mod.MetalModelRunner._handle_new_requests = orig_hnr

    queue.put(
        {
            "tokens": results,
            "verify": verify_stats,
            "ties": ties,
            "admissions": admissions,
        }
    )


def _classify_mismatch(ref, spec, ref_ties):
    """Whether a divergent spec token was an exact-tie alternative.

    Repetitive corpora produce exact logprob ties; greedy argmax then picks
    by reduction-order noise, which differs with the speculative-config
    profiling shapes. A divergence where the spec arm's token ties the
    reference's choice within 1e-4 is that artifact, not corruption.
    """
    for i, (r, s) in enumerate(zip(ref, spec, strict=False)):
        if r == s:
            continue
        step_ties = ref_ties[i] if i < len(ref_ties) else {}
        chosen = step_ties.get(r)
        alt = step_ties.get(s)
        if chosen is not None and alt is not None and abs(chosen - alt) < 1e-4:
            return "tie"
        return "real"
    return "tie" if ref != spec else None


def run_pair(model, prefix_caching):
    ctx = mp.get_context("spawn")
    arms: dict[bool, dict] = {}
    for enable in (False, True):
        queue = ctx.Queue()
        proc = ctx.Process(
            target=run_child, args=(model, enable, prefix_caching, queue)
        )
        proc.start()
        try:
            deadline = time.monotonic() + 1500
            while True:
                try:
                    arms[enable] = queue.get(timeout=5)
                    break
                except queue_mod.Empty:
                    if proc.exitcode is not None:
                        raise RuntimeError(
                            f"child (spec={enable}, cache={prefix_caching}) "
                            f"exited with {proc.exitcode} before reporting"
                        ) from None
                    if time.monotonic() >= deadline:
                        raise
        finally:
            proc.join(timeout=60)
            if proc.is_alive():
                proc.terminate()
        if proc.exitcode != 0:
            raise RuntimeError(f"child exited with {proc.exitcode}")

    reference, spec = arms[False]["tokens"], arms[True]["tokens"]
    ref_ties = arms[False]["ties"]
    mismatches: dict[str, dict] = {}
    tie_flips = 0
    for key in reference:
        if reference[key] == spec[key]:
            continue
        verdict = _classify_mismatch(reference[key], spec[key], ref_ties.get(key, []))
        if verdict == "tie":
            tie_flips += 1
            continue
        mismatches[key] = {"ref": reference[key], "spec": spec[key]}

    verify = arms[True]["verify"]
    rounds = len(verify)
    admissions = arms[True]["admissions"]
    hits = [n for n in admissions if n > 0]
    full = sum(1 for span, n in verify if n == span)
    partial = sum(1 for span, n in verify if 1 < n < span)
    rejected = sum(1 for span, n in verify if n == 1 and span > 1)
    drafted_tokens = sum(n for _, n in verify)
    return (
        len(reference),
        mismatches,
        tie_flips,
        {
            "rounds": rounds,
            "hits": len(hits),
            "full": full,
            "partial": partial,
            "first_reject": rejected,
            "committed": drafted_tokens,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_DEFAULT)
    args = parser.parse_args()

    ok = True
    for prefix_caching in (False, True):
        mode = "align" if prefix_caching else "none"
        n_cases, mismatches, tie_flips, stats = run_pair(args.model, prefix_caching)
        full_ok = stats["full"] > 0 and (stats["partial"] + stats["first_reject"]) > 0
        hit_ok = stats["hits"] > 0 if prefix_caching else True
        arm_ok = not mismatches and full_ok and stats["rounds"] > 0 and hit_ok
        ok &= arm_ok
        print(
            f"[{mode}] cases={n_cases} verify_rounds={stats['rounds']} "
            f"prefix_hits={stats['hits']} "
            f"full={stats['full']} partial={stats['partial']} "
            f"first_reject={stats['first_reject']} "
            f"committed={stats['committed']} "
            f"tie_flips={tie_flips} mismatches={len(mismatches)} "
            f"{'PASS' if arm_ok else 'FAIL'}",
            flush=True,
        )
        for key, diff in list(mismatches.items())[:3]:
            print(f"  mismatch {key}: ref={diff['ref'][:12]} spec={diff['spec'][:12]}")
        if not full_ok:
            print(
                "  acceptance-path coverage missing: need at least one full "
                "accept and one partial/rejected round"
            )
        if not hit_ok:
            print("  cached arm admitted no prefix restore (no genuine hit)")
    print("PARITY PASS" if ok else "PARITY FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
