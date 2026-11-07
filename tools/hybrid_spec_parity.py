# SPDX-License-Identifier: Apache-2.0
"""End-to-end lossless check: n-gram speculative decode on hybrid GDN models.

Runs two arms of the same generate matrix in child processes (Metal is not
fork-safe): a plain greedy reference, and the same run with
``--speculative-config method=ngram``. Asserts token identity across arms,
and — via a spy on verify — that the speculative arm actually exercised both
acceptance paths (a full-span accept and a partial accept), so the GDN state
staging/fixup in ``gdn_spec`` is on the tested path, not just configured.

Arm pairs run under both mamba cache modes: ``none`` (per-request state) and
``align`` (dense block checkpoints with the draft block-boundary cap).

Usage:
    python tools/hybrid_spec_parity.py --model /path/to/Qwen3.5-0.8B
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue as queue_mod
import sys
import time
from pathlib import Path

MODEL_DEFAULT = os.environ.get("QWEN35_MODEL_PATH", "Qwen/Qwen3.5-0.8B")
GPU_MEMORY_UTILIZATION = 0.5
NUM_SPECULATIVE_TOKENS = 3
SAMPLING_KWARGS = {"temperature": 0, "max_tokens": 48, "ignore_eos": True}

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
    # The spies must run in the same process as the engine.
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"


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

    batch = [
        (f"batch_{i}", ids_a[:short_len] + sids)
        for i, sids in enumerate(
            tok(s, add_special_tokens=False)["input_ids"] for s in SUFFIXES
        )
    ]
    return singles, batch


def build_restore_prompt(prompt, generated, checkpoint):
    """Leave one uncached token beyond the checkpoint restored by the consumer."""
    history = prompt + generated
    if len(history) < checkpoint + 1:
        raise ValueError("boundary producer did not generate enough tokens")
    return history[: checkpoint + 1]


def run_child(model, enable_spec, prefix_caching, queue, canonical_prompts=None):
    _child_env()
    from vllm import LLM, SamplingParams

    import vllm_metal.v1.ngram_proposer as ng_mod
    import vllm_metal.v1.spec_decode as sd_mod

    active_tag = "initialization"
    verify_stats: list[dict] = []
    orig = sd_mod.SpeculativeDecodeController.verify

    def spy(self, logits, decode_reqs, decode_segments, *args, **kwargs):
        out = orig(self, logits, decode_reqs, decode_segments, *args, **kwargs)
        for segment, ids in zip(decode_segments, out, strict=True):
            verify_stats.append(
                {
                    "case": active_tag,
                    "start": segment.cache_start_pos,
                    "span": segment.num_query_tokens,
                    "committed": len(ids),
                }
            )
        return out

    sd_mod.SpeculativeDecodeController.verify = spy
    # Reach spy: admissions' num_computed_tokens prove the cached arms really
    # restored a prefix (a genuine hit), not just a config that fell back.
    admissions: list[dict] = []
    import vllm_metal.v1.model_runner as mr_mod

    orig_hnr = mr_mod.MetalModelRunner._handle_new_requests

    def hnr_spy(self, batch, new_reqs, scheduler_output):
        admissions.extend(
            {"case": active_tag, "computed": req.num_computed_tokens}
            for req in new_reqs
        )
        return orig_hnr(self, batch, new_reqs, scheduler_output)

    mr_mod.MetalModelRunner._handle_new_requests = hnr_spy
    orig_propose = ng_mod.NgramProposer.propose
    orig_lookup = ng_mod.VllmNgramProposer.propose
    raw_drafts: list[list[int]] = []
    cap_events: list[dict] = []

    def lookup_spy(self, *args, **kwargs):
        nonlocal raw_drafts
        out = orig_lookup(self, *args, **kwargs)
        raw_drafts = out
        return out

    def propose_spy(self, ctx):
        nonlocal raw_drafts
        raw_drafts = []
        out = orig_propose(self, ctx)
        # This probe has exactly one request. Observe the real lookup once;
        # re-running it would not prove which draft the proposer actually used.
        if active_tag == "edge_tail" and ctx.draft_caps and raw_drafts:
            assert len(raw_drafts) == 1 and len(ctx.draft_caps) == 1
            req_id, cap = next(iter(ctx.draft_caps.items()))
            final = (
                dict(zip(out.req_ids, out.draft_token_ids, strict=True)) if out else {}
            )
            cap_events.append(
                {
                    "start": len(ctx.request_states[req_id].token_ids) - 1,
                    "raw": len(raw_drafts[0]),
                    "cap": cap,
                    "drafts": len(final.get(req_id, [])),
                }
            )
        return out

    ng_mod.VllmNgramProposer.propose = lookup_spy
    ng_mod.NgramProposer.propose = propose_spy
    try:
        kwargs = {
            "model": model,
            "max_model_len": 2048,
            "max_num_seqs": 4,
            "enable_prefix_caching": prefix_caching,
            # The boundary probe restores a generated checkpoint, which the
            # default semantic-only retention policy is allowed to discard.
            "prefix_cache_retention_interval": None,
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
        cache_config = llm.llm_engine.vllm_config.cache_config
        block_size = cache_config.block_size
        singles, batch = build_cases(tok, block_size)
        # Requesting logprobs changes the greedy sampling path and disables
        # n-gram drafting. Both arms must use these identical settings.
        sp = SamplingParams(**SAMPLING_KWARGS)

        results: dict[str, list[int]] = {}
        prompts: dict[str, list[int]] = {}

        def generate_single(tag, token_ids):
            nonlocal active_tag
            active_tag = tag
            if canonical_prompts is not None:
                token_ids = canonical_prompts[tag]
            prompts[tag] = list(token_ids)
            out = llm.generate([{"prompt_token_ids": token_ids}], sp)[0]
            results[tag] = list(out.outputs[0].token_ids)

        for tag, token_ids in singles:
            generate_single(tag, token_ids)
        if canonical_prompts is not None:
            batch = [(tag, canonical_prompts[tag]) for tag, _ in batch]
        active_tag = "batch"
        prompts.update(batch)
        outs = llm.generate([{"prompt_token_ids": t} for _, t in batch], sp)
        for (tag, _), out in zip(batch, outs, strict=True):
            results[tag] = list(out.outputs[0].token_ids)

        # Isolate the producer so an earlier case cannot supply its checkpoint.
        # Use the resolved block size: the platform may round the requested size.
        checkpoint = 2 * block_size
        if checkpoint + 1 + SAMPLING_KWARGS["max_tokens"] > kwargs["max_model_len"]:
            raise ValueError("resolved block size is too large for the boundary probe")
        if prefix_caching and not llm.reset_prefix_cache():
            raise RuntimeError("could not isolate the boundary producer's prefix cache")
        edge_prompt = tok(CORPUS_A)["input_ids"][: checkpoint - 2]
        generate_single("edge_tail", edge_prompt)
        # The reference produces the canonical consumer IDs. The speculative
        # child reuses them even if its producer diverges, so inputs stay equal.
        restore_prompt = (
            canonical_prompts["edge_restore"]
            if canonical_prompts is not None
            else build_restore_prompt(
                prompts["edge_tail"], results["edge_tail"], checkpoint
            )
        )
        generate_single("edge_restore", restore_prompt)
    finally:
        sd_mod.SpeculativeDecodeController.verify = orig
        mr_mod.MetalModelRunner._handle_new_requests = orig_hnr
        ng_mod.NgramProposer.propose = orig_propose
        ng_mod.VllmNgramProposer.propose = orig_lookup

    queue.put(
        {
            "tokens": results,
            "prompts": prompts,
            "block_size": block_size,
            "prefix_cache_retention_interval": cache_config.prefix_cache_retention_interval,
            "sampling": SAMPLING_KWARGS,
            "verify": verify_stats,
            "admissions": admissions,
            "caps": cap_events,
        }
    )


def compare_arms(reference, spec, prefix_caching):
    """Require exact outputs on identical inputs and observed boundary coverage."""
    errors = []
    for key in ("prompts", "sampling", "block_size", "prefix_cache_retention_interval"):
        if reference[key] != spec[key]:
            errors.append(f"arms differ in {key}")
    mismatches = {}
    for tag in sorted(reference["tokens"].keys() | spec["tokens"].keys()):
        ref_ids, spec_ids = reference["tokens"].get(tag), spec["tokens"].get(tag)
        if ref_ids != spec_ids:
            mismatches[tag] = {"ref": ref_ids, "spec": spec_ids}
    if mismatches:
        errors.append(f"{len(mismatches)} exact token mismatches")
    for label, arm in (("reference", reference), ("spec", spec)):
        if arm["tokens"].keys() != arm["prompts"].keys() or not arm["tokens"]:
            errors.append(f"{label} has missing case outputs")
        if any(
            len(ids) != SAMPLING_KWARGS["max_tokens"] for ids in arm["tokens"].values()
        ):
            errors.append(f"{label} has incomplete generation")

    verify = spec["verify"]
    full = sum(v["span"] > 1 and v["committed"] == v["span"] for v in verify)
    partial = sum(1 < v["committed"] < v["span"] for v in verify)
    rejected = sum(v["committed"] == 1 and v["span"] > 1 for v in verify)
    if not full or not (partial + rejected):
        errors.append("need full acceptance and partial/first-reject verification")

    checkpoint = 2 * reference["block_size"]
    if prefix_caching:
        for label, arm in (("reference", reference), ("spec", spec)):
            cold = [
                a["computed"] for a in arm["admissions"] if a["case"] == "edge_tail"
            ]
            hits = [
                a["computed"] for a in arm["admissions"] if a["case"] == "edge_restore"
            ]
            if cold != [0] or hits != [checkpoint]:
                errors.append(
                    f"{label} boundary admissions: producer={cold}, consumer={hits}; "
                    f"expected [0] and [{checkpoint}]"
                )
        if not any(
            e["start"] == checkpoint - 2 and e["raw"] > e["cap"] == e["drafts"] == 1
            for e in spec["caps"]
        ):
            errors.append("boundary probe did not trim a real draft to one token")
        if not any(
            v["case"] == "edge_tail"
            and v["start"] == checkpoint - 2
            and v["span"] == v["committed"] == 2
            for v in verify
        ):
            errors.append("boundary probe did not fully accept the capped two-row span")
    return {
        "ok": not errors,
        "errors": errors,
        "mismatches": mismatches,
        "cases": len(reference["tokens"]),
        "block_size": reference["block_size"],
        "rounds": len(verify),
        "full": full,
        "partial": partial,
        "first_reject": rejected,
        "committed": sum(v["committed"] for v in verify),
        "hits": sum(a["computed"] > 0 for a in spec["admissions"]),
    }


def run_pair(model, prefix_caching):
    ctx = mp.get_context("spawn")
    arms: dict[bool, dict] = {}
    for enable in (False, True):
        queue = ctx.Queue()
        proc = ctx.Process(
            target=run_child,
            args=(
                model,
                enable,
                prefix_caching,
                queue,
                arms[False]["prompts"] if enable else None,
            ),
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
                proc.join(timeout=10)
        if proc.exitcode != 0:
            raise RuntimeError(f"child exited with {proc.exitcode}")

    return {
        "summary": compare_arms(arms[False], arms[True], prefix_caching),
        "reference": arms[False],
        "spec": arms[True],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_DEFAULT)
    parser.add_argument(
        "--output", type=Path, help="Save token IDs and coverage evidence as JSON"
    )
    args = parser.parse_args()

    ok = True
    reports = {"model": args.model, "modes": {}}
    for prefix_caching in (False, True):
        mode = "align" if prefix_caching else "none"
        report = run_pair(args.model, prefix_caching)
        reports["modes"][mode] = report
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(reports, indent=2) + "\n")
        stats = report["summary"]
        ok &= stats["ok"]
        print(
            f"[{mode}] cases={stats['cases']} block_size={stats['block_size']} "
            f"verify_rounds={stats['rounds']} "
            f"prefix_hits={stats['hits']} "
            f"full={stats['full']} partial={stats['partial']} "
            f"first_reject={stats['first_reject']} "
            f"committed={stats['committed']} "
            f"mismatches={len(stats['mismatches'])} "
            f"{'PASS' if stats['ok'] else 'FAIL'}",
            flush=True,
        )
        for error in stats["errors"]:
            print(f"  {error}")
        for key, diff in list(stats["mismatches"].items())[:3]:
            print(f"  mismatch {key}: ref={diff['ref']} spec={diff['spec']}")
    print("PARITY PASS" if ok else "PARITY FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
