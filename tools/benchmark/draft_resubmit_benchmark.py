# SPDX-License-Identifier: Apache-2.0
"""Benchmark for draft-model KV cache cross-request reuse and losslessness.

Two things are measured on a *single* tree (no cross-tree comparison):

**Reuse.** Two speculative-decode generates of the same prompt, under fresh
request ids each time:

  gen1_cold      -- fresh request, cold draft cache
  gen2_resubmit  -- identical prompt, new request id

A tree without cross-request reuse re-ingests the full prompt into the draft
model's KV inside every first propose; with the scheduler-managed draft cache
the resubmit reuses the committed prefix blocks and ingests only the uncached
suffix. Greedy throughout.

**Losslessness.** A separate non-speculative reference generate of the same
prompt establishes ground-truth output tokens. The two spec-decode runs are
compared against it; ``lossless`` is ``true`` when the token ids match exactly.
This proves the spec-decode path does not alter greedy output without relying
on a comparison across different code trees (where a broken base path can
produce degenerate output, making the comparison meaningless).

Structural numbers come from monkeypatching ``DraftModelProposer``'s plan
construction: ``first_plan_reused_tokens`` / ``first_plan_ingest_tokens`` are
the first plan's ``draft_seq_len`` (where existing KV starts) and its ingest
length, read directly from internal proposer state.

Notes on methodology:

- Requires ``VLLM_ENABLE_V1_MULTIPROCESSING=0`` so the monkeypatch reaches
  the engine; the script sets it if unset. Set ``--gpu-memory-utilization``
  as needed for the machine (0.4 is the default).
- The warm 1-token generate heats the *target* prefix cache only on trees
  where ``propose()`` is skipped for prefill-only steps. On the
  scheduler-managed tree the runner calls ``propose()`` every step, so the
  warm request also ingests draft KV and gen1_cold is draft-warm there. Use
  ``--skip-warm`` for a draft-cold gen1.
- ``propose_first_ms`` times the first ``propose()`` call, which on the
  scheduler-managed tree can be an empty call that precedes the first plan;
  prefer the plan columns and wall/tpot for conclusions.
- The reference run executes in its own subprocess: an in-process
  ``del llm`` / ``gc.collect()`` does not release Metal memory
  (``mx.clear_cache()`` does not either), so the reference's KV would still
  be resident
  when the spec run profiles its budget. The subprocess exit releases
  everything. Pass ``--skip-lossless`` to skip it if memory is tight.

Output: one JSON object with ``reference`` (or ``null``) and ``spec_runs``
(array of per-generate records), prefixed with ``RESULT ``.

Usage:

    python tools/benchmark/draft_resubmit_benchmark.py --prefix 8192 \\
        --gpu-memory-utilization 0.4
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time

import mlx.core as mx

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

from vllm import LLM, SamplingParams  # noqa: E402

from vllm_metal.v1 import draft_model_proposer as dmp  # noqa: E402

DURS: list[float] = []
PLANS: list[tuple[int, int]] = []  # (draft_seq_len, n_ingest) per plan
FIRST_PROPOSE_PEAKS: list[float] = []  # GiB high-water at end of first propose

_orig_propose = dmp.DraftModelProposer.propose


def _timed_propose(self, ctx):
    if len(DURS) == 0:
        mx.reset_peak_memory()  # isolate the first propose's own allocations
    t0 = time.perf_counter()
    r = _orig_propose(self, ctx)  # self-evals its drafts before returning
    DURS.append(time.perf_counter() - t0)
    if len(DURS) == 1:
        FIRST_PROPOSE_PEAKS.append(mx.get_peak_memory() / (2**30))
    return r


def _note_plan(plan) -> None:
    if plan is not None:
        PLANS.append((plan.draft_seq_len, len(plan.ingest_tokens)))


if hasattr(dmp.DraftModelProposer, "_make_plan"):
    # Single plan constructor (pre-scheduler-cache trees).
    _orig_make_plan = dmp.DraftModelProposer._make_plan

    def _logged_make_plan(self, req_id, state, num_speculative_tokens):
        plan = _orig_make_plan(self, req_id, state, num_speculative_tokens)
        _note_plan(plan)
        return plan

    dmp.DraftModelProposer._make_plan = _logged_make_plan
else:
    # Plan construction split into decode/prefill variants (scheduler-managed
    # cache tree).
    _orig_decode = dmp.DraftModelProposer._make_decode_plan
    _orig_prefill = dmp.DraftModelProposer._make_prefill_plan

    def _logged_decode(self, req_id, state, k, drafting_req_ids):
        plan = _orig_decode(self, req_id, state, k, drafting_req_ids)
        _note_plan(plan)
        return plan

    def _logged_prefill(self, prefill, result_mode, k, drafting_req_ids):
        plan = _orig_prefill(self, prefill, result_mode, k, drafting_req_ids)
        _note_plan(plan)
        return plan

    dmp.DraftModelProposer._make_decode_plan = _logged_decode
    dmp.DraftModelProposer._make_prefill_plan = _logged_prefill

dmp.DraftModelProposer.propose = _timed_propose


def _build_prompt(model: str, prefix: int) -> str:
    """Build a deterministic prompt of exactly ``prefix`` tokens."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model)
    unit = (
        "The engineer examined the trace carefully, noting where the latency "
        "spiked and which subsystem held the lock the longest before yielding. "
    )
    prompt = unit
    while len(tok.encode(prompt)) < prefix:
        prompt += unit
    prompt = tok.decode(tok.encode(prompt)[:prefix])
    return prompt


def _run_reference(args, prompt: str) -> dict | None:
    """Generate without speculative decoding to establish ground truth."""
    sys.stderr.write("=== reference (no spec) start ===\n")
    sys.stderr.flush()
    llm = LLM(
        model=args.model,
        max_model_len=args.prefix + 256,
        max_num_seqs=1,
        enable_prefix_caching=True,
        async_scheduling=False,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    sp = SamplingParams(temperature=0, max_tokens=args.gen)
    t0 = time.perf_counter()
    out = llm.generate([prompt], sp)
    dt = time.perf_counter() - t0
    ids = list(out[0].outputs[0].token_ids)
    sys.stderr.write("=== reference done ===\n")
    sys.stderr.flush()
    return {
        "label": "reference",
        "gen": len(ids),
        "wall_s": round(dt, 3),
        "tpot_ms": round(dt / max(len(ids), 1) * 1000, 2),
        "token_ids": ids,
    }


def _run_reference_subprocess(args) -> dict:
    """Run the reference generate in its own process.

    The reference's Metal allocations (weights + KV) are only guaranteed
    released once its process exits; in-process teardown of the ``LLM``
    object does not free them before the spec run profiles its KV budget,
    so with a tight KV budget the spec run would
    see a reduced or negative ``kv_budget`` and fail.
    """
    cmd = [
        sys.executable,
        __file__,
        "--reference-only",
        "--model",
        args.model,
        "--prefix",
        str(args.prefix),
        "--gen",
        str(args.gen),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
    ]
    # stderr passes through so the child's progress markers stay visible.
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"reference subprocess failed with exit code {proc.returncode}"
            " (see its stderr above)"
        )
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT") :].lstrip())
    raise RuntimeError("reference subprocess produced no RESULT line")


def _run_spec(args, prompt: str, reference_ids: list[int] | None) -> list[dict]:
    """Run speculative-decode generates, comparing against reference."""
    llm = LLM(
        model=args.model,
        max_model_len=args.prefix + 256,
        max_num_seqs=1,
        enable_prefix_caching=True,
        async_scheduling=False,
        gpu_memory_utilization=args.gpu_memory_utilization,
        speculative_config={
            "method": "draft_model",
            "model": args.model,
            "num_speculative_tokens": args.num_speculative_tokens,
        },
    )

    if not args.skip_warm:
        llm.generate(
            [prompt], SamplingParams(temperature=0, max_tokens=1)
        )  # warm target

    results = []
    sp = SamplingParams(temperature=0, max_tokens=args.gen)
    for label in ("gen1_cold", "gen2_resubmit"):
        DURS.clear()
        PLANS.clear()
        FIRST_PROPOSE_PEAKS.clear()
        mx.reset_peak_memory()
        t0 = time.perf_counter()
        out = llm.generate([prompt], sp)
        dt = time.perf_counter() - t0
        peak_mem_gb = mx.get_peak_memory() / (2**30)
        ids = list(out[0].outputs[0].token_ids)
        ms = sorted(d * 1000 for d in DURS)
        first_plan = PLANS[0] if PLANS else (None, None)
        lossless = (ids == reference_ids) if reference_ids is not None else None
        results.append(
            {
                "label": label,
                "prefix": args.prefix,
                "gen": len(ids),
                "wall_s": round(dt, 3),
                "tpot_ms": round(dt / max(len(ids), 1) * 1000, 2),
                "peak_mem_gb": round(peak_mem_gb, 2),
                "first_propose_peak_gb": (
                    round(FIRST_PROPOSE_PEAKS[0], 2) if FIRST_PROPOSE_PEAKS else None
                ),
                "propose_calls": len(ms),
                "propose_first_ms": round(DURS[0] * 1000, 1) if DURS else None,
                "propose_median_ms": round(ms[len(ms) // 2], 1) if ms else None,
                "first_plan_reused_tokens": first_plan[0],
                "first_plan_ingest_tokens": first_plan[1],
                "lossless": lossless,
                "token_ids": ids,
            }
        )
        sys.stderr.write(f"=== {label} done ===\n")
        sys.stderr.flush()

    del llm
    gc.collect()
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", type=int, default=8192)
    ap.add_argument("--gen", type=int, default=96)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.4)
    ap.add_argument("--num-speculative-tokens", type=int, default=3)
    ap.add_argument("--skip-warm", action="store_true")
    ap.add_argument(
        "--skip-lossless",
        action="store_true",
        help="Skip the non-speculative reference run (saves memory/time).",
    )
    ap.add_argument(
        "--reference-only",
        action="store_true",
        help=argparse.SUPPRESS,  # internal: spawned by the parent process
    )
    args = ap.parse_args()

    prompt = _build_prompt(args.model, args.prefix)

    if args.reference_only:
        print("RESULT " + json.dumps(_run_reference(args, prompt)))
        return

    # -- Reference: no speculative decoding ------------------------------------
    # In its own subprocess so its Metal memory is fully released before the
    # spec run profiles the KV budget.
    reference = None
    reference_ids = None
    if not args.skip_lossless:
        reference = _run_reference_subprocess(args)
        reference_ids = reference["token_ids"]

    # -- Spec-decode runs with instrumentation ---------------------------------
    spec_runs = _run_spec(args, prompt, reference_ids)

    output = {"reference": reference, "spec_runs": spec_runs}
    print("RESULT " + json.dumps(output))


if __name__ == "__main__":
    main()
