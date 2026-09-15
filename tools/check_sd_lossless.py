# SPDX-License-Identifier: Apache-2.0
"""Verify speculative decoding is lossless under greedy sampling (Metal).

Greedy SD must produce token-identical output to greedy non-SD: the verifier
accepts a draft token iff it equals the target argmax, so a divergence means
either the verification forward (or the draft/KV plumbing behind it) is wrong,
or the target itself chose differently between two near-tied tokens because
the verify batch has a different shape than the target-only decode batch. This
exercises the whole spec-decode stack end to end — including the attention
kernel's multi-token verify windows — with no golden files to maintain.

Each engine runs in its own subprocess (clean Metal state between configs). The
base engine also records its top-K logprobs, so every first divergence is
classified: ``TOP_K_MATCH`` when the SD token is among the base engine's top-K
at that position (printed with the logit gap; later tokens are not compared,
as in tools/check_parity.py), ``FAIL`` otherwise. A run whose SD engine never
drafted is ``INCONCLUSIVE`` (exit 2), not a pass: it compares target-only
output against itself. Strict token identity is the
default; ``--top-k K`` accepts TOP_K_MATCH. ``--max-num-seqs 1`` serves the
prompts one at a time, which removes the batch-shape difference between the
two engines and is the control for a suspected near-tie.

Upstream vLLM implements DSpark only in its GPU V2 model runner (Triton); on
Metal the platform hook presents a DSpark pair as the draft_model method and
vllm-metal builds its own DSparkProposer. The base engine is otherwise identical.

Usage:
  python tools/check_sd_lossless.py                         # Qwen3-0.6B self-draft K=3
  python tools/check_sd_lossless.py --method dspark \\
      --model mlx-community/Qwen3-4B-4bit \\
      --draft deepseek-ai/dspark_qwen3_4b_block7 -k 4 --top-k 2
  python tools/check_sd_lossless.py --method ngram --model Qwen/Qwen3-0.6B -k 3
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
engine_kwargs = {}
if cfg["max_num_seqs"] is not None:
    engine_kwargs["max_num_seqs"] = cfg["max_num_seqs"]
llm = LLM(
    model=cfg["model"],
    max_model_len=cfg["max_model_len"],
    gpu_memory_utilization=cfg["gpu_memory_utilization"],
    enable_prefix_caching=False,
    speculative_config=cfg["speculative_config"],
    # Both engines run synchronous scheduling: SD on Metal requires it, and
    # keeping the base engine identical isolates the SD stack as the only
    # variable in the token-identity comparison.
    async_scheduling=False,
    disable_log_stats=False,  # keep spec-decode acceptance counters available
    **engine_kwargs,
)
# Only the base engine records logprobs: a logprobs request is not drafted on
# Metal, so asking the SD engine for them would disable the path under test.
params = SamplingParams(
    temperature=0.0,
    max_tokens=cfg["max_tokens"],
    ignore_eos=True,
    logprobs=cfg["logprobs"],
)
start = time.perf_counter()
outputs = llm.generate(cfg["prompts"], params)
elapsed = time.perf_counter() - start
result = {o.prompt: list(o.outputs[0].token_ids) for o in outputs}
topk = {}
for o in outputs:
    rows = o.outputs[0].logprobs or []
    topk[o.prompt] = [
        sorted(
            ((tid, lp.logprob) for tid, lp in row.items()),
            key=lambda item: -item[1],
        )
        for row in rows
    ]
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
json.dump({"tokens": result, "topk": topk, "stats": stats}, open(cfg["out"], "w"))
"""


def drafted_tokens(stats: dict) -> int | None:
    """Draft tokens the speculative engine actually produced, or None if unknown.

    A speculative run that never drafted emits exactly the target's tokens, so
    comparing them proves nothing about verification. That happens whenever the
    method declines a request and it falls back to target-only generation: an
    unmatched pair, a prompt served from the prefix cache, features that were
    not captured, or a request the proposer does not draft. Such a run is
    inconclusive, not a pass, so the caller has to be able to tell.

    The counters arrive as whatever ``llm.get_metrics()`` names them, keyed
    here by the part after the last colon (``spec_decode_num_draft_tokens``
    today). Match on the suffix so a prefix or a ``_total`` spelling does not
    silently turn every run into "no metrics".
    """
    if "metrics_error" in stats:
        return None
    for suffix in ("num_draft_tokens", "num_drafts"):
        for key, value in stats.items():
            if key.endswith((suffix, f"{suffix}_total")):
                if isinstance(value, (list, tuple)):
                    value = sum(value)
                return int(value)
    return None


def run_engine(
    args, spec_config: dict | None, out_path: str, *, logprobs: int | None
) -> dict:
    cfg = {
        "model": args.model,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "max_tokens": args.max_tokens,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "prompts": args.prompts,
        "speculative_config": spec_config,
        "logprobs": logprobs,
        "out": out_path,
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(cfg, f)
        cfg_path = f.name
    env = os.environ.copy()
    env.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
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
        "--method",
        default="draft_model",
        choices=["draft_model", "dspark", "ngram", "mtp"],
    )
    p.add_argument("-k", "--num-speculative-tokens", type=int, default=3)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--num-prompts", type=int, default=len(PROMPTS))
    p.add_argument(
        "--prompt",
        action="append",
        dest="prompts",
        help="Prompt text (repeatable); replaces the built-in prompt list.",
    )
    p.add_argument("--max-model-len", type=int, default=512)
    p.add_argument(
        "--max-num-seqs",
        type=int,
        default=None,
        help="Sequence limit for both engines; 1 serves the prompts one at a time.",
    )
    p.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.35,
        help="Metal memory allowance for each engine (vLLM --gpu-memory-utilization)",
    )
    p.add_argument(
        "--control-max-num-seqs",
        type=int,
        default=None,
        help="Also run the base engine at this sequence limit and report where "
        "target-only output differs from the base run: a divergence the target "
        "reproduces on its own is a batch-shape near-tie, not a verifier fault.",
    )
    p.add_argument(
        "--dump",
        default=None,
        help="Directory to keep each engine's tokens, base top-K logprobs and stats "
        "(base.json, sd.json, control.json) for follow-up analysis.",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Accept a first divergence whose SD token is within the base engine's "
        "top-K at that position (TOP_K_MATCH); 0 (default) requires token identity.",
    )
    args = p.parse_args()
    if not args.prompts:
        args.prompts = PROMPTS[: args.num_prompts]

    spec = {
        "method": args.method,
        "model": args.draft,
        "num_speculative_tokens": args.num_speculative_tokens,
    }
    record_k = max(2, args.top_k)
    with tempfile.TemporaryDirectory() as td:
        base_result = run_engine(
            args, None, os.path.join(td, "base.json"), logprobs=record_k
        )
        sd_result = run_engine(args, spec, os.path.join(td, "sd.json"), logprobs=None)
        control = None
        if args.control_max_num_seqs is not None:
            control_args = argparse.Namespace(**vars(args))
            control_args.max_num_seqs = args.control_max_num_seqs
            control = run_engine(
                control_args, None, os.path.join(td, "control.json"), logprobs=None
            )
    if args.dump:
        os.makedirs(args.dump, exist_ok=True)
        for name, payload in (
            ("base", base_result),
            ("sd", sd_result),
            ("control", control),
        ):
            if payload is not None:
                with open(os.path.join(args.dump, f"{name}.json"), "w") as f:
                    json.dump({"args": vars(args), **payload}, f)
    base, sd = base_result["tokens"], sd_result["tokens"]
    print(f"[stats] base: {json.dumps(base_result['stats'])}")
    print(f"[stats] sd:   {json.dumps(sd_result['stats'])}")

    if control is not None:
        moved = [
            (
                prompt,
                next(
                    (i for i, (x, y) in enumerate(zip(b, c, strict=False)) if x != y),
                    min(len(b), len(c)),
                ),
            )
            for prompt, b in base.items()
            if (c := control["tokens"].get(prompt) or []) != b
        ]
        print(
            f"[control] target-only at max_num_seqs={args.control_max_num_seqs} "
            f"differs from the base run on {len(moved)}/{len(base)} prompts"
            + "".join(f"\n  @ token {i}: {prompt!r}" for prompt, i in moved)
        )

    exact = near = failed = 0
    for prompt in base:
        b, s = base[prompt], sd.get(prompt) or []
        if b == s:
            exact += 1
            continue
        div = next(
            (i for i, (x, y) in enumerate(zip(b, s, strict=False)) if x != y),
            min(len(b), len(s)),
        )
        ranked = (base_result["topk"].get(prompt) or [[]] * (div + 1))[div]
        top_ids = [tid for tid, _ in ranked[:record_k]]
        sd_token = s[div] if div < len(s) else None
        within = args.top_k > 0 and sd_token in top_ids[: args.top_k]
        verdict = "TOP_K_MATCH" if within else "FAIL"
        if within:
            near += 1
        else:
            failed += 1
        detail = ""
        if ranked:
            detail = (
                f"; base top-{record_k} logprobs "
                f"{[(tid, round(lp, 4)) for tid, lp in ranked[:record_k]]}"
            )
            if sd_token in dict(ranked):
                detail += (
                    f", gap to the SD token {ranked[0][1] - dict(ranked)[sd_token]:.4f}"
                )
            else:
                detail += f", SD token {sd_token} not in the base top-{record_k}"
        print(f"{verdict} @ token {div}: {prompt!r}{detail}")
        print(f"  base[{div}:{div + 5}] = {b[div : div + 5]}")
        print(f"  sd  [{div}:{div + 5}] = {s[div : div + 5]}")
    summary = (
        f"{len(base)} prompts x {args.max_tokens} greedy tokens, "
        f"method={args.method}, K={args.num_speculative_tokens}, "
        f"max_num_seqs={args.max_num_seqs}: {exact} exact, "
        f"{near} top-{args.top_k} near-tie, {failed} failed"
    )
    # A comparison only says something about verification if the drafter ran.
    drafted = drafted_tokens(sd_result["stats"])
    if drafted is None:
        print(
            f"INCONCLUSIVE: {summary} — the SD engine reported no spec-decode "
            f"metrics, so this run cannot show that the drafter ran"
        )
        return 2
    if drafted == 0:
        print(
            f"INCONCLUSIVE: {summary} — the SD engine drafted 0 tokens, so this "
            f"compares target-only output against itself. Check that the pair is "
            f"matched and that the requests are eligible for this method."
        )
        return 2
    if failed:
        print(f"FAIL: {summary}, {drafted} draft tokens — SD is NOT lossless")
        return 1
    print(f"PASS: {summary}, {drafted} draft tokens")
    return 0


if __name__ == "__main__":
    sys.exit(main())
