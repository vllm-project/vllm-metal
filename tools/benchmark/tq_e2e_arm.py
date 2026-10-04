# SPDX-License-Identifier: Apache-2.0
"""Warm paired model TTFT or teacher-forced perplexity through real vLLM.

Both TQ arms share a loaded model and quantized cache settings; the reference
disables only the prefill planner. Prefix caching is off except in the explicit
prefix-reuse probe. vLLM request metrics
provide TTFT separately from total generation time. This is an in-process
benchmark, excluding tokenization, HTTP and concurrent serving queues.

    PYTHONPATH=. python tools/benchmark/tq_e2e_arm.py --model /path/to/model
    PYTHONPATH=. python tools/benchmark/tq_e2e_arm.py --model /path/to/model \
        --quality-text /path/to/wikitext-test.txt --output quality.json
    PYTHONPATH=. python tools/benchmark/tq_e2e_arm.py --model /path/to/model \
        --prefix-probe --prefix-tokens 8192 --query-tokens 32 64 96 128
"""

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

PARA = (
    "The city library opened its doors at eight in the morning, and by nine "
    "the reading rooms were already half full. Students spread their notes "
    "across the long oak tables, older visitors settled into the armchairs "
    "by the tall windows, and the librarians moved quietly between the "
    "shelves, returning books to their places. "
)


def prefix_probe_layout(prefix_tokens, query_tokens, block_size, max_prompt):
    """Keep an exact, block-aligned cached prefix and leave query rows uncached."""
    cached = 2 * block_size if prefix_tokens is None else prefix_tokens
    if cached <= 0 or cached % block_size:
        raise ValueError(f"--prefix-tokens must be a positive multiple of {block_size}")
    required = cached + max(query_tokens)
    if required > max_prompt:
        raise ValueError(f"Prefix probe needs --prompt-tokens >= {required}")
    return cached, required


def validate_prefix_reuse(seed, row, cached_tokens, query_tokens):
    """Reject a cache miss or a different query count instead of mislabelling it."""
    if seed["num_cached_tokens"] != 0 or row["num_cached_tokens"] != cached_tokens:
        raise RuntimeError("Prefix probe did not reuse the exact requested prefix")
    if row["prompt_tokens"] - row["num_cached_tokens"] != query_tokens:
        raise RuntimeError("Prefix probe executed a different query length")
    if row["arm"] == "tq":
        dispatch = row["dispatch"]
        eligible = dispatch["threshold_eligible_layer_calls"]
        selected = dispatch["lane_layer_calls"]
        if selected > eligible or (eligible and not selected):
            raise RuntimeError(
                "Prefix probe dispatch disagrees with the query threshold; "
                "check hardware opt-in and workspace budget"
            )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument(
        "--arm", choices=("paired", "bf16", "tq", "tq-reference"), default="paired"
    )
    ap.add_argument("--prompt-tokens", type=int, nargs="+", default=[1153, 8192, 16384])
    ap.add_argument("--max-tokens", type=int, default=1)
    ap.add_argument("--batch-tokens", type=int, default=2048)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--k-quant", default="q8_0")
    ap.add_argument("--v-quant", default="q3_0")
    ap.add_argument("--quality-text", type=Path)
    ap.add_argument("--prefix-probe", action="store_true")
    ap.add_argument(
        "--prefix-tokens",
        type=int,
        help="Exact cached prefix length, block-aligned; default: two cache blocks",
    )
    ap.add_argument(
        "--query-tokens",
        type=int,
        nargs="+",
        help="Uncached query lengths for the prefix probe; default: 1 9 257",
    )
    ap.add_argument("--quality-windows", type=int, default=16)
    ap.add_argument("--quality-window", type=int, default=1024)
    ap.add_argument("--progress-interval", type=float, default=0)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    if args.query_tokens is not None and not args.prefix_probe:
        ap.error("--query-tokens requires --prefix-probe")
    if args.query_tokens is None:
        args.query_tokens = [1, 9, 257]
    if (
        min(
            *args.prompt_tokens,
            args.max_tokens,
            args.batch_tokens,
            args.reps,
            args.quality_windows,
            args.quality_window,
            *args.query_tokens,
        )
        < 1
        or args.warmup < 0
        or args.progress_interval < 0
    ):
        ap.error("lengths/repetitions must be positive and warmup nonnegative")
    if args.quality_text and args.quality_window < 256:
        ap.error("quality scoring requires window >= 256")
    if args.prefix_probe and (args.arm != "paired" or args.quality_text):
        ap.error("prefix probe requires --arm paired without --quality-text")
    if args.prefix_tokens is not None and (
        not args.prefix_probe or args.prefix_tokens <= 0
    ):
        ap.error("--prefix-tokens requires --prefix-probe and a positive length")
    if args.prefix_probe and args.max_tokens != 1:
        ap.error("prefix probe requires --max-tokens 1 to preserve the seeded prefix")
    if args.prefix_probe and max(args.query_tokens) > args.batch_tokens:
        ap.error("prefix probe query lengths must fit in one --batch-tokens step")

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    import mlx.core as mx
    import numpy as np
    from vllm import LLM, SamplingParams

    from tools.attention_bench_utils import package_versions
    from vllm_metal.attention.caches.turboquant import prefill_workspace_bytes
    from vllm_metal.attention.impls import sdpa
    from vllm_metal.attention.impls import turboquant_prefill as tq_prefill
    from vllm_metal.metal import get_ops

    original_planner = sdpa._turboquant_prefill_plan
    dispatch = {}
    active_arm = args.arm
    progress_start = last_progress = time.perf_counter()

    def counted_planner(*planner_args, **kwargs):
        nonlocal last_progress
        dispatch["prefill_layer_calls"] += 1
        dispatch["workspace_limit_bytes"] = planner_args[1].tq_prefill_workspace_bytes
        threshold = tq_prefill.min_prefill_tokens(*planner_args[4:7])
        cu = planner_args[0].cu_seqlens
        dispatch["threshold_eligible_layer_calls"] += int(
            any(b - a >= threshold for a, b in zip(cu[:-1], cu[1:], strict=True))
        )
        if threshold not in dispatch["query_thresholds"]:
            dispatch["query_thresholds"].append(threshold)
        plan = (
            None
            if active_arm == "tq-reference"
            else original_planner(*planner_args, **kwargs)
        )
        if plan is not None:
            dispatch["lane_layer_calls"] += 1
            dispatch["lane_segments"] += plan.prefill.seq_lens.shape[0]
            dispatch["max_gathered_tokens"] = max(
                dispatch["max_gathered_tokens"], plan.pool_pages.shape[0]
            )
            dispatch["max_workspace_bytes"] = max(
                dispatch["max_workspace_bytes"], plan.workspace_bytes
            )
        if args.progress_interval:
            now = time.perf_counter()
            if now - last_progress >= args.progress_interval:
                print(
                    json.dumps(
                        {
                            "progress": {
                                "arm": active_arm,
                                "elapsed_s": now - progress_start,
                                "context_tokens": max(planner_args[0].context_lens),
                                "dispatch": dict(dispatch),
                                "active_bytes": mx.get_active_memory(),
                                "peak_active_bytes": mx.get_peak_memory(),
                            }
                        }
                    ),
                    flush=True,
                )
                last_progress = now
        return plan

    sdpa._turboquant_prefill_plan = counted_planner

    def reset_dispatch():
        dispatch.update(
            prefill_layer_calls=0,
            threshold_eligible_layer_calls=0,
            query_thresholds=[],
            lane_layer_calls=0,
            lane_segments=0,
            max_gathered_tokens=0,
            max_workspace_bytes=0,
            workspace_limit_bytes=None,  # Unknown until a planner call is observed.
        )

    reset_dispatch()

    kwargs = {}
    if args.arm != "bf16":
        kwargs["additional_config"] = {
            "turboquant": True,
            "k_quant": args.k_quant,
            "v_quant": args.v_quant,
        }
    max_prompt = args.quality_window if args.quality_text else max(args.prompt_tokens)
    if args.prefix_probe and args.prefix_tokens is not None:
        max_prompt = args.prefix_tokens + max(args.query_tokens)
    t0 = time.perf_counter()
    llm = LLM(
        model=os.path.expanduser(args.model),
        max_model_len=max_prompt + args.max_tokens,
        max_num_batched_tokens=args.batch_tokens,
        max_num_seqs=1,
        gpu_memory_utilization=0.7,
        enable_prefix_caching=args.prefix_probe,
        disable_log_stats=False,
        **kwargs,
    )
    load_s = time.perf_counter() - t0
    tokenizer = llm.get_tokenizer()
    arms = ["tq-reference", "tq"] if args.arm == "paired" else [args.arm]
    metadata = {
        "model": args.model,
        "device": mx.device_info()["device_name"],
        "nax_ready": get_ops().nax_ready(),
        # The worker may cap this ceiling by the model and scheduler geometry.
        "workspace_ceiling_bytes": prefill_workspace_bytes(),
        "versions": package_versions(
            "vllm", "mlx", "mlx-lm", "mlx-vlm", "torch", "numpy"
        ),
        "mlx_enable_tf32": os.getenv("MLX_ENABLE_TF32"),
        "prefix_caching": args.prefix_probe,
        "arguments": vars(args),
        "load_s": load_s,
    }
    records = []
    summaries = []
    print(json.dumps({"metadata": metadata}, default=str), flush=True)

    def run(ids, arm, phase, trial, *, quality=False, allow_no_lane=False):
        nonlocal active_arm, progress_start, last_progress
        active_arm = arm
        reset_dispatch()
        mx.synchronize()
        active_bytes = mx.get_active_memory()
        mx.reset_peak_memory()
        start = time.perf_counter()
        progress_start = last_progress = start
        result = llm.generate(
            [{"prompt_token_ids": ids}],
            SamplingParams(
                temperature=0,
                max_tokens=1 if quality else args.max_tokens,
                ignore_eos=True,
                prompt_logprobs=1 if quality else None,
            ),
            use_tqdm=False,
        )[0]
        elapsed = time.perf_counter() - start
        mx.synchronize()
        metrics = result.metrics
        ttft = getattr(metrics, "first_token_latency", None)
        if ttft is None or ttft <= 0:
            raise RuntimeError("vLLM did not report a valid first_token_latency")
        row = {
            "phase": phase,
            "trial": trial,
            "arm": arm,
            "prompt_tokens": len(ids),
            "tokens": list(result.outputs[0].token_ids),
            "ttft_s": ttft,
            "gen_wall_s": elapsed,
            "num_cached_tokens": result.num_cached_tokens,
            "active_before_bytes": active_bytes,
            "peak_active_bytes": mx.get_peak_memory(),
            "peak_extra_bytes": max(0, mx.get_peak_memory() - active_bytes),
            "dispatch": dict(dispatch),
        }
        if arm == "tq" and not allow_no_lane and not dispatch["lane_layer_calls"]:
            raise RuntimeError(
                "No prefill lane calls: check hardware opt-in, lengths and workspace budget"
            )
        if quality:
            if result.prompt_logprobs is None or len(result.prompt_logprobs) != len(
                ids
            ):
                raise RuntimeError("Missing teacher-forced prompt logprobs")
            logprobs = []
            for target, probs in zip(ids[1:], result.prompt_logprobs[1:], strict=True):
                if not probs or target not in probs:
                    raise RuntimeError("Missing ground-truth prompt token logprob")
                logprobs.append(probs[target].logprob)
            if not all(math.isfinite(p) for p in logprobs):
                raise RuntimeError("Nonfinite prompt logprobs")
            row.update(
                scored_tokens=len(logprobs),
                nll_sum=-sum(logprobs),
                logprobs=logprobs,
            )
        records.append(row)
        print(
            json.dumps({k: v for k, v in row.items() if k != "logprobs"}),
            flush=True,
        )
        return row

    try:
        if args.prefix_probe:
            block_size = llm.llm_engine.vllm_config.cache_config.block_size
            cached_tokens, required = prefix_probe_layout(
                args.prefix_tokens, args.query_tokens, block_size, max_prompt
            )
            all_ids = tokenizer.encode(
                PARA * (required // 32 + 2), add_special_tokens=False
            )[:required]
            if len(all_ids) != required:
                raise RuntimeError(
                    "Prefix probe did not construct enough prompt tokens"
                )
            for query_tokens in args.query_tokens:
                measured = []
                for trial in range(-args.warmup, args.reps):
                    for arm in arms if trial % 2 == 0 else arms[::-1]:
                        if not llm.reset_prefix_cache():
                            raise RuntimeError("Unable to reset the prefix cache")
                        # The extra token lets vLLM cache every requested prefix
                        # block while retaining a query row to compute logits.
                        # Seed both arms with the same production path so their
                        # cached hidden states do not depend on the measured arm.
                        seed = run(
                            all_ids[: cached_tokens + 1],
                            "tq",
                            "prefix-seed",
                            trial,
                            allow_no_lane=True,
                        )
                        row = run(
                            all_ids[: cached_tokens + query_tokens],
                            arm,
                            "prefix-warmup" if trial < 0 else "prefix-reuse",
                            trial,
                            allow_no_lane=True,
                        )
                        validate_prefix_reuse(seed, row, cached_tokens, query_tokens)
                        if trial >= 0:
                            measured.append(row)
                summary = {
                    "kind": "prefix-reuse",
                    "comparison": "compressed_vs_production_policy",
                    "block_size": block_size,
                    "num_cached_tokens": cached_tokens,
                    "remaining_query_tokens": query_tokens,
                    "median_ttft_s": {
                        arm: statistics.median(
                            row["ttft_s"] for row in measured if row["arm"] == arm
                        )
                        for arm in arms
                    },
                    "lane_layer_calls": {
                        arm: sorted(
                            {
                                row["dispatch"]["lane_layer_calls"]
                                for row in measured
                                if row["arm"] == arm
                            }
                        )
                        for arm in arms
                    },
                    "greedy_tokens_match": all(
                        len(
                            {
                                tuple(row["tokens"])
                                for row in measured
                                if row["trial"] == trial
                            }
                        )
                        == 1
                        for trial in range(args.reps)
                    ),
                }
                summaries.append(summary)
                print(json.dumps(summary), flush=True)
        elif args.quality_text:
            text = args.quality_text.read_text()
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            needed = args.quality_windows * args.quality_window
            if len(token_ids) < needed:
                raise ValueError(
                    f"Quality corpus has {len(token_ids)} tokens; need {needed}"
                )
            metadata["corpus_sha256"] = hashlib.sha256(text.encode()).hexdigest()
            metadata["scored_token_ids_sha256"] = hashlib.sha256(
                json.dumps(token_ids[:needed]).encode()
            ).hexdigest()
            nll = {arm: [] for arm in arms}
            for window in range(args.quality_windows):
                ids = token_ids[
                    window * args.quality_window : (window + 1) * args.quality_window
                ]
                for arm in arms if window % 2 == 0 else arms[::-1]:
                    row = run(ids, arm, "quality", window, quality=True)
                    nll[arm].append(row["nll_sum"])
            count = args.quality_windows * (args.quality_window - 1)
            ppl = {arm: math.exp(sum(values) / count) for arm, values in nll.items()}
            summary = {
                "kind": "quality",
                "scored_tokens": count,
                "perplexity": ppl,
            }
            if len(arms) == 2:
                # Resample paired windows, preserving token alignment and context.
                deltas = (np.array(nll["tq"]) - np.array(nll["tq-reference"])) / (
                    args.quality_window - 1
                )
                bootstrap = (
                    np.random.default_rng(0)
                    .choice(deltas, (10000, len(deltas)))
                    .mean(axis=1)
                )
                summary.update(
                    relative_ppl_change_percent=(ppl["tq"] / ppl["tq-reference"] - 1)
                    * 100,
                    mean_delta_nll=float(deltas.mean()),
                    delta_nll_bootstrap_95ci=np.quantile(
                        bootstrap, [0.025, 0.975]
                    ).tolist(),
                )
            summaries.append(summary)
            print(json.dumps(summary), flush=True)
        else:
            for length in args.prompt_tokens:
                ids = tokenizer.encode(
                    PARA * (length // 32 + 2), add_special_tokens=False
                )[:length]
                assert len(ids) == length
                measured = []
                for trial in range(-args.warmup, args.reps):
                    for arm in arms if trial % 2 == 0 else arms[::-1]:
                        row = run(
                            ids, arm, "warmup" if trial < 0 else "measured", trial
                        )
                        if trial >= 0:
                            measured.append(row)
                summary = {
                    "kind": "latency",
                    "prompt_tokens": length,
                    "output_tokens": args.max_tokens,
                    "median_ttft_s": {},
                    "median_gen_wall_s": {},
                }
                for arm in arms:
                    for field in ("ttft_s", "gen_wall_s"):
                        summary[f"median_{field}"][arm] = statistics.median(
                            row[field] for row in measured if row["arm"] == arm
                        )
                summaries.append(summary)
                print(json.dumps(summary), flush=True)
        if args.output:
            args.output.write_text(
                json.dumps(
                    {"metadata": metadata, "summaries": summaries, "records": records},
                    indent=2,
                    default=str,
                )
                + "\n"
            )
    finally:
        sdpa._turboquant_prefill_plan = original_planner


if __name__ == "__main__":
    sys.exit(main())
