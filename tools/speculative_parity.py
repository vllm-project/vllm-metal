#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Speculative workers for tools.check_parity.

Unlike requesting sample logprobs, observing target rows inside the runner
keeps greedy drafting enabled. This diagnostic is not a performance benchmark.
Each engine runs in a separate process to release Metal allocations.
Use --audit-continuations for strict native replay on every emitted prefix.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

from tools.attention_bench_utils import source_file_hashes
from tools.check_parity import mlx_generate
from tools.parity_prompts import PROMPTS
from tools.parity_utils import parity_report
from vllm_metal.config import (
    DSPARK_DRAFT_QUANTIZATION_KEY,
)


def run_engine(args):
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    import mlx.core as mx
    import numpy as np
    from vllm import LLM, SamplingParams

    from tools.continuation_audit import decision_kinds, score_token

    reference = json.loads((args.output_dir / "native.json").read_text())
    spec = (
        None
        if args.worker == "target"
        else {
            "method": args.method,
            "num_speculative_tokens": args.num_draft_tokens,
            "num_speculative_tokens_per_batch_size": args.draft_schedule,
        }
    )
    if spec is not None:
        # Model-free methods (suffix) draft from token history alone and
        # take no draft checkpoint.
        if args.draft is not None:
            spec["model"] = args.draft
        if args.dspark_draft_topk is not None:
            spec["dspark_draft_topk"] = args.dspark_draft_topk
    llm = LLM(
        model=args.target,
        max_model_len=1024,
        max_num_seqs=max(args.batch_size),
        max_num_batched_tokens=256,
        block_size=16,
        gpu_memory_utilization=args.gpu_memory_utilization,
        num_gpu_blocks_override=1 + 64 * max(args.batch_size),
        enable_prefix_caching=False,
        async_scheduling=False,
        speculative_config=spec,
        additional_config=(
            {DSPARK_DRAFT_QUANTIZATION_KEY: args.dspark_draft_quantization}
            if spec is not None and args.dspark_draft_quantization is not None
            else {}
        ),
    )
    runner = llm.llm_engine.model_executor.driver_worker.model_runner
    sample = runner._sample_paged_batch
    records, prompts, decisions = {}, {}, {}
    request_ids = {}
    stats = {"drafted": 0, "accepted": 0, "scheduled_widths": {}}
    if runner._drafter is not None:
        propose = runner._drafter.propose

        def observe_width(ctx):
            widths = stats["scheduled_widths"]
            width = ctx.num_speculative_tokens
            widths[width] = widths.get(width, 0) + 1
            return propose(ctx)

        runner._drafter.propose = observe_width

    def record(req_id, row, token, kind, query_tokens):
        decision = score_token(
            np.array(row.astype(mx.float32)), token, top_k=args.top_k
        )
        records[req_id].append(decision["top_logprobs"])
        if args.audit_continuations:
            decision.update(kind=kind, query_tokens=query_tokens)
            decisions[req_id].append(decision)

    def observe(*positional, **keywords):
        state = runner._execute_model_state
        for new in state.scheduler_output.scheduled_new_reqs:
            output_state = llm.llm_engine.output_processor.request_states[new.req_id]
            request_ids[output_state.external_req_id] = new.req_id
            records[new.req_id] = []
            decisions[new.req_id] = []
            prompts[new.req_id] = new.prompt_token_ids
        lengths = {req_id: len(req.token_ids) for req_id, req in state.decode_reqs}
        result = sample(*positional, **keywords)
        for (req_id, req), segment in zip(
            state.decode_reqs, state.decode_segments, strict=True
        ):
            count = len(req.token_ids) - lengths[req_id]
            if segment.draft_token_ids:
                stats["drafted"] += len(segment.draft_token_ids)
                stats["accepted"] += count - 1
            emitted = req.token_ids[lengths[req_id] :]
            kinds = decision_kinds(list(segment.draft_token_ids), emitted)
            for i, (token, kind) in enumerate(zip(emitted, kinds, strict=True)):
                record(
                    req_id,
                    state.logits[0, segment.start_row + i],
                    token,
                    kind,
                    segment.num_query_tokens,
                )
        for i, (prefill, entry) in enumerate(
            zip(state.prefill_reqs, state.batch.paged_prefill_entries, strict=True)
        ):
            if entry.result_mode != "intermediate":
                row = state.logits_cu_seqlens[len(state.decode_segments) + i + 1] - 1
                token = runner._request_states[prefill.req_id].token_ids[-1]
                record(
                    prefill.req_id,
                    state.logits[0, row],
                    token,
                    "prefill",
                    len(prefill.token_ids),
                )
        return result

    runner._sample_paged_batch = observe
    try:
        for batch_size in args.batch_size:
            stats.update(drafted=0, accepted=0, scheduled_widths={})
            outputs = []
            for start in range(0, len(reference), batch_size):
                records.clear()
                prompts.clear()
                decisions.clear()
                request_ids.clear()
                refs = reference[start : start + batch_size]
                results = llm.generate(
                    [{"prompt_token_ids": r["input_ids"]} for r in refs],
                    SamplingParams(
                        temperature=0, max_tokens=args.max_tokens, ignore_eos=True
                    ),
                    use_tqdm=False,
                )
                for ref, result in zip(refs, results, strict=True):
                    req_id = request_ids[result.request_id]
                    if prompts[req_id] != ref["input_ids"]:
                        raise ValueError("Serving changed the input token IDs")
                    output = result.outputs[0]
                    rows = records[req_id][: args.max_tokens]
                    if len(rows) != len(output.token_ids):
                        raise AssertionError("Missing target verification rows")
                    outputs.append(
                        {
                            "input_ids": prompts[req_id],
                            "tokens": list(output.token_ids),
                            "text": output.text,
                            "top_logprobs": rows,
                            **(
                                {"decisions": decisions[req_id][: args.max_tokens]}
                                if args.audit_continuations
                                else {}
                            ),
                        }
                    )
            path = args.output_dir / f"{args.worker}-b{batch_size}.json"
            path.write_text(json.dumps({"outputs": outputs, "stats": stats}))
            if (
                args.worker == args.method
                and args.schedule_lookup[batch_size] > 0
                and not stats["drafted"]
            ):
                raise AssertionError("Parity run did not exercise drafting")
    finally:
        llm.llm_engine.engine_core.shutdown()


def validate_args(args, parser):
    if args.method == "suffix" and args.draft is not None:
        parser.error("--draft cannot be used with --method suffix")
    if args.dspark_draft_quantization is not None and args.method != "dspark":
        parser.error("--dspark-draft-quantization requires --method dspark")
    if args.dspark_draft_topk is not None and (
        args.method != "dspark" or args.dspark_draft_topk < 1
    ):
        parser.error(
            "--dspark-draft-topk requires --method dspark and a positive value"
        )
    if args.draft is None:
        args.draft = {
            "dflash": "z-lab/Qwen3-4B-DFlash-b16",
            "dspark": "deepseek-ai/dspark_qwen3_4b_block7",
            "draft_model": "Qwen/Qwen3-0.6B",
            # Suffix drafts from token history; there is no draft checkpoint.
            "suffix": None,
        }[args.method]
    if args.worker not in (None, "native", "target", args.method, "replay"):
        parser.error("Draft worker must match --method")
    if args.worker == "replay" and not args.audit_continuations:
        parser.error("Replay requires --audit-continuations")
    if (
        min(args.batch_size) < 1
        or len(set(args.batch_size)) != len(args.batch_size)
        or args.num_draft_tokens < 1
        or not 1 <= args.max_tokens <= 512
    ):
        parser.error(
            "batch sizes must be unique and positive and max-tokens must be 1–512"
        )
    from vllm.v1.spec_decode.dynamic.utils import build_dynamic_sd_schedule_lookup

    try:
        args.schedule_lookup = build_dynamic_sd_schedule_lookup(
            args.draft_schedule
            if args.draft_schedule is not None
            else [[1, max(args.batch_size), args.num_draft_tokens]],
            max(args.batch_size),
            args.num_draft_tokens,
        )
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))


def run(args) -> bool:
    """Run correctness workers serially, then preserve every comparison."""
    os.environ["MLX_ENABLE_TF32"] = "0"
    if args.worker == "native":
        reference = mlx_generate(
            args.target, args.prompt or PROMPTS, args.max_tokens, args.top_k
        )
        (args.output_dir / "native.json").write_text(json.dumps(reference))
        return True
    if args.worker == "replay":
        from tools.continuation_audit import audit_serving_run

        report = audit_serving_run(args)
        (args.output_dir / "continuation-audit.json").write_text(
            json.dumps(report, indent=2, allow_nan=False)
        )
        return True
    if args.worker:
        run_engine(args)
        return True
    args.output_dir.mkdir(parents=True, exist_ok=False)
    from huggingface_hub import snapshot_download

    # Resolve once so every worker loads the same checkpoint revision.
    for name in ("target", "draft"):
        checkpoint = getattr(args, name)
        if checkpoint is None:
            continue
        path = Path(checkpoint)
        if not path.is_dir():
            path = Path(snapshot_download(checkpoint))
        setattr(args, name, str(path.resolve()))
    root = Path(__file__).resolve().parents[1]
    sources = sorted(
        p
        for directory in (root / "vllm_metal", root / "tools")
        for p in directory.rglob("*")
        if p.suffix in (".py", ".metal", ".cpp", ".h")
    )
    (args.output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "source_sha256": source_file_hashes(root, sources),
                "target": args.target,
                "draft": args.draft,
                "method": args.method,
                "num_draft_tokens": args.num_draft_tokens,
                "dspark_draft_topk": args.dspark_draft_topk,
                "dspark_draft_quantization": args.dspark_draft_quantization,
                "max_tokens": args.max_tokens,
                "top_k": args.top_k,
                "batch_sizes": args.batch_size,
                "draft_schedule": args.draft_schedule,
                "audit_continuations": args.audit_continuations,
                "environment": {
                    name: os.environ.get(name)
                    for name in ("MLX_ENABLE_TF32", "VLLM_METAL_SPEC_VERIFY_WINDOW")
                },
                "versions": {
                    name: importlib.metadata.version(name)
                    for name in ("vllm", "mlx", "mlx-lm", "transformers")
                },
            },
            indent=2,
        )
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    workers = ["native", "target", args.method]
    if args.audit_continuations:
        workers.append("replay")
    for worker in workers:
        with (args.output_dir / f"{worker}.log").open("w") as log:
            command = [
                sys.executable,
                "-m",
                "tools.check_parity",
                *sys.argv[1:],
                "--model",
                args.target,
            ]
            if args.draft is not None:
                command += ["--draft", args.draft]
            command += ["--output-dir", str(args.output_dir), "--worker", worker]
            subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                cwd=root,
                env=env,
                # Replay includes a native control and both serving arms for
                # every batch size, each with a full continuation per prompt.
                timeout=600 * (1 + 2 * len(args.batch_size))
                if worker == "replay"
                else 600,
            )
    reference = json.loads((args.output_dir / "native.json").read_text())
    reports = {}
    target_suffix_exact = True
    for batch_size in args.batch_size:
        arms = {}
        for worker in ("target", args.method):
            result = json.loads(
                (args.output_dir / f"{worker}-b{batch_size}.json").read_text()
            )
            arms[worker] = result["outputs"]
            print(f"{worker}, batch {batch_size}: {result['stats']}")
            reports[f"native-vs-{worker}-b{batch_size}"] = parity_report(
                reference,
                result["outputs"],
                max_tokens=args.max_tokens,
                top_k=args.top_k,
                labels=("native", worker),
            )
        reports[f"target-vs-{args.method}-b{batch_size}"] = parity_report(
            arms["target"],
            arms[args.method],
            max_tokens=args.max_tokens,
            top_k=args.top_k,
            labels=("target", args.method),
        )
        if args.method == "suffix":
            target_suffix_exact &= len(arms["target"]) == len(arms["suffix"]) and all(
                target["input_ids"] == suffix["input_ids"]
                and target["tokens"] == suffix["tokens"]
                for target, suffix in zip(arms["target"], arms["suffix"], strict=True)
            )
    passed = all(report["passed"] for report in reports.values())
    summary = {"top_k": args.top_k, "comparisons": reports}
    if args.method == "suffix":
        summary["target_suffix_exact_passed"] = target_suffix_exact
        passed &= target_suffix_exact
    if args.audit_continuations:
        report = json.loads((args.output_dir / "continuation-audit.json").read_text())
        for name, audit in report["reports"].items():
            message = (
                f"{name}: {audit['tokens']} tokens audited, "
                f"{audit['native_argmax_mismatches']} native greedy mismatches"
            )
            if audit["serving_argmax_mismatches"] is not None:
                message += (
                    f", {audit['serving_argmax_mismatches']} serving greedy mismatches"
                )
            print(message)
        summary["continuation_audit_passed"] = report["passed"]
        passed &= report["passed"]
    summary["passed"] = passed
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    return passed
