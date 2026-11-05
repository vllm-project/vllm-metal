# SPDX-License-Identifier: Apache-2.0
"""Collect and fit a local cost/confidence profile for greedy DSpark admission."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import time
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np


def write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fit_confidence(records: list[dict[str, Any]], width: int) -> dict[str, Any]:
    """Fit prefix survival on the fitting split; only report held-out errors."""
    logits = np.zeros((len(records), width), dtype=np.float64)
    valid = np.zeros_like(logits, dtype=bool)
    labels = np.zeros_like(logits)
    fitting = np.array([r["split"] == "fit" for r in records])
    for i, row in enumerate(records):
        if row.get("generated_width") != width:
            raise ValueError(
                "Calibrate confidence from the full configured draft block"
            )
        n = row["verified"]
        if not 0 <= row["accepted"] <= n <= width or n != len(row["confidence"]):
            raise ValueError("Invalid observed acceptance prefix")
        logits[i, :n] = row["confidence"]
        valid[i, :n] = True
        labels[i, : row["accepted"]] = 1
    if not np.isfinite(logits).all():
        raise ValueError("Confidence observations must be finite")

    def sigmoid(values: np.ndarray) -> np.ndarray:
        return np.exp(-np.logaddexp(0, -values))

    # Prefer a temperature near one when losses tie. The grid is independent
    # of holdout scores and subsequent serving measurements.
    grid = sorted(np.logspace(-1, 1, 81).tolist(), key=lambda t: abs(math.log(t)))
    temperatures = []
    survival = np.ones(len(records))
    for position in range(width):
        selected = fitting & valid[:, position]
        if not selected.any():
            raise ValueError(f"No fitting observations for position {position + 1}")
        losses = [
            np.mean(
                (
                    survival[selected] * sigmoid(logits[selected, position] / t)
                    - labels[selected, position]
                )
                ** 2
            )
            for t in grid
        ]
        temperature = grid[int(np.argmin(losses))]
        temperatures.append(temperature)
        survival *= sigmoid(logits[:, position] / temperature)
    raw = np.cumprod(sigmoid(logits), axis=1)
    calibrated = np.cumprod(sigmoid(logits / np.array(temperatures)), axis=1)
    reports: dict[str, list[dict[str, Any]]] = {}
    for split, partition in (("fit", fitting), ("holdout", ~fitting)):
        reports[split] = []
        for position in range(width):
            selected = partition & valid[:, position]
            if not selected.any():
                raise ValueError(f"No {split} observations for position {position + 1}")
            actual = labels[selected, position]
            reports[split].append(
                {
                    "position": position + 1,
                    "blocks": int(selected.sum()),
                    "actual_survival": float(np.mean(actual)),
                    "raw_predicted": float(np.mean(raw[selected, position])),
                    "calibrated_predicted": float(
                        np.mean(calibrated[selected, position])
                    ),
                    "raw_brier": float(
                        np.mean((raw[selected, position] - actual) ** 2)
                    ),
                    "calibrated_brier": float(
                        np.mean((calibrated[selected, position] - actual) ** 2)
                    ),
                }
            )
    return {"temperatures": temperatures, "reports": reports}


def load_inputs(path: Path, max_length: int, output_cap: int) -> list[dict[str, Any]]:
    records = json.loads(path.read_text())
    if not isinstance(records, list) or not records:
        raise ValueError("Inputs must be a nonempty JSON list")
    partitions: dict[str, set[tuple[int, ...]]] = {"fit": set(), "holdout": set()}
    for row in records:
        if not isinstance(row, dict) or row.get("split") not in partitions:
            raise ValueError("Each input needs split='fit' or split='holdout'")
        ids = row.get("input_ids")
        if (
            not isinstance(ids, list)
            or not ids
            or any(type(token) is not int or token < 0 for token in ids)
            or len(ids) + output_cap > max_length
        ):
            raise ValueError(
                "Input token IDs and output cap must fit the context limit"
            )
        tokens = tuple(ids)
        if tokens in partitions[row["split"]]:
            raise ValueError("Duplicate input token IDs in a calibration split")
        partitions[row["split"]].add(tokens)
    if not all(partitions.values()) or not partitions["fit"].isdisjoint(
        partitions["holdout"]
    ):
        raise ValueError("Use nonempty, disjoint fitting and held-out inputs")
    return records


def collect_costs(runner: Any, args: argparse.Namespace) -> list[dict[str, Any]]:
    import mlx.core as mx

    from vllm_metal.attention.context import OffsetCache, clear_context, prepare_grouped

    proposer = runner._drafter
    cache = proposer.cache
    assert cache is not None
    group_sizes = runner._paged_group_block_sizes
    width = proposer.num_draft_tokens
    batches = runner.scheduler_config.max_num_seqs
    tables, raw = [], []
    # This engine has not admitted requests. Initialize the synthetic context
    # before timing reads, using the same shared layout as native serving.
    for tensor in cache.storage.tensors.values():
        tensor.zero_()
    mx.synchronize()

    def timed(
        context: int, batch: int, phase: str, function: Any, **details: Any
    ) -> float:
        measurements = []
        for repeat in range(args.warmups + args.repeats):
            mx.synchronize()
            start = time.perf_counter_ns()
            arrays = function()
            mx.eval(*arrays)
            mx.synchronize()
            elapsed = time.perf_counter_ns() - start
            warmup = repeat < args.warmups
            raw.append(
                {
                    "context": context,
                    "batch": batch,
                    "phase": phase,
                    "repeat": repeat,
                    "warmup": warmup,
                    "ns": elapsed,
                    **details,
                }
            )
            if not warmup:
                measurements.append(elapsed / 1e6)
        if not bool(mx.all(mx.isfinite(arrays[0])).item()):
            raise ValueError("Non-finite profiling output")
        return statistics.median(measurements)

    def target(context: int, groups: Any, lengths: list[int]) -> list[Any]:
        prepare_grouped(
            [(g, context, n) for g, n in zip(groups, lengths, strict=True)],
            [],
            group_sizes,
            merge_verify_windows=runner.merge_verify_windows,
            tq_prefill_workspace_bytes=runner.tq_prefill_workspace_bytes,
        )
        try:
            result = runner._target_forward(
                mx.ones((1, sum(lengths)), dtype=mx.int32),
                cache=[OffsetCache(0) for _ in range(runner.num_layers)],
            )
            arrays = [result.logits, *result.aux_hidden_states]
            runner._paged_attention_runtime.extend_forward_eval_outputs(arrays)
            return arrays
        finally:
            clear_context()

    def proposal(draft: Any, batch: int, groups: Any, context: int) -> list[Any]:
        ids, _, confidence = draft(
            mx.ones((batch,), dtype=mx.int32),
            [(g[proposer._group_index], context) for g in groups],
        )
        return [ids, confidence, *cache.storage.buffers]

    try:
        for context in args.contexts:
            table: dict[str, Any] = {
                "context": context,
                "draft_ms": [],
                "verify_ms": [],
            }
            for batch in range(1, batches + 1):
                pages = math.ceil((context + width + 1) / min(group_sizes))
                if batch * pages > runner._paged_attention_runtime.num_blocks():
                    raise ValueError("Cost grid exceeds allocated KV blocks")
                groups = [
                    [list(range(i * pages, (i + 1) * pages)) for _ in group_sizes]
                    for i in range(batch)
                ]
                costs = []
                for admitted in range(batch * width + 1):
                    lengths = [
                        1 + admitted // batch + int(i < admitted % batch)
                        for i in range(batch)
                    ]
                    costs.append(
                        timed(
                            context,
                            batch,
                            "target",
                            partial(target, context, groups, lengths),
                            admitted=admitted,
                            lengths=lengths,
                        )
                    )
                table["verify_ms"].append(costs)
                draft = cache.compile_draft(
                    num_draft_tokens=width,
                    draft_topk=proposer.draft_topk,
                    corrected_logits=False,
                )
                table["draft_ms"].append(
                    timed(
                        context,
                        batch,
                        "draft",
                        partial(proposal, draft, batch, groups, context),
                    )
                )
                write(args.output / "raw-costs.json", raw)
                print(f"Measured context {context}, batch {batch}", flush=True)
            tables.append(table)
            write(args.output / "costs.json", tables)
        return tables
    finally:
        write(args.output / "raw-costs.json", raw)


def collect_confidence(
    llm: Any, inputs: list[dict[str, Any]], args: argparse.Namespace
) -> list[dict[str, Any]]:
    from vllm import SamplingParams

    proposer = llm.llm_engine.model_executor.driver_worker.model_runner._drafter
    cache = proposer.cache
    original_compile, original_admit, original_propose = (
        proposer._compile_draft,
        proposer._admit_drafts,
        proposer.propose,
    )
    original_width = proposer._draft_width

    def full_width(limit: int) -> int:
        return proposer.num_draft_tokens

    proposer._draft_width = full_width
    pending: dict[str, Any] = {}
    latest: dict[str, Any] = {}
    observed, outputs = [], []
    current: dict[str, Any] = {}

    def compile_observer(width: int) -> Any:
        assert width == proposer.num_draft_tokens
        draft = cache.compile_draft(
            num_draft_tokens=width,
            draft_topk=proposer.draft_topk,
            corrected_logits=False,
        )

        def forward(anchors: Any, rows: Any) -> Any:
            ids, _, confidence = draft(anchors, rows)
            return ids, confidence

        return forward

    def admit_observer(confidence: Any, rows: Any, limits: list[int]) -> list[int]:
        latest["confidence"] = confidence.tolist()
        return limits

    def propose_observer(ctx: Any) -> Any:
        for segment, tokens in zip(
            ctx.decode_segments, ctx.decode_token_ids, strict=True
        ):
            previous = pending.pop(segment.req_id, None)
            n = len(segment.draft_token_ids)
            if n:
                assert previous is not None and previous["ids"][:n] == list(
                    segment.draft_token_ids
                )
                accepted = len(tokens) - 1
                assert 0 <= accepted <= n
                observed.append(
                    {
                        **current,
                        "confidence": previous["confidence"][:n],
                        "accepted": accepted,
                        "verified": n,
                        "generated_width": proposer.num_draft_tokens,
                        "context": segment.cache_start_pos,
                    }
                )
        latest.clear()
        result = original_propose(ctx)
        if result is not None:
            for request, ids, confidence in zip(
                result.req_ids,
                result.draft_token_ids,
                latest["confidence"],
                strict=True,
            ):
                pending[request] = {"ids": ids, "confidence": confidence}
        return result

    def generate(row: dict[str, Any]) -> dict[str, Any]:
        result = llm.generate(
            [{"prompt_token_ids": row["input_ids"]}],
            SamplingParams(temperature=0, max_tokens=args.output_cap),
            use_tqdm=False,
        )[0]
        output = result.outputs[0]
        return {
            "input_ids": result.prompt_token_ids,
            "tokens": list(output.token_ids),
            "finish_reason": output.finish_reason,
        }

    try:
        for index, row in enumerate(inputs):
            current = {"index": index, "split": row["split"]}
            proposer._compile_draft, proposer._admit_drafts, proposer.propose = (
                original_compile,
                original_admit,
                original_propose,
            )
            proposer._drafts.clear()
            control = generate(row)
            proposer._compile_draft, proposer._admit_drafts, proposer.propose = (
                compile_observer,
                admit_observer,
                propose_observer,
            )
            proposer._drafts.clear()
            pending.clear()
            trace = generate(row)
            outputs.append(
                {
                    **current,
                    "control": control,
                    "observed": trace,
                    "exact": control == trace,
                }
            )
            write(args.output / "outputs.json", outputs)
            write(args.output / "confidence.json", observed)
            if control != trace:
                raise AssertionError(
                    "Confidence observation changed the generated output"
                )
            print(
                f"Collected {index + 1}/{len(inputs)} exact control/observer pairs",
                flush=True,
            )
        return observed
    finally:
        proposer._draft_width = original_width
        proposer._compile_draft, proposer._admit_drafts, proposer.propose = (
            original_compile,
            original_admit,
            original_propose,
        )
        write(args.output / "outputs.json", outputs)
        write(args.output / "confidence.json", observed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--llm-arguments",
        type=Path,
        required=True,
        help="JSON object of matching vllm.LLM arguments",
    )
    parser.add_argument(
        "--inputs",
        type=Path,
        required=True,
        help="JSON rows with input_ids and split (fit or holdout)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New output directory; existing results are never replaced",
    )
    parser.add_argument("--contexts", type=int, nargs="+", default=[128, 1024, 3072])
    parser.add_argument("--output-cap", type=int, default=128)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    config = json.loads(args.llm_arguments.read_text())
    if not isinstance(config, dict):
        parser.error("LLM arguments must be a JSON object")
    spec = config.get("speculative_config", {})
    if not isinstance(spec, dict):
        parser.error("speculative_config must be a JSON object")
    if spec.get("method") != "dspark" or spec.get("enable_adaptive_verification"):
        parser.error(
            "Collect profiles with fixed-width DSpark, before enabling adaptive verification"
        )
    if spec.get("num_speculative_tokens_per_batch_size") is not None:
        parser.error(
            "Collect a complete full-width profile without a batch-size draft schedule"
        )
    if (
        type(spec.get("num_speculative_tokens")) is not int
        or spec["num_speculative_tokens"] < 1
    ):
        parser.error("Set an explicit positive num_speculative_tokens")
    if (
        args.warmups < 1
        or args.repeats < 1
        or args.output_cap < 1
        or args.contexts != sorted(set(args.contexts))
        or min(args.contexts) < 0
    ):
        parser.error(
            "Use positive repetition/output counts and unique increasing nonnegative contexts"
        )
    for name in ("max_model_len", "max_num_seqs"):
        if type(config.get(name)) is not int or config[name] < 1:
            parser.error(f"Set an explicit positive {name} for the measured grid")
    if config.get("enable_prefix_caching", True):
        parser.error("Set enable_prefix_caching=false for matched calibration controls")
    maximum = config["max_model_len"]
    if max(args.contexts) + spec["num_speculative_tokens"] + 1 > maximum:
        parser.error("Cost grid exceeds the configured context limit")
    inputs = load_inputs(args.inputs, maximum, args.output_cap)
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "arguments.json", config)
    write(args.output / "inputs.json", inputs)
    write(
        args.output / "settings.json",
        {
            "contexts": args.contexts,
            "output_cap": args.output_cap,
            "warmups": args.warmups,
            "repeats": args.repeats,
            "generated_width": spec["num_speculative_tokens"],
            "collector_sha256": digest(Path(__file__)),
        },
    )
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    os.environ["MLX_ENABLE_TF32"] = "0"
    from vllm import LLM

    from vllm_metal.v1.dspark_proposer import DSparkProposer

    llm = LLM(**config)
    try:
        runner = llm.llm_engine.model_executor.driver_worker.model_runner
        proposer = runner._drafter
        if (
            not isinstance(proposer, DSparkProposer)
            or proposer.draft_model.confidence_head is None
        ):
            raise ValueError(
                "Profiling requires native Metal DSpark with a trained confidence head"
            )
        identity = DSparkProposer.profile_identity(
            runner,
            proposer._checkpoint_path(runner),
            proposer.draft_topk,
            config.get("additional_config", {}).get("dspark_draft_quantization"),
        )
        write(args.output / "identity.json", identity)
        costs = collect_costs(runner, args)
        observations = collect_confidence(llm, inputs, args)
        fit = fit_confidence(observations, proposer.num_draft_tokens)
        profile = {
            "version": 1,
            "identity": identity,
            "temperatures": fit["temperatures"],
            "costs": costs,
            "calibration": {
                "reports": fit["reports"],
                "method": "Sequential prefix Brier scaling on the fitting split; fixed 81-point temperature grid 0.1..10.",
            },
            "evidence": {
                str(p.resolve()): digest(p) for p in args.output.glob("*.json")
            },
        }
        write(args.output / "profile.json", profile)
        write(
            args.output / "complete.json",
            {"complete": True, "profile_sha256": digest(args.output / "profile.json")},
        )
        print(
            f"Profile saved to {args.output / 'profile.json'}. Validate held-out correctness and end-to-end serving performance before use."
        )
    finally:
        llm.llm_engine.model_executor.shutdown()


if __name__ == "__main__":
    main()
