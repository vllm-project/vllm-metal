# SPDX-License-Identifier: Apache-2.0
"""Offline native MTP-head throughput benchmark (GLM-4.7-Flash first).

Thin sibling of ``gemma4_mtp_benchmark.py`` (whose prompt/measurement helpers it
reuses) for native MTP heads run by ``NativeMTPProposer``. Differences from the
Gemma4 tool, both forced by the native-MTP contract:

- ``enable_prefix_caching=False`` is always passed: cached prompt tokens skip
  the target forward that produces the hidden states the head consumes, so the
  runner rejects native MTP with prefix caching on (Gemma4 MTP tolerates it by
  reading target paged KV instead).
- ``--chat`` renders each prompt through the tokenizer's chat template
  (``add_generation_prompt=True``); raw continuation prompts drive thinking
  models into degenerate loops that distort acceptance rates.

It also records the engine's ``spec_decode`` counters (drafts / accepted) in
the output JSON so acceptance travels with every benchmark artifact.

Usage (baseline, then MTP; one mode per process):
  python tools/benchmark/native_mtp_benchmark.py \
      --model mlx-community/GLM-4.7-Flash-4bit \
      --batch-size 1 --max-tokens 128 --repeats 3 --warmup 1 \
      --ignore-eos --chat --output-json baseline_bs1.json
  python tools/benchmark/native_mtp_benchmark.py \
      --model mlx-community/GLM-4.7-Flash-4bit \
      --assistant-model samithaj/GLM-4.7-Flash-MTP-4bit \
      --batch-size 1 --max-tokens 128 --repeats 3 --warmup 1 \
      --ignore-eos --chat --output-json mtp_bs1.json
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    # Executed as a script: make the repo root importable for tools.benchmark.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.benchmark.gemma4_mtp_benchmark import (  # noqa: E402
    RunResult,
    _build_llm_kwargs,
    _run_generate,
    _summary,
    environment_metadata,
    select_prompts,
)
from tools.benchmark.gemma4_mtp_benchmark import (  # noqa: E402
    build_parser as _gemma4_build_parser,
)


def spec_decode_metrics(llm: Any, *, require: bool) -> dict[str, Any]:
    """Read the engine's cumulative spec-decode counters (drafts / accepted).

    ``require`` fails loud when the draft/accepted counters are absent: for an MTP
    run they are the benchmark's promised evidence, so a silent miss must not pass
    as a result. A baseline run (no drafter) reports none and passes
    ``require=False``.
    """
    metrics: dict[str, Any] = {}
    for metric in llm.get_metrics():
        if "spec_decode" not in metric.name:
            continue
        value = getattr(metric, "value", None)
        if value is None:
            value = getattr(metric, "values", None)
        metrics[metric.name.split(":")[-1]] = value
    drafted = metrics.get("spec_decode_num_draft_tokens")
    accepted = metrics.get("spec_decode_num_accepted_tokens")
    if require and (drafted is None or accepted is None):
        raise RuntimeError(
            "Native MTP benchmark expected the engine's spec-decode draft/accepted "
            "counters (spec_decode_num_draft_tokens / "
            "spec_decode_num_accepted_tokens) but they were not reported; got "
            f"{sorted(metrics)}."
        )
    if drafted and accepted is not None:
        metrics["acceptance_rate"] = round(accepted / drafted, 4)
    return metrics


def _measured_delta(
    cumulative: dict[str, Any], warmup: dict[str, Any]
) -> dict[str, Any]:
    """Subtract the post-warmup counter snapshot from the final counters.

    Numeric counters (and per-position lists) become measured-runs-only values;
    ``acceptance_rate`` is recomputed from the deltas.
    """
    delta: dict[str, Any] = {}
    for name, value in cumulative.items():
        if name == "acceptance_rate":
            continue
        base = warmup.get(name)
        if isinstance(value, (int, float)) and isinstance(base, (int, float)):
            delta[name] = value - base
        elif isinstance(value, list) and isinstance(base, list):
            delta[name] = [
                v - b
                if isinstance(v, (int, float)) and isinstance(b, (int, float))
                else v
                for v, b in zip(value, base, strict=True)
            ]
        else:
            delta[name] = value
    drafted = delta.get("spec_decode_num_draft_tokens")
    accepted = delta.get("spec_decode_num_accepted_tokens")
    if drafted and accepted is not None:
        delta["acceptance_rate"] = round(accepted / drafted, 4)
    return delta


def chat_render(llm: Any, prompts: Sequence[str]) -> list[str]:
    """Render each prompt as a single-turn chat with a generation prompt."""
    tokenizer = llm.get_tokenizer()
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    """Run the requested benchmark and return JSON-serializable results."""
    from vllm import LLM, SamplingParams

    prompts = select_prompts(
        batch_size=args.num_prompts or args.batch_size,
        prompts=args.prompt,
        prompt_file=args.prompt_file,
    )
    llm_kwargs = _build_llm_kwargs(args)
    # Native MTP hard requirement (see module docstring); the runner would
    # reject the speculative config otherwise.
    llm_kwargs["enable_prefix_caching"] = False
    # LLM() defaults stats off; keep them on so the spec-decode acceptance
    # counters land in the output JSON alongside the throughput numbers.
    llm_kwargs["disable_log_stats"] = False
    llm = LLM(**llm_kwargs)
    if args.chat:
        prompts = chat_render(llm, prompts)
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        ignore_eos=args.ignore_eos,
    )

    # A repeat runs the whole prompt corpus in slices of ``batch_size`` (one
    # generate() per slice), so ``--num-prompts 8 --batch-size 1`` measures 8
    # distinct serialized requests instead of the first prompt alone.
    slices = [
        prompts[i : i + args.batch_size]
        for i in range(0, len(prompts), args.batch_size)
    ]

    def run_corpus(engine: Any, *, include_text: bool) -> RunResult:
        parts = [
            _run_generate(engine, chunk, sampling_params, include_text=include_text)
            for chunk in slices
        ]
        elapsed_s = sum(part.elapsed_s for part in parts)
        prompt_tokens = sum(part.prompt_tokens for part in parts)
        output_tokens = sum(part.output_tokens for part in parts)
        total_tokens = prompt_tokens + output_tokens
        return RunResult(
            elapsed_s=elapsed_s,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            output_tokens_per_s=output_tokens / elapsed_s if elapsed_s else 0.0,
            total_tokens_per_s=total_tokens / elapsed_s if elapsed_s else 0.0,
            outputs=[out for part in parts for out in part.outputs],
        )

    for _ in range(args.warmup):
        run_corpus(llm, include_text=False)

    # The engine's spec-decode counters are cumulative, so snapshot after warmup
    # and report the measured-runs delta — otherwise warmup drafts pollute the
    # reported acceptance rate.
    warmup_metrics = spec_decode_metrics(llm, require=False)

    results = [
        run_corpus(llm, include_text=args.include_text) for _ in range(args.repeats)
    ]
    cumulative = spec_decode_metrics(llm, require=bool(args.assistant_model))
    metrics = _measured_delta(cumulative, warmup_metrics)

    # Let short-lived benchmark processes release MLX buffers before exit logs.
    del llm
    gc.collect()

    mode = "native_mtp" if args.assistant_model else "baseline"
    return {
        "schema": "native-mtp-benchmark-v1",
        "mode": mode,
        "label": args.label or mode,
        "config": {
            "model": args.model,
            "assistant_model": args.assistant_model,
            "batch_size": args.batch_size,
            "num_prompts": args.num_prompts or args.batch_size,
            "max_tokens": args.max_tokens,
            "max_model_len": args.max_model_len,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "speculative_method": "mtp" if args.assistant_model else None,
            "num_speculative_tokens": (
                args.num_speculative_tokens if args.assistant_model else None
            ),
            "enable_prefix_caching": False,
            "chat": args.chat,
            "ignore_eos": args.ignore_eos,
            "async_scheduling": args.async_scheduling,
            "warmup": args.warmup,
            "repeats": args.repeats,
        },
        "environment": environment_metadata(),
        "prompts": prompts,
        "spec_decode_metrics": metrics,
        "spec_decode_metrics_cumulative": cumulative,
        "summary": _summary(results),
        "runs": [asdict(result) for result in results],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = _gemma4_build_parser()
    parser.prog = Path(__file__).name
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=None,
        help="Total distinct prompts per repeat, run serialized in batch_size "
        "slices (default: batch_size).",
    )
    parser.description = __doc__
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Render prompts through the tokenizer chat template.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_benchmark(args)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
