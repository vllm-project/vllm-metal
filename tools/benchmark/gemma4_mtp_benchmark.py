# SPDX-License-Identifier: Apache-2.0
"""Offline Gemma4 MTP throughput benchmark.

Run one target model with or without a Gemma4 MTP assistant and emit structured
JSON that can be attached to PRs. The script intentionally runs one mode per
process so target/assistant model state does not leak between baseline and MTP
measurements.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
import platform
import statistics
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DEFAULT_PROMPTS = (
    "The capital of France is",
    "The weather today is not",
    "One plus one equals",
    "The largest planet in our solar system is",
    "Water boils at a temperature of",
)


@dataclass(frozen=True, slots=True)
class RunResult:
    """One measured generate() call."""

    elapsed_s: float
    prompt_tokens: int
    output_tokens: int
    output_tokens_per_s: float
    total_tokens_per_s: float
    outputs: list[dict[str, Any]]


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value}")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError(
            f"expected a non-negative integer, got {value}"
        )
    return parsed


def _load_prompt_file(path: Path) -> list[str]:
    prompts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [prompt for prompt in prompts if prompt]


def select_prompts(
    *,
    batch_size: int,
    prompts: Sequence[str] | None = None,
    prompt_file: Path | None = None,
) -> list[str]:
    """Return exactly batch_size prompts, cycling shorter prompt sets."""
    if prompt_file is not None:
        source = _load_prompt_file(prompt_file)
    elif prompts is not None:
        source = list(prompts)
    else:
        source = list(DEFAULT_PROMPTS)

    if not source:
        raise ValueError("at least one prompt is required")

    return [source[i % len(source)] for i in range(batch_size)]


def _package_version(package_name: str) -> str | None:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment_metadata() -> dict[str, Any]:
    """Return enough environment detail to make benchmark JSON auditable."""
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "vllm": _package_version("vllm"),
        "vllm-metal": _package_version("vllm-metal"),
        "mlx": _package_version("mlx"),
        "mlx-lm": _package_version("mlx-lm"),
        "transformers": _package_version("transformers"),
        "env": {
            name: os.environ.get(name)
            for name in (
                "VLLM_METAL_MEMORY_FRACTION",
                "VLLM_METAL_USE_PAGED_ATTENTION",
                "VLLM_ENABLE_V1_MULTIPROCESSING",
            )
            if os.environ.get(name) is not None
        },
    }


def _output_token_ids(request_output: Any) -> list[int]:
    if not request_output.outputs:
        return []
    return list(request_output.outputs[0].token_ids)


def summarize_outputs(
    outputs: Sequence[Any],
    *,
    include_text: bool,
) -> tuple[int, int, list[dict[str, Any]]]:
    """Return prompt/output token totals plus optional per-output samples."""
    prompt_tokens = 0
    output_tokens = 0
    samples: list[dict[str, Any]] = []

    for output in outputs:
        prompt_token_ids = list(getattr(output, "prompt_token_ids", ()) or ())
        token_ids = _output_token_ids(output)
        prompt_tokens += len(prompt_token_ids)
        output_tokens += len(token_ids)

        sample: dict[str, Any] = {
            "prompt_tokens": len(prompt_token_ids),
            "output_tokens": len(token_ids),
            "token_ids": token_ids,
        }
        if include_text and output.outputs:
            sample["text"] = output.outputs[0].text
        samples.append(sample)

    return prompt_tokens, output_tokens, samples


def _run_generate(
    llm: Any,
    prompts: Sequence[str],
    sampling_params: Any,
    *,
    include_text: bool,
) -> RunResult:
    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    elapsed_s = time.perf_counter() - start
    prompt_tokens, output_tokens, samples = summarize_outputs(
        outputs,
        include_text=include_text,
    )
    total_tokens = prompt_tokens + output_tokens
    return RunResult(
        elapsed_s=elapsed_s,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        output_tokens_per_s=output_tokens / elapsed_s if elapsed_s else 0.0,
        total_tokens_per_s=total_tokens / elapsed_s if elapsed_s else 0.0,
        outputs=samples,
    )


def _summary(results: Sequence[RunResult]) -> dict[str, float]:
    output_rates = [result.output_tokens_per_s for result in results]
    total_rates = [result.total_tokens_per_s for result in results]
    elapsed = [result.elapsed_s for result in results]
    return {
        "mean_output_tokens_per_s": statistics.fmean(output_rates),
        "best_output_tokens_per_s": max(output_rates),
        "mean_total_tokens_per_s": statistics.fmean(total_rates),
        "mean_elapsed_s": statistics.fmean(elapsed),
    }


def _build_llm_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": args.model,
        "dtype": args.dtype,
        "seed": args.seed,
        "trust_remote_code": args.trust_remote_code,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.batch_size,
        "async_scheduling": args.async_scheduling,
    }
    if args.gpu_memory_utilization is not None:
        kwargs["gpu_memory_utilization"] = args.gpu_memory_utilization
    if args.model_revision is not None:
        kwargs["revision"] = args.model_revision
    if args.max_num_batched_tokens is not None:
        kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens
    if args.assistant_model is not None:
        speculative_config: dict[str, Any] = {
            "method": "mtp",
            "model": args.assistant_model,
            "num_speculative_tokens": args.num_speculative_tokens,
        }
        if args.assistant_revision is not None:
            speculative_config["revision"] = args.assistant_revision
        kwargs["speculative_config"] = speculative_config
    return kwargs


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    """Run the requested benchmark and return JSON-serializable results."""
    from vllm import LLM, SamplingParams

    prompts = select_prompts(
        batch_size=args.batch_size,
        prompts=args.prompt,
        prompt_file=args.prompt_file,
    )
    llm = LLM(**_build_llm_kwargs(args))
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        ignore_eos=args.ignore_eos,
    )

    for _ in range(args.warmup):
        _run_generate(
            llm,
            prompts,
            sampling_params,
            include_text=False,
        )

    results = [
        _run_generate(
            llm,
            prompts,
            sampling_params,
            include_text=args.include_text,
        )
        for _ in range(args.repeats)
    ]

    # Let short-lived benchmark processes release MLX buffers before exit logs.
    del llm
    gc.collect()

    mode = "gemma4_mtp" if args.assistant_model else "baseline"
    return {
        "schema": "gemma4-mtp-benchmark-v1",
        "mode": mode,
        "label": args.label or mode,
        "config": {
            "model": args.model,
            "assistant_model": args.assistant_model,
            "batch_size": args.batch_size,
            "max_tokens": args.max_tokens,
            "max_model_len": args.max_model_len,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "speculative_method": "mtp" if args.assistant_model else None,
            "num_speculative_tokens": (
                args.num_speculative_tokens if args.assistant_model else None
            ),
            "ignore_eos": args.ignore_eos,
            "async_scheduling": args.async_scheduling,
            "warmup": args.warmup,
            "repeats": args.repeats,
        },
        "environment": environment_metadata(),
        "prompts": prompts,
        "summary": _summary(results),
        "runs": [asdict(result) for result in results],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Target Gemma4 model path")
    parser.add_argument("--assistant-model", help="Gemma4 MTP assistant model path")
    parser.add_argument("--model-revision", help="Target model revision")
    parser.add_argument("--assistant-revision", help="Assistant model revision")
    parser.add_argument("--batch-size", type=_positive_int, default=4)
    parser.add_argument("--max-tokens", type=_positive_int, default=128)
    parser.add_argument("--max-model-len", type=_positive_int, default=2048)
    parser.add_argument("--max-num-batched-tokens", type=_positive_int)
    parser.add_argument("--num-speculative-tokens", type=_positive_int, default=1)
    parser.add_argument("--repeats", type=_positive_int, default=3)
    parser.add_argument("--warmup", type=_nonnegative_int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument(
        "--async-scheduling",
        action="store_true",
        help="Enable vLLM async scheduling. Gemma4 MTP currently requires this off.",
    )
    parser.add_argument("--prompt", action="append", help="Prompt to include")
    parser.add_argument("--prompt-file", type=Path, help="One prompt per line")
    parser.add_argument("--include-text", action="store_true")
    parser.add_argument("--label", help="Free-form label stored in output JSON")
    parser.add_argument("--output-json", type=Path, help="Write results as JSON")
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
