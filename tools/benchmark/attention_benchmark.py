# SPDX-License-Identifier: Apache-2.0
"""Benchmark Metal attention backends on shared synthetic workloads.

Benchmarked backends:
- `v1` (decode-only paged attention)
- `v2` (Metal unified attention)
- `textbook` (pure-MLX reference)
- `sdpa-compute-only` (dense MLX SDPA only)
- `sdpa` (paged gather + dense MLX SDPA)

Running with no arguments executes the built-in `all` preset group. Built-in
presets run `v1`, `v2`, `textbook`, and `sdpa` by default. Use
`--backend all` when you also want to include `sdpa-compute-only`.

Examples:
  python -m tools.benchmark.attention_benchmark
  python -m tools.benchmark.attention_benchmark --group decode
  python -m tools.benchmark.attention_benchmark --group small
  python -m tools.benchmark.attention_benchmark --cases decode-small,varlen-light
  python -m tools.benchmark.attention_benchmark --group decode --num-layers 32
  python -m tools.benchmark.attention_benchmark --group all --backend all
  python -m tools.benchmark.attention_benchmark --output-json /tmp/attention.json
  python -m tools.benchmark.attention_benchmark --output-csv /tmp/attention.csv
  python -m tools.benchmark.attention_benchmark --mode decode --batch-size 8 --kv-lens 2048
  python -m tools.benchmark.attention_benchmark --mode varlen --q-lens 1,4,16,64 --kv-lens 128,256,512,1024
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np

if __package__ in (None, ""):
    raise SystemExit(
        "Run this benchmark as a module: python -m tools.benchmark.attention_benchmark"
    )

from tools.attention_bench_utils import ref_paged_attn, run_v1_paged_attention
from vllm_metal.metal import metal_unified_attention

ALL_BACKENDS = ["v1", "v2", "textbook", "sdpa-compute-only", "sdpa"]
DTYPE_MAP = {
    "float16": mx.float16,
    "bfloat16": mx.bfloat16,
    "float32": mx.float32,
}
DEFAULTS: dict[str, object] = {
    "backend": "v1,v2,textbook,sdpa",
    "warmup": 10,
    "iters": 100,
    "seed": 0,
    "num_layers": 1,
    "num_q_heads": 8,
    "num_kv_heads": 8,
    "head_dim": 128,
    "block_size": 16,
    "num_blocks": 256,
    "dtype": "float16",
}
CASES: dict[str, dict[str, object]] = {
    "decode-small": {
        "mode": "decode",
        "batch_size": 1,
        "kv_lens": (128,),
    },
    "decode-typical": {
        "mode": "decode",
        "batch_size": 8,
        "kv_lens": (2048,),
    },
    "decode-big-head": {
        "mode": "decode",
        "batch_size": 8,
        "kv_lens": (2048,),
        "num_q_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 256,
    },
    "decode-long": {
        "mode": "decode",
        "batch_size": 32,
        "kv_lens": (8192,),
        "num_blocks": 512,
    },
    "varlen-light": {
        "mode": "varlen",
        "q_lens": (1, 4, 16, 64),
        "kv_lens": (128, 256, 512, 1024),
    },
    "varlen-typical": {
        "mode": "varlen",
        "q_lens": (32, 64, 128, 256),
        "kv_lens": (512, 1024, 2048, 4096),
    },
    "varlen-single-long": {
        "mode": "varlen",
        "q_lens": (256,),
        "kv_lens": (4096,),
    },
    "varlen-ragged-longtail": {
        "mode": "varlen",
        "q_lens": (1, 1, 8, 128),
        "kv_lens": (4096, 8192, 512, 2048),
        "num_blocks": 512,
    },
}
GROUPS: dict[str, tuple[str, ...]] = {
    "all": tuple(CASES),
    "decode": tuple(name for name in CASES if name.startswith("decode-")),
    "varlen": tuple(name for name in CASES if name.startswith("varlen-")),
    "small": ("decode-small", "varlen-light"),
    "typical": ("decode-typical", "varlen-typical"),
    "long": (
        "decode-big-head",
        "decode-long",
        "varlen-single-long",
        "varlen-ragged-longtail",
    ),
}
PRESET_FIELDS = (
    "backend",
    "warmup",
    "iters",
    "seed",
    "num_layers",
    "num_q_heads",
    "num_kv_heads",
    "head_dim",
    "block_size",
    "num_blocks",
    "dtype",
    "mode",
    "batch_size",
    "q_lens",
    "kv_lens",
)


def parse_int_list(value: str | tuple[int, ...] | list[int] | None) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, str):
        values = [chunk.strip() for chunk in value.split(",") if chunk.strip()]
        if not values:
            return None
        return [int(v) for v in values]
    return [int(v) for v in value]


def has_cli_override(flag: str) -> bool:
    cli_args = tuple(sys.argv[1:])
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in cli_args)


def parse_name_list(text: str, kind: str) -> list[str]:
    values = [chunk.strip() for chunk in text.split(",") if chunk.strip()]
    if not values:
        raise ValueError(f"--{kind}s must include at least one {kind}")
    return values


@dataclass(frozen=True)
class Workload:
    mode: str
    query_lens: list[int]
    kv_lens: list[int]
    num_layers: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    block_size: int
    num_blocks: int
    dtype_name: str
    seed: int

    @property
    def dtype(self) -> mx.Dtype:
        return DTYPE_MAP[self.dtype_name]

    @property
    def num_seqs(self) -> int:
        return len(self.query_lens)

    @property
    def total_q_tokens(self) -> int:
        return sum(self.query_lens)

    @property
    def max_q_len(self) -> int:
        return max(self.query_lens)

    @property
    def max_kv_len(self) -> int:
        return max(self.kv_lens)

    @property
    def scale(self) -> float:
        return self.head_dim**-0.5


@dataclass
class WorkloadData:
    workload: Workload
    queries: list[mx.array]
    key_caches: list[mx.array]
    value_caches: list[mx.array]
    block_tables: mx.array
    block_tables_np: np.ndarray
    kv_lens_arr: mx.array
    cu_query_lens: mx.array


@dataclass
class Result:
    backend: str
    mean_ms: float | None
    p50_ms: float | None
    p95_ms: float | None
    tokens_per_s: float | None
    notes: str = ""


@dataclass
class CaseRun:
    case_name: str
    workload: Workload
    results: list[Result]


def apply_preset(args: argparse.Namespace, preset: dict[str, object]) -> None:
    for attr in PRESET_FIELDS:
        flag = f"--{attr.replace('_', '-')}"
        if not has_cli_override(flag) and attr in preset:
            setattr(args, attr, preset[attr])


def manual_workload_requested(args: argparse.Namespace) -> bool:
    return any(
        value is not None
        for value in (args.mode, args.batch_size, args.q_lens, args.kv_lens)
    )


def resolve_case_names(args: argparse.Namespace) -> list[str]:
    if args.group is not None and args.cases is not None:
        raise ValueError("Choose either --group or --cases, not both")

    if args.cases is not None:
        case_names = parse_name_list(args.cases, "case")
        unknown = [name for name in case_names if name not in CASES]
        if unknown:
            raise ValueError(f"Unknown case(s): {', '.join(unknown)}")
        return case_names

    group_name = args.group or "all"
    if group_name not in GROUPS:
        raise ValueError(f"Unknown group: {group_name}")
    return list(GROUPS[group_name])


def build_case_invocations(
    args: argparse.Namespace,
) -> list[tuple[str, argparse.Namespace]]:
    if manual_workload_requested(args):
        if args.group is not None or args.cases is not None:
            raise ValueError(
                "Cannot combine manual workload flags with --group or --cases"
            )
        case_args = argparse.Namespace(**vars(args))
        apply_preset(case_args, DEFAULTS)
        return [("custom", case_args)]

    case_names = resolve_case_names(args)
    invocations: list[tuple[str, argparse.Namespace]] = []
    for case_name in case_names:
        case_args = argparse.Namespace(**vars(args))
        apply_preset(case_args, DEFAULTS)
        apply_preset(case_args, CASES[case_name])
        invocations.append((case_name, case_args))
    return invocations


def build_workload(args: argparse.Namespace) -> Workload:
    q_lens = parse_int_list(args.q_lens)
    kv_lens = parse_int_list(args.kv_lens)

    required_fields = (
        "num_layers",
        "num_q_heads",
        "num_kv_heads",
        "head_dim",
        "block_size",
        "num_blocks",
        "dtype",
        "seed",
    )
    missing = [field for field in required_fields if getattr(args, field) is None]
    if missing:
        raise ValueError(f"Missing required benchmark settings: {', '.join(missing)}")

    if args.mode == "decode":
        if q_lens is None:
            if args.batch_size is None:
                raise ValueError("--batch-size is required for decode mode")
            q_lens = [1] * args.batch_size
        if any(q != 1 for q in q_lens):
            raise ValueError("decode mode requires all q_lens to be 1")

        if kv_lens is None:
            raise ValueError("--kv-lens is required")
        if len(kv_lens) == 1:
            kv_lens = kv_lens * len(q_lens)
        elif len(kv_lens) != len(q_lens):
            raise ValueError("decode mode requires kv_lens length to match batch size")
    else:
        if q_lens is None or kv_lens is None:
            raise ValueError("varlen mode requires both --q-lens and --kv-lens")
        if len(q_lens) != len(kv_lens):
            raise ValueError("--q-lens and --kv-lens must have the same length")

    if args.num_q_heads % args.num_kv_heads != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")
    if args.num_layers < 1:
        raise ValueError("num_layers must be at least 1")
    if args.dtype not in DTYPE_MAP:
        raise ValueError(f"Unsupported dtype: {args.dtype}")
    return Workload(
        mode=args.mode,
        query_lens=q_lens,
        kv_lens=kv_lens,
        num_layers=args.num_layers,
        num_q_heads=args.num_q_heads,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        block_size=args.block_size,
        num_blocks=args.num_blocks,
        dtype_name=args.dtype,
        seed=args.seed,
    )


def make_workload_data(workload: Workload) -> WorkloadData:
    max_blocks_per_seq = math.ceil(workload.max_kv_len / workload.block_size)
    if max_blocks_per_seq > workload.num_blocks:
        raise ValueError(
            f"num_blocks={workload.num_blocks} is too small for max_kv_len="
            f"{workload.max_kv_len} and block_size={workload.block_size}; need at least "
            f"{max_blocks_per_seq}"
        )

    mx.random.seed(workload.seed)
    block_tables = mx.random.randint(
        0,
        workload.num_blocks,
        shape=(workload.num_seqs, max_blocks_per_seq),
    ).astype(mx.int32)
    kv_lens_arr = mx.array(workload.kv_lens, dtype=mx.int32)
    cu_query_lens = mx.cumsum(mx.array([0] + workload.query_lens, dtype=mx.int32))
    queries: list[mx.array] = []
    key_caches: list[mx.array] = []
    value_caches: list[mx.array] = []
    for layer_idx in range(workload.num_layers):
        mx.random.seed(workload.seed + layer_idx)
        queries.append(
            mx.random.normal(
                shape=(workload.total_q_tokens, workload.num_q_heads, workload.head_dim)
            ).astype(workload.dtype)
        )
        key_caches.append(
            mx.random.normal(
                shape=(
                    workload.num_blocks,
                    workload.block_size,
                    workload.num_kv_heads,
                    workload.head_dim,
                )
            ).astype(workload.dtype)
        )
        value_caches.append(
            mx.random.normal(
                shape=(
                    workload.num_blocks,
                    workload.block_size,
                    workload.num_kv_heads,
                    workload.head_dim,
                )
            ).astype(workload.dtype)
        )
    mx.eval(
        *queries,
        *key_caches,
        *value_caches,
        block_tables,
        kv_lens_arr,
        cu_query_lens,
    )

    return WorkloadData(
        workload=workload,
        queries=queries,
        key_caches=key_caches,
        value_caches=value_caches,
        block_tables=block_tables,
        block_tables_np=np.array(block_tables),
        kv_lens_arr=kv_lens_arr,
        cu_query_lens=cu_query_lens,
    )


def make_sdpa_mask(
    query_len: int,
    kv_len: int,
) -> mx.array:
    empty_mask = mx.ones((query_len, kv_len))
    mask = mx.triu(empty_mask, k=kv_len - query_len + 1).astype(mx.bool_)
    return mask[None, None, :, :]


def gather_dense_sdpa_inputs(
    data: WorkloadData,
    layer_idx: int,
) -> list[tuple[mx.array, mx.array, mx.array, mx.array]]:
    workload = data.workload
    query = data.queries[layer_idx]
    key_cache = data.key_caches[layer_idx]
    value_cache = data.value_caches[layer_idx]
    prepared: list[tuple[mx.array, mx.array, mx.array, mx.array]] = []
    start = 0
    for i, query_len in enumerate(workload.query_lens):
        kv_len = workload.kv_lens[i]
        q = query[start : start + query_len].transpose(1, 0, 2)[None, ...]

        num_kv_blocks = math.ceil(kv_len / workload.block_size)
        block_indices = data.block_tables[i, :num_kv_blocks]
        k = key_cache[block_indices].reshape(
            -1, workload.num_kv_heads, workload.head_dim
        )[:kv_len]
        v = value_cache[block_indices].reshape(
            -1, workload.num_kv_heads, workload.head_dim
        )[:kv_len]
        k = k.transpose(1, 0, 2)[None, ...]
        v = v.transpose(1, 0, 2)[None, ...]
        mask = make_sdpa_mask(query_len, kv_len)
        prepared.append((q, k, v, mask))
        start += query_len

    mx.eval(*(arr for item in prepared for arr in item))
    return prepared


def run_sdpa_from_prepared(
    prepared: list[tuple[mx.array, mx.array, mx.array, mx.array]],
    scale: float,
) -> mx.array:
    outputs: list[mx.array] = []
    for q, k, v, mask in prepared:
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
        outputs.append(out[0].transpose(1, 0, 2))
    return mx.concatenate(outputs, axis=0)


def time_backend(
    fn: Callable[[], mx.array],
    warmup: int,
    iters: int,
    total_q_tokens: int,
    num_layers: int,
) -> tuple[float, float, float, float]:
    for _ in range(warmup):
        out = fn()
        if out is not None:
            mx.eval(out)

    timings_ms: list[float] = []
    for _ in range(iters):
        mx.synchronize()
        t0 = time.perf_counter()
        out = fn()
        if out is not None:
            mx.eval(out)
        mx.synchronize()
        timings_ms.append((time.perf_counter() - t0) * 1000.0)

    mean_ms = statistics.fmean(timings_ms) / num_layers
    p50_ms = float(np.percentile(timings_ms, 50)) / num_layers
    p95_ms = float(np.percentile(timings_ms, 95)) / num_layers
    tokens_per_s = total_q_tokens / (mean_ms / 1000.0)
    return mean_ms, p50_ms, p95_ms, tokens_per_s


def benchmark_backend(
    backend: str,
    data: WorkloadData,
    warmup: int,
    iters: int,
) -> Result:
    workload = data.workload
    notes = ""

    if backend == "v1":
        if workload.mode != "decode":
            return Result(
                backend=backend,
                mean_ms=None,
                p50_ms=None,
                p95_ms=None,
                tokens_per_s=None,
                notes="unsupported in varlen mode",
            )

        def fn() -> mx.array:
            out = None
            for layer_idx in range(workload.num_layers):
                out = run_v1_paged_attention(
                    query=data.queries[layer_idx],
                    key_cache=data.key_caches[layer_idx],
                    value_cache=data.value_caches[layer_idx],
                    num_kv_heads=workload.num_kv_heads,
                    scale=workload.scale,
                    block_tables=data.block_tables,
                    seq_lens=data.kv_lens_arr,
                    block_size=workload.block_size,
                    max_seq_len=workload.max_kv_len,
                )
            assert out is not None
            return out

        notes = "decode-only"
    elif backend == "v2":

        def fn() -> mx.array:
            out = None
            for layer_idx in range(workload.num_layers):
                out = _run_v2(data, layer_idx)
            assert out is not None
            return out
    elif backend == "textbook":

        def fn() -> mx.array:
            out = None
            for layer_idx in range(workload.num_layers):
                out = ref_paged_attn(
                    query=data.queries[layer_idx],
                    key_cache=data.key_caches[layer_idx],
                    value_cache=data.value_caches[layer_idx],
                    query_lens=workload.query_lens,
                    kv_lens=workload.kv_lens,
                    block_tables=data.block_tables_np,
                    scale=workload.scale,
                    sliding_window=None,
                    soft_cap=None,
                )
            assert out is not None
            return out
    elif backend == "sdpa-compute-only":
        prepared_per_layer = [
            gather_dense_sdpa_inputs(data, layer_idx)
            for layer_idx in range(workload.num_layers)
        ]

        def fn() -> mx.array:
            out = None
            for prepared in prepared_per_layer:
                out = run_sdpa_from_prepared(prepared, workload.scale)
            assert out is not None
            return out

        notes = "dense compute only"
    elif backend == "sdpa":

        def fn() -> mx.array:
            out = None
            for layer_idx in range(workload.num_layers):
                out = run_sdpa_from_prepared(
                    gather_dense_sdpa_inputs(data, layer_idx), workload.scale
                )
            assert out is not None
            return out

        notes = "includes gather"
    else:
        raise ValueError(f"Unknown backend: {backend}")

    try:
        mean_ms, p50_ms, p95_ms, tokens_per_s = time_backend(
            fn, warmup, iters, workload.total_q_tokens, workload.num_layers
        )
        return Result(
            backend=backend,
            mean_ms=mean_ms,
            p50_ms=p50_ms,
            p95_ms=p95_ms,
            tokens_per_s=tokens_per_s,
            notes=notes,
        )
    except Exception as exc:
        error_note = f"error: {type(exc).__name__}: {exc}"
        notes = error_note if not notes else f"{notes}; {error_note}"
        return Result(
            backend=backend,
            mean_ms=None,
            p50_ms=None,
            p95_ms=None,
            tokens_per_s=None,
            notes=notes,
        )


def _run_v2(data: WorkloadData, layer_idx: int) -> mx.array:
    workload = data.workload
    out = mx.zeros_like(data.queries[layer_idx])
    metal_unified_attention(
        q=data.queries[layer_idx],
        k=data.key_caches[layer_idx],
        v=data.value_caches[layer_idx],
        out=out,
        cu_seqlens_q=data.cu_query_lens,
        seqused_k=data.kv_lens_arr,
        max_seqlen_q=workload.max_q_len,
        max_seqlen_k=workload.max_kv_len,
        softmax_scale=workload.scale,
        causal=True,
        window_size=(-1, -1),
        block_table=data.block_tables,
        softcap=0,
    )
    return out


def format_query_spec(workload: Workload) -> str:
    if workload.mode == "decode":
        return f"batch={workload.num_seqs}, q_len=1, kv_len={workload.kv_lens}"
    return "seq_lens=" + str(
        list(zip(workload.query_lens, workload.kv_lens, strict=False))
    )


def short_query_spec(workload: Workload) -> str:
    if workload.mode == "decode":
        kv = (
            workload.kv_lens[0] if len(set(workload.kv_lens)) == 1 else workload.kv_lens
        )
        return f"B={workload.num_seqs}, q=1, kv={kv}"
    pairs = list(zip(workload.query_lens, workload.kv_lens, strict=False))
    if len(pairs) <= 4:
        return " ".join(f"{q}/{kv}" for q, kv in pairs)
    return (
        f"{len(pairs)} seqs; max_q={workload.max_q_len}; max_kv={workload.max_kv_len}"
    )


def valid_results(results: list[Result]) -> list[Result]:
    return [result for result in results if result.mean_ms is not None]


def mean_ms_key(result: Result) -> float:
    assert result.mean_ms is not None
    return result.mean_ms


def ordered_backends(case_runs: list[CaseRun]) -> list[str]:
    present = {result.backend for case_run in case_runs for result in case_run.results}
    return [backend for backend in ALL_BACKENDS if backend in present]


def case_kind(workload: Workload) -> str:
    return "decode" if workload.mode == "decode" else "varlen"


def display_case_name(case_run: CaseRun) -> str:
    prefix = f"{case_kind(case_run.workload)}-"
    if case_run.case_name.startswith(prefix):
        return case_run.case_name[len(prefix) :]
    return case_run.case_name


def backend_label(backend: str) -> str:
    return {
        "v1": "v1",
        "v2": "v2",
        "textbook": "textbook",
        "sdpa-compute-only": "sdpa-compute-only",
        "sdpa": "sdpa",
    }.get(backend, backend.replace("-", "_"))


def format_time_ms(result: Result | None) -> str:
    if result is None:
        return "-"
    if result.mean_ms is None:
        if result.notes.startswith("error:"):
            return "ERROR"
        return "N/A"
    return f"{result.mean_ms:.3f}"


def format_vs_best(result: Result | None, best: Result | None) -> str:
    if result is None or result.mean_ms is None or best is None or best.mean_ms is None:
        return "-"
    pct = result.mean_ms / best.mean_ms * 100.0
    if math.isclose(result.mean_ms, best.mean_ms, rel_tol=0.0, abs_tol=1e-9):
        return f"{pct:.1f}% best"
    return f"{pct:.1f}%"


def comparison_headers(backends: list[str], compare_to_fastest: bool) -> list[str]:
    headers = ["case", "type", "batch", "shape"]
    for backend in backends:
        label = backend_label(backend)
        headers.append(label)
        if compare_to_fastest:
            headers.append(f"{label}_vs_best")
    return headers


def comparison_rows(case_runs: list[CaseRun], backends: list[str]) -> list[list[str]]:
    compare_to_fastest = len(backends) > 1
    rows: list[list[str]] = []
    for case_run in case_runs:
        results_by_backend = {result.backend: result for result in case_run.results}
        best = min(
            valid_results(case_run.results),
            key=mean_ms_key,
            default=None,
        )
        row = [
            display_case_name(case_run),
            case_kind(case_run.workload),
            str(case_run.workload.num_seqs),
            short_query_spec(case_run.workload),
        ]
        for backend in backends:
            result = results_by_backend.get(backend)
            row.append(format_time_ms(result))
            if compare_to_fastest:
                row.append(format_vs_best(result, best))
        rows.append(row)
    return rows


def print_text_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [len(header) for header in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    print(" | ".join(header.ljust(widths[i]) for i, header in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))


def summary_dict(
    case_runs: list[CaseRun], args: argparse.Namespace
) -> dict[str, object]:
    block_sizes = sorted({run.workload.block_size for run in case_runs})
    dtypes = sorted({run.workload.dtype_name for run in case_runs})
    num_layers = sorted({run.workload.num_layers for run in case_runs})
    seeds = sorted({run.workload.seed for run in case_runs})
    return {
        "cases": [run.case_name for run in case_runs],
        "num_layers": num_layers[0] if len(num_layers) == 1 else num_layers,
        "block_size": block_sizes[0] if len(block_sizes) == 1 else block_sizes,
        "dtype": dtypes[0] if len(dtypes) == 1 else dtypes,
        "warmup": args.warmup,
        "iters": args.iters,
        "seed": seeds[0] if len(seeds) == 1 else seeds,
    }


def comparison_rows_dict(
    case_runs: list[CaseRun], backends: list[str]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for case_run in case_runs:
        row: dict[str, object] = {
            "case": display_case_name(case_run),
            "case_name": case_run.case_name,
            "type": case_kind(case_run.workload),
            "batch": case_run.workload.num_seqs,
            "shape": short_query_spec(case_run.workload),
        }
        results_by_backend = {result.backend: result for result in case_run.results}
        best = min(
            valid_results(case_run.results),
            key=mean_ms_key,
            default=None,
        )
        for backend in backends:
            result = results_by_backend.get(backend)
            label = backend_label(backend)
            row[label] = (
                None
                if result is None or result.mean_ms is None
                else round(result.mean_ms, 3)
            )
            row[f"{label}_vs_best"] = (
                None
                if result is None
                or result.mean_ms is None
                or best is None
                or best.mean_ms is None
                else round(result.mean_ms / best.mean_ms * 100.0, 1)
            )
        rows.append(row)
    return rows


def json_payload(
    case_runs: list[CaseRun], args: argparse.Namespace
) -> dict[str, object]:
    backends = ordered_backends(case_runs)
    return {
        "summary": summary_dict(case_runs, args),
        "rows": comparison_rows_dict(case_runs, backends),
    }


def write_json(path: Path, case_runs: list[CaseRun], args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_payload(case_runs, args), indent=2) + "\n")


def write_csv(path: Path, case_runs: list[CaseRun]) -> None:
    backends = ordered_backends(case_runs)
    rows = comparison_rows_dict(case_runs, backends)
    fieldnames = (
        list(rows[0].keys())
        if rows
        else ["case", "case_name", "type", "batch", "shape"]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_exports(case_runs: list[CaseRun], args: argparse.Namespace) -> None:
    if args.output_json:
        write_json(Path(args.output_json).expanduser(), case_runs, args)
    if args.output_csv:
        write_csv(Path(args.output_csv).expanduser(), case_runs)


def print_summary(case_runs: list[CaseRun], args: argparse.Namespace) -> None:
    summary = summary_dict(case_runs, args)
    summary_parts = [
        f"num_layers: {summary['num_layers']}"
        if isinstance(summary["num_layers"], int)
        else "num_layers: mixed",
        f"block_size: {summary['block_size']}"
        if isinstance(summary["block_size"], int)
        else "block_size: mixed",
        f"dtype: {summary['dtype']}"
        if isinstance(summary["dtype"], str)
        else "dtype: mixed",
        f"warmup: {args.warmup}",
        f"iters: {args.iters}",
        f"seed: {summary['seed']}"
        if isinstance(summary["seed"], int)
        else "seed: mixed",
    ]
    print("  ".join(summary_parts))


def print_case_header(case_run: CaseRun, args: argparse.Namespace) -> None:
    print("\nMetal Attention Benchmark")
    print(f"case: {case_run.case_name}")
    print(f"mode: {case_run.workload.mode}")
    print(f"workload: {format_query_spec(case_run.workload)}")
    print(
        "heads(q/kv): "
        f"{case_run.workload.num_q_heads}/{case_run.workload.num_kv_heads}  "
        f"head_dim: {case_run.workload.head_dim}  "
        f"block_size: {case_run.workload.block_size}  "
        f"num_blocks: {case_run.workload.num_blocks}  "
        f"num_layers: {case_run.workload.num_layers}"
    )
    print(
        f"dtype: {case_run.workload.dtype_name}  warmup: {args.warmup}  "
        f"iters: {args.iters}  seed: {case_run.workload.seed}"
    )


def print_results(case_run: CaseRun, args: argparse.Namespace) -> None:
    print_case_header(case_run, args)
    print()
    backends = ordered_backends([case_run])
    headers = comparison_headers(backends, compare_to_fastest=len(backends) > 1)
    rows = comparison_rows([case_run], backends)
    print_text_table(headers, rows)
    write_exports([case_run], args)


def print_combined_results(case_runs: list[CaseRun], args: argparse.Namespace) -> None:
    print("\nMetal Attention Benchmark")
    print(f"cases: {', '.join(run.case_name for run in case_runs)}")
    print_summary(case_runs, args)
    print()

    backends = ordered_backends(case_runs)
    headers = comparison_headers(backends, compare_to_fastest=len(backends) > 1)
    rows = comparison_rows(case_runs, backends)
    print_text_table(headers, rows)
    write_exports(case_runs, args)


def resolve_backends(text: str, mode: str) -> list[str]:
    if text == "all":
        backends = list(ALL_BACKENDS)
    else:
        backends = [chunk.strip() for chunk in text.split(",") if chunk.strip()]

    invalid = [backend for backend in backends if backend not in ALL_BACKENDS]
    if invalid:
        raise ValueError(f"Unknown backend(s): {', '.join(invalid)}")
    return backends


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--group",
        choices=sorted(GROUPS),
        default=None,
        help="Built-in preset group to run; defaults to all when no manual workload is given",
    )
    parser.add_argument(
        "--cases",
        default=None,
        help="Comma-separated explicit preset case names to run",
    )
    parser.add_argument(
        "--mode",
        choices=["decode", "varlen"],
        default=None,
        help="Manual workload mode; when set, runs one custom case instead of preset cases",
    )
    parser.add_argument(
        "--backend",
        default=None,
        help="all|v1|v2|textbook|sdpa-compute-only|sdpa or a comma-separated subset",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Decode mode only: number of sequences; implies q_len=1 for each sequence",
    )
    parser.add_argument(
        "--q-lens",
        default=None,
        help="Comma-separated query lengths; required for manual varlen mode",
    )
    parser.add_argument(
        "--kv-lens",
        default=None,
        help="Comma-separated KV lengths; one value may be repeated across all decode sequences",
    )
    parser.add_argument(
        "--num-q-heads",
        type=int,
        default=None,
        help="Number of query heads",
    )
    parser.add_argument(
        "--num-kv-heads",
        type=int,
        default=None,
        help="Number of key/value heads; must divide num-q-heads",
    )
    parser.add_argument(
        "--head-dim",
        type=int,
        default=None,
        help="Attention head dimension",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=None,
        help="Paged KV block size",
    )
    parser.add_argument(
        "--num-blocks",
        type=int,
        default=None,
        help="Number of blocks in the synthetic paged KV cache",
    )
    parser.add_argument(
        "--dtype",
        choices=sorted(DTYPE_MAP),
        default=None,
        help="Element dtype for synthetic inputs",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=None,
        help="Number of warmup iterations before timing",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=None,
        help="Number of measured iterations",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible synthetic inputs",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="Number of attention layers to benchmark; timings are reported per layer",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Write structured benchmark results to a JSON file",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Write row-oriented benchmark results to a CSV file",
    )
    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    try:
        case_invocations = build_case_invocations(args)
    except ValueError as exc:
        parser.error(str(exc))

    case_runs: list[CaseRun] = []
    for case_name, case_args in case_invocations:
        if case_args.mode is None:
            parser.error("--mode is required for manual workloads")

        try:
            workload = build_workload(case_args)
            backends = resolve_backends(case_args.backend, workload.mode)
        except ValueError as exc:
            parser.error(str(exc))
        if not backends:
            raise ValueError("No backends selected")

        data = make_workload_data(workload)
        results: list[Result] = []
        for backend in backends:
            result = benchmark_backend(backend, data, case_args.warmup, case_args.iters)
            results.append(result)
        case_runs.append(CaseRun(case_name or "custom", workload, results))

    if len(case_runs) == 1:
        print_results(case_runs[0], case_args)
    else:
        display_args = argparse.Namespace(**vars(args))
        apply_preset(display_args, DEFAULTS)
        print_combined_results(case_runs, display_args)


if __name__ == "__main__":
    main()
