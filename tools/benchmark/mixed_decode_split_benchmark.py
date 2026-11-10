# SPDX-License-Identifier: Apache-2.0
"""Kernel benchmark for splitting decode rows out of mixed batches.

A mixed prefill + decode batch can run its leading ordinary decode rows on the
decode kernel (per-token or GQA) and the rest on the prefill kernel (tiled, or
NAX on M5), or run every row on the prefill kernel, where each one-row decode
sequence occupies a whole prefill tile with no KV-length parallelism. This
script times both at controlled decode lengths:

  unsplit         one primitive with ``num_decode_requests=0``: every row runs
                  on the prefill kernel.
  split_native    one primitive with the production decode metadata, as
                  ``sdpa_forward`` passes it; native code splits the batch.
  split_emulated  the same split as two primitives over sliced inputs: the
                  decode rows as a pure decode batch, then the prefill rows.
                  Native code runs exactly these two kernels in one
                  primitive. Use ``--compare split_emulated`` to measure a
                  build whose native split has a context threshold, such as
                  the former 4096-token rule.

``split_emulated_pertoken`` (opt-in) repeats the emulated split with GQA
disabled for the decode rows. Every arm records its executed dispatch family.

Each timed evaluation queues ``--layers`` independent attention calls, as one
forward pass does, and reports milliseconds per layer. Arms are interleaved
in random order each rep. Outputs of all arms are checked against each other
before timing.

Run from the repository root (rebuild native artifacts first when testing a
kernel change, e.g. ``python -m vllm_metal.metal.build``):

    python tools/benchmark/mixed_decode_split_benchmark.py \\
        --output mixed_split.jsonl

``--disable-nax`` measures the tiled prefill route on M5. Confirm routing
changes with the real serving benchmark (``vllm bench serve``); kernel
timings alone do not establish a serving speedup.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Match the plugin's decode-step command-buffer default before MLX loads.
os.environ.setdefault("MLX_MAX_OPS_PER_BUFFER", "2000")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mlx.core as mx  # noqa: E402

from tools.attention_bench_utils import (  # noqa: E402
    attention_tolerances,
    package_versions,
)
from vllm_metal.attention.nax_split import (  # noqa: E402
    nax_split_kwargs,
    nax_tile_work,
)
from vllm_metal.metal import get_ops, paged_attention_capabilities  # noqa: E402

ARMS = ("unsplit", "split_native", "split_emulated", "split_emulated_pertoken")
DEFAULT_ARMS = ("unsplit", "split_native")
SPLIT_ARMS = ("split_native", "split_emulated")
DTYPES = {"bfloat16": mx.bfloat16, "float16": mx.float16}


@dataclass(frozen=True)
class Shape:
    q_heads: int
    kv_heads: int
    head_size: int
    block_size: int
    dtype: str
    num_decode: int
    decode_len: int  # KV length per decode row, including its new token
    prefill_tokens: int
    prefill_context: int  # cached prefix before this prefill chunk

    @property
    def prefill_len(self) -> int:
        return self.prefill_context + self.prefill_tokens

    @property
    def kv_lens(self) -> list[int]:
        return [self.decode_len] * self.num_decode + [self.prefill_len]

    @property
    def query_lens(self) -> list[int]:
        return [1] * self.num_decode + [self.prefill_tokens]


@dataclass
class Inputs:
    query: mx.array
    key_cache: mx.array
    value_cache: mx.array
    block_tables: mx.array
    seq_lens: mx.array
    cu_seqlens: mx.array


@dataclass
class Result:
    shape: Shape
    ms_per_layer: dict[str, float] = field(default_factory=dict)
    ms_min: dict[str, float] = field(default_factory=dict)
    routes: dict[str, str] = field(default_factory=dict)
    gqa_partition: dict[str, int] = field(default_factory=dict)


def _cumsum(lengths: Sequence[int]) -> list[int]:
    out = [0]
    for length in lengths:
        out.append(out[-1] + length)
    return out


def _cache_bytes(shape: Shape) -> int:
    blocks = sum(-(-n // shape.block_size) for n in shape.kv_lens) + 1
    item = 2  # fp16 / bf16
    return 2 * blocks * shape.block_size * shape.kv_heads * shape.head_size * item


def _build_inputs(shape: Shape, seed: int) -> Inputs:
    """Distinct physical pages per sequence, as separate requests would hold."""
    mx.random.seed(seed)
    dtype = DTYPES[shape.dtype]
    tables: list[list[int]] = []
    next_block = 1  # page 0 is padding
    for kv_len in shape.kv_lens:
        n = -(-kv_len // shape.block_size)
        tables.append(list(range(next_block, next_block + n)))
        next_block += n
    width = max(len(t) for t in tables)
    padded = [t + [0] * (width - len(t)) for t in tables]
    cache_shape = (next_block, shape.block_size, shape.kv_heads, shape.head_size)
    inputs = Inputs(
        query=mx.random.normal(
            (sum(shape.query_lens), shape.q_heads, shape.head_size)
        ).astype(dtype),
        key_cache=mx.random.normal(cache_shape).astype(dtype),
        value_cache=mx.random.normal(cache_shape).astype(dtype),
        block_tables=mx.array(padded, dtype=mx.int32),
        seq_lens=mx.array(shape.kv_lens, dtype=mx.int32),
        cu_seqlens=mx.array(_cumsum(shape.query_lens), dtype=mx.int32),
    )
    mx.eval(
        inputs.query,
        inputs.key_cache,
        inputs.value_cache,
        inputs.block_tables,
        inputs.seq_lens,
        inputs.cu_seqlens,
    )
    return inputs


class Arms:
    """Lazy primitive builders for one shape, mirroring ``sdpa_forward``."""

    def __init__(self, shape: Shape, inputs: Inputs, capabilities: dict[str, bool]):
        self.shape = shape
        self.x = inputs
        self.ops = get_ops()
        self.caps = capabilities
        self.scale = shape.head_size**-0.5
        d = shape.num_decode
        cu = _cumsum(shape.query_lens)
        self.decode_lens = shape.kv_lens[:d]
        # Whole-batch and prefill-only NAX hints, as sdpa_forward computes them.
        self.work_all = nax_tile_work(cu, shape.kv_lens)
        self.work_prefill = nax_tile_work(cu[d:], shape.kv_lens[d:])
        prefill_cu = [c - cu[d] for c in cu[d:]]
        self.work_prefill_only = nax_tile_work(prefill_cu, shape.kv_lens[d:])
        x = inputs
        self.decode_x = Inputs(
            query=x.query[:d],
            key_cache=x.key_cache,
            value_cache=x.value_cache,
            block_tables=x.block_tables[:d],
            seq_lens=x.seq_lens[:d],
            cu_seqlens=mx.arange(d + 1, dtype=mx.int32),
        )
        self.prefill_x = Inputs(
            query=x.query[d:],
            key_cache=x.key_cache,
            value_cache=x.value_cache,
            block_tables=x.block_tables[d:],
            seq_lens=x.seq_lens[d:],
            cu_seqlens=mx.array(prefill_cu, dtype=mx.int32),
        )
        mx.eval(*vars(self.decode_x).values(), *vars(self.prefill_x).values())

    def _call(self, x: Inputs, max_seq_len: int, **kwargs) -> mx.array:
        return self.ops.paged_attention_primitive(
            x.query,
            x.key_cache,
            x.value_cache,
            self.shape.kv_heads,
            self.scale,
            0.0,
            x.block_tables,
            x.seq_lens,
            x.cu_seqlens,
            self.shape.block_size,
            max_seq_len,
            -1,
            **kwargs,
        )

    def _nax(self, work, prefill_work=None) -> dict[str, int]:
        return nax_split_kwargs(
            self.caps, work, self.shape.head_size, prefill_work=prefill_work
        )

    def unsplit(self) -> list[mx.array]:
        kwargs = {"num_decode_requests": 0, "num_decode_tokens": 0}
        kwargs.update(self._nax(self.work_all))
        return [self._call(self.x, max(self.shape.kv_lens), **kwargs)]

    def split_native(self) -> list[mx.array]:
        d = self.shape.num_decode
        kwargs = {
            "num_decode_requests": d,
            "num_decode_tokens": d,
            "max_decode_context_len": max(self.decode_lens),
            "gqa_length_plan": self.ops.gqa_decode_length_plan(self.decode_lens),
        }
        kwargs.update(self._nax(self.work_all, self.work_prefill))
        return [self._call(self.x, max(self.shape.kv_lens), **kwargs)]

    def _decode_only(self, gqa_disabled: bool) -> mx.array:
        d = self.shape.num_decode
        kwargs: dict = {
            "num_decode_requests": d,
            "num_decode_tokens": d,
            "max_decode_context_len": max(self.decode_lens),
        }
        if gqa_disabled:
            kwargs["gqa_disabled"] = True
        elif d > 1:
            # sdpa_forward plans only multi-request pure decode batches.
            kwargs["gqa_length_plan"] = self.ops.gqa_decode_length_plan(
                self.decode_lens
            )
        return self._call(self.decode_x, max(self.decode_lens), **kwargs)

    def _prefill_only(self) -> mx.array:
        kwargs = {"num_decode_requests": 0, "num_decode_tokens": 0}
        kwargs.update(self._nax(self.work_prefill_only))
        return self._call(self.prefill_x, self.shape.prefill_len, **kwargs)

    def split_emulated(self) -> list[mx.array]:
        return [self._decode_only(False), self._prefill_only()]

    def split_emulated_pertoken(self) -> list[mx.array]:
        return [self._decode_only(True), self._prefill_only()]


def _route(arms: Arms, name: str) -> tuple[str, int]:
    """Native dispatch family (decode part first for emulated arms)."""
    ops = arms.ops
    mx.synchronize()
    previous = ops._set_paged_dispatch_diagnostics(True)
    try:
        if name.startswith("split_emulated"):
            mx.eval(arms._decode_only(name.endswith("pertoken")))
            decode_family = ops.last_paged_dispatch()
            partition = ops.last_gqa_partition_size()
            mx.eval(arms._prefill_only())
            return f"{decode_family}+{ops.last_paged_dispatch()}", partition
        mx.eval(getattr(arms, name)())
        return ops.last_paged_dispatch(), ops.last_gqa_partition_size()
    finally:
        mx.synchronize()
        ops._set_paged_dispatch_diagnostics(previous)


def _check_outputs(arms: Arms, names: Sequence[str]) -> None:
    dtype = DTYPES[arms.shape.dtype]
    atol, rtol = attention_tolerances(dtype)
    outputs = {}
    for name in names:
        parts = getattr(arms, name)()
        outputs[name] = parts[0] if len(parts) == 1 else mx.concatenate(parts)
    reference_name = names[0]
    reference = outputs[reference_name].astype(mx.float32)
    for name, out in outputs.items():
        if name == reference_name:
            continue
        if not mx.allclose(out.astype(mx.float32), reference, atol=atol, rtol=rtol):
            err = mx.max(mx.abs(out.astype(mx.float32) - reference)).item()
            raise AssertionError(
                f"{name} differs from {reference_name} for {arms.shape} "
                f"(max abs err {err:.3g})"
            )


def _time(
    builders: dict[str, Callable[[], list[mx.array]]],
    layers: int,
    warmup: int,
    reps: int,
    rng: random.Random,
) -> dict[str, tuple[float, float]]:
    """Median and min milliseconds per layer for each arm.

    Arms are interleaved in a fresh random order every rep, so warm-up,
    thermal drift and ordering affect every arm alike.
    """

    def one(build: Callable[[], list[mx.array]]) -> float:
        outs = [o for _ in range(layers) for o in build()]
        mx.synchronize()
        t0 = time.perf_counter()
        mx.eval(outs)
        return (time.perf_counter() - t0) * 1e3 / layers

    names = list(builders)
    samples: dict[str, list[float]] = {name: [] for name in names}
    for rep in range(warmup + reps):
        rng.shuffle(names)
        for name in names:
            elapsed = one(builders[name])
            if rep >= warmup:
                samples[name].append(elapsed)
    return {n: (statistics.median(s), min(s)) for n, s in samples.items()}


def _metadata(args: argparse.Namespace, ops) -> dict:
    try:
        device = mx.device_info()
    except AttributeError:  # older MLX
        device = mx.metal.device_info()
    try:
        commit = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(ROOT), "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {
        "kind": "metadata",
        "nax_ready": ops.nax_ready(),
        "gpu_cores": ops.detected_gpu_core_count(),
        "device": {k: str(v) for k, v in device.items()},
        "machine": platform.machine(),
        "macos": platform.mac_ver()[0],
        "commit": commit,
        "dirty": dirty,
        "versions": package_versions("mlx", "mlx-lm", "vllm", "vllm-metal"),
        "env": {k: os.environ.get(k) for k in ("MLX_MAX_OPS_PER_BUFFER",)},
        "args": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
    }


def _parse_ints(text: str) -> list[int]:
    return [int(v) for v in text.split(",") if v]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--geometry",
        default="32,8,128",
        help="query heads, KV heads, head size (default: 32,8,128)",
    )
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--dtype", choices=sorted(DTYPES), default="bfloat16")
    p.add_argument(
        "--num-decode",
        type=_parse_ints,
        default=[1, 4, 16, 32],
        help="decode rows per batch, comma-separated",
    )
    p.add_argument(
        "--decode-lens",
        type=_parse_ints,
        default=[512, 1024, 1536, 2048, 3072, 4096, 6144, 8192],
        help="KV length per decode row, comma-separated",
    )
    p.add_argument(
        "--prefill-tokens",
        type=_parse_ints,
        default=[512, 2048],
        help="query tokens in the prefill chunk, comma-separated",
    )
    p.add_argument(
        "--prefill-context",
        type=int,
        default=0,
        help="cached tokens before the prefill chunk",
    )
    p.add_argument(
        "--arms",
        default=",".join(DEFAULT_ARMS),
        help=f"comma-separated subset of {', '.join(ARMS)}",
    )
    p.add_argument(
        "--layers", type=int, default=32, help="attention calls per timed evaluation"
    )
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--reps", type=int, default=10)
    p.add_argument(
        "--max-cache-gib",
        type=float,
        default=8.0,
        help="skip shapes whose K+V cache exceeds this size",
    )
    p.add_argument(
        "--min-gain",
        type=float,
        default=0.03,
        help="relative gain the split must keep to count as faster",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--compare",
        choices=SPLIT_ARMS,
        default="split_native",
        help="split arm reported against unsplit (added to --arms if missing)",
    )
    p.add_argument(
        "--disable-nax",
        action="store_true",
        help="route prefill to the tiled kernel on M5 (the pre-M5 route)",
    )
    p.add_argument("--output", type=Path, help="write JSON lines here")
    args = p.parse_args()
    requested = [a for a in args.arms.split(",") if a]
    if set(requested) - set(ARMS) or "unsplit" not in requested:
        p.error(f"--arms must include 'unsplit' and only use {ARMS}; got {requested}")
    requested.append(args.compare)
    # unsplit first: it is the reference for the output check.
    args.arms = [a for a in ARMS if a in requested]
    q, kv, hs = _parse_ints(args.geometry)
    if q % kv:
        p.error("query heads must be a multiple of KV heads")
    args.q_heads, args.kv_heads, args.head_size = q, kv, hs
    return args


def _break_even(results: list[Result], arm: str, min_gain: float) -> int | None:
    """Smallest decode length from which ``arm`` stays faster than unsplit."""
    rows = sorted(results, key=lambda r: r.shape.decode_len)
    for i, r in enumerate(rows):
        if all(
            s.ms_per_layer[arm] <= (1 - min_gain) * s.ms_per_layer["unsplit"]
            for s in rows[i:]
        ):
            return r.shape.decode_len
    return None


def main() -> int:
    args = _parse_args()
    ops = get_ops()
    caps = paged_attention_capabilities(ops)
    missing = [
        k
        for k in ("decode_routing_metadata", "gqa_mixed_decode_plan")
        if not caps.get(k)
    ]
    if missing:
        print(f"Native extension lacks {missing}; rebuild it from this checkout.")
        return 1
    if args.disable_nax:
        ops.set_nax_enabled(False)

    meta = _metadata(args, ops)
    rng = random.Random(args.seed)
    sink = args.output.open("w") if args.output else None
    if sink:
        sink.write(json.dumps(meta) + "\n")
    print(
        f"device={meta['device'].get('device_name', '?')} "
        f"arch={meta['device'].get('architecture', '?')} "
        f"cores={meta['gpu_cores']} nax_ready={meta['nax_ready']}"
    )

    for prefill_tokens in args.prefill_tokens:
        for num_decode in args.num_decode:
            group: list[Result] = []
            print(
                f"\nprefill={prefill_tokens} (+{args.prefill_context} cached) "
                f"decode_rows={num_decode} geometry={args.geometry} "
                f"{args.dtype} block={args.block_size}  [ms per layer, median]"
            )
            header = f"{'decode_len':>10}" + "".join(f"{a:>26}" for a in args.arms)
            print(header + f"{args.compare + '/unsplit':>24}")
            for decode_len in args.decode_lens:
                shape = Shape(
                    args.q_heads,
                    args.kv_heads,
                    args.head_size,
                    args.block_size,
                    args.dtype,
                    num_decode,
                    decode_len,
                    prefill_tokens,
                    args.prefill_context,
                )
                if _cache_bytes(shape) > args.max_cache_gib * 2**30:
                    print(
                        f"{decode_len:>10}  skipped: cache exceeds "
                        f"{args.max_cache_gib} GiB"
                    )
                    continue
                inputs = _build_inputs(shape, args.seed)
                arms = Arms(shape, inputs, caps)
                _check_outputs(arms, args.arms)
                result = Result(shape)
                for name in args.arms:
                    route, partition = _route(arms, name)
                    result.routes[name] = route
                    result.gqa_partition[name] = partition
                timings = _time(
                    {name: getattr(arms, name) for name in args.arms},
                    args.layers,
                    args.warmup,
                    args.reps,
                    rng,
                )
                for name, (median, best) in timings.items():
                    result.ms_per_layer[name] = median
                    result.ms_min[name] = best
                cells = "".join(
                    f"{result.ms_per_layer[a]:>8.3f} {result.routes[a][:17]:>17}"
                    for a in args.arms
                )
                split = result.ms_per_layer[args.compare]
                ratio = f"{split / result.ms_per_layer['unsplit']:>24.3f}"
                print(f"{decode_len:>10}{cells}{ratio}")
                group.append(result)
                if sink:
                    sink.write(json.dumps({"kind": "result", **asdict(result)}) + "\n")
                    sink.flush()
                del arms, inputs
                mx.clear_cache()
            if group:
                be = _break_even(group, args.compare, args.min_gain)
                print(
                    f"  {args.compare} faster by >= {args.min_gain:.0%} from "
                    f"decode_len={be if be is not None else 'never (in range)'}"
                )
    if sink:
        sink.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
