# SPDX-License-Identifier: Apache-2.0
"""Compare default MLX MLA decode with the opt-in Metal MLA path.

This is a synthetic wrapper-level benchmark for RFC #360 / PR #601. It uses
the real ``MLAPagedAttentionWrapper`` around a small absorbed-MLA module, so the
timed region includes query projection, KV projection for the new token, RoPE,
cache write, attention, unembed, and output projection.

Example:
    VLLM_METAL_BUILD_FROM_SOURCE=1 \
      .venv-vllm-metal/bin/python tools/benchmark/mla_split_vs_mlx_wrapper.py \
      --contexts 32768 49152 65536 81920 98304 131072
"""

from __future__ import annotations

import argparse
import csv
import gc
import math
import os
import random
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import mlx.core as mx
import mlx.nn as nn

from vllm_metal import envs
from vllm_metal.attention import context as pac
from vllm_metal.attention.caches.mla_cache import MLAPagedLatentCache
from vllm_metal.attention.impls.mla import MLAPagedAttentionWrapper
from vllm_metal.metal import metal_mla_split_plan


@dataclass(frozen=True)
class Shape:
    ctx_len: int
    batch_size: int
    num_heads: int
    block_size: int
    dtype: mx.Dtype


class AbsorbedMLAInner(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        kv_lora_rank: int,
        v_head_dim: int,
        dtype: mx.Dtype,
    ) -> None:
        super().__init__()
        self.q_lora_rank = None
        self.num_heads = num_heads
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.kv_lora_rank = kv_lora_rank
        self.v_head_dim = v_head_dim
        self.q_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.scale = 1.0 / math.sqrt(kv_lora_rank)

        self.q_proj = nn.Linear(hidden_size, num_heads * self.q_head_dim, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(
            hidden_size, kv_lora_rank + qk_rope_head_dim, bias=False
        )
        self.kv_a_layernorm = nn.LayerNorm(kv_lora_rank)
        self.embed_q = nn.Linear(qk_nope_head_dim, kv_lora_rank, bias=False)
        self.unembed_out = nn.Linear(kv_lora_rank, v_head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * v_head_dim, hidden_size, bias=False)
        self._cast_linear_weights(dtype)

    def _cast_linear_weights(self, dtype: mx.Dtype) -> None:
        for module in (
            self.q_proj,
            self.kv_a_proj_with_mqa,
            self.embed_q,
            self.unembed_out,
            self.o_proj,
        ):
            module.weight = module.weight.astype(dtype)

    def rope(self, x: mx.array, offset: int = 0) -> mx.array:
        return x


def _sync() -> None:
    mx.eval(mx.array(0))
    if hasattr(mx, "synchronize"):
        mx.synchronize()


def _clear_allocator_cache() -> None:
    """Drop cached allocations between shape rows.

    The benchmark times repeated runs for one shape at a time. Without clearing
    between rows, a previous long-context shape can leave enough cached memory
    behind to perturb the next row, making cross-shape sweeps look choppier than
    isolated per-shape runs.
    """
    gc.collect()
    if hasattr(mx, "clear_cache"):
        mx.clear_cache()


def _make_context(shape: Shape) -> SimpleNamespace:
    blocks_per_seq = math.ceil(shape.ctx_len / shape.block_size)
    block_tables = []
    slot_mapping = []
    for seq_idx in range(shape.batch_size):
        start_block = seq_idx * blocks_per_seq
        blocks = list(range(start_block, start_block + blocks_per_seq))
        block_tables.append(blocks)
        slot_mapping.append(start_block * shape.block_size + shape.ctx_len - 1)
    return SimpleNamespace(
        slot_mapping=slot_mapping,
        block_tables=block_tables,
        context_lens=[shape.ctx_len] * shape.batch_size,
        cu_seqlens=list(range(shape.batch_size + 1)),
        offsets=[shape.ctx_len - 1] * shape.batch_size,
    )


def _make_wrapper(
    *,
    shape: Shape,
    hidden_size: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    kv_lora_rank: int,
    v_head_dim: int,
    inner: AbsorbedMLAInner,
) -> tuple[MLAPagedAttentionWrapper, MLAPagedLatentCache, mx.array, mx.array]:
    blocks_per_seq = math.ceil(shape.ctx_len / shape.block_size)
    cache = MLAPagedLatentCache(
        num_layers=1,
        latent_dim=kv_lora_rank + qk_rope_head_dim,
        num_blocks=blocks_per_seq * shape.batch_size,
        block_size=shape.block_size,
        dtype=shape.dtype,
    )
    base_cache = mx.random.normal(cache.latent_caches[0].shape).astype(shape.dtype)
    x = mx.random.normal((1, shape.batch_size, hidden_size)).astype(shape.dtype)
    mx.eval(base_cache, x)
    wrapper = MLAPagedAttentionWrapper(inner=inner, layer_idx=0, latent_cache=cache)
    return wrapper, cache, base_cache, x


def _run_once(
    *,
    wrapper: MLAPagedAttentionWrapper,
    cache: MLAPagedLatentCache,
    base_cache: mx.array,
    x: mx.array,
    ctx: SimpleNamespace,
    mode: str,
) -> mx.array:
    envs.VLLM_METAL_MLA_KERNEL = mode != "mlx"
    if mode == "metal_single":
        os.environ["VLLM_METAL_MLA_SPLIT_KV"] = "0"
    else:
        os.environ.pop("VLLM_METAL_MLA_SPLIT_KV", None)
    cache.latent_caches[0] = base_cache
    pac.set_context(ctx)
    try:
        out = wrapper(x, mask=None, cache=None)
        mx.eval(out)
        _sync()
        return out
    finally:
        pac.clear_context()


def _time_mode(
    *,
    wrapper: MLAPagedAttentionWrapper,
    cache: MLAPagedLatentCache,
    base_cache: mx.array,
    x: mx.array,
    ctx: SimpleNamespace,
    warmup: int,
    iters: int,
    mode: str,
) -> float:
    for _ in range(warmup):
        _run_once(
            wrapper=wrapper,
            cache=cache,
            base_cache=base_cache,
            x=x,
            ctx=ctx,
            mode=mode,
        )

    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _run_once(
            wrapper=wrapper,
            cache=cache,
            base_cache=base_cache,
            x=x,
            ctx=ctx,
            mode=mode,
        )
        samples.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(samples)


def _bench_shape(args: argparse.Namespace, shape: Shape) -> dict[str, object]:
    mx.random.seed(args.seed)
    random.seed(args.seed)
    inner = AbsorbedMLAInner(
        hidden_size=args.hidden_size,
        num_heads=shape.num_heads,
        qk_nope_head_dim=args.qk_nope_head_dim,
        qk_rope_head_dim=args.qk_rope_head_dim,
        kv_lora_rank=args.kv_lora_rank,
        v_head_dim=args.v_head_dim,
        dtype=shape.dtype,
    )
    mlx_wrapper, mlx_cache, mlx_base_cache, x = _make_wrapper(
        shape=shape,
        hidden_size=args.hidden_size,
        qk_nope_head_dim=args.qk_nope_head_dim,
        qk_rope_head_dim=args.qk_rope_head_dim,
        kv_lora_rank=args.kv_lora_rank,
        v_head_dim=args.v_head_dim,
        inner=inner,
    )
    metal_single_wrapper, metal_single_cache, _, _ = _make_wrapper(
        shape=shape,
        hidden_size=args.hidden_size,
        qk_nope_head_dim=args.qk_nope_head_dim,
        qk_rope_head_dim=args.qk_rope_head_dim,
        kv_lora_rank=args.kv_lora_rank,
        v_head_dim=args.v_head_dim,
        inner=inner,
    )
    metal_split_wrapper, metal_split_cache, _, _ = _make_wrapper(
        shape=shape,
        hidden_size=args.hidden_size,
        qk_nope_head_dim=args.qk_nope_head_dim,
        qk_rope_head_dim=args.qk_rope_head_dim,
        kv_lora_rank=args.kv_lora_rank,
        v_head_dim=args.v_head_dim,
        inner=inner,
    )
    ctx = _make_context(shape)
    heads_per_tg = MLAPagedAttentionWrapper._pick_heads_per_tg(
        shape.num_heads, shape.batch_size
    )
    os.environ.pop("VLLM_METAL_MLA_SPLIT_KV", None)
    plan = metal_mla_split_plan(
        total_q_tokens=shape.batch_size,
        num_seqs=shape.batch_size,
        num_heads=shape.num_heads,
        heads_per_tg=heads_per_tg,
        max_seq_len=math.ceil(shape.ctx_len / shape.block_size) * shape.block_size,
    )

    mlx_out = _run_once(
        wrapper=mlx_wrapper,
        cache=mlx_cache,
        base_cache=mlx_base_cache,
        x=x,
        ctx=ctx,
        mode="mlx",
    )
    metal_single_out = _run_once(
        wrapper=metal_single_wrapper,
        cache=metal_single_cache,
        base_cache=mlx_base_cache,
        x=x,
        ctx=ctx,
        mode="metal_single",
    )
    metal_split_out = _run_once(
        wrapper=metal_split_wrapper,
        cache=metal_split_cache,
        base_cache=mlx_base_cache,
        x=x,
        ctx=ctx,
        mode="metal_split",
    )
    max_diff_single = float(
        mx.max(
            mx.abs(mlx_out.astype(mx.float32) - metal_single_out.astype(mx.float32))
        ).item()
    )
    max_diff_split = float(
        mx.max(
            mx.abs(mlx_out.astype(mx.float32) - metal_split_out.astype(mx.float32))
        ).item()
    )

    trial_pairs = []
    for _ in range(args.trials):
        modes = ["mlx", "metal_single", "metal_split"]
        random.shuffle(modes)
        trial = {}
        for mode in modes:
            trial[mode] = _time_mode(
                wrapper={
                    "mlx": mlx_wrapper,
                    "metal_single": metal_single_wrapper,
                    "metal_split": metal_split_wrapper,
                }[mode],
                cache={
                    "mlx": mlx_cache,
                    "metal_single": metal_single_cache,
                    "metal_split": metal_split_cache,
                }[mode],
                base_cache=mlx_base_cache,
                x=x,
                ctx=ctx,
                warmup=args.warmup,
                iters=args.iters,
                mode=mode,
            )
        trial_pairs.append(trial)

    mlx_ms = statistics.median(t["mlx"] for t in trial_pairs)
    metal_single_ms = statistics.median(t["metal_single"] for t in trial_pairs)
    metal_split_ms = statistics.median(t["metal_split"] for t in trial_pairs)
    single_vs_mlx = (mlx_ms / metal_single_ms - 1.0) * 100.0
    split_vs_mlx = (mlx_ms / metal_split_ms - 1.0) * 100.0
    split_vs_single = (metal_single_ms / metal_split_ms - 1.0) * 100.0
    route = "metal_split" if plan["partition"] else "mlx_fallback"
    routed_ms = metal_split_ms if plan["partition"] else mlx_ms
    routed_vs_mlx = split_vs_mlx if plan["partition"] else 0.0
    routed_max_diff = max_diff_split if plan["partition"] else 0.0
    return {
        "ctx_len": shape.ctx_len,
        "batch": shape.batch_size,
        "heads": shape.num_heads,
        "dtype": "bf16" if shape.dtype == mx.bfloat16 else "fp16",
        "block_size": shape.block_size,
        "heads_per_tg": heads_per_tg,
        "split": plan["partition"],
        "partition_size": plan["partition_size"],
        "partitions": plan["max_num_partitions"],
        "mlx_ms": mlx_ms,
        "route": route,
        "routed_ms": routed_ms,
        "metal_single_ms": metal_single_ms,
        "metal_split_ms": metal_split_ms,
        "routed_vs_mlx_pct": routed_vs_mlx,
        "single_vs_mlx_pct": single_vs_mlx,
        "split_vs_mlx_pct": split_vs_mlx,
        "split_vs_single_pct": split_vs_single,
        "max_abs_diff_single": max_diff_single,
        "max_abs_diff_split": max_diff_split,
        "routed_max_abs_diff": routed_max_diff,
    }


def _print_markdown(rows: list[dict[str, object]]) -> None:
    print(
        "\n| ctx | B | H | dtype | route | parts | MLX ms | "
        "routed ms | routed vs MLX | max diff |"
    )
    print("|---:|---:|---:|:---|:---|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['ctx_len']} | {row['batch']} | {row['heads']} | "
            f"{row['dtype']} | {row['route']} | {row['partitions']} | "
            f"{float(row['mlx_ms']):.3f} | "
            f"{float(row['routed_ms']):.3f} | "
            f"{float(row['routed_vs_mlx_pct']):+.1f}% | "
            f"{float(row['routed_max_abs_diff']):.5f} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contexts",
        nargs="+",
        type=int,
        default=[32768, 49152, 65536, 81920, 98304, 131072],
    )
    parser.add_argument("--batches", nargs="+", type=int, default=[1])
    parser.add_argument("--heads", nargs="+", type=int, default=[16])
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--qk-nope-head-dim", type=int, default=128)
    parser.add_argument("--qk-rope-head-dim", type=int, default=64)
    parser.add_argument("--kv-lora-rank", type=int, default=512)
    parser.add_argument("--v-head-dim", type=int, default=128)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv-out", type=Path)
    args = parser.parse_args()

    dtype = mx.bfloat16 if args.dtype == "bf16" else mx.float16
    rows = []
    for batch_size in args.batches:
        for num_heads in args.heads:
            for ctx_len in args.contexts:
                _clear_allocator_cache()
                row = _bench_shape(
                    args,
                    Shape(
                        ctx_len=ctx_len,
                        batch_size=batch_size,
                        num_heads=num_heads,
                        block_size=args.block_size,
                        dtype=dtype,
                    ),
                )
                rows.append(row)
                print(
                    f"ctx={ctx_len} B={batch_size} H={num_heads}: "
                    f"MLX {float(row['mlx_ms']):.3f} ms, "
                    f"{row['route']} {float(row['routed_ms']):.3f} ms, "
                    f"routed-vs-MLX "
                    f"{float(row['routed_vs_mlx_pct']):+.1f}%"
                )

    _print_markdown(rows)
    if args.csv_out:
        with args.csv_out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {args.csv_out}")


if __name__ == "__main__":
    main()
