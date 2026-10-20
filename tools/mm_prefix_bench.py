# SPDX-License-Identifier: Apache-2.0
"""Op-level A/B for Gemma 4 image-block attention: tiled kernel vs recompute.

Times one Gemma 4 26B-A4B sliding-window layer (16 query heads, 8 KV heads,
head_dim 256, window 1024, bf16 paged cache) over a prefill chunk that holds
image blocks, three ways:

  causal     the paged kernel alone, without the image blocks: what a text
             chunk of the same shape costs (wrong for image rows)
  kernel     the tiled prefill kernel with per-row block ranges
             (VLLM_METAL_MM_PREFIX_PATH=kernel, the default)
  recompute  the causal kernel, then the MLX SDPA recompute of the block
             rows spliced into its output (VLLM_METAL_MM_PREFIX_PATH=recompute)

Each rep chains LAYERS_PER_REP calls between mx.synchronize fences, feeding
one layer's output to the next as its query, the way a forward runs its
layers.  The kernel's row ranges are built once per rep, as the attention
impl builds them once per forward.  Before timing a scenario, ``max|k-r|``
compares the two image paths on one layer, and the script stops if they
differ by more than MAX_ABS_DIFF.

On an M5 the calls without ranges (causal, and the recompute's kernel call)
run on NAX, while a batch with ranges stays on the tiled kernel.  ``--no-nax``
keeps every call on the tiled kernel, which is what M1-M4 run.  Measured
numbers live in the PR that added this script, not here, so this header
cannot go stale.

Run from the repo root:

    PYTHONPATH=$PWD python tools/mm_prefix_bench.py [--no-nax]
"""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass

import mlx.core as mx

from vllm_metal.attention.context import PagedAttentionContext
from vllm_metal.attention.impls.bidi_prefill import apply_bidirectional_segments
from vllm_metal.attention.impls.mm_prefix import build_mm_prefix_rows
from vllm_metal.metal import get_ops

# Gemma 4 26B-A4B text_config: 25 of its 30 layers are sliding-window layers,
# the only kind whose image blocks attend bidirectionally.
HEADS, KV_HEADS, HEAD_DIM, WINDOW = 16, 8, 256, 1024
SCALE = HEAD_DIM**-0.5
BLOCK_SIZE = 16
DTYPE = mx.bfloat16
SOFT_TOKENS = 280  # vision_soft_tokens_per_image
TEXT_GAP = 20  # text tokens before, between and after the images
SLIDING_LAYERS = 25
LAYERS_PER_REP = SLIDING_LAYERS
WARMUP_REPS = 3
TIMED_REPS = 10
# The two image paths round differently in bf16; measured runs stay within
# 2**-6, one bf16 ulp for outputs in [2, 4).  A path that attends to the
# wrong keys misses by far more, and its timings would compare different
# computations.
MAX_ABS_DIFF = 2**-5


@dataclass(frozen=True)
class Scenario:
    name: str
    images: tuple[int, ...]  # soft tokens per image
    history: int = 0  # tokens already in the cache before the chunk
    rows: int | None = None  # pad the chunk's text tail to this many rows


SCENARIOS = [
    Scenario("1 image", (SOFT_TOKENS,)),
    Scenario("1 image, 8K history", (SOFT_TOKENS,), history=8192),
    Scenario("4 images", (SOFT_TOKENS,) * 4),
    Scenario("6 images, 2048 rows", (SOFT_TOKENS,) * 6, rows=2048),
    Scenario("1120-token image", (1120,)),
]


def _layout(s: Scenario) -> tuple[int, list[tuple[int, int]]]:
    """Chunk rows and the half-open absolute blocks of the soft tokens.

    Text, then ``boi + soft tokens + eoi`` and more text per image.
    """
    pos = s.history + TEXT_GAP
    blocks = []
    for soft in s.images:
        start = pos + 1  # after boi
        blocks.append((start, start + soft))
        pos = start + soft + 1 + TEXT_GAP  # eoi, then text
    rows = pos - s.history
    if s.rows is not None:
        if rows > s.rows:
            raise SystemExit(f"{s.name}: the layout needs {rows} rows, over {s.rows}")
        rows = s.rows
    return rows, blocks


def _median_us(run) -> float:
    for _ in range(WARMUP_REPS):
        mx.eval(run())
    mx.synchronize()
    times = []
    for _ in range(TIMED_REPS):
        t0 = time.perf_counter()
        mx.eval(run())
        mx.synchronize()
        times.append((time.perf_counter() - t0) / LAYERS_PER_REP * 1e6)
    return statistics.median(times)


def _bench(ops, s: Scenario) -> tuple[int, float, float, float, float]:
    rows, blocks = _layout(s)
    seq_len = s.history + rows
    num_blocks = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    shape = (num_blocks + 1, BLOCK_SIZE, KV_HEADS, HEAD_DIM)
    k_cache = mx.random.normal(shape).astype(DTYPE)
    v_cache = mx.random.normal(shape).astype(DTYPE)
    query = mx.random.normal((rows, HEADS, HEAD_DIM)).astype(DTYPE)
    table = mx.array([list(range(1, num_blocks + 1))], dtype=mx.int32)
    seq_lens = mx.array([seq_len], dtype=mx.int32)
    cu = [0, rows]
    cu_q = mx.array(cu, dtype=mx.int32)
    mx.eval(k_cache, v_cache, query, table, seq_lens, cu_q)

    def paged(q, ranges=None):
        kwargs = {} if ranges is None else {"mm_prefix_ranges": ranges}
        out = ops.paged_attention_primitive(
            q,
            k_cache,
            v_cache,
            KV_HEADS,
            SCALE,
            0.0,
            table,
            seq_lens,
            cu_q,
            BLOCK_SIZE,
            seq_len,
            WINDOW,
            **kwargs,
        )
        return out

    def ranges():
        found = build_mm_prefix_rows(cu, [seq_len], [blocks])
        if found is None:
            raise SystemExit(f"{s.name}: no query row lies inside an image block")
        return mx.array(found)

    def recompute_layer(q, ctx):
        return apply_bidirectional_segments(
            paged(q),
            q,
            k_cache,
            v_cache,
            block_tables=table,
            block_size=BLOCK_SIZE,
            cu_seqlens=cu,
            context_lens=[seq_len],
            ctx=ctx,
            window=WINDOW,
            scale=SCALE,
            head_dim=HEAD_DIM,
            softcap=0.0,
            sinks=None,
            turboquant=False,
        )

    def context():
        ctx = PagedAttentionContext(
            slot_mapping=[],
            cu_seqlens=cu,
            context_lens=[seq_len],
            segment_bidi_ranges=[blocks],
            bidi_layer_kinds=frozenset({"sliding"}),
        )
        ctx.bidi_logged = True  # keep the per-forward log line out of the timing
        return ctx

    def run_causal():
        q = query
        for _ in range(LAYERS_PER_REP):
            q = paged(q)
        return q

    def run_kernel():
        rows_buf = ranges()
        q = query
        for _ in range(LAYERS_PER_REP):
            q = paged(q, rows_buf)
        return q

    def run_recompute():
        ctx = context()
        q = query
        for _ in range(LAYERS_PER_REP):
            q = recompute_layer(q, ctx)
        return q

    got_kernel = paged(query, ranges()).astype(mx.float32)
    got_recompute = recompute_layer(query, context()).astype(mx.float32)
    diff = mx.max(mx.abs(got_kernel - got_recompute)).item()
    if not diff <= MAX_ABS_DIFF:  # also stops on NaN
        raise SystemExit(
            f"{s.name}: max|k-r| {diff:.4f} exceeds {MAX_ABS_DIFF}: the image "
            "paths disagree, so their timings are not comparable"
        )
    causal = _median_us(run_causal)
    kernel = _median_us(run_kernel)
    recompute = _median_us(run_recompute)
    return rows, causal, kernel, recompute, diff


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--no-nax",
        action="store_true",
        help="keep every call on the tiled kernel, as on M1-M4",
    )
    args = parser.parse_args()

    ops = get_ops()
    if args.no_nax:
        ops.set_nax_enabled(False)
    nax = "on" if ops.nax_ready() else "off"
    print(f"device: {mx.device_info()['device_name']}  nax: {nax}")
    print(
        f"layer: {HEADS}q/{KV_HEADS}kv hd{HEAD_DIM} window {WINDOW} {DTYPE}  "
        f"layers/rep: {LAYERS_PER_REP}  reps: {TIMED_REPS}"
    )
    print()
    mx.random.seed(0)

    header = (
        f"{'scenario':<22} {'rows':>5} | {'causal us':>9} {'kernel us':>9} "
        f"{'recomp us':>9} | {'rec/ker':>7} {'ker/caus':>8} | "
        f"{f'x{SLIDING_LAYERS} ker ms':>10} {f'x{SLIDING_LAYERS} rec ms':>10} | "
        f"{'max|k-r|':>8}"
    )
    print(header)
    print("-" * len(header))
    for s in SCENARIOS:
        rows, causal, kernel, recompute, diff = _bench(ops, s)
        print(
            f"{s.name:<22} {rows:>5} | {causal:>9.0f} {kernel:>9.0f} "
            f"{recompute:>9.0f} | {recompute / kernel:>7.2f} "
            f"{kernel / causal:>8.2f} | {kernel * SLIDING_LAYERS / 1e3:>10.2f} "
            f"{recompute * SLIDING_LAYERS / 1e3:>10.2f} | {diff:>8.4f}"
        )


if __name__ == "__main__":
    main()
