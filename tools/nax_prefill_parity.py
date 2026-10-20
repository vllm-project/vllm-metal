#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Manual M5 parity check for NAX versus tiled paged prefill attention.

Deliberately not wired into CI: no GitHub runner has an M5, and a check that
silently skips itself is worse than no check. Run it on an M5 before touching
``pagedattention_nax.metal`` or the NAX dispatch::

    python tools/nax_prefill_parity.py

Covers every compiled specialization -- {float16, bfloat16} x head_size
{64, 96, 128, 256, 512} x block_size {8, 16, 32} -- in both masking regimes:

``fast``
    Softcap off, no sliding window, sinks off. Reaches the ``tile_no_mask``
    arm (where ``min_q_abs_sg`` and the ``row_lim >= 16`` guard are
    load-bearing) and the ``_sk0`` pipeline, whose softmax init is the one that
    can end a row with a zero denominator.
``masked``
    Softcap, sliding window and sinks all on: the per-element mask arm and the
    ``_sk1`` pipeline.

Block size matters beyond the kernel name: at 8 the four half-row bases of a
32-token KV tile land on four different pages, at 32 they collapse onto one.
Head sizes 64/96 (TD=4/6) skip the mid-loop scheduling barrier. Head sizes
128/256/512 (TD=8/16/32) split the PV loop at its midpoint; their output
accumulators hold 64/128/256 fp32 values per thread, respectively.

Additional boundary cases poison unused cache slots with NaNs and independently
exercise small sliding windows, softcap and sinks. Dispatch diagnostics verify
that both kernels ran; zero- and one-token windows can legitimately produce
identical outputs.

Without a hint these batches take the original loop. A subset -- every
boundary case (NaN padding, windows, softcap, sinks), including one each at
head sizes 64/96, and block size 16 at head sizes 128/256/512 -- also runs
with ``nax_split_hint`` forced to 0 and 1 and checks the pipeline that actually
ran. Forced-hint cases exercise both pipelines except at head_size 512, where
both hints use the original loop; the final counts distinguish these fallbacks.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass

import mlx.core as mx

from vllm_metal.metal import get_ops

# (query tokens, context tokens) per sequence. One-token rows exercise decode
# passengers in a mixed prefill batch.
SEQS = ((129, 130), (1, 999), (64, 0), (1, 333))
# Full/partial 16-row Q tiles and KV lengths around 32-token boundaries.
BOUNDARY_SEQS = (
    (16, 15),  # 31 KV tokens
    (17, 15),  # 32
    (32, 1),  # 33
    (33, 31),  # 64
    (48, 17),  # 65
    (63, 0),
    (64, 0),
    (64, 1),
    (16, 32),
    (17, 33),
)
Q_HEADS = 32
KV_HEADS = 8

SOFTCAP = 30.0
SLIDING_WINDOW = 64

# NAX truncates the fp32 P operand of PV (relaxed_precision) and accumulates in
# a different order than the tiled kernel, so exact agreement is not the bar --
# but output-dtype rounding is. Divergence is measured in ULPs of the output
# dtype at the batch's peak magnitude, which keeps one bound meaningful for both
# fp16 and bf16 (a plain absolute bound loose enough for bf16 is 8x too loose
# for fp16, and a per-element relative error is meaningless where attention
# outputs approach zero).
#
# Measured across the full matrix on an M5 Pro (macOS 26.6): worst case
# 0.88 ULP max / 0.022 ULP mean. The mean is the one that catches a mask or
# geometry bug -- those shift whole rows, while rounding noise stays scattered.
_DTYPE_EPS = {mx.float16: 2**-10, mx.bfloat16: 2**-7}
MAX_ULP = 2.0
MEAN_ULP = 0.05


@dataclass(frozen=True)
class Case:
    dtype: mx.Dtype
    head_size: int
    block_size: int
    regime: str
    softcap: float = 0.0
    sliding_window: int = -1
    use_sinks: bool = False
    seqs: tuple[tuple[int, int], ...] = SEQS
    poison_padding: bool = False

    def __str__(self) -> str:
        return (
            f"{str(self.dtype).removeprefix('mlx.core.'):9s} "
            f"hs={self.head_size:<3d} bs={self.block_size:<2d} {self.regime}"
        )


def _cases() -> list[Case]:
    """The full instantiated matrix: instantiate_paged_attention_nax_all emits
    {half, bfloat16_t} x hs{64, 96, 128, 256, 512} x bs{8, 16, 32}, and the
    dispatcher picks a separate pipeline per sinks constant."""
    return [
        Case(
            dtype,
            head_size,
            block_size,
            regime,
            softcap=SOFTCAP if regime == "masked" else 0.0,
            sliding_window=SLIDING_WINDOW if regime == "masked" else -1,
            use_sinks=regime == "masked",
        )
        for dtype in (mx.float16, mx.bfloat16)
        for head_size in (64, 96, 128, 256, 512)
        for block_size in (8, 16, 32)
        for regime in ("fast", "masked")
    ]


def _boundary_cases() -> list[Case]:
    cases = []
    for dtype, head_size in ((mx.float16, 128), (mx.bfloat16, 256)):
        for block_size in (8, 16, 32):
            cases.append(
                Case(
                    dtype,
                    head_size,
                    block_size,
                    "boundary-fast",
                    seqs=BOUNDARY_SEQS,
                    poison_padding=True,
                )
            )
        # Keep masking independent without multiplying the full shape matrix.
        for name, window, softcap, sinks in [
            *[(f"window-{w}", w, 0.0, False) for w in (0, 1, 31, 32, 33, 63, 64, 65)],
            ("softcap", -1, SOFTCAP, False),
            ("sinks", -1, 0.0, True),
        ]:
            cases.append(
                Case(
                    dtype,
                    head_size,
                    16,
                    f"boundary-{name}",
                    softcap=softcap,
                    sliding_window=window,
                    use_sinks=sinks,
                    seqs=BOUNDARY_SEQS,
                    poison_padding=True,
                )
            )
    # Cover the TD=4/6 specializations on both pipelines without expanding
    # the dtype/block-size/masking matrix.
    cases.extend(
        Case(
            dtype,
            head_size,
            16,
            "boundary-fast",
            seqs=BOUNDARY_SEQS,
            poison_padding=True,
        )
        for dtype, head_size in ((mx.float16, 64), (mx.bfloat16, 96))
    )
    return cases


def _build_case(case: Case):
    mx.random.seed(3)
    block_size, head_size = case.block_size, case.head_size
    totals = [q + c for q, c in case.seqs]
    blocks_per = [(t + block_size - 1) // block_size for t in totals]
    num_blocks = sum(blocks_per) + 1  # block 0 is padding

    shape = (num_blocks, block_size, KV_HEADS, head_size)
    key_cache = mx.random.normal(shape).astype(case.dtype)
    value_cache = mx.random.normal(shape).astype(case.dtype)
    query = (
        mx.random.normal((sum(q for q, _ in case.seqs), Q_HEADS, head_size)) * 0.5
    ).astype(case.dtype)

    # Non-contiguous physical pages exercise the block-table gather.
    block_ids = [*range(1, num_blocks, 2), *range(2, num_blocks, 2)]
    rows, offset = [], 0
    for count in blocks_per:
        rows.append(block_ids[offset : offset + count])
        offset += count
    if case.poison_padding:
        valid_lens = [0] * num_blocks
        for total, row in zip(totals, rows, strict=True):
            for logical_block, physical_block in enumerate(row):
                valid_lens[physical_block] = min(
                    block_size, total - logical_block * block_size
                )
        valid = mx.arange(block_size)[None, :] < mx.array(valid_lens)[:, None]
        valid = valid[:, :, None, None]
        padding = mx.array(float("nan"), dtype=case.dtype)
        key_cache = mx.where(valid, key_cache, padding)
        value_cache = mx.where(valid, value_cache, padding)
    max_blocks = max(blocks_per)
    block_tables = mx.array(
        [row + [0] * (max_blocks - len(row)) for row in rows], dtype=mx.int32
    )
    seq_lens = mx.array(totals, dtype=mx.int32)
    cu_seqlens = mx.array(
        [0, *mx.cumsum(mx.array([q for q, _ in case.seqs], dtype=mx.int32)).tolist()],
        dtype=mx.int32,
    )
    sinks = (
        (mx.random.normal((Q_HEADS,)) * 0.5).astype(mx.float32)
        if case.use_sinks
        else None
    )
    mx.eval(key_cache, value_cache, query, block_tables, seq_lens, cu_seqlens)
    if sinks is not None:
        mx.eval(sinks)
    return key_cache, value_cache, query, block_tables, seq_lens, cu_seqlens, sinks


def _run(
    ops, case: Case, built, *, enabled: bool, split_hint: int | None = None
) -> tuple[mx.array, str, int]:
    key_cache, value_cache, query, block_tables, seq_lens, cu_seqlens, sinks = built
    ops.set_nax_enabled(enabled)
    out = ops.paged_attention_primitive(
        query,
        key_cache,
        value_cache,
        KV_HEADS,
        case.head_size**-0.5,
        case.softcap,
        block_tables,
        seq_lens,
        cu_seqlens,
        case.block_size,
        max(int(x) for x in seq_lens.tolist()),
        case.sliding_window,
        sinks=sinks,
        **({} if split_hint is None else {"nax_split_hint": split_hint}),
    )
    mx.eval(out)
    return (
        out.astype(mx.float32),
        ops.last_paged_dispatch(),
        ops.last_nax_split_pipeline(),
    )


def _compare(case: Case, nax: mx.array, tiled: mx.array) -> tuple[bool, str]:
    delta = mx.abs(nax - tiled)
    abs_err = delta.max().item()
    ulp = _DTYPE_EPS[case.dtype] * max(mx.abs(tiled).max().item(), 1.0)
    max_ulp = abs_err / ulp
    mean_ulp = delta.mean().item() / ulp

    if not math.isfinite(abs_err):
        return False, f"non-finite output (abs={abs_err})"
    detail = f"max={max_ulp:.2f}ulp mean={mean_ulp:.4f}ulp"
    if max_ulp > MAX_ULP or mean_ulp > MEAN_ULP:
        return False, f"{detail} exceeds bound"
    return True, detail


def _forced_hints(case: Case) -> bool:
    return case.regime.startswith("boundary") or (
        case.block_size == 16 and case.head_size >= 128
    )


def _check(ops, case: Case) -> tuple[bool, str]:
    built = _build_case(case)
    tiled, tiled_dispatch, _ = _run(ops, case, built, enabled=False)
    if tiled_dispatch != "tiled_prefill":
        return False, f"unexpected dispatch: tiled={tiled_dispatch}"
    details = []
    for hint in (0, 1) if _forced_hints(case) else (None,):
        nax, nax_dispatch, pipeline = _run(
            ops, case, built, enabled=True, split_hint=hint
        )
        if nax_dispatch != "nax_prefill":
            return False, f"hint={hint}: unexpected dispatch NAX={nax_dispatch}"
        if hint is not None:
            expected = hint if case.head_size < 512 else 0
            if pipeline != expected:
                return False, f"hint={hint}: ran pipeline {pipeline}, want {expected}"
        mx.eval(nax, tiled)
        ok, detail = _compare(case, nax, tiled)
        details.append(f"{'split' if pipeline == 1 else 'loop'}: {detail}")
        if not ok:
            return False, "; ".join(details)
    return True, "; ".join(details)


def main() -> int:
    ops = get_ops()
    if not (ops.nax_supported() and ops.nax_ready()):
        print(
            "NAX is unavailable; run this check on an M5 with the NAX metallib.",
            file=sys.stderr,
        )
        return 2

    cases = _cases() + _boundary_cases()
    failures = []
    previous_diagnostics = ops._set_paged_dispatch_diagnostics(True)
    try:
        for case in cases:
            ok, detail = _check(ops, case)
            print(f"{'ok  ' if ok else 'FAIL'} {case}  {detail}")
            if not ok:
                failures.append(f"{case}: {detail}")
    finally:
        ops.set_nax_enabled(True)
        ops._set_paged_dispatch_diagnostics(previous_diagnostics)

    if failures:
        print(
            f"\nPARITY FAIL ({len(failures)} case(s), "
            f"limits max={MAX_ULP:.3g}ulp mean={MEAN_ULP:.3g}ulp):",
            file=sys.stderr,
        )
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1
    forced = sum(_forced_hints(case) for case in cases)
    fallback = sum(_forced_hints(case) and case.head_size == 512 for case in cases)
    print(
        f"\nPARITY PASS ({len(cases)} cases, {forced} forced-hint cases: "
        f"{forced - fallback} on both NAX pipelines, "
        f"{fallback} D512 unsplit fallback cases)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
