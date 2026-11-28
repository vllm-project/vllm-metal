# SPDX-License-Identifier: Apache-2.0
"""Host-side choice between the two NAX prefill pipelines.

``pagedattention_nax.metal`` compiles a split pipeline, which skips bounds
checks on complete KV tiles of full 64-row Q blocks, and an unsplit one that
keeps the original single guarded loop.  Outputs are identical; only speed
differs.  The split pipeline speeds up full Q blocks but slows the guarded
tiles of partial blocks (a sequence's last block when its length is not a
multiple of 64, and decode rows), and a launch with only a few full blocks
does not fill the GPU enough to hide that.

The choice needs every sequence's query and KV length, which the kernel
dispatcher can only read from the GPU.  The attention context holds them on
the CPU, so the rule is evaluated here once per forward and passed down as
``nax_split_hint``. The D64-D256 thresholds were calibrated on an M5 Max
(BF16, block size 16). D512 uses two output slices to reduce the split kernel's live
accumulator state; short and partial-dominated launches keep the original loop.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

_BQ = 64  # Q rows per NAX threadgroup
_TILE_KV = 32  # KV tokens per tile
# Fewer full Q blocks than this leave too little split work to pay for the
# slower partial blocks (measured: one to three full blocks lose 1-12%).
MIN_FULL_BLOCKS = 4
# Split only when full-block KV tiles outnumber partial-block KV tiles by
# this factor.  head_size 256 gains ~45% per full tile, head_size <= 128
# only ~15%, so the latter needs a much larger share of full-block work.
_MIN_FULL_TO_PARTIAL = {256: 1, 512: 8}
_MIN_FULL_TO_PARTIAL_DEFAULT = 8


@dataclass(frozen=True)
class NaxTileWork:
    """KV-tile counts per Q-block kind for one forward's batch."""

    full_blocks: int = 0
    full_tiles: int = 0
    partial_tiles: int = 0


def _tiles(kv_end: int) -> int:
    return (kv_end + _TILE_KV - 1) // _TILE_KV


def nax_tile_work(
    cu_seqlens: Sequence[int], kv_lens: Sequence[int], sliding_window: int = -1
) -> NaxTileWork:
    """Count tiles for one view using its CPU query and KV lengths."""
    return nax_tile_work_views(cu_seqlens, kv_lens, sliding_window)[0]


def nax_tile_work_views(
    cu_seqlens: Sequence[int],
    kv_lens: Sequence[int],
    sliding_window: int = -1,
    *,
    prefill_start: int | None = None,
) -> tuple[NaxTileWork, NaxTileWork | None]:
    """Count the whole batch and an optional proper prefill suffix in one pass.

    Each sequence contributes independently. Restart the counters at the
    decode/prefill boundary so the prefill rows are only visited once.
    With no boundary, only the whole-batch count is needed.
    """
    if len(cu_seqlens) != len(kv_lens) + 1:
        raise ValueError("cu_seqlens must have one more entry than kv_lens")

    if prefill_start is not None and not 0 < prefill_start < len(kv_lens):
        raise ValueError("prefill_start must identify a nonempty suffix")

    def first_tile(q_abs: int) -> int:
        if sliding_window < 0:
            return 0
        return max(0, (q_abs + 1 - sliding_window) // _TILE_KV)

    prefix = None
    full_blocks = full_tiles = partial_tiles = 0
    for i, kv in enumerate(kv_lens):
        if i == prefill_start:
            prefix = NaxTileWork(full_blocks, full_tiles, partial_tiles)
            full_blocks = full_tiles = partial_tiles = 0
        q = cu_seqlens[i + 1] - cu_seqlens[i]
        if q <= 0:
            continue
        context = kv - q
        n_full = q // _BQ
        full_blocks += n_full
        for j in range(n_full):
            start = context + j * _BQ
            full_tiles += _tiles(start + _BQ) - first_tile(start)
        if q % _BQ:
            partial_tiles += _tiles(kv) - first_tile(context + n_full * _BQ)
    work = NaxTileWork(full_blocks, full_tiles, partial_tiles)
    if prefix is None:
        return work, None
    whole = NaxTileWork(
        prefix.full_blocks + full_blocks,
        prefix.full_tiles + full_tiles,
        prefix.partial_tiles + partial_tiles,
    )
    return whole, work


def nax_split_hint(work: NaxTileWork, head_size: int) -> int:
    """1 to use the split pipeline, 0 for the original loop."""
    if head_size > 512 or work.full_blocks < MIN_FULL_BLOCKS:
        return 0
    factor = _MIN_FULL_TO_PARTIAL.get(head_size, _MIN_FULL_TO_PARTIAL_DEFAULT)
    return int(work.full_tiles >= factor * work.partial_tiles)


def nax_split_kwargs(
    capabilities: Mapping[str, bool],
    work: NaxTileWork | None,
    head_size: int,
    *,
    prefill_work: NaxTileWork | None = None,
) -> dict[str, int]:
    """``nax_split_hint`` keyword for one prefill ``paged_attention_primitive``.

    Empty when the native build predates the hint.  Without per-sequence
    metadata (``work`` is None) the original loop is used. A separate hint
    for the prefill subset is consumed only when native splits the batch.
    """
    if not capabilities.get("nax_split_hint", False):
        return {}
    kwargs = {"nax_split_hint": 0 if work is None else nax_split_hint(work, head_size)}
    if capabilities.get("nax_prefill_split_hint", False) and prefill_work is not None:
        kwargs["nax_prefill_split_hint"] = nax_split_hint(prefill_work, head_size)
    return kwargs
