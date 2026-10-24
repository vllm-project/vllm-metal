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
``nax_split_hint``. Single-request full attention uses the same conservative
tile-work score for full and cached prefills at every supported head size.
It credits complete query blocks and the kernel's output slices separately.
Windowed, multi-request and legacy callers retain the established rule.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

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
    sequences: int = 0
    query_tokens: int = 0
    sliding_window: int = -1


def nax_gpu_core_count(ops: Any) -> int:
    """Detected capacity, without a device-name policy or guessed fallback."""
    detect = getattr(ops, "detected_gpu_core_count", None)
    return int(detect()) if detect is not None else 0


def _tiles(kv_end: int) -> int:
    return (kv_end + _TILE_KV - 1) // _TILE_KV


def nax_tile_work(
    cu_seqlens: Sequence[int], kv_lens: Sequence[int], sliding_window: int = -1
) -> NaxTileWork:
    """Count the KV tiles the NAX kernel visits for full and partial Q blocks.

    ``cu_seqlens`` are cumulative query rows and ``kv_lens`` each sequence's
    KV length including this step's tokens (``PagedAttentionContext``).
    Causal tiles past a block's last row are skipped, and with
    ``sliding_window >= 0`` so are the tiles before the block's first row's
    window, mirroring ``kb_start`` / ``kb_end`` in ``pagedattention_nax.metal``.
    """
    if len(cu_seqlens) != len(kv_lens) + 1:
        raise ValueError("cu_seqlens must have one more entry than kv_lens")

    def first_tile(q_abs: int) -> int:
        if sliding_window < 0:
            return 0
        return max(0, (q_abs + 1 - sliding_window) // _TILE_KV)

    full_blocks = full_tiles = partial_tiles = sequences = query_tokens = 0
    for i, kv in enumerate(kv_lens):
        q = cu_seqlens[i + 1] - cu_seqlens[i]
        if q <= 0:
            continue
        context = kv - q
        sequences += 1
        query_tokens += q
        n_full = q // _BQ
        full_blocks += n_full
        for j in range(n_full):
            start = context + j * _BQ
            full_tiles += _tiles(start + _BQ) - first_tile(start)
        if q % _BQ:
            partial_tiles += _tiles(kv) - first_tile(context + n_full * _BQ)
    return NaxTileWork(
        full_blocks,
        full_tiles,
        partial_tiles,
        sequences,
        query_tokens,
        sliding_window,
    )


def nax_split_hint(
    work: NaxTileWork,
    head_size: int,
    *,
    gpu_core_count: int = 0,
    num_heads: int = 0,
) -> int:
    """1 to use the split pipeline, 0 for the original loop."""
    if head_size > 512:
        return 0
    if (
        num_heads <= 0
        or gpu_core_count <= 0
        or work.sequences != 1
        or work.query_tokens < 2
        or work.sliding_window >= 0
    ):
        factor = _MIN_FULL_TO_PARTIAL.get(head_size, _MIN_FULL_TO_PARTIAL_DEFAULT)
        return int(
            work.full_blocks >= MIN_FULL_BLOCKS
            and work.full_tiles >= factor * work.partial_tiles
        )

    # A tile-work heuristic, not a latency prediction. Small heads share a
    # minimum width unit because fixed tile costs dominate their QK/PV work.
    width = max(head_size, 128) / 128
    full_groups = num_heads * work.full_blocks
    full_credit = 0.0
    startup_margin = 8
    if work.full_blocks >= MIN_FULL_BLOCKS:
        # Enough complete blocks retain their full credit, including tails.
        # Discounting them can disable a previously faster pipeline.
        full_credit = float(work.full_tiles)
        startup_margin = 0
    elif width * work.full_blocks >= 4 and 2 * full_groups >= 3 * gpu_core_count:
        full_credit = work.full_tiles * min(1.0, full_groups / (4 * gpu_core_count))

    # This is the actual output partition, not a shape-specific admission list.
    # Wider grids receive less credit for duplicating QK and softmax.
    slices = (head_size + 255) // 256
    slice_credit = (
        (slices - 1)
        * min(1.0, gpu_core_count / num_heads)
        * (work.full_tiles + work.partial_tiles)
    )
    tail_cost = 8 * work.partial_tiles / (width * width)
    return int(full_credit + slice_credit >= tail_cost + startup_margin)


def nax_split_kwargs(
    capabilities: Mapping[str, bool],
    work: NaxTileWork | None,
    head_size: int,
    *,
    prefill_work: NaxTileWork | None = None,
    gpu_core_count: int = 0,
    num_heads: int = 0,
) -> dict[str, int]:
    """``nax_split_hint`` keyword for one prefill ``paged_attention_primitive``.

    Empty when the native build predates the hint.  Without per-sequence
    metadata (``work`` is None) the original loop is used. A separate hint
    for the prefill subset is consumed only when native splits the batch.
    """
    if not capabilities.get("nax_split_hint", False):
        return {}

    def hint(batch_work: NaxTileWork | None) -> int:
        if batch_work is None:
            return 0
        return nax_split_hint(
            batch_work,
            head_size,
            gpu_core_count=gpu_core_count,
            num_heads=num_heads,
        )

    kwargs = {"nax_split_hint": hint(work)}
    if capabilities.get("nax_prefill_split_hint", False) and prefill_work is not None:
        kwargs["nax_prefill_split_hint"] = hint(prefill_work)
    return kwargs
