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
``nax_split_hint``. Full attention uses one batch work score for full and
cached prefills at every supported head size. Per-sequence loop depth keeps
many short requests distinct from one long request with the same tile count.
Native dispatch decides whether to use the whole-batch or prefill-subset score.
Windowed attention and callers missing layout metadata retain the legacy rule.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import sqrt
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
    # Sum of each sequence's mean KV-loop length over its complete Q blocks.
    full_stream_tiles: float = 0.0
    # Cached page footprint counted again by another complete-block stream.
    repeated_prefix_tiles: float = 0.0


def nax_gpu_core_count(ops: Any) -> int:
    """Detected capacity, without a device-name policy or guessed fallback."""
    detect = getattr(ops, "detected_gpu_core_count", None)
    return int(detect()) if detect is not None else 0


def _tiles(kv_end: int) -> int:
    return (kv_end + _TILE_KV - 1) // _TILE_KV


def nax_tile_work(
    cu_seqlens: Sequence[int],
    kv_lens: Sequence[int],
    sliding_window: int = -1,
    *,
    block_tables: Sequence[Sequence[int]] | None = None,
    block_size: int = 0,
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
    if block_tables is not None and (
        len(block_tables) != len(kv_lens) or block_size <= 0
    ):
        raise ValueError("block tables require one row per sequence and a block size")

    def first_tile(q_abs: int) -> int:
        if sliding_window < 0:
            return 0
        return max(0, (q_abs + 1 - sliding_window) // _TILE_KV)

    full_blocks = full_tiles = partial_tiles = sequences = query_tokens = 0
    full_stream_tiles = 0.0
    repeated_prefix_tiles = 0.0
    prefix_pages: set[int] = set()
    can_share = (
        sliding_window < 0
        and sum(cu_seqlens[i + 1] - cu_seqlens[i] >= _BQ for i in range(len(kv_lens)))
        > 1
    )
    for i, kv in enumerate(kv_lens):
        q = cu_seqlens[i + 1] - cu_seqlens[i]
        if q <= 0:
            continue
        context = kv - q
        sequences += 1
        query_tokens += q
        n_full = q // _BQ
        full_blocks += n_full
        sequence_full_tiles = 0
        for j in range(n_full):
            start = context + j * _BQ
            sequence_full_tiles += _tiles(start + _BQ) - first_tile(start)
        full_tiles += sequence_full_tiles
        if n_full:
            full_stream_tiles += sequence_full_tiles / n_full
            if block_tables is not None and can_share:
                # Only complete cached pages can be shared. Count them in the
                # existing CPU layout, before translation to kernel pages.
                count = context // block_size
                if len(block_tables[i]) < count:
                    raise ValueError("block table does not cover the cached prefix")
                for page in block_tables[i][:count]:
                    if page in prefix_pages:
                        repeated_prefix_tiles += block_size / _TILE_KV
                    else:
                        prefix_pages.add(page)
        if q % _BQ:
            partial_tiles += _tiles(kv) - first_tile(context + n_full * _BQ)
    return NaxTileWork(
        full_blocks,
        full_tiles,
        partial_tiles,
        sequences,
        query_tokens,
        sliding_window,
        full_stream_tiles,
        repeated_prefix_tiles,
    )


def nax_split_hint(
    work: NaxTileWork,
    head_size: int,
    *,
    gpu_core_count: int = 0,
    num_heads: int = 0,
    num_kv_heads: int = 0,
) -> int:
    """1 to use the split pipeline, 0 for the original loop."""
    if head_size > 512:
        return 0
    if (
        num_heads <= 0
        or gpu_core_count <= 0
        or num_kv_heads <= 0
        or work.query_tokens <= work.sequences
        or work.sliding_window >= 0
    ):
        factor = _MIN_FULL_TO_PARTIAL.get(head_size, _MIN_FULL_TO_PARTIAL_DEFAULT)
        return int(
            work.full_blocks >= MIN_FULL_BLOCKS
            and work.full_tiles >= factor * work.partial_tiles
        )

    # A calibrated work score, not a latency prediction. The complete-block
    # grid and actual output slices share one capacity estimate across shapes.
    width = max(head_size, 128) / 128
    slices = (head_size + 255) // 256
    parallelism = width * num_heads * work.full_blocks / gpu_core_count + 3 * (
        slices - 1
    )
    if parallelism <= 0:
        return 0

    slice_credit = (slices - 1) * min(1.0, gpu_core_count / num_heads)
    benefit = (
        width
        * width
        * (work.full_tiles + slice_credit * (work.full_tiles + work.partial_tiles))
    )
    reuse = sqrt(num_heads / num_kv_heads) + 1
    # A larger grid cannot make the guarded KV traffic disappear.
    guarded_cost = (
        8
        * work.partial_tiles
        * max(3 * (1 + num_kv_heads / num_heads) / parallelism, 1 / reuse)
    )
    # This is a working-set estimate, not a claim that cache lines are fetched
    # only once. Shared cached pages occupy one physical stream footprint.
    stream_tiles = work.full_stream_tiles - work.repeated_prefix_tiles
    # Stream bytes scale with D; cap the capacity discount at saturation.
    stream_cost = 8 * (head_size / 128) * stream_tiles / (reuse * min(3.0, parallelism))
    return int(benefit > max(guarded_cost, stream_cost))


def nax_split_kwargs(
    capabilities: Mapping[str, bool],
    work: NaxTileWork | None,
    head_size: int,
    *,
    prefill_work: NaxTileWork | None = None,
    gpu_core_count: int = 0,
    num_heads: int = 0,
    num_kv_heads: int = 0,
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
            num_kv_heads=num_kv_heads,
        )

    kwargs = {"nax_split_hint": hint(work)}
    if capabilities.get("nax_prefill_split_hint", False) and prefill_work is not None:
        kwargs["nax_prefill_split_hint"] = hint(prefill_work)
    return kwargs
