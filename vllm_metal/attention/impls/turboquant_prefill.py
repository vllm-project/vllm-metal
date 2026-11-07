# SPDX-License-Identifier: Apache-2.0
"""Bounded TurboQuant prefill admission using scheduler-owned page metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import mlx.core as mx
from vllm.logger import init_logger

from vllm_metal.attention.caches.turboquant import (
    FWHT_SUPPORTED_HEAD_DIMS,
    QUANT_PARAMS,
    prefill_bytes_per_token,
)
from vllm_metal.attention.context import PagedAttentionContext
from vllm_metal.attention.nax_split import NaxTileWork, nax_tile_work
from vllm_metal.metal.constants import KERNEL_BLOCK_SIZES

if TYPE_CHECKING:
    from vllm_metal.attention.impls.sdpa import _KernelMetadata

logger = init_logger(__name__)

# Dequantizing cached context must be amortized by enough query rows. Keep
# short suffixes (including prefix hits) on the compressed kernel. MHA needs
# more rows because it cannot amortize dequant across grouped query heads.
# See tools/benchmark/tq_lane_verify.py's production-path crossover sweep.
_TQ_MIN_PREFILL_TOKENS = 128
_TQ_MIN_QUERIES_PER_KV_HEAD = 256
# The hd128 crossover was measured for signed K8/V3 and these head geometries.
# Keep other formats, short histories and unmeasured shapes on the old policy.
_TQ_HD128_CALIBRATED_GEOMETRIES = frozenset(
    {(8, 2, 128), (8, 8, 128), (32, 8, 128), (16, 2, 128)}
)
_TQ_HD128_MIN_CONTEXT = 8192
_TQ_HD128_MIN_PREFILL_TOKENS = 64


def min_prefill_tokens(
    num_query_heads: int,
    num_kv_heads: int,
    head_dim: int,
    *,
    context_len: int | None = None,
    key_quant_type: str | None = None,
    value_bits: int | None = None,
) -> int:
    """Minimum new queries for a geometry and total visible KV context.

    An unspecified context or format retains the original policy. For a fixed
    geometry and format, the threshold never increases with context; workspace
    sizing relies on the threshold at max_model_len being the smallest used.
    """
    key_format = QUANT_PARAMS.get(key_quant_type or "")
    if (
        context_len is not None
        and context_len >= _TQ_HD128_MIN_CONTEXT
        and (num_query_heads, num_kv_heads, head_dim) in _TQ_HD128_CALIBRATED_GEOMETRIES
        and key_format is not None
        and key_format["bits"] == 8
        and key_format["signed"]
        and value_bits == 3
    ):
        return _TQ_HD128_MIN_PREFILL_TOKENS
    return max(
        _TQ_MIN_PREFILL_TOKENS,
        # Wide-head tiled prefill needs more rows to amortize materialization.
        head_dim // 2,
        (_TQ_MIN_QUERIES_PER_KV_HEAD * num_kv_heads + num_query_heads - 1)
        // num_query_heads,
    )


def dtype_head_reason(dtype: mx.Dtype, head_dim: int) -> str | None:
    """Why a model's dtype/head_dim rule out compressed-KV prefill, or ``None``."""
    if dtype not in (mx.bfloat16, mx.float16):
        return f"activation dtype {dtype} requires FP16/BF16"
    if head_dim not in FWHT_SUPPORTED_HEAD_DIMS:
        return f"head_dim {head_dim} is unsupported"
    return None


def unsupported_reason(
    *,
    dtype: mx.Dtype,
    head_dim: int,
    kernel_block_size: int,
    cache_block_size: int,
    stored_block_size: int,
) -> str | None:
    if reason := dtype_head_reason(dtype, head_dim):
        return reason
    if (
        kernel_block_size not in KERNEL_BLOCK_SIZES
        or stored_block_size != cache_block_size
        or cache_block_size % kernel_block_size
    ):
        return (
            f"cache layout (stored={stored_block_size}, scheduler={cache_block_size}, "
            f"kernel={kernel_block_size}) is unsupported"
        )
    return None


def _routing_bytes(queries: int, num_query_heads: int, head_dim: int) -> int:
    # Splitting also gathers Q and concatenates/restores the outputs. Charge
    # those three FP16/BF16 copies plus query-gather and output-restore indices.
    query_row_bytes = num_query_heads * head_dim * 2  # FP16/BF16 element width.
    return queries * (3 * query_row_bytes + 2 * 4)


def _split_metadata_bytes(
    num_sequences: int, fallback_count: int, table_width: int
) -> int:
    # Split seq_lens have N int32s; the two cu_seqlens_q have N + 2.
    # Each fallback needs a padded table row and its int32 gather index.
    return (2 * num_sequences + 2) * 4 + fallback_count * (table_width + 1) * 4


def workspace_upper_bound(
    *,
    max_model_len: int,
    max_num_seqs: int,
    max_num_batched_tokens: int,
    num_query_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    key_quant_type: str | None = None,
    value_bits: int | None = None,
) -> int:
    """Conservative per-layer cap, including independent histories and split copies.

    Queries are limited by a scheduler step; the historical KV each can read
    is limited by max_model_len. Layers reuse one allowance, so do not sum
    this bound over model layers. The minimum kernel page is eight tokens;
    round histories to a full scheduler page to cover every translated view.
    """
    queries = min(max_num_batched_tokens, max_num_seqs * max_model_len)
    minimum = min_prefill_tokens(
        num_query_heads,
        num_kv_heads,
        head_dim,
        context_len=max_model_len,
        key_quant_type=key_quant_type,
        value_bits=value_bits,
    )
    if max_model_len < minimum:
        return 0
    candidates = min(max_num_seqs, queries // minimum)
    if not candidates:
        return 0
    tokens = ((max_model_len + block_size - 1) // block_size) * block_size
    table_width = tokens // min(KERNEL_BLOCK_SIZES)
    materialized = candidates * (
        tokens * prefill_bytes_per_token(num_kv_heads, head_dim) + table_width * 4
    )
    if max_num_seqs == 1:
        return materialized
    routing = _routing_bytes(queries, num_query_heads, head_dim)
    split_metadata = _split_metadata_bytes(max_num_seqs, max_num_seqs, table_width)
    return materialized + routing + split_metadata


@dataclass(frozen=True, eq=False)
class _AttentionBatch:
    query_indices: mx.array | None
    block_tables: mx.array
    seq_lens: mx.array
    cu_seqlens_q: mx.array
    max_seq_len: int
    # KV-tile counts for the NAX pipeline choice (dequantized prefill only).
    nax_work: NaxTileWork | None = None


@dataclass(frozen=True, eq=False)
class _TurboQuantPrefillPlan:
    prefill: _AttentionBatch
    fallback: _AttentionBatch | None
    restore_indices: mx.array | None
    pool_pages: mx.array
    pool_offsets: mx.array
    workspace_bytes: int


def _turboquant_prefill_plan(
    ctx: PagedAttentionContext,
    meta: _KernelMetadata,
    raw_block_tables: list[list[int]],
    cache_block_size: int,
    num_query_heads: int,
    num_kv_heads: int,
    head_dim: int,
    *,
    key_quant_type: str | None = None,
    value_bits: int | None = None,
    compute_nax_work: bool = False,
) -> _TurboQuantPrefillPlan | None:
    """Plan bounded materialization using the existing CPU scheduler metadata.

    Only referenced kernel blocks are gathered; shared prefixes are deduplicated.
    Admit independent histories as well as shared prefixes while the combined
    gather fits the absolute workspace reserved before KV allocation. An
    oversized candidate is skipped before building its gather indices.
    """
    workspace_limit = meta.tq_prefill_workspace_bytes
    if not workspace_limit:
        return None
    if ctx.cu_seqlens is None:
        raise ValueError("TurboQuant prefill requires cumulative query lengths")
    # The policy is part of the key, so a swapped policy is never served
    # thresholds computed by another one.
    threshold_key = (
        min_prefill_tokens,
        num_query_heads,
        num_kv_heads,
        head_dim,
        key_quant_type,
        value_bits,
    )
    min_tokens = meta.tq_min_tokens.get(threshold_key)
    if min_tokens is None:
        min_tokens = tuple(
            min_prefill_tokens(
                num_query_heads,
                num_kv_heads,
                head_dim,
                context_len=context_len,
                key_quant_type=key_quant_type,
                value_bits=value_bits,
            )
            for context_len in ctx.context_lens
        )
        meta.tq_min_tokens[threshold_key] = min_tokens
    key = (
        cache_block_size,
        num_query_heads,
        num_kv_heads,
        head_dim,
        workspace_limit,
        compute_nax_work,
        *min_tokens,
    )
    if key in meta.tq_prefill_plans:
        return meta.tq_prefill_plans[key]
    cu_seqlens = ctx.cu_seqlens
    lengths = [b - a for a, b in zip(cu_seqlens[:-1], cu_seqlens[1:], strict=True)]
    candidates = [
        i
        for i, (length, minimum) in enumerate(zip(lengths, min_tokens, strict=True))
        if length >= minimum
    ]
    if not candidates:
        meta.tq_prefill_plans[key] = None
        return None

    kernel_bs = meta.block_size
    ratio = cache_block_size // kernel_bs
    bytes_per_block = kernel_bs * prefill_bytes_per_token(num_kv_heads, head_dim)
    routing_bytes = _routing_bytes(cu_seqlens[-1], num_query_heads, head_dim)

    def split_metadata_bytes(fallback_count: int) -> int:
        return _split_metadata_bytes(
            len(lengths), fallback_count, meta.block_tables.shape[1]
        )

    def select_rows(
        limit: int,
        *,
        split: bool,
    ) -> tuple[dict[int, int], list[int], list[list[int]], int]:
        block_limit = max(0, limit) // bytes_per_block
        source_blocks: dict[int, int] = {}
        prefill_ids: list[int] = []
        rows: list[list[int]] = []
        table_width = 0
        for i in candidates:
            num_blocks = (ctx.context_lens[i] + kernel_bs - 1) // kernel_bs
            if num_blocks > block_limit:
                continue
            sources = [
                raw_block_tables[i][j // ratio] * ratio + j % ratio
                for j in range(num_blocks)
            ]
            # Inspect only new blocks, without copying all admitted histories.
            additions = dict.fromkeys(
                source for source in sources if source not in source_blocks
            )
            next_width = max(table_width, num_blocks)
            next_bytes = (len(source_blocks) + len(additions)) * bytes_per_block
            next_bytes += (len(rows) + 1) * next_width * 4
            if split:
                next_bytes += split_metadata_bytes(len(lengths) - len(rows) - 1)
            if next_bytes > limit:
                continue
            row = []
            for source in sources:
                if source not in source_blocks:
                    source_blocks[source] = len(source_blocks)
                row.append(source_blocks[source])
            prefill_ids.append(i)
            rows.append(row)
            table_width = next_width
        return source_blocks, prefill_ids, rows, table_width

    # First allow the whole batch to fit without copies. If a split becomes
    # necessary, repeat CPU-only admission with its routing cost reserved.
    must_split = len(candidates) < len(lengths)
    source_blocks, prefill_ids, rows, table_width = select_rows(
        workspace_limit - (routing_bytes if must_split else 0), split=must_split
    )
    if not must_split and 0 < len(prefill_ids) < len(lengths):
        source_blocks, prefill_ids, rows, table_width = select_rows(
            workspace_limit - routing_bytes, split=True
        )
    if len(prefill_ids) < len(candidates):
        logger.info_once(
            "Metal: TurboQuant prefill workspace limit reached; "
            "overflow histories use compressed attention."
        )
    if not prefill_ids:
        meta.tq_prefill_plans[key] = None
        return None
    selected = set(prefill_ids)
    fallback_ids = [i for i in range(len(lengths)) if i not in selected]
    query_order: list[int] = []

    def batch(
        ids: list[int], tables: mx.array, host_tables: list[list[int]] | None = None
    ) -> _AttentionBatch:
        if not fallback_ids:
            # The whole batch keeps its original query order and metadata.
            return _AttentionBatch(
                None,
                tables,
                meta.seq_lens,
                meta.cu_seqlens_q,
                meta.max_seq_len,
                nax_tile_work(
                    cu_seqlens,
                    ctx.context_lens,
                    block_tables=host_tables,
                    block_size=kernel_bs,
                )
                if compute_nax_work and host_tables is not None
                else None,
            )
        indices = [q for i in ids for q in range(cu_seqlens[i], cu_seqlens[i + 1])]
        query_order.extend(indices)
        cu = [0]
        for i in ids:
            cu.append(cu[-1] + lengths[i])
        kv_lens = [ctx.context_lens[i] for i in ids]
        return _AttentionBatch(
            mx.array(indices, dtype=mx.int32),
            tables,
            mx.array(kv_lens, dtype=mx.int32),
            mx.array(cu, dtype=mx.int32),
            max(kv_lens),
            nax_tile_work(cu, kv_lens, block_tables=host_tables, block_size=kernel_bs)
            if compute_nax_work and host_tables is not None
            else None,
        )

    tables = mx.array([row + [0] * (table_width - len(row)) for row in rows], mx.int32)
    prefill = batch(prefill_ids, tables, rows)
    fallback = (
        batch(fallback_ids, meta.block_tables[mx.array(fallback_ids, mx.int32)])
        if fallback_ids
        else None
    )
    restore = None
    if fallback_ids:
        inverse = [0] * len(query_order)
        for packed, original in enumerate(query_order):
            inverse[original] = packed
        restore = mx.array(inverse, dtype=mx.int32)
    blocks = mx.array(list(source_blocks), dtype=mx.int32)
    # Kernel block IDs, including translated IDs, must fit int32 as required
    # by the primitive. Multiplying those IDs by kernel_bs may still overflow;
    # gather instead with scheduler-page and within-page token coordinates.
    pages = mx.repeat(blocks // ratio, kernel_bs)
    offsets = (
        (blocks % ratio)[:, None] * kernel_bs + mx.arange(kernel_bs, dtype=mx.int32)
    ).reshape(-1)
    plan = _TurboQuantPrefillPlan(
        prefill,
        fallback,
        restore,
        pages,
        offsets,
        len(source_blocks) * bytes_per_block
        + len(rows) * table_width * 4
        + (
            routing_bytes + split_metadata_bytes(len(fallback_ids))
            if fallback_ids
            else 0
        ),
    )
    meta.tq_prefill_plans[key] = plan
    return plan
