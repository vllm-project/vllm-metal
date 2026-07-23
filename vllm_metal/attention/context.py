# SPDX-License-Identifier: Apache-2.0
"""Per-step paged-attention context.

The thread-local ``PagedAttentionContext`` carries per-request metadata
(slot_mapping, block_tables, cu_seqlens, …) to the attention wrappers buried
inside the model; ``OffsetCache`` is the shim that keeps mlx_lm's RoPE and
masking working with no real K/V; ``prepare_grouped`` stages the context before
a forward pass.

Usage:
    1. Before each forward pass call ``prepare_grouped()``
    2. Run ``model(input_ids, cache=offset_caches)`` as normal
    3. The attention wrapper reads ``get_context()`` for paged metadata
    4. Call ``clear_context()`` after the forward pass
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from mlx_lm.models.base import create_causal_mask

# ---------------------------------------------------------------------------
# Global context (thread-local)
# ---------------------------------------------------------------------------

# Thread-local storage used to pass per-request metadata (slot_mapping,
# block_tables, etc.) to attention wrappers buried inside the model.
# We cannot add extra arguments to the mlx_lm forward signature, so
# instead: prepare_grouped() stashes context here before the
# forward pass, each attention wrapper reads it via get_context(), and
# clear_context() cleans up afterwards.
_thread_local = threading.local()


@dataclass
class PagedKVGroupContext:
    """Per-cache-group paged metadata for one packed forward pass."""

    slot_mapping: list[int]
    block_tables: list[list[int]]
    block_size: int


@dataclass
class PagedAttentionContext:
    """Context set before each forward pass, read by patched attention.

    All forward passes use the varlen kernel with ``cu_seqlens`` to handle
    variable-length subsequences (both prefill and decode tokens packed
    into a single flat sequence).
    """

    slot_mapping: list[int]
    block_tables: list[list[int]] = field(default_factory=list)
    context_lens: list[int] = field(default_factory=list)
    # Per-segment RoPE offsets: 0 for fresh prefill, seq_len for decode.
    offsets: list[int] = field(default_factory=list)
    # Cumulative sequence length array: [0, len0, len0+len1, ...]
    # (length = num_requests + 1).
    cu_seqlens: list[int] | None = None
    # GDN state pool slot mapping: request batch position → stable slot ID.
    # Populated by model_runner for hybrid models; None for non-hybrid.
    gdn_slot_mapping: list[int] | None = None
    # Number of decode requests packed at the front of the batch.
    # This lets attention wrappers distinguish pure prefill from mixed prefill+decode
    # without reverse-engineering one-token segments from ``cu_seqlens``.
    num_decode_requests: int = 0
    # Per-segment caller-supplied M-RoPE positions: each entry is either
    # ``None`` (use ``offsets[i]`` with sequential arange) or an
    # ``(3, 1, seg_len)`` array.  ``None`` for the whole field skips
    # per-segment handling entirely.
    segment_positions: list[Any] | None = None
    # Longest spec-decode verification window in this batch, or 1.  Set
    # above 1 only when every multi-token segment is a decode window (no
    # prefill segments), so the kernel dispatcher may route the batch to
    # the per-token kernel's window mode; prefill batches keep the tiled
    # kernel and stay numerically identical to the non-speculative path.
    verify_window_q: int = 1
    # Per scheduler KV-cache-group slot mappings and block tables.
    # ``slot_mapping`` / ``block_tables`` above mirror group zero for the
    # legacy single-group path.
    kv_groups: tuple[PagedKVGroupContext, ...] | None = None


def set_context(ctx: PagedAttentionContext) -> None:
    _thread_local.paged_ctx = ctx


def get_context() -> PagedAttentionContext | None:
    return getattr(_thread_local, "paged_ctx", None)


def clear_context() -> None:
    _thread_local.paged_ctx = None


# ---------------------------------------------------------------------------
# OffsetCache — thin shim so the model's create_attention_mask / RoPE work
# ---------------------------------------------------------------------------


class OffsetCache:
    """Fake KV cache that stores no data — only satisfies mlx_lm's protocol.

    The mlx_lm model expects ``cache=`` to be a list of cache objects (one
    per layer) and reads two things from each:
    - ``cache.offset``: RoPE position index for the current token(s).
    - ``cache.make_mask(N)``: attention mask (``"causal"`` for multi-token
      prefill, ``None`` for single-token decode where no mask is needed).

    With paged attention, real K/V lives in the MPS paged cache (managed by
    the attention wrapper), NOT in these objects.  OffsetCache is a shim so
    that mlx_lm's RoPE and masking logic work without changes.

    Note: during batched decode the model runner passes a single shared
    ``OffsetCache(max_offset)`` per layer.  The actual per-request RoPE
    offsets come from ``ctx.offsets`` inside the attention wrapper, not
    from this object.
    """

    def __init__(self, offset: int) -> None:
        self.offset = offset

    # --- satisfy KVCache protocol expected by create_attention_mask ---------

    def make_mask(
        self,
        N: int,  # noqa: N803
        return_array: bool = False,
        window_size: int | None = None,
    ) -> Any:
        if N == 1:
            return None
        if return_array:
            return create_causal_mask(N, self.offset, window_size=window_size)
        return "causal"


# ---------------------------------------------------------------------------
# Prepare functions — called before each forward pass
# ---------------------------------------------------------------------------


def prepare_grouped(
    decode_requests: Sequence[
        tuple[Sequence[Sequence[int]], int] | tuple[Sequence[Sequence[int]], int, int]
    ],
    prefill_requests: Sequence[tuple[Sequence[Sequence[int]], int, int]],
    block_sizes: Sequence[int],
    *,
    merge_verify_windows: bool = False,
) -> None:
    """Compute metadata for every scheduler KV-cache group in one forward pass.

    Packs decode tokens followed by prefill tokens into a single flattened
    sequence. ``cu_seqlens`` marks attention segments so the varlen kernel
    handles decode and prefill subsequences in one dispatch.

    Args:
        decode_requests: list of ``(block_ids, seq_len)`` or
            ``(block_ids, seq_len, num_tokens)`` for decode requests.
            ``block_ids`` carries one list per scheduler cache group.
            ``seq_len`` = tokens already cached before this step. ``num_tokens``
            defaults to 1; a larger value is a speculative-verification
            window, kept as one multi-token segment.
        prefill_requests: list of ``(block_ids, num_tokens, start_pos)`` for
            prefill.  ``start_pos`` is the position of the first token in this
            chunk (0 for a fresh prefill, >0 for continuation chunks).
        block_sizes: tokens per KV cache block, ordered by scheduler group.
        merge_verify_windows: keep a multi-token decode request as ONE
            cu_seqlens segment (the window-mode layout).  Defaults to
            False — the expanded per-token layout — because window mode
            is opt-in runtime policy (the runner passes its
            merge_verify_windows property); a caller must ask for the
            merged shape explicitly.  Keep False for models the window
            path does not serve: MLA native decode and the GDN layers of
            hybrids admit only one-row decode segments, and heads past
            PA_WINDOW_MAX_HEAD_SIZE would leave the decode kernel for
            the tiled one.
    """
    group_slot_mappings: list[list[int]] = [[] for _ in block_sizes]
    group_block_tables: list[list[list[int]]] = [[] for _ in block_sizes]
    cu_seqlens: list[int] = [0]
    context_lens: list[int] = []
    offsets: list[int] = []

    max_decode_window = 1

    # Decode requests first.  Multi-token decode requests are speculative
    # verification windows.  The default expanded layout keeps one segment
    # per row; the window-mode layout keeps one segment per request.
    for decode_request in decode_requests:
        if len(decode_request) == 2:
            block_ids_by_group, seq_len = decode_request
            num_tokens = 1
        else:
            block_ids_by_group, seq_len, num_tokens = decode_request

        for group_index, block_size in enumerate(block_sizes):
            block_ids = block_ids_by_group[group_index]
            block_table = list(block_ids)
            for pos in range(seq_len, seq_len + num_tokens):
                block_idx = block_ids[pos // block_size]
                group_slot_mappings[group_index].append(
                    block_idx * block_size + (pos % block_size)
                )
            if num_tokens > 1 and not merge_verify_windows:
                group_block_tables[group_index].extend(
                    block_table for _ in range(num_tokens)
                )
            else:
                group_block_tables[group_index].append(block_table)

        if num_tokens > 1 and not merge_verify_windows:
            # Expanded layout: one single-token segment per window row,
            # byte-for-byte the pre-window-mode metadata.
            for pos in range(seq_len, seq_len + num_tokens):
                cu_seqlens.append(cu_seqlens[-1] + 1)
                context_lens.append(pos + 1)  # including this decode token
                offsets.append(pos)  # RoPE position
            continue
        cu_seqlens.append(cu_seqlens[-1] + num_tokens)
        context_lens.append(seq_len + num_tokens)  # including window tokens
        offsets.append(seq_len)  # RoPE position of the first window token
        max_decode_window = max(max_decode_window, num_tokens)

    # Prefill requests (variable tokens each, starting at start_pos)
    for block_ids_by_group, num_tokens, start_pos in prefill_requests:
        for group_index, block_size in enumerate(block_sizes):
            block_ids = block_ids_by_group[group_index]
            block_table = list(block_ids)
            for pos in range(start_pos, start_pos + num_tokens):
                block_idx = block_ids[pos // block_size]
                group_slot_mappings[group_index].append(
                    block_idx * block_size + (pos % block_size)
                )
            group_block_tables[group_index].append(block_table)

        cu_seqlens.append(cu_seqlens[-1] + num_tokens)
        context_lens.append(start_pos + num_tokens)
        offsets.append(start_pos)

    kv_groups = tuple(
        PagedKVGroupContext(
            slot_mapping=group_slot_mappings[group_index],
            block_tables=group_block_tables[group_index],
            block_size=block_size,
        )
        for group_index, block_size in enumerate(block_sizes)
    )

    set_context(
        PagedAttentionContext(
            slot_mapping=kv_groups[0].slot_mapping,
            block_tables=kv_groups[0].block_tables,
            context_lens=context_lens,
            cu_seqlens=cu_seqlens,
            offsets=offsets,
            num_decode_requests=len(decode_requests),
            # Window routing applies only to pure-verification batches;
            # any prefill segment keeps the whole batch on the tiled
            # kernel, numerically identical to the non-speculative path.
            verify_window_q=1 if prefill_requests else max_decode_window,
            kv_groups=kv_groups,
        )
    )


def prepare_unified(
    decode_requests: list[tuple[list[int], int] | tuple[list[int], int, int]],
    prefill_requests: list[tuple[list[int], int, int]],
    block_size: int,
    *,
    merge_verify_windows: bool = False,
) -> None:
    """Prepare one legacy KV-cache group through :func:`prepare_grouped`."""
    grouped_decode: list[
        tuple[list[list[int]], int] | tuple[list[list[int]], int, int]
    ] = []
    for decode_request in decode_requests:
        if len(decode_request) == 2:
            block_ids, seq_len = decode_request
            grouped_decode.append(([block_ids], seq_len))
        else:
            block_ids, seq_len, num_tokens = decode_request
            grouped_decode.append(([block_ids], seq_len, num_tokens))

    grouped_prefill = [
        ([block_ids], num_tokens, start_pos)
        for block_ids, num_tokens, start_pos in prefill_requests
    ]
    prepare_grouped(
        grouped_decode,
        grouped_prefill,
        (block_size,),
        merge_verify_windows=merge_verify_windows,
    )
