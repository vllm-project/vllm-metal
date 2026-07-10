# SPDX-License-Identifier: Apache-2.0
"""Per-step paged-attention context.

The thread-local ``PagedAttentionContext`` carries per-request metadata
(slot_mapping, block_tables, cu_seqlens, …) to the attention wrappers buried
inside the model; ``OffsetCache`` is the shim that keeps mlx_lm's RoPE and
masking working with no real K/V; ``prepare_unified`` stages the context before
a forward pass.

Usage:
    1. Before each forward pass call ``prepare_unified()``
    2. Run ``model(input_ids, cache=offset_caches)`` as normal
    3. The attention wrapper reads ``get_context()`` for paged metadata
    4. Call ``clear_context()`` after the forward pass
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from mlx_lm.models.base import create_causal_mask

# ---------------------------------------------------------------------------
# Global context (thread-local)
# ---------------------------------------------------------------------------

# Thread-local storage used to pass per-request metadata (slot_mapping,
# block_tables, etc.) to attention wrappers buried inside the model.
# We cannot add extra arguments to the mlx_lm forward signature, so
# instead: prepare_unified() stashes context here before the
# forward pass, each attention wrapper reads it via get_context(), and
# clear_context() cleans up afterwards.
_thread_local = threading.local()


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
# Slot mapping helpers
# ---------------------------------------------------------------------------


def _slot_for_pos(block_ids: list[int], pos: int, block_size: int) -> int:
    block_index, block_offset = divmod(pos, block_size)
    return block_ids[block_index] * block_size + block_offset


def _has_contiguous_physical_blocks(
    block_ids: list[int],
    start_block: int,
    end_block: int,
) -> bool:
    expected = block_ids[start_block]
    for table_idx in range(start_block + 1, end_block + 1):
        expected += 1
        if block_ids[table_idx] != expected:
            return False
    return True


def _append_slot_range(
    slot_mapping: list[int],
    block_ids: list[int],
    start_pos: int,
    num_tokens: int,
    block_size: int,
) -> None:
    """Append slots for a contiguous logical token span.

    Most prefill chunks in practice are backed by physically contiguous KV
    blocks.  In that case, every token slot is one arithmetic range even when
    the logical span crosses block boundaries, so avoid the per-token loop from
    the original implementation.
    """
    if num_tokens <= 0:
        return

    start_block, block_offset = divmod(start_pos, block_size)
    start_slot = block_ids[start_block] * block_size + block_offset

    if num_tokens == 1:
        slot_mapping.append(start_slot)
        return

    end_block = (start_pos + num_tokens - 1) // block_size
    if _has_contiguous_physical_blocks(block_ids, start_block, end_block):
        slot_mapping.extend(range(start_slot, start_slot + num_tokens))
        return

    pos = start_pos
    remaining = num_tokens
    while remaining:
        block_index, block_offset = divmod(pos, block_size)
        chunk = min(block_size - block_offset, remaining)
        start_slot = block_ids[block_index] * block_size + block_offset
        slot_mapping.extend(range(start_slot, start_slot + chunk))
        pos += chunk
        remaining -= chunk


# ---------------------------------------------------------------------------
# Prepare functions — called before each forward pass
# ---------------------------------------------------------------------------


def prepare_unified(
    decode_requests: list[tuple[list[int], int] | tuple[list[int], int, int]],
    prefill_requests: list[tuple[list[int], int, int]],
    block_size: int,
) -> None:
    """Compute metadata for a unified prefill + decode forward pass.

    Packs decode tokens followed by prefill tokens into a single flattened
    sequence. ``cu_seqlens`` marks attention segments so the varlen kernel
    handles decode and prefill subsequences in one dispatch.

    Args:
        decode_requests: list of ``(block_ids, seq_len)`` or
            ``(block_ids, seq_len, num_tokens)`` for decode requests.
            ``seq_len`` = tokens already cached before this step. ``num_tokens``
            defaults to 1 and expands the decode row span for speculative
            verification.
        prefill_requests: list of ``(block_ids, num_tokens, start_pos)`` for
            prefill.  ``start_pos`` is the position of the first token in this
            chunk (0 for a fresh prefill, >0 for continuation chunks).
        block_size: tokens per KV cache block.
    """
    single_token_decode_only = not prefill_requests
    for decode_request in decode_requests:
        if len(decode_request) == 3:
            num_tokens = decode_request[2]
            if num_tokens != 1:
                single_token_decode_only = False

    num_decode_requests = len(decode_requests)

    # Very common serving hot path: pure batched decode with one token per
    # request.  A 3-tuple with num_tokens == 1 is metadata-equivalent to the
    # 2-tuple form, so keep it on this fast path.
    if single_token_decode_only:
        n = num_decode_requests
        slot_mapping = [0] * n
        cu_seqlens = list(range(n + 1))
        block_tables: list[list[int]] = []
        context_lens = [0] * n
        offsets = [0] * n

        for i, decode_request in enumerate(decode_requests):
            block_ids = decode_request[0]
            seq_len = decode_request[1]
            slot_mapping[i] = _slot_for_pos(block_ids, seq_len, block_size)
            block_tables.append(block_ids)
            context_lens[i] = seq_len + 1
            offsets[i] = seq_len

        set_context(
            PagedAttentionContext(
                slot_mapping=slot_mapping,
                block_tables=block_tables,
                context_lens=context_lens,
                cu_seqlens=cu_seqlens,
                offsets=offsets,
                num_decode_requests=num_decode_requests,
            )
        )
        return

    slot_mapping: list[int] = []
    cu_seqlens: list[int] = [0]
    block_tables: list[list[int]] = []
    context_lens: list[int] = []
    offsets: list[int] = []
    cu_len = 0

    # Decode requests first.  Multi-token decode is represented as one token
    # per varlen segment for speculative verification.
    for decode_request in decode_requests:
        if len(decode_request) == 2:
            block_ids, seq_len = decode_request
            num_tokens = 1
        else:
            block_ids, seq_len, num_tokens = decode_request

        _append_slot_range(slot_mapping, block_ids, seq_len, num_tokens, block_size)

        for pos in range(seq_len, seq_len + num_tokens):
            cu_len += 1
            cu_seqlens.append(cu_len)
            block_tables.append(block_ids)
            context_lens.append(pos + 1)
            offsets.append(pos)

    # Prefill requests (variable tokens each, starting at start_pos).
    for block_ids, num_tokens, start_pos in prefill_requests:
        _append_slot_range(slot_mapping, block_ids, start_pos, num_tokens, block_size)
        cu_len += num_tokens
        cu_seqlens.append(cu_len)
        block_tables.append(block_ids)
        context_lens.append(start_pos + num_tokens)
        offsets.append(start_pos)

    set_context(
        PagedAttentionContext(
            slot_mapping=slot_mapping,
            block_tables=block_tables,
            context_lens=context_lens,
            cu_seqlens=cu_seqlens,
            offsets=offsets,
            num_decode_requests=num_decode_requests,
        )
    )
