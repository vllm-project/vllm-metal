# SPDX-License-Identifier: Apache-2.0
"""Paged attention shared utilities — context, prepare functions, and helpers.

Provides the thread-local ``PagedAttentionContext`` and ``OffsetCache`` used by
both the Metal kernel paged attention backend and the model runner.

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
# Model introspection
# ---------------------------------------------------------------------------


def find_layers(model: Any) -> list[Any]:
    """Find transformer layers in an mlx_lm / mlx-vlm model.

    Supports model structures like:
        model.language_model.model.layers   (VLMs)
        model.model.layers
        model.layers
    """
    # Unwrap VLM wrapper (e.g. LLaVA, Pixtral via mlx-vlm)
    root = getattr(model, "language_model", model)
    # Try root.model.layers (Qwen3 Model wrapper)
    layers_container = getattr(root, "model", root)
    if hasattr(layers_container, "layers"):
        return layers_container.layers
    elif hasattr(root, "layers"):
        return root.layers
    else:
        raise ValueError(
            f"Cannot find transformer layers in model of type {type(model)}"
        )


# Attribute names to probe on each layer, in priority order.
_ATTN_ATTR_NAMES = ("self_attn", "linear_attn", "attention")


def find_attn_attr(layer: Any) -> str | None:
    """Return the attention attribute name for a single layer, or None."""
    for name in _ATTN_ATTR_NAMES:
        if hasattr(layer, name):
            return name
    return None


# ---------------------------------------------------------------------------
# Prepare functions — called before each forward pass
# ---------------------------------------------------------------------------


def prepare_unified(
    decode_requests: list[tuple[list[int], int]],
    prefill_requests: list[tuple[list[int], int, int]],
    block_size: int,
) -> None:
    """Compute metadata for a unified prefill + decode forward pass.

    Packs decode tokens (1 per request) followed by prefill tokens into a
    single flattened sequence.  ``cu_seqlens`` marks request boundaries so
    the varlen kernel handles both decode (length-1) and prefill (length-N)
    subsequences in one dispatch.

    Args:
        decode_requests: list of ``(block_ids, seq_len)`` for decode requests.
            ``seq_len`` = tokens already cached before this step.
        prefill_requests: list of ``(block_ids, num_tokens, start_pos)`` for
            prefill.  ``start_pos`` is the position of the first token in this
            chunk (0 for a fresh prefill, >0 for continuation chunks).
        block_size: tokens per KV cache block.
    """
    slot_mapping: list[int] = []
    cu_seqlens: list[int] = [0]
    block_tables: list[list[int]] = []
    context_lens: list[int] = []
    offsets: list[int] = []

    # Decode requests first (1 token each)
    for block_ids, seq_len in decode_requests:
        new_pos = seq_len
        block_idx = block_ids[new_pos // block_size]
        slot = block_idx * block_size + (new_pos % block_size)
        slot_mapping.append(slot)
        cu_seqlens.append(cu_seqlens[-1] + 1)
        block_tables.append(block_ids)
        context_lens.append(seq_len + 1)  # including new token
        offsets.append(seq_len)  # RoPE position

    # Prefill requests (variable tokens each, starting at start_pos)
    for block_ids, num_tokens, start_pos in prefill_requests:
        for pos in range(start_pos, start_pos + num_tokens):
            block_idx = block_ids[pos // block_size]
            slot = block_idx * block_size + (pos % block_size)
            slot_mapping.append(slot)
        cu_seqlens.append(cu_seqlens[-1] + num_tokens)
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
        )
    )
