# SPDX-License-Identifier: Apache-2.0
"""Paged attention shared utilities — context, prepare functions, and helpers.

Provides the thread-local ``PagedAttentionContext`` and ``OffsetCache`` used by
both the Metal kernel paged attention backend and the model runner.

Usage:
    1. Before each forward pass call ``prepare_prefill_packed()`` or ``prepare_decode()``
    2. Run ``model(input_ids, cache=offset_caches)`` as normal
    3. The attention wrapper reads ``get_context()`` to decide prefill vs decode
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
# instead: prepare_prefill_packed/decode() stashes context here before the
# forward pass, each attention wrapper reads it via get_context(), and
# clear_context() cleans up afterwards.
_thread_local = threading.local()


@dataclass
class PagedAttentionContext:
    """Context set before each forward pass, read by patched attention."""

    is_prefill: bool
    slot_mapping: list[int]
    # decode-only fields
    block_tables: list[list[int]] = field(default_factory=list)
    context_lens: list[int] = field(default_factory=list)
    offsets: list[int] = field(default_factory=list)
    # packed prefill fields — set when multiple requests are packed into
    # a single forward pass.  cu_seqlens is a cumulative sequence length
    # array: [0, len0, len0+len1, ...] (length = num_requests + 1).
    cu_seqlens: list[int] | None = None


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


def find_layers_and_attr(model: Any) -> tuple[list[Any], str]:
    """Find transformer layers and the attention attribute name.

    Returns (layer_list, attn_attr_name) where each layer has
    getattr(layer, attn_attr_name) pointing to the attention module.

    Supports mlx_lm model structures like:
        model.language_model.model.layers[i].self_attn   (VLMs)
        model.model.layers[i].self_attn
        model.layers[i].self_attn
    """
    # Unwrap VLM wrapper (e.g. LLaVA, Pixtral via mlx-vlm)
    root = getattr(model, "language_model", model)
    # Try root.model.layers (Qwen3 Model wrapper)
    layers_container = getattr(root, "model", root)
    if hasattr(layers_container, "layers"):
        layer_list = layers_container.layers
    elif hasattr(root, "layers"):
        layer_list = root.layers
    else:
        raise ValueError(
            f"Cannot find transformer layers in model of type {type(model)}"
        )

    # Determine attribute name
    if layer_list:
        sample = layer_list[0]
        if hasattr(sample, "self_attn"):
            return layer_list, "self_attn"
        elif hasattr(sample, "attention"):
            return layer_list, "attention"
        else:
            raise ValueError(f"Cannot find attention module in layer {type(sample)}")
    return layer_list, "self_attn"


# ---------------------------------------------------------------------------
# Prepare functions — called before each forward pass
# ---------------------------------------------------------------------------


def prepare_prefill_packed(
    requests: list[tuple[list[int], int]],
    block_size: int,
) -> None:
    """Compute slot_mapping, cu_seqlens, block_tables, context_lens for prefill.

    Packs one or more prefill requests into a single forward pass.  The
    varlen Metal kernel uses ``cu_seqlens`` to locate each sequence's
    query tokens and ``block_tables`` / ``context_lens`` to read K/V
    from the paged cache.

    Args:
        requests: list of (block_ids, num_tokens) per request.
        block_size: tokens per block.
    """
    slot_mapping: list[int] = []
    cu_seqlens: list[int] = [0]
    block_tables: list[list[int]] = []
    context_lens: list[int] = []

    for block_ids, num_tokens in requests:
        for pos in range(num_tokens):
            block_idx = block_ids[pos // block_size]
            slot = block_idx * block_size + (pos % block_size)
            slot_mapping.append(slot)
        cu_seqlens.append(cu_seqlens[-1] + num_tokens)
        block_tables.append(block_ids)
        context_lens.append(num_tokens)

    set_context(
        PagedAttentionContext(
            is_prefill=True,
            slot_mapping=slot_mapping,
            block_tables=block_tables,
            context_lens=context_lens,
            cu_seqlens=cu_seqlens,
        )
    )


def prepare_decode(
    requests: list[tuple[list[int], int]],
    block_size: int,
) -> None:
    """Compute slot_mapping, block_tables, context_lens, offsets for decode.

    Args:
        requests: list of (block_ids, seq_len) per request.
                  seq_len = number of tokens already stored (before this step).
        block_size: tokens per block
    """
    slot_mapping: list[int] = []
    block_tables: list[list[int]] = []
    context_lens: list[int] = []
    offsets: list[int] = []

    for block_ids, seq_len in requests:
        # Slot for the new token at position seq_len
        new_pos = seq_len
        block_idx = block_ids[new_pos // block_size]
        slot = block_idx * block_size + (new_pos % block_size)
        slot_mapping.append(slot)
        block_tables.append(block_ids)
        context_lens.append(seq_len + 1)  # including new token
        offsets.append(seq_len)  # RoPE position = seq_len

    set_context(
        PagedAttentionContext(
            is_prefill=False,
            slot_mapping=slot_mapping,
            block_tables=block_tables,
            context_lens=context_lens,
            offsets=offsets,
        )
    )
