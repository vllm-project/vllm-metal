# SPDX-License-Identifier: Apache-2.0
"""Paged attention shared utilities — context, prepare functions, and helpers.

Provides the thread-local ``PagedAttentionContext`` and ``OffsetCache`` used by
both the Metal kernel paged attention backend and the model runner.

Usage:
    1. Before each forward pass call ``prepare_prefill()`` or ``prepare_decode()``
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
    """Minimal cache-like object that provides ``offset`` and ``make_mask``.

    The mlx_lm model reads ``cache.offset`` for RoPE and calls
    ``cache.make_mask(N)`` (delegated from ``create_attention_mask``).
    We satisfy both without storing any KV data.
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


def _find_layers_and_attr(model: Any) -> tuple[list[Any], str]:
    """Find transformer layers and the attention attribute name.

    Returns (layer_list, attn_attr_name) where each layer has
    getattr(layer, attn_attr_name) pointing to the attention module.

    Supports mlx_lm model structures like:
        model.model.layers[i].self_attn
        model.layers[i].self_attn
    """
    # Try model.model.layers (Qwen3 Model wrapper)
    layers_container = getattr(model, "model", model)
    if hasattr(layers_container, "layers"):
        layer_list = layers_container.layers
    elif hasattr(model, "layers"):
        layer_list = model.layers
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


def prepare_prefill(
    block_ids: list[int],
    num_tokens: int,
    block_size: int,
) -> None:
    """Compute slot_mapping for prefill and set global context."""
    slot_mapping = []
    for pos in range(num_tokens):
        block_idx = block_ids[pos // block_size]
        slot = block_idx * block_size + (pos % block_size)
        slot_mapping.append(slot)

    set_context(
        PagedAttentionContext(
            is_prefill=True,
            slot_mapping=slot_mapping,
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
