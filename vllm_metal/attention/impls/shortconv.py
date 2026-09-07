# SPDX-License-Identifier: Apache-2.0
"""LFM2 ShortConv with per-request or scheduler-block-indexed state.

The checkpoint is the last ``L_cache - 1`` rows of the gated input B*x,
before convolution. Align-mode state motion happens before this wrapper;
the wrapper writes only the destination selected for each request.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from vllm_metal.attention.caches.shortconv_cache import ShortConvStateCache
from vllm_metal.attention.context import get_context


def is_shortconv(module: nn.Module) -> bool:
    return (
        hasattr(module, "conv")
        and hasattr(module, "in_proj")
        and hasattr(module, "out_proj")
        and hasattr(module, "L_cache")
        and not hasattr(module, "q_proj")
    )


class ShortConvPagedWrapper(nn.Module):
    """Apply the native LFM2 operation independently to packed requests."""

    def __init__(
        self,
        inner: nn.Module,
        layer_idx: int,
        cache_idx: int,
        state_cache: ShortConvStateCache,
    ) -> None:
        super().__init__()
        object.__setattr__(self, "_inner", inner)
        self.rebind_state_cache(state_cache, cache_idx=cache_idx)

    def rebind_state_cache(
        self, state_cache: ShortConvStateCache, *, cache_idx: int
    ) -> None:
        object.__setattr__(self, "_state_cache", state_cache)
        object.__setattr__(self, "_cache_idx", cache_idx)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: nn.Module | None = None,
    ) -> mx.array:
        ctx = get_context()
        if ctx is None:
            return self._inner(x, mask=mask, cache=cache)

        cache_idx = self._cache_idx
        state_cache = self._state_cache
        if ctx.state_group_slot_mappings is not None:
            ordinal = state_cache.layer_group_ordinal(cache_idx)
            slots = ctx.state_group_slot_mappings[ordinal]
        else:
            slots = ctx.state_slot_mapping
        boundaries = ctx.cu_seqlens
        if slots is None or len(slots) != len(boundaries) - 1:
            raise RuntimeError("ShortConv requires one state slot per packed request")
        if len(set(slots)) != len(slots):
            raise RuntimeError("ShortConv requires distinct destination state slots")
        state_cache.require_allocated_slots(slots)

        inner = self._inner
        b_gate, c_gate, x_gate = mx.split(inner.in_proj(x), 3, axis=-1)
        bx = b_gate * x_gate
        pool = state_cache.conv_states[cache_idx]
        ids = mx.array(slots, dtype=mx.int32)
        n_keep = inner.L_cache - 1
        if boundaries == list(range(len(slots) + 1)):
            # One token per request: a single batched convolution.
            conv_input = mx.concatenate(
                [pool[ids], bx.reshape(len(slots), 1, -1)], axis=1
            )
            updates = conv_input[:, -n_keep:, :]
            conv_out = inner.conv(conv_input).reshape(1, len(slots), -1)
        else:
            outputs = []
            tails = []
            for request, slot in enumerate(slots):
                start, end = boundaries[request : request + 2]
                conv_input = mx.concatenate(
                    [pool[slot : slot + 1], bx[:, start:end, :]], axis=1
                )
                outputs.append(inner.conv(conv_input))
                tails.append(conv_input[:, -n_keep:, :])
            conv_out = mx.concatenate(outputs, axis=1)
            updates = mx.concatenate(tails, axis=0)

        # All tokens in this forward are committed. Speculative verification
        # remains rejected because it would require rolling this state back.
        state_cache.write_conv_rows(cache_idx, updates, ids)
        return inner.out_proj(c_gate * conv_out)
