# SPDX-License-Identifier: Apache-2.0
"""ShortConv (LFM2-style) causal-conv layers with paged state dispatch.

Decomposes the mlx_lm LFM2 ``ShortConv`` module's forward pass and runs the
stateful causal conv per request against a paged state pool, mirroring how
``GDNPagedAttentionWrapper`` decomposes the GDN module:

1. ``in_proj`` + gate (``Bx = B * x``) — stateless, batched
2. Causal conv over ``Bx`` with per-request state — stateful, per segment
3. ``y = C * conv_out`` + ``out_proj`` — stateless, batched

Pure MLX ops throughout; pure-decode steps run one batched gather / conv /
scatter, prefill-containing batches keep the per-segment Python loop.  When
no ``PagedAttentionContext`` is active, delegates to the original module
unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from vllm_metal.attention.caches.shortconv_cache import ShortConvStateCache
from vllm_metal.attention.context import PagedAttentionContext, get_context


def is_shortconv(module: nn.Module) -> bool:
    """Return True if *module* is an LFM2-style ShortConv layer.

    Checks for the depthwise causal conv (``conv``) together with the
    ``in_proj`` / ``out_proj`` gate projections and the ``L_cache`` kernel
    size, and the absence of ``q_proj`` (which would indicate SDPA).  GDN
    linear attention names its conv ``conv1d`` and so does not match.
    """
    return (
        hasattr(module, "conv")
        and hasattr(module, "in_proj")
        and hasattr(module, "out_proj")
        and hasattr(module, "L_cache")
        and not hasattr(module, "q_proj")
    )


@dataclass(frozen=True)
class _ShortConvForwardState:
    cu_seqlens: list[int]
    num_requests: int
    slot_ids: list[int]
    num_decode_requests: int


class ShortConvPagedWrapper(nn.Module):
    """Wraps an LFM2 ShortConv module for paged state dispatch."""

    def __init__(
        self,
        inner: nn.Module,
        cache_idx: int,
        state_cache: ShortConvStateCache,
    ) -> None:
        super().__init__()
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_state_cache_idx", cache_idx)
        object.__setattr__(self, "_state_cache", state_cache)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: nn.Module | None = None,
        **kwargs: Any,
    ) -> mx.array:
        ctx = get_context()
        if ctx is None:
            return self._inner(x, mask=mask, cache=cache)

        state = self._prepare_forward_state(ctx)
        # === Step 1: input projection + gate (stateless, batched) ===
        # Mirrors mlx_lm lfm2.ShortConv.__call__: BCx -> split -> Bx = B * x.
        # The packed layout has no padding rows, so the mlx_lm ssm mask
        # (which zeroes padded Bx rows) is not needed here.
        inner = self._inner
        bcx = inner.in_proj(x)
        b_gate, c_gate, x_gate = mx.split(bcx, 3, axis=-1)
        bx = b_gate * x_gate

        # === Step 2: per-request stateful causal conv ===
        conv_out = self._run_conv(bx, state)

        # === Step 3: output gate + projection (stateless, batched) ===
        y = c_gate * conv_out
        return inner.out_proj(y)

    def _prepare_forward_state(
        self, ctx: PagedAttentionContext
    ) -> _ShortConvForwardState:
        cu_seqlens = ctx.cu_seqlens
        if cu_seqlens is None or len(cu_seqlens) < 2:
            raise RuntimeError("ShortConv wrapper requires cu_seqlens in context")

        num_requests = len(cu_seqlens) - 1
        # Conv hybrids run mamba_cache_mode="none", whose contract is the flat
        # per-request slot mapping (align mode's per-group mappings arrive with
        # conv prefix caching).
        slot_ids = ctx.gdn_slot_mapping
        if slot_ids is None:
            raise RuntimeError("ShortConv wrapper requires gdn_slot_mapping in context")
        if len(slot_ids) != num_requests:
            raise RuntimeError("ShortConv wrapper requires one slot per request")
        if len(set(slot_ids)) != len(slot_ids):
            raise RuntimeError("ShortConv wrapper requires unique slots per request")
        self._state_cache.require_allocated_slots(slot_ids)

        return _ShortConvForwardState(
            cu_seqlens=cu_seqlens,
            num_requests=num_requests,
            slot_ids=slot_ids,
            num_decode_requests=ctx.num_decode_requests,
        )

    def _run_conv(self, bx: mx.array, state: _ShortConvForwardState) -> mx.array:
        # INVARIANT: the new conv state is written into the stable pool during
        # the forward, with no pending-state staging.  This is only sound while
        # every forward's tokens are committed — conv hybrids reject
        # speculative decoding (whose rejected drafts would need state
        # rollback) and none-mode never replays a step.  Revisit if either
        # constraint is lifted.
        inner = self._inner
        state_cache = self._state_cache
        cache_idx = self._state_cache_idx
        n_keep = inner.L_cache - 1

        pool = state_cache.conv_states[cache_idx]
        num_requests = state.num_requests

        # Pure-decode fast path: every segment is a single token, so one
        # batched gather / conv / scatter replaces the per-request Python
        # loop (O(B) kernel launches per layer per step otherwise).
        if state.num_decode_requests == num_requests and state.cu_seqlens == list(
            range(num_requests + 1)
        ):
            slots_arr = mx.array(state.slot_ids, dtype=mx.int32)
            states = pool[slots_arr]  # [B, K-1, D]
            conv_input = mx.concatenate(
                [states, bx.reshape(num_requests, 1, -1)], axis=1
            )
            pool[slots_arr] = conv_input[:, 1:, :]
            state_cache.conv_states[cache_idx] = pool
            conv_out = inner.conv(conv_input)[:, -1:, :]
            return conv_out.reshape(1, num_requests, -1)

        conv_outputs: list[mx.array] = []
        for req_idx in range(num_requests):
            slot = state.slot_ids[req_idx]
            start = state.cu_seqlens[req_idx]
            end = state.cu_seqlens[req_idx + 1]
            seg_bx = bx[:, start:end, :]

            # Load conv state from the stable slab, prepend, run the conv.
            conv_state = pool[slot : slot + 1]
            conv_input = mx.concatenate([conv_state, seg_bx], axis=1)

            # Keep the last L_cache - 1 rows of Bx as the new state.
            pool[slot : slot + 1] = conv_input[:, -n_keep:, :]

            conv_out = inner.conv(conv_input)
            # Take only the output tokens (not the conv state prefix).
            conv_outputs.append(conv_out[:, -(end - start) :, :])

        state_cache.conv_states[cache_idx] = pool
        return mx.concatenate(conv_outputs, axis=1)
