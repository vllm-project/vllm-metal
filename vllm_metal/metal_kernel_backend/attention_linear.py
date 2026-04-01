# SPDX-License-Identifier: Apache-2.0
"""Linear attention (Gated DeltaNet) wrapper for paged path.

Wraps the original mlx_lm GDN module with managed recurrent state from
``GDNPagedStateCache``.  Each forward call loads the request's conv and
recurrent state, delegates to the original module, then writes back.

Uses MLX-native array indexing (same pattern as MLA backend), not a
custom C++/Metal kernel.  The original mlx_lm Metal kernel for the
GDN recurrence is reused as-is.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import ArraysCache

from vllm_metal.mlx_backend.gdn_cache import GDNPagedStateCache
from vllm_metal.paged_attention_common import get_context


def is_linear_attention(module: nn.Module) -> bool:
    """Return True if *module* is a linear attention layer (e.g. GatedDeltaNet).

    Checks for ``conv1d`` (present in all known GatedDeltaNet variants) and
    the absence of ``q_proj`` (which would indicate SDPA).
    """
    return hasattr(module, "conv1d") and not hasattr(module, "q_proj")


class GDNPagedAttentionWrapper(nn.Module):
    """Wraps a GDN linear attention module with managed paged state.

    For each request in the packed batch, loads conv_state and
    recurrent_state from ``GDNPagedStateCache``, runs the original
    mlx_lm GDN forward, then writes the updated state back.

    When no ``PagedAttentionContext`` is active, delegates to the
    original module unchanged.
    """

    def __init__(
        self,
        inner: nn.Module,
        layer_idx: int,
        cache_idx: int,
        state_cache: GDNPagedStateCache,
    ) -> None:
        super().__init__()
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_gdn_layer_idx", layer_idx)
        object.__setattr__(self, "_gdn_cache_idx", cache_idx)
        object.__setattr__(self, "_gdn_state_cache", state_cache)

    def __call__(self, x: mx.array, mask: Any = None, cache: Any = None) -> mx.array:
        ctx = get_context()
        if ctx is None:
            return self._inner(x, mask=mask, cache=cache)

        inner = self._inner
        cache_idx: int = self._gdn_cache_idx
        state_cache: GDNPagedStateCache = self._gdn_state_cache

        cu_seqlens = ctx.cu_seqlens
        if cu_seqlens is None or len(cu_seqlens) < 2:
            raise RuntimeError("GDN wrapper requires cu_seqlens in context")

        num_requests = len(cu_seqlens) - 1
        outputs = []

        for req_idx in range(num_requests):
            start = cu_seqlens[req_idx]
            end = cu_seqlens[req_idx + 1]
            req_x = x[:, start:end, :]  # [1, req_tokens, D]

            # Load state from managed cache into ArraysCache for mlx_lm
            arrays_cache = ArraysCache(2)
            arrays_cache[0] = state_cache.conv_states[cache_idx][req_idx : req_idx + 1]
            arrays_cache[1] = state_cache.recurrent_states[cache_idx][
                req_idx : req_idx + 1
            ]

            # Delegate to mlx_lm's GDN forward — it uses the existing
            # Metal kernel (gated_delta_kernel) internally
            req_out = inner(req_x, mask=None, cache=arrays_cache)

            # Write updated state back (MLX indexed assignment returns
            # a new array; reassign the layer's cache entry)
            conv = state_cache.conv_states[cache_idx]
            conv[req_idx : req_idx + 1] = arrays_cache[0]
            state_cache.conv_states[cache_idx] = conv

            rec = state_cache.recurrent_states[cache_idx]
            rec[req_idx : req_idx + 1] = arrays_cache[1]
            state_cache.recurrent_states[cache_idx] = rec

            outputs.append(req_out)

        if len(outputs) == 1:
            return outputs[0]
        return mx.concatenate(outputs, axis=1)
