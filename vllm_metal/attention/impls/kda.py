# SPDX-License-Identifier: Apache-2.0
"""Bailing KDA attention with scheduler-managed recurrent state."""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import ArraysCache

from vllm_metal.attention.caches.gdn_cache import GDNPagedStateCache
from vllm_metal.attention.context import PagedAttentionContext, get_context


def is_bailing_kda(module: nn.Module) -> bool:
    """Return whether a module exposes the supported Bailing KDA layout."""
    return all(
        hasattr(module, name)
        for name in (
            "q_proj",
            "k_proj",
            "v_proj",
            "q_conv1d",
            "k_conv1d",
            "v_conv1d",
            "projection_size",
            "conv_kernel_size",
        )
    )


class KDAPagedAttentionWrapper(nn.Module):
    """Run Bailing KDA segments against scheduler-managed recurrent slots."""

    def __init__(
        self,
        inner: nn.Module,
        layer_idx: int,
        cache_idx: int,
        state_cache: GDNPagedStateCache,
    ) -> None:
        super().__init__()
        if not is_bailing_kda(inner):
            raise TypeError(f"{type(inner).__name__} is not a Bailing KDA module")
        projection_size = int(inner.projection_size)
        conv_kernel_size = int(inner.conv_kernel_size)
        if 3 * projection_size != state_cache.conv_dim:
            raise RuntimeError(
                f"Bailing KDA projection_size={projection_size} does not match "
                f"state cache conv_dim={state_cache.conv_dim} (expected "
                f"3 * projection_size)"
            )
        if conv_kernel_size != state_cache.conv_kernel_dim:
            raise RuntimeError(
                f"Bailing KDA conv_kernel_size={conv_kernel_size} does not match "
                f"state cache conv_kernel_dim={state_cache.conv_kernel_dim}"
            )
        object.__setattr__(self, "_inner", inner)
        self.rebind_state_cache(state_cache, cache_idx=cache_idx)

    def rebind_state_cache(
        self, state_cache: GDNPagedStateCache, *, cache_idx: int
    ) -> None:
        """Refresh pooled state refs in place (cached model reuse)."""
        object.__setattr__(self, "_kda_cache_idx", cache_idx)
        object.__setattr__(self, "_kda_state_cache", state_cache)

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

        cu_seqlens = ctx.cu_seqlens
        if cu_seqlens is None or len(cu_seqlens) < 2:
            raise RuntimeError("Bailing KDA wrapper requires cu_seqlens")
        num_requests = len(cu_seqlens) - 1
        slot_ids = self._slot_ids(ctx)
        if len(slot_ids) != num_requests:
            raise RuntimeError("Bailing KDA wrapper requires one slot per request")
        if len(set(slot_ids)) != len(slot_ids):
            raise RuntimeError("Bailing KDA wrapper requires unique slots")

        state_cache = self._kda_state_cache
        cache_idx = self._kda_cache_idx
        state_cache.require_allocated_slots(slot_ids)
        state_cache.apply_pending_conv_state(cache_idx)
        state_cache.apply_pending_recurrent_state(cache_idx)

        conv_pool = state_cache.conv_states[cache_idx]
        recurrent_pool = state_cache.recurrent_states[cache_idx]
        outputs: list[mx.array] = []
        conv_updates: list[mx.array] = []
        recurrent_updates: list[mx.array] = []
        projection_size = int(self._inner.projection_size)

        for req_idx, slot in enumerate(slot_ids):
            start = cu_seqlens[req_idx]
            end = cu_seqlens[req_idx + 1]
            local_cache = ArraysCache(size=4)
            packed_conv = conv_pool[slot : slot + 1]
            local_cache[0], local_cache[1], local_cache[2] = mx.split(
                packed_conv,
                [projection_size, 2 * projection_size],
                axis=-1,
            )
            local_cache[3] = recurrent_pool[slot : slot + 1]
            outputs.append(
                self._inner(x[:, start:end, :], mask=None, cache=local_cache)
            )
            conv_updates.append(
                mx.concatenate(
                    [local_cache[0], local_cache[1], local_cache[2]], axis=-1
                )
            )
            recurrent_updates.append(local_cache[3])

        slots = mx.array(slot_ids, dtype=mx.int32)
        state_cache.write_conv_rows(
            cache_idx, mx.concatenate(conv_updates, axis=0), slots
        )
        state_cache.write_recurrent_rows(
            cache_idx, mx.concatenate(recurrent_updates, axis=0), slots
        )
        return mx.concatenate(outputs, axis=1)

    def _slot_ids(self, ctx: PagedAttentionContext) -> list[int]:
        if ctx.gdn_group_slot_mappings is not None:
            ordinal = self._kda_state_cache.layer_group_ordinal(self._kda_cache_idx)
            return ctx.gdn_group_slot_mappings[ordinal]
        if ctx.gdn_slot_mapping is not None:
            return ctx.gdn_slot_mapping
        raise RuntimeError("Bailing KDA wrapper requires a state slot mapping")
