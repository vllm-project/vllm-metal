# SPDX-License-Identifier: Apache-2.0
"""Kimi Delta Attention with scheduler-managed recurrent state."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
from mlx_vlm.models.cache import ArraysCache

from vllm_metal.attention.caches.state_cache import (
    PagedStateCache,
    StateSlotArraysCache,
)
from vllm_metal.attention.context import PagedAttentionContext, get_context


@dataclass(frozen=True, slots=True)
class _KDALayout:
    projection_size: int
    conv_kernel_size: int
    conv_dtype: mx.Dtype
    # Per-stream split of the packed conv state; ``None`` keeps one fused
    # q/k/v stream.
    conv_widths: tuple[int, ...] | None


def _resolve_kda_layout(module: nn.Module) -> _KDALayout | None:
    """Read the KDA dims from MLX-LM's split or mlx-vlm's fused q/k/v layout."""
    if all(
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
    ):
        projection_size = int(module.projection_size)
        return _KDALayout(
            projection_size=projection_size,
            conv_kernel_size=int(module.conv_kernel_size),
            conv_dtype=module.q_conv1d.weight.dtype,
            conv_widths=(projection_size,) * 3,
        )
    if all(
        hasattr(module, name)
        for name in ("qkv_proj", "qkv_conv", "projection_dim", "conv_kernel")
    ):
        return _KDALayout(
            projection_size=int(module.projection_dim),
            conv_kernel_size=int(module.conv_kernel),
            conv_dtype=module.qkv_conv.conv.weight.dtype,
            conv_widths=None,
        )
    return None


def is_kda_attention(module: nn.Module) -> bool:
    """Return whether a module exposes the KDA interface used by the wrapper."""
    return _resolve_kda_layout(module) is not None


@dataclass(frozen=True, slots=True)
class KDAVerifySpan:
    """One request's speculative verify span in one KDA layer, kept until
    acceptance.

    ``cache`` is mlx-vlm's ``ArraysCache`` over the request's state row, which
    recorded the state after every span row.
    """

    cache_idx: int
    req_id: str
    slot_ids: mx.array
    cache: ArraysCache


def accept_kda_verify_spans(
    state_cache: PagedStateCache, committed_rows: Mapping[str, int]
) -> None:
    """Roll spans that rejected a draft back to their committed depth.

    A verify step writes the span-final state, as if every draft were accepted.
    ``committed_rows`` maps a request to its committed rows (accepted drafts
    plus the bonus row); a request without an entry keeps its spans.
    """
    kept = []
    for span in state_cache.kda_verify_spans:
        committed = committed_rows.get(span.req_id)
        if committed is None:
            kept.append(span)
        elif committed < span.cache.history_capacity:
            span.cache.commit_speculation([committed])
            state_cache.write_conv_rows(span.cache_idx, span.cache[0], span.slot_ids)
            state_cache.write_recurrent_rows(
                span.cache_idx, span.cache[1], span.slot_ids
            )
    state_cache.kda_verify_spans = kept


@dataclass(frozen=True, slots=True)
class _KDAStep:
    cu_seqlens: list[int]
    slot_ids: list[int]
    num_requests: int
    num_decode_requests: int
    req_ids: list[str]


class KDAPagedAttentionWrapper(nn.Module):
    """Run packed KDA segments against scheduler-managed state slots."""

    def __init__(
        self,
        inner: nn.Module,
        layer_idx: int,
        cache_idx: int,
        state_cache: PagedStateCache,
    ) -> None:
        super().__init__()
        layout = _resolve_kda_layout(inner)
        if layout is None:
            raise TypeError(f"{type(inner).__name__} is not a KDA module")
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_kda_layout", layout)
        object.__setattr__(self, "_kda_layer_idx", layer_idx)
        self.rebind_state_cache(state_cache, cache_idx=cache_idx)

    def _validate_state_cache(
        self, state_cache: PagedStateCache, cache_idx: int
    ) -> None:
        inner = self._inner
        layout = self._kda_layout
        expected_conv = (layout.conv_kernel_size - 1, 3 * layout.projection_size)
        actual_conv = state_cache.conv_states[cache_idx].shape[1:]
        if actual_conv != expected_conv:
            raise RuntimeError(
                f"KDA conv state shape {actual_conv} does not match {expected_conv}"
            )
        expected_recurrent = (
            int(inner.num_heads),
            int(inner.head_dim),
            int(inner.head_dim),
        )
        actual_recurrent = state_cache.recurrent_states[cache_idx].shape[1:]
        if actual_recurrent != expected_recurrent:
            raise RuntimeError(
                f"KDA recurrent state shape {actual_recurrent} does not match "
                f"{expected_recurrent}"
            )
        state_cache.require_mixer_dtype(
            layout.conv_dtype, layer_idx=self._kda_layer_idx
        )

    def rebind_state_cache(
        self, state_cache: PagedStateCache, *, cache_idx: int
    ) -> None:
        """Refresh pooled state references for a cached model."""
        self._validate_state_cache(state_cache, cache_idx)
        object.__setattr__(self, "_kda_cache_idx", cache_idx)
        object.__setattr__(self, "_kda_state_cache", state_cache)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: nn.Module | None = None,
    ) -> mx.array:
        ctx = get_context()
        if ctx is None:
            return self._inner(x, mask=mask, cache=cache)

        step = self._prepare_step(ctx)
        self._kda_state_cache.apply_pending_conv_state(self._kda_cache_idx)
        self._kda_state_cache.apply_pending_recurrent_state(self._kda_cache_idx)
        if step.num_decode_requests == step.num_requests == step.cu_seqlens[-1]:
            return self._run_decode(x, step)
        return self._run_requests(x, step)

    def _prepare_step(self, ctx: PagedAttentionContext) -> _KDAStep:
        cu_seqlens = ctx.cu_seqlens
        if cu_seqlens is None or len(cu_seqlens) < 2:
            raise RuntimeError("KDA wrapper requires cu_seqlens")
        num_requests = len(cu_seqlens) - 1
        return _KDAStep(
            cu_seqlens=cu_seqlens,
            slot_ids=self._kda_state_cache.step_slot_ids(
                ctx, self._kda_cache_idx, num_requests
            ),
            num_requests=num_requests,
            num_decode_requests=ctx.num_decode_requests,
            req_ids=ctx.req_ids,
        )

    def _run_decode(self, x: mx.array, step: _KDAStep) -> mx.array:
        rows = x.reshape(step.num_requests, 1, x.shape[-1])
        cache = self._slot_cache(mx.array(step.slot_ids, dtype=mx.int32))
        output = self._inner(rows, mask=None, cache=cache)
        cache.flush()
        return output.reshape(1, step.num_requests, -1)

    def _run_requests(self, x: mx.array, step: _KDAStep) -> mx.array:
        outputs = []
        for req_idx, slot in enumerate(step.slot_ids):
            start = step.cu_seqlens[req_idx]
            end = step.cu_seqlens[req_idx + 1]
            if req_idx < step.num_decode_requests and end - start > 1:
                outputs.append(
                    self._run_verify(x[:, start:end, :], slot, step.req_ids[req_idx])
                )
                continue
            cache = self._slot_cache(mx.array([slot], dtype=mx.int32))
            outputs.append(self._inner(x[:, start:end, :], mask=None, cache=cache))
            cache.flush()
        return mx.concatenate(outputs, axis=1)

    def _run_verify(self, x: mx.array, slot: int, req_id: str) -> mx.array:
        """Run a speculative verify span on mlx-vlm's ``ArraysCache``.

        The span-final state is written now; the cache keeps the state after
        every row for :func:`accept_kda_verify_spans`.
        """
        state_cache = self._kda_state_cache
        cache_idx = self._kda_cache_idx
        slot_ids = mx.array([slot], dtype=mx.int32)
        cache = ArraysCache(2)
        cache[0] = state_cache.gather_conv_rows(cache_idx, slot_ids)
        cache[1] = state_cache.gather_recurrent_rows(cache_idx, slot_ids)
        cache.start_speculation(x.shape[1])
        output = self._inner(x, mask=None, cache=cache)
        state_cache.write_conv_rows(cache_idx, cache[0], slot_ids)
        state_cache.write_recurrent_rows(cache_idx, cache[1], slot_ids)
        state_cache.kda_verify_spans.append(
            KDAVerifySpan(cache_idx, req_id, slot_ids, cache)
        )
        return output

    def _slot_cache(self, slot_ids: mx.array) -> StateSlotArraysCache:
        return StateSlotArraysCache(
            self._kda_state_cache,
            self._kda_cache_idx,
            slot_ids,
            conv_widths=self._kda_layout.conv_widths,
        )
