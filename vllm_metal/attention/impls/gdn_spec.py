# SPDX-License-Identifier: Apache-2.0
"""Speculative-verify state staging and acceptance fixup for GDN layers.

A hybrid verification forward packs ``[last_token, *drafts]`` per decode
request, so every GDN layer's recurrent scan advances the request's state
``num_query_tokens`` steps in one shot. The deferred-update path parks that
span-final state as *pending* instead of scattering it into the stable pool,
which keeps the pre-forward state (the depth-0 rollback point) intact.

When every request accepts its scheduled drafts, the pending updates are
already correct and no re-scan runs. When a request rejects a draft, its
committed prefix ends ``accepted + 1`` rows into the span, so
:func:`apply_spec_decode_acceptance` re-scans only those partial verify
prefixes and replaces their pending updates. Other requests keep the forward's
state. The re-scan reads the untouched depth-0 state and reuses compact
verify-only projections stashed during the forward.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mlx.core as mx

from vllm_metal.attention.caches.state_cache import PagedStateCache
from vllm_metal.attention.impls.gdn_lazy import (
    GDNLazyKernels,
    GDNRecurrentDecodeRequest,
    GDNRecurrentPrefillRequest,
)


@dataclass(slots=True)
class GDNSpecVerifyStash:
    """One GDN layer's verify-span activations, held until acceptance."""

    cache_idx: int
    request_indices: list[int]
    slot_ids: list[int]
    cu_seqlens: list[int]
    span_lengths: tuple[int, ...]
    compute_dtype: mx.Dtype | None
    decode_threadgroup_dv: int
    q: mx.array
    k: mx.array
    v: mx.array
    g: mx.array
    beta: mx.array
    mixed_qkv: mx.array


@dataclass(slots=True)
class GDNSpecVerifyStep:
    """The stash for one forward; owned by the state cache, cleared per step."""

    layers: list[GDNSpecVerifyStash] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.layers)


def stash_spec_verify_layer(
    stash: GDNSpecVerifyStep,
    *,
    cache_idx: int,
    slot_ids: list[int],
    cu_seqlens: list[int],
    num_decode_requests: int,
    compute_dtype: mx.Dtype | None,
    decode_threadgroup_dv: int,
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    mixed_qkv: mx.array,
) -> None:
    request_indices = [
        i for i in range(num_decode_requests) if cu_seqlens[i + 1] - cu_seqlens[i] > 1
    ]
    if not request_indices:
        return
    lengths = tuple(cu_seqlens[i + 1] - cu_seqlens[i] for i in request_indices)
    compact_cu = [0]
    for length in lengths:
        compact_cu.append(compact_cu[-1] + length)
    # Reuse inputs for full-batch verification; otherwise gather into owned
    # buffers so a subset does not retain the full packed allocation.
    rows = None
    if compact_cu[-1] != q.shape[1]:
        rows = mx.array(
            [
                row
                for i in request_indices
                for row in range(cu_seqlens[i], cu_seqlens[i + 1])
            ],
            dtype=mx.int32,
        )

    def _stash_rows(x: mx.array) -> mx.array:
        return x if rows is None else mx.take(x, rows, axis=1)

    stash.layers.append(
        GDNSpecVerifyStash(
            cache_idx=cache_idx,
            request_indices=request_indices,
            slot_ids=[slot_ids[i] for i in request_indices],
            cu_seqlens=compact_cu,
            span_lengths=lengths,
            compute_dtype=compute_dtype,
            decode_threadgroup_dv=decode_threadgroup_dv,
            q=_stash_rows(q),
            k=_stash_rows(k),
            v=_stash_rows(v),
            g=_stash_rows(g),
            beta=_stash_rows(beta),
            mixed_qkv=_stash_rows(mixed_qkv),
        )
    )


def is_spec_verify_batch(num_decode_requests: int, cu_seqlens: list[int]) -> bool:
    """Whether the packed batch carries any multi-token decode (verify) span.

    Packed decode segments come first, followed by prefill segments. Plain
    decode packs one token per decode segment; a speculative verification
    batch packs ``[last_token, *drafts]`` per drafting request, so any decode
    segment longer than one row means this step must stage its GDN state for
    the acceptance fixup. Mixed spec-decode + prefill steps qualify too.
    """
    return any(
        cu_seqlens[i + 1] - cu_seqlens[i] > 1 for i in range(num_decode_requests)
    )


def apply_spec_decode_acceptance(
    state_cache: PagedStateCache,
    lazy: GDNLazyKernels,
    committed_rows: list[int],
) -> bool:
    """Rewrite pending GDN state to the accepted depth after verification.

    ``committed_rows[i]`` is the number of span rows decode request ``i`` committed
    (accepted drafts plus the residual/bonus row). An all-full-span batch keeps
    its pending state; only partial verify requests need a re-scan in each
    layer. Returns whether a re-scan ran.
    """
    stash = state_cache.spec_verify_stash
    if stash is None or not stash.layers:
        return False
    try:
        first = stash.layers[0]
        if any(
            not 1 <= committed_rows[i] <= length
            for i, length in zip(first.request_indices, first.span_lengths, strict=True)
        ):
            raise ValueError("committed rows must be within the GDN verify span")
        if all(
            committed_rows[i] == length
            for i, length in zip(first.request_indices, first.span_lengths, strict=True)
        ):
            return False
        for layer_stash in stash.layers:
            _rescan_layer_to_accepted(state_cache, lazy, layer_stash, committed_rows)
        return True
    finally:
        state_cache.spec_verify_stash = None


def _merge_state_rows(
    state: mx.array,
    state_slots: list[int],
    updates: mx.array,
    update_slots: list[int],
) -> mx.array:
    """Replace selected rows without changing the saved forward state."""
    rows = mx.array([state_slots.index(slot) for slot in update_slots], mx.int32)
    merged = mx.array(state)
    merged[rows] = updates
    return merged


def _rescan_layer_to_accepted(
    state_cache: PagedStateCache,
    lazy: GDNLazyKernels,
    layer_stash: GDNSpecVerifyStash,
    committed_rows: list[int],
) -> None:
    partial = [
        i
        for i, (req_idx, length) in enumerate(
            zip(layer_stash.request_indices, layer_stash.span_lengths, strict=True)
        )
        if committed_rows[req_idx] < length
    ]
    committed_rows = [committed_rows[layer_stash.request_indices[i]] for i in partial]
    slot_ids = [layer_stash.slot_ids[i] for i in partial]
    # Preserve forward results for all other requests, but do not scatter:
    # the re-scan must still read the partial requests' depth-0 pool state.
    recurrent = state_cache.pending_recurrent_states[layer_stash.cache_idx]
    recurrent_slots = state_cache.pending_recurrent_slot_ids[layer_stash.cache_idx]
    conv = state_cache.pending_conv_states[layer_stash.cache_idx]
    conv_slots = state_cache.pending_conv_slot_ids[layer_stash.cache_idx]
    state_cache.clear_pending_recurrent_state(layer_stash.cache_idx)
    state_cache.clear_pending_conv_state(layer_stash.cache_idx)

    cu = [0]
    for length in committed_rows:
        cu.append(cu[-1] + length)
    total = cu[-1]

    # Truncated rows are sliced out of the ORIGINAL packed layout: request
    # i's kept prefix starts at its original span start, not at the
    # truncated batch's running offset.
    original_starts = [layer_stash.cu_seqlens[i] for i in partial]

    def slice_span(arr: mx.array) -> mx.array:
        rows = [
            arr[0, start : start + length]
            for start, length in zip(original_starts, committed_rows, strict=True)
        ]
        return mx.concatenate(rows, axis=0)[None]

    num_requests = len(committed_rows)
    if total > num_requests:
        request: GDNRecurrentPrefillRequest | GDNRecurrentDecodeRequest = (
            GDNRecurrentPrefillRequest(
                q=slice_span(layer_stash.q),
                k=slice_span(layer_stash.k),
                v=slice_span(layer_stash.v),
                g=slice_span(layer_stash.g),
                beta=slice_span(layer_stash.beta),
                state_cache=state_cache,
                cache_idx=layer_stash.cache_idx,
                slot_ids=slot_ids,
                output_dtype=layer_stash.q.dtype,
                cu_seqlens=cu,
                compute_dtype=layer_stash.compute_dtype,
                defer_state_scatter=True,
            )
        )
        rescan = lazy.try_recurrent_prefill(request)
    else:
        # Every request rejected its first draft: the accepted state is one
        # row per request, which is exactly the T=1 decode-kernel batch shape.
        request = GDNRecurrentDecodeRequest(
            q=slice_span(layer_stash.q),
            k=slice_span(layer_stash.k),
            v=slice_span(layer_stash.v),
            g=slice_span(layer_stash.g),
            beta=slice_span(layer_stash.beta),
            state_cache=state_cache,
            cache_idx=layer_stash.cache_idx,
            slot_ids=slot_ids,
            output_dtype=layer_stash.q.dtype,
            threadgroup_dv=layer_stash.decode_threadgroup_dv,
        )
        rescan = lazy.try_recurrent_decode(request)
    if rescan is None:
        raise RuntimeError(
            "spec-decode GDN acceptance re-scan was ineligible; the verify "
            "forward must constrain spans so acceptance fixups stay "
            "kernel-eligible"
        )

    if recurrent is not None:
        assert recurrent_slots is not None
        recurrent = _merge_state_rows(
            recurrent,
            recurrent_slots,
            state_cache.pending_recurrent_states[layer_stash.cache_idx],
            slot_ids,
        )
        state_cache.clear_pending_recurrent_state(layer_stash.cache_idx)
        state_cache.set_pending_recurrent_state(
            layer_stash.cache_idx, recurrent_slots, recurrent
        )

    # The conv tail at the accepted depth is the last (kernel - 1) rows of
    # (depth-0 conv state + accepted span prefix) — pure slicing, no kernel.
    conv_state = state_cache.conv_states[layer_stash.cache_idx]
    state_len = conv_state.shape[1]
    tails = []
    for req_idx, (slot, start) in enumerate(
        zip(slot_ids, original_starts, strict=True)
    ):
        prefix = layer_stash.mixed_qkv[0, start : start + committed_rows[req_idx]]
        window = mx.concatenate([conv_state[slot], prefix], axis=0)
        tails.append(window[-state_len:])
    updated_conv = mx.stack(tails, axis=0).astype(conv_state.dtype)
    if conv is not None:
        assert conv_slots is not None
        updated_conv = _merge_state_rows(conv, conv_slots, updated_conv, slot_ids)
        slot_ids = conv_slots
    state_cache.set_pending_conv_state(layer_stash.cache_idx, slot_ids, updated_conv)
    mx.eval(
        state_cache.pending_recurrent_states[layer_stash.cache_idx],
        state_cache.pending_conv_states[layer_stash.cache_idx],
    )
