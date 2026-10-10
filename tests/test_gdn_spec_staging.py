# SPDX-License-Identifier: Apache-2.0
"""Speculative-verify state staging and acceptance fixup for GDN layers."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from tests.stub_runner import make_state_cache
from vllm_metal.attention.impls.gdn_lazy import (
    GDNLazyKernels,
    GDNRecurrentDecodeRequest,
    GDNRecurrentPrefillRequest,
)
from vllm_metal.attention.impls.gdn_spec import (
    GDNSpecVerifyStep,
    apply_spec_decode_acceptance,
    is_spec_verify_batch,
    stash_spec_verify_layer,
)


def _require_metal() -> None:
    if not mx.metal.is_available():
        pytest.skip("Metal GPU required")


def _inputs(total_tokens: int, *, n_hk=2, n_hv=4, d_k=32, d_v=8):
    q = mx.random.normal((1, total_tokens, n_hk, d_k))
    k = mx.random.normal((1, total_tokens, n_hk, d_k))
    v = mx.random.normal((1, total_tokens, n_hv, d_v))
    g = mx.random.normal((1, total_tokens, n_hv))
    beta = mx.random.normal((1, total_tokens, n_hv))
    return q, k, v, g, beta


def _scan(lazy, cache, slot_ids, cu, q, k, v, g, beta, *, defer=True):
    request = GDNRecurrentPrefillRequest(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        state_cache=cache,
        cache_idx=0,
        slot_ids=list(slot_ids),
        output_dtype=q.dtype,
        cu_seqlens=list(cu),
        compute_dtype=None,
        defer_state_scatter=defer,
    )
    out = lazy.try_recurrent_prefill(request)
    assert out is not None
    mx.eval(out, cache.pending_recurrent_states[0])
    return out


class TestSpecVerifyPredicate:
    def test_plain_decode_is_not_spec(self) -> None:
        # 3 decode requests, one token each.
        assert not is_spec_verify_batch(3, [0, 1, 2, 3])

    def test_multi_token_decode_span_is_spec(self) -> None:
        assert is_spec_verify_batch(2, [0, 4, 8])  # two 4-row verify spans

    def test_mixed_spec_and_prefill_is_spec(self) -> None:
        # One 3-row verify span followed by one prefill segment.
        assert is_spec_verify_batch(1, [0, 3, 10])

    def test_prefill_only_is_not_spec(self) -> None:
        assert not is_spec_verify_batch(0, [0, 5, 12])


class TestAcceptanceFixup:
    def test_partial_accept_rescans_to_accepted_depth(self) -> None:
        _require_metal()
        mx.random.seed(0)
        lazy = GDNLazyKernels(enabled=True)
        cache = make_state_cache(
            num_layers=1,
            max_seqs=4,
            conv_kernel_dim=4,
            conv_dim=64,
            num_v_heads=4,
            value_head_dim=8,
            key_head_dim=32,
        )
        slot_ids = [1, 3]
        span = 5  # [last, d1..d4]
        cu = [0, span, 2 * span]
        q, k, v, g, beta = _inputs(2 * span)
        mixed_qkv = mx.random.normal((1, 2 * span, 64))

        # Verify forward: full-span scan defers its state update.
        _scan(lazy, cache, slot_ids, cu, q, k, v, g, beta)
        full_pending = cache.pending_recurrent_states[0]
        assert full_pending is not None

        # Reference: a direct truncated scan (rows [0:3] per request) from
        # the same untouched base state is what acceptance must produce.
        cache.clear_pending_recurrent_state(0)
        cache2 = make_state_cache(
            num_layers=1,
            max_seqs=4,
            conv_kernel_dim=4,
            conv_dim=64,
            num_v_heads=4,
            value_head_dim=8,
            key_head_dim=32,
        )
        cache2.conv_states[0] = cache.conv_states[0]
        cache2.recurrent_states[0] = cache.recurrent_states[0]
        keep = 3
        # Each request's kept prefix starts at its ORIGINAL span offset
        # (rows 0-2 of request 0, rows 5-7 of request 1).
        rows = [0, 1, 2, 5, 6, 7]
        cu_ref = [0, keep, 2 * keep]
        _scan(
            lazy,
            cache2,
            slot_ids,
            cu_ref,
            q[:, rows],
            k[:, rows],
            v[:, rows],
            g[:, rows],
            beta[:, rows],
        )
        expected = cache2.pending_recurrent_states[0]

        # Stash + fixup for [3, 3] committed rows (both requests partial).
        stash = GDNSpecVerifyStep()
        cache.spec_verify_stash = stash
        stash_spec_verify_layer(
            stash,
            cache_idx=0,
            slot_ids=slot_ids,
            cu_seqlens=cu,
            span_lengths=(span, span),
            compute_dtype=None,
            decode_threadgroup_dv=4,
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            mixed_qkv=mixed_qkv,
        )
        assert apply_spec_decode_acceptance(cache, lazy, [keep, keep])

        fixed = cache.pending_recurrent_states[0]
        assert fixed is not None
        # Pending updates are compact: row order matches the request order in
        # slot_ids, not the pool's slot numbering.
        np.testing.assert_allclose(np.array(fixed), np.array(expected), atol=1e-4)
        # The stable pool still holds the untouched depth-0 state.
        assert cache.spec_verify_stash is None

    def test_full_accept_keeps_staged_state(self) -> None:
        _require_metal()
        mx.random.seed(1)
        lazy = GDNLazyKernels(enabled=True)
        cache = make_state_cache(
            num_layers=1,
            max_seqs=4,
            conv_kernel_dim=4,
            conv_dim=64,
            num_v_heads=4,
            value_head_dim=8,
            key_head_dim=32,
        )
        slot_ids = [0]
        span = 4
        cu = [0, span]
        q, k, v, g, beta = _inputs(span)
        mixed_qkv = mx.random.normal((1, span, 64))
        base = cache.recurrent_states[0]

        _scan(lazy, cache, slot_ids, cu, q, k, v, g, beta)
        full_pending = cache.pending_recurrent_states[0]

        stash = GDNSpecVerifyStep()
        cache.spec_verify_stash = stash
        stash_spec_verify_layer(
            stash,
            cache_idx=0,
            slot_ids=slot_ids,
            cu_seqlens=cu,
            span_lengths=(span,),
            compute_dtype=None,
            decode_threadgroup_dv=4,
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            mixed_qkv=mixed_qkv,
        )
        # All drafts accepted: committed rows equal the span; no re-scan.
        assert not apply_spec_decode_acceptance(cache, lazy, [span])
        assert cache.pending_recurrent_states[0] is full_pending
        assert cache.recurrent_states[0] is base  # pool untouched

    def test_conv_tail_rolls_back_by_slicing(self) -> None:
        _require_metal()
        mx.random.seed(2)
        lazy = GDNLazyKernels(enabled=True)
        cache = make_state_cache(
            num_layers=1,
            max_seqs=4,
            conv_kernel_dim=4,
            conv_dim=64,
            num_v_heads=4,
            value_head_dim=8,
            key_head_dim=32,
        )
        slot = 2
        slot_ids = [slot]
        span = 4
        cu = [0, span]
        q, k, v, g, beta = _inputs(span)
        mixed_qkv = mx.random.normal((1, span, 64))

        _scan(lazy, cache, slot_ids, cu, q, k, v, g, beta)

        stash = GDNSpecVerifyStep()
        cache.spec_verify_stash = stash
        stash_spec_verify_layer(
            stash,
            cache_idx=0,
            slot_ids=slot_ids,
            cu_seqlens=cu,
            span_lengths=(span,),
            compute_dtype=None,
            decode_threadgroup_dv=4,
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            mixed_qkv=mixed_qkv,
        )
        keep = 2
        assert apply_spec_decode_acceptance(cache, lazy, [keep])

        state_len = cache.conv_states[0].shape[1]
        window = mx.concatenate(
            [cache.conv_states[0][slot], mixed_qkv[0, :keep]], axis=0
        )
        # The pending tail must match the pool's dtype (the stub pool is fp16).
        expected = window[-state_len:].astype(cache.conv_states[0].dtype)
        got = cache.pending_conv_states[0][0]
        np.testing.assert_allclose(np.array(got), np.array(expected), atol=1e-3)

    def test_mixed_accept_depths_rescan_batched(self) -> None:
        _require_metal()
        mx.random.seed(3)
        lazy = GDNLazyKernels(enabled=True)
        cache = make_state_cache(
            num_layers=1,
            max_seqs=4,
            conv_kernel_dim=4,
            conv_dim=64,
            num_v_heads=4,
            value_head_dim=8,
            key_head_dim=32,
        )
        slot_ids = [0, 1]
        spans = (4, 2)
        cu = [0, 4, 6]
        q, k, v, g, beta = _inputs(6)
        mixed_qkv = mx.random.normal((1, 6, 64))

        _scan(lazy, cache, slot_ids, cu, q, k, v, g, beta)

        # Reference per-request truncated scans from the same base.
        cache2 = make_state_cache(
            num_layers=1,
            max_seqs=4,
            conv_kernel_dim=4,
            conv_dim=64,
            num_v_heads=4,
            value_head_dim=8,
            key_head_dim=32,
        )
        cache2.conv_states[0] = cache.conv_states[0]
        cache2.recurrent_states[0] = cache.recurrent_states[0]
        keep = (3, 1)
        _scan(
            lazy,
            cache2,
            slot_ids,
            [0, 3, 4],
            q[:, [0, 1, 2, 4]],  # rows 0-2 of r0, row 0 of r1
            k[:, [0, 1, 2, 4]],
            v[:, [0, 1, 2, 4]],
            g[:, [0, 1, 2, 4]],
            beta[:, [0, 1, 2, 4]],
        )
        expected = cache2.pending_recurrent_states[0]

        stash = GDNSpecVerifyStep()
        cache.spec_verify_stash = stash
        stash_spec_verify_layer(
            stash,
            cache_idx=0,
            slot_ids=slot_ids,
            cu_seqlens=cu,
            span_lengths=spans,
            compute_dtype=None,
            decode_threadgroup_dv=4,
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            mixed_qkv=mixed_qkv,
        )
        assert apply_spec_decode_acceptance(cache, lazy, list(keep))
        got = cache.pending_recurrent_states[0]
        np.testing.assert_allclose(np.array(got), np.array(expected), atol=1e-4)

    def test_all_first_reject_rescans_via_decode_kernel(self) -> None:
        _require_metal()
        mx.random.seed(4)
        lazy = GDNLazyKernels(enabled=True)
        cache = make_state_cache(
            num_layers=1,
            max_seqs=4,
            conv_kernel_dim=4,
            conv_dim=64,
            num_v_heads=4,
            value_head_dim=8,
            key_head_dim=32,
        )
        slot_ids = [1, 3]
        span = 4
        cu = [0, span, 2 * span]
        q, k, v, g, beta = _inputs(2 * span)
        mixed_qkv = mx.random.normal((1, 2 * span, 64))

        _scan(lazy, cache, slot_ids, cu, q, k, v, g, beta)

        # Every request rejected its first draft: the accepted state is one
        # row per request, which routes the fixup to the T=1 decode kernel
        # (the prefill scan requires total > num_requests). Reference: a
        # direct decode scan over each request's kept first span row (rows 0
        # and span) from the same untouched base state.
        cache2 = make_state_cache(
            num_layers=1,
            max_seqs=4,
            conv_kernel_dim=4,
            conv_dim=64,
            num_v_heads=4,
            value_head_dim=8,
            key_head_dim=32,
        )
        cache2.conv_states[0] = cache.conv_states[0]
        cache2.recurrent_states[0] = cache.recurrent_states[0]
        kept = [0, span]
        reference = lazy.try_recurrent_decode(
            GDNRecurrentDecodeRequest(
                q=q[:, kept],
                k=k[:, kept],
                v=v[:, kept],
                g=g[:, kept],
                beta=beta[:, kept],
                state_cache=cache2,
                cache_idx=0,
                slot_ids=slot_ids,
                output_dtype=q.dtype,
            )
        )
        assert reference is not None
        expected = cache2.pending_recurrent_states[0]
        assert expected is not None

        stash = GDNSpecVerifyStep()
        cache.spec_verify_stash = stash
        stash_spec_verify_layer(
            stash,
            cache_idx=0,
            slot_ids=slot_ids,
            cu_seqlens=cu,
            span_lengths=(span, span),
            compute_dtype=None,
            decode_threadgroup_dv=4,
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            mixed_qkv=mixed_qkv,
        )
        assert apply_spec_decode_acceptance(cache, lazy, [1, 1])
        got = cache.pending_recurrent_states[0]
        np.testing.assert_allclose(np.array(got), np.array(expected), atol=1e-4)

        # The conv tail at depth 1 is the last (kernel - 1) rows of
        # (depth-0 conv state + the kept first span row).
        state_len = cache.conv_states[0].shape[1]
        for row, (slot, start) in enumerate(zip(slot_ids, kept, strict=True)):
            window = mx.concatenate(
                [cache.conv_states[0][slot], mixed_qkv[0, start : start + 1]], axis=0
            )
            expected_tail = window[-state_len:].astype(cache.conv_states[0].dtype)
            np.testing.assert_allclose(
                np.array(cache.pending_conv_states[0][row]),
                np.array(expected_tail),
                atol=1e-3,
            )
