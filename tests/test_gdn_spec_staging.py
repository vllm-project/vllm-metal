# SPDX-License-Identifier: Apache-2.0
"""Speculative-verify state staging and acceptance fixup for GDN layers."""

from __future__ import annotations

from unittest.mock import patch

import mlx.core as mx
import numpy as np
import pytest

from tests.stub_runner import make_state_cache
from vllm_metal.attention.caches.state_cache import PagedStateCache
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


def _state_cache(max_seqs: int = 4) -> PagedStateCache:
    """Create a state pool matching the input geometry below."""
    return make_state_cache(
        num_layers=1,
        max_seqs=max_seqs,
        conv_kernel_dim=4,
        conv_dim=64,
        num_v_heads=4,
        value_head_dim=8,
        key_head_dim=32,
    )


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


class TestVerifyStash:
    def test_stash_contains_only_multi_token_decode_rows(self) -> None:
        mx.random.seed(0)
        arrays = _inputs(18)
        mixed = mx.random.normal((1, 18, 64))
        stash = GDNSpecVerifyStep()

        stash_spec_verify_layer(
            stash,
            cache_idx=0,
            slot_ids=[5, 2, 6, 0, 3, 1],
            cu_seqlens=[0, 1, 5, 7, 8, 11, 18],
            num_decode_requests=5,
            compute_dtype=None,
            decode_threadgroup_dv=4,
            q=arrays[0],
            k=arrays[1],
            v=arrays[2],
            g=arrays[3],
            beta=arrays[4],
            mixed_qkv=mixed,
        )

        layer = stash.layers[0]
        assert layer.request_indices == [1, 2, 4]
        assert layer.slot_ids == [2, 6, 3]
        assert layer.cu_seqlens == [0, 4, 6, 9]
        expected_rows = [1, 2, 3, 4, 5, 6, 8, 9, 10]
        for actual, original in zip(
            (layer.q, layer.k, layer.v, layer.g, layer.beta, layer.mixed_qkv),
            (*arrays, mixed),
            strict=True,
        ):
            np.testing.assert_array_equal(
                np.array(actual), np.array(original[:, expected_rows])
            )

    def test_full_verify_batch_aliases_inputs(self) -> None:
        arrays = _inputs(6)
        mixed = mx.random.normal((1, 6, 64))
        stash = GDNSpecVerifyStep()
        stash_spec_verify_layer(
            stash,
            cache_idx=0,
            slot_ids=[3, 1],
            cu_seqlens=[0, 3, 6],
            num_decode_requests=2,
            compute_dtype=None,
            decode_threadgroup_dv=4,
            q=arrays[0],
            k=arrays[1],
            v=arrays[2],
            g=arrays[3],
            beta=arrays[4],
            mixed_qkv=mixed,
        )
        layer = stash.layers[0]
        stashed = (layer.q, layer.k, layer.v, layer.g, layer.beta, layer.mixed_qkv)
        for actual, original in zip(stashed, (*arrays, mixed), strict=True):
            assert actual is original

        cache = _state_cache()
        cache.spec_verify_stash = stash
        updated = cache.updated_state_arrays()
        for actual in stashed:
            assert any(actual is array for array in updated)


class TestAcceptanceFixup:
    @pytest.mark.parametrize(
        "kept, expected_calls, expected_rescanned",
        [
            ((3, 1, 3), [([2, 6], 4)], True),
            ((1, 1, 3), [([2, 6], 2)], True),
            ((4, 2, 3), [], False),
        ],
        ids=["partial-accept", "first-draft-reject", "full-accept"],
    )
    def test_acceptance_preserves_other_requests_across_prefill_chunks(
        self,
        kept: tuple[int, int, int],
        expected_calls: list[tuple[list[int], int]],
        expected_rescanned: bool,
    ) -> None:
        """Keep other requests' states across two mixed chunks."""
        _require_metal()
        mx.random.seed(0)
        lazy = GDNLazyKernels(enabled=True)
        cache = _state_cache(max_seqs=7)
        reference = _state_cache(max_seqs=7)
        cache.conv_states[0] = mx.random.normal(cache.conv_states[0].shape).astype(
            mx.float16
        )
        cache.recurrent_states[0] = mx.random.normal(cache.recurrent_states[0].shape)
        slots = [5, 2, 6, 0, 3, 1]
        cu = [0, 1, 5, 7, 8, 11, 18]  # plain, verify, verify, plain, verify, prefill
        committed = [1, kept[0], kept[1], 1, kept[2], 7]
        for _ in range(2):
            reference.conv_states[0] = cache.conv_states[0]
            reference.recurrent_states[0] = cache.recurrent_states[0]
            arrays = _inputs(cu[-1])
            mixed = mx.random.normal((1, cu[-1], 64))
            _scan(lazy, cache, slots, cu, *arrays)

            def tails(lengths, mixed=mixed):
                return mx.stack(
                    [
                        mx.concatenate(
                            [cache.conv_states[0][slot], mixed[0, start : start + n]]
                        )[-3:]
                        for slot, start, n in zip(slots, cu[:-1], lengths, strict=True)
                    ]
                ).astype(mx.float16)

            cache.set_pending_conv_state(0, slots, tails(np.diff(cu)))
            forward_recurrent = cache.pending_recurrent_states[0]
            forward_conv = cache.pending_conv_states[0]
            # Original algorithm: rescan every request's committed prefix.
            rows = [
                r
                for start, n in zip(cu[:-1], committed, strict=True)
                for r in range(start, start + n)
            ]
            _scan(
                lazy,
                reference,
                slots,
                np.cumsum([0, *committed]).tolist(),
                *[a[:, rows] for a in arrays],
            )
            expected_conv = tails(committed)

            cache.spec_verify_stash = GDNSpecVerifyStep()
            stash_spec_verify_layer(
                cache.spec_verify_stash,
                cache_idx=0,
                slot_ids=slots,
                cu_seqlens=cu,
                num_decode_requests=5,
                compute_dtype=None,
                decode_threadgroup_dv=4,
                q=arrays[0],
                k=arrays[1],
                v=arrays[2],
                g=arrays[3],
                beta=arrays[4],
                mixed_qkv=mixed,
            )
            mx.eval(*cache.updated_state_arrays())
            with (
                patch.object(
                    lazy, "try_recurrent_prefill", wraps=lazy.try_recurrent_prefill
                ) as prefill_scan,
                patch.object(
                    lazy, "try_recurrent_decode", wraps=lazy.try_recurrent_decode
                ) as decode_scan,
            ):
                rescanned = apply_spec_decode_acceptance(cache, lazy, committed[:5])

            calls = [
                (call.args[0].slot_ids, call.args[0].q.shape[1])
                for call in prefill_scan.call_args_list + decode_scan.call_args_list
            ]
            assert calls == expected_calls
            assert rescanned is expected_rescanned
            np.testing.assert_allclose(
                np.array(cache.pending_recurrent_states[0]),
                np.array(reference.pending_recurrent_states[0]),
                atol=1e-4,
            )
            np.testing.assert_array_equal(
                np.array(cache.pending_conv_states[0]), np.array(expected_conv)
            )
            unchanged = [i for i, n in enumerate(committed) if n == cu[i + 1] - cu[i]]
            for actual, original in (
                (cache.pending_recurrent_states[0], forward_recurrent),
                (cache.pending_conv_states[0], forward_conv),
            ):
                np.testing.assert_array_equal(
                    np.array(actual[unchanged]), np.array(original[unchanged])
                )
            # Start the next chunk from the corrected states without native scatter.
            cache.recurrent_states[0] = mx.array(cache.recurrent_states[0])
            cache.recurrent_states[0][mx.array(slots)] = cache.pending_recurrent_states[
                0
            ]
            cache.conv_states[0] = mx.array(cache.conv_states[0])
            cache.conv_states[0][mx.array(slots)] = cache.pending_conv_states[0]
            cache.discard_pending_states()
            reference.discard_pending_states()

    def test_partial_accept_rescans_to_accepted_depth(self) -> None:
        _require_metal()
        mx.random.seed(0)
        lazy = GDNLazyKernels(enabled=True)
        cache = _state_cache()
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
        cache2 = _state_cache()
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
            num_decode_requests=2,
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
        cache = _state_cache()
        slot_ids = [0]
        span = 4
        cu = [0, span]
        q, k, v, g, beta = _inputs(span)
        mixed_qkv = mx.random.normal((1, span, 64))
        base = cache.recurrent_states[0]

        _scan(lazy, cache, slot_ids, cu, q, k, v, g, beta)
        full_pending = cache.pending_recurrent_states[0]
        state_len = cache.conv_states[0].shape[1]
        full_conv_pending = mixed_qkv[:, -state_len:].astype(cache.dtype)
        cache.set_pending_conv_state(0, slot_ids, full_conv_pending)

        stash = GDNSpecVerifyStep()
        cache.spec_verify_stash = stash
        stash_spec_verify_layer(
            stash,
            cache_idx=0,
            slot_ids=slot_ids,
            cu_seqlens=cu,
            num_decode_requests=1,
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
        assert cache.pending_conv_states[0] is full_conv_pending
        assert cache.recurrent_states[0] is base  # pool untouched

    def test_conv_tail_rolls_back_by_slicing(self) -> None:
        _require_metal()
        mx.random.seed(2)
        lazy = GDNLazyKernels(enabled=True)
        cache = _state_cache()
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
            num_decode_requests=1,
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
        cache = _state_cache()
        slot_ids = [0, 1]
        cu = [0, 4, 6]
        q, k, v, g, beta = _inputs(6)
        mixed_qkv = mx.random.normal((1, 6, 64))

        _scan(lazy, cache, slot_ids, cu, q, k, v, g, beta)

        # Reference per-request truncated scans from the same base.
        cache2 = _state_cache()
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
            num_decode_requests=2,
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
        cache = _state_cache()
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
        cache2 = _state_cache()
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
            num_decode_requests=2,
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
