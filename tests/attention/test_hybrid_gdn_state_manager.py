# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from tests.stub_runner import make_bailing_hybrid_plan, make_gdn_hybrid_plan
from vllm_metal.attention.caches.gdn_cache import GDNPagedStateCache
from vllm_metal.attention.context import (
    PagedAttentionContext,
    clear_context,
    set_context,
)
from vllm_metal.attention.impls.kda import KDAPagedAttentionWrapper
from vllm_metal.attention.impls.mla import MLAPagedAttentionWrapper
from vllm_metal.attention.runtime.hybrid import (
    BailingHybridPagedAttentionRuntime,
    HybridPagedAttentionRuntime,
)
from vllm_metal.attention.state import HybridGDNStateManager


def _make_cache(*, num_layers: int = 2, max_seqs: int = 2) -> GDNPagedStateCache:
    return GDNPagedStateCache(
        num_layers=num_layers,
        max_seqs=max_seqs,
        conv_kernel_dim=4,
        conv_dim=64,
        num_v_heads=4,
        value_head_dim=16,
        key_head_dim=16,
        initial_seqs=0,
        dtype=mx.float16,
    )


def _make_context() -> PagedAttentionContext:
    return PagedAttentionContext(slot_mapping=[])


def _make_bailing_runtime(num_layers: int) -> BailingHybridPagedAttentionRuntime:
    return BailingHybridPagedAttentionRuntime(
        hybrid_plan=make_bailing_hybrid_plan(num_layers),
        max_num_seqs=2,
        num_kv_heads=1,
        head_dim=6,
        block_size=4,
        dtype=mx.float32,
    )


class TestHybridGDNStateManager:
    def test_assign_step_slots_grows_state_cache_once(self) -> None:
        cache = _make_cache(max_seqs=3)
        manager = HybridGDNStateManager(cache)

        slots = manager.assign_step_slots(["req-A", "req-B"])

        assert slots == [0, 1]
        assert cache.allocated_seqs == 2
        assert manager.request_slots == {"req-A": 0, "req-B": 1}
        assert manager.free_slots == ()

    def test_assign_step_slots_is_atomic_on_grow_failure(self) -> None:
        cache = _make_cache(max_seqs=1)
        manager = HybridGDNStateManager(cache)

        with pytest.raises(RuntimeError, match="more slots than max_num_seqs"):
            manager.assign_step_slots(["req-A", "req-B"])

        assert cache.allocated_seqs == 0
        assert manager.request_slots == {}
        assert manager.free_slots == ()

    def test_populate_step_context_sets_gdn_slot_mapping(self) -> None:
        cache = _make_cache()
        manager = HybridGDNStateManager(cache)
        ctx = _make_context()

        manager.populate_step_context(req_ids=["req-A", "req-B"], ctx=ctx)

        assert ctx.gdn_slot_mapping == [0, 1]
        assert manager.request_slots == {"req-A": 0, "req-B": 1}

    def test_extend_forward_eval_outputs_uses_pending_compact_state(self) -> None:
        cache = GDNPagedStateCache(
            num_layers=1,
            max_seqs=2,
            conv_kernel_dim=2,
            conv_dim=4,
            num_v_heads=1,
            value_head_dim=4,
            key_head_dim=32,
            initial_seqs=0,
            dtype=mx.float32,
        )
        manager = HybridGDNStateManager(cache)
        manager.assign_step_slots(["req-A", "req-B"])
        cache.set_pending_conv_state(0, [1], mx.full((1, 1, 4), 7, dtype=mx.float32))
        cache.set_pending_recurrent_state(
            0,
            [1],
            mx.full((1, 1, 4, 32), 9, dtype=mx.float32),
        )
        logits = mx.array([0], dtype=mx.float32)
        outputs = [logits]

        manager.extend_forward_eval_outputs(outputs)

        assert outputs[0] is logits
        # Pending compact updates ride the submission; stable pool arrays may
        # accompany them (shared pools carry sibling layers' state).
        assert any(a is cache.pending_conv_states[0] for a in outputs[1:])
        assert any(a is cache.pending_recurrent_states[0] for a in outputs[1:])

    def test_release_requests_applies_pending_states_before_reuse(self) -> None:
        cache = GDNPagedStateCache(
            num_layers=1,
            max_seqs=2,
            conv_kernel_dim=2,
            conv_dim=4,
            num_v_heads=1,
            value_head_dim=4,
            key_head_dim=32,
            initial_seqs=0,
            dtype=mx.float32,
        )
        manager = HybridGDNStateManager(cache)
        manager.assign_step_slots(["done"])
        slot = manager.request_slots["done"]

        cache.set_pending_conv_state(0, [slot], mx.full((1, 1, 4), 7, dtype=mx.float32))
        cache.set_pending_recurrent_state(
            0,
            [slot],
            mx.full((1, 1, 4, 32), 9, dtype=mx.float32),
        )

        manager.release_requests({"done"})

        assert not cache.has_pending_conv_state(0)
        assert not cache.has_pending_recurrent_state(0)
        mx.eval(cache.conv_states[0], cache.recurrent_states[0])
        np.testing.assert_array_equal(np.array(cache.conv_states[0][slot]), 7)
        np.testing.assert_array_equal(np.array(cache.recurrent_states[0][slot]), 9)
        assert manager.free_slots == (slot,)
        assert manager.needs_materialize is True

    def test_materialize_pending_state_clears_flag_once(self) -> None:
        cache = _make_cache(max_seqs=2)
        manager = HybridGDNStateManager(cache)
        manager.assign_step_slots(["req-A", "req-B"])

        manager.release_requests({"req-A", "req-B"})
        manager.materialize_pending_state()
        manager.materialize_pending_state()

        assert manager.request_slots == {}
        assert sorted(manager.free_slots) == [0, 1]
        assert manager.needs_materialize is False

    def test_materialize_pending_state_applies_same_step_reused_slot_updates(
        self,
    ) -> None:
        cache = GDNPagedStateCache(
            num_layers=1,
            max_seqs=2,
            conv_kernel_dim=2,
            conv_dim=4,
            num_v_heads=1,
            value_head_dim=4,
            key_head_dim=32,
            initial_seqs=0,
            dtype=mx.float32,
        )
        manager = HybridGDNStateManager(cache)
        released_slot = manager.assign_step_slots(["done"])[0]

        manager.release_requests({"done"})
        reused_slot = manager.assign_step_slots(["next"])[0]
        assert reused_slot == released_slot

        next_conv_state = mx.full((1, 1, 4), 7, dtype=mx.float32)
        next_recurrent_state = mx.full((1, 1, 4, 32), 9, dtype=mx.float32)
        cache.set_pending_conv_state(0, [reused_slot], next_conv_state)
        cache.set_pending_recurrent_state(0, [reused_slot], next_recurrent_state)

        manager.materialize_pending_state()

        assert not cache.has_pending_conv_state(0)
        assert not cache.has_pending_recurrent_state(0)
        mx.eval(cache.conv_states[0], cache.recurrent_states[0])
        np.testing.assert_array_equal(np.array(cache.conv_states[0][reused_slot]), 7)
        np.testing.assert_array_equal(
            np.array(cache.recurrent_states[0][reused_slot]),
            9,
        )
        assert manager.needs_materialize is False

    def test_reused_slot_is_zeroed_before_new_request_uses_it(self) -> None:
        cache = _make_cache(num_layers=1, max_seqs=2)
        manager = HybridGDNStateManager(cache)
        slot = manager.assign_step_slots(["req-A"])[0]

        conv = cache.conv_states[0]
        conv[slot] = 1
        cache.conv_states[0] = conv
        recurrent = cache.recurrent_states[0]
        recurrent[slot] = 1
        cache.recurrent_states[0] = recurrent
        mx.eval(cache.conv_states[0], cache.recurrent_states[0])

        manager.release_requests({"req-A"})
        reused_slot = manager.assign_step_slots(["req-B"])[0]
        mx.eval(cache.conv_states[0], cache.recurrent_states[0])

        assert reused_slot == slot
        assert np.all(np.array(cache.conv_states[0][slot]) == 0)
        assert np.all(np.array(cache.recurrent_states[0][slot]) == 0)

    def test_reused_slot_does_not_touch_other_live_slot(self) -> None:
        cache = _make_cache(num_layers=1, max_seqs=2)
        manager = HybridGDNStateManager(cache)
        slot_a, slot_b = manager.assign_step_slots(["req-A", "req-B"])

        conv_states = cache.conv_states[0]
        conv_states[slot_a] = 5
        conv_states[slot_b] = 11
        cache.conv_states[0] = conv_states

        recurrent_states = cache.recurrent_states[0]
        recurrent_states[slot_a] = 3
        recurrent_states[slot_b] = 13
        cache.recurrent_states[0] = recurrent_states
        mx.eval(cache.conv_states[0], cache.recurrent_states[0])

        manager.release_requests({"req-A"})
        reused_slot = manager.assign_step_slots(["req-C"])[0]
        mx.eval(cache.conv_states[0], cache.recurrent_states[0])

        assert reused_slot == slot_a
        assert np.all(np.array(cache.conv_states[0][slot_a]) == 0)
        assert np.all(np.array(cache.recurrent_states[0][slot_a]) == 0)
        np.testing.assert_array_equal(np.array(cache.conv_states[0][slot_b]), 11)
        np.testing.assert_array_equal(np.array(cache.recurrent_states[0][slot_b]), 13)


class TestHybridPagedAttentionRuntime:
    def test_initialize_wires_gdn_state_manager_delegation(self) -> None:
        runtime = HybridPagedAttentionRuntime(
            hybrid_plan=make_gdn_hybrid_plan(
                2,
                range(1, 2, 2),
                conv_kernel_dim=2,
                conv_dim=4,
                num_v_heads=1,
                value_head_dim=4,
                key_head_dim=32,
            ),
            max_num_seqs=2,
            num_kv_heads=1,
            head_dim=4,
            block_size=4,
            dtype=mx.float32,
        )
        runtime.initialize(num_blocks=2)

        ctx = _make_context()
        runtime.populate_step_context(req_ids=["req-A"], ctx=ctx)

        assert ctx.gdn_slot_mapping == [0]

        cache = runtime.state_cache
        slot = ctx.gdn_slot_mapping[0]
        cache.set_pending_conv_state(0, [slot], mx.full((1, 1, 4), 7, dtype=mx.float32))
        cache.set_pending_recurrent_state(
            0,
            [slot],
            mx.full((1, 1, 4, 32), 9, dtype=mx.float32),
        )

        runtime.release_requests({"req-A"})
        runtime.materialize_pending_state()

        assert not cache.has_pending_conv_state(0)
        assert not cache.has_pending_recurrent_state(0)
        assert runtime.gdn_state_manager.needs_materialize is False


class _FakeBailingKDA(nn.Module):
    q_proj = object()
    k_proj = object()
    v_proj = object()
    q_conv1d = object()
    k_conv1d = object()
    v_conv1d = object()
    projection_size = 4
    conv_kernel_size = 2


class _FakeBailingMLA(nn.Module):
    pass


class _FakeBailingLayer:
    def __init__(self, attention: nn.Module) -> None:
        self.attention = attention


class TestKDAPagedAttentionWrapper:
    def test_multi_request_prefill_and_decode_match_mlx_lm(self) -> None:
        from mlx_lm.models.bailing_moe_v3 import BailingKDA, ModelArgs
        from mlx_lm.models.cache import ArraysCache

        mx.random.seed(17)
        args = ModelArgs(
            hidden_size=16,
            num_attention_heads=2,
            head_dim=4,
            short_conv_kernel_size=3,
        )
        inner = BailingKDA(args)
        projection_size = args.num_attention_heads * args.head_dim
        state_cache = GDNPagedStateCache(
            num_layers=1,
            max_seqs=2,
            conv_kernel_dim=args.short_conv_kernel_size,
            conv_dim=3 * projection_size,
            num_v_heads=args.num_attention_heads,
            value_head_dim=args.head_dim,
            key_head_dim=args.head_dim,
            initial_seqs=2,
            dtype=mx.float32,
        )
        wrapper = KDAPagedAttentionWrapper(inner, 0, 0, state_cache)
        references = [ArraysCache(size=4), ArraysCache(size=4)]

        for cu_seqlens, grouped_slots in (([0, 3, 5], False), ([0, 1, 2], True)):
            x = mx.random.normal((1, cu_seqlens[-1], args.hidden_size)).astype(
                mx.float32
            )
            expected = mx.concatenate(
                [
                    inner(x[:, start:end], cache=reference)
                    for start, end, reference in zip(
                        cu_seqlens[:-1], cu_seqlens[1:], references, strict=True
                    )
                ],
                axis=1,
            )

            set_context(
                PagedAttentionContext(
                    slot_mapping=[],
                    cu_seqlens=cu_seqlens,
                    gdn_slot_mapping=None if grouped_slots else [0, 1],
                    gdn_group_slot_mappings=[[0, 1]] if grouped_slots else None,
                )
            )
            try:
                actual = wrapper(x)
            finally:
                clear_context()

            mx.eval(actual, expected)
            np.testing.assert_allclose(
                np.array(actual), np.array(expected), rtol=1e-5, atol=1e-5
            )

        expected_conv = mx.concatenate(
            [mx.concatenate(reference.cache[:3], axis=-1) for reference in references]
        )
        expected_recurrent = mx.concatenate([reference[3] for reference in references])
        np.testing.assert_array_equal(
            np.array(state_cache.conv_states[0]), np.array(expected_conv)
        )
        np.testing.assert_array_equal(
            np.array(state_cache.recurrent_states[0]), np.array(expected_recurrent)
        )


class TestBailingHybridPagedAttentionRuntime:
    def test_patches_interleaved_layers_with_compact_cache_indices(self) -> None:
        runtime = _make_bailing_runtime(4)
        runtime.initialize(num_blocks=3)
        model = SimpleNamespace(
            model=SimpleNamespace(
                layers=[
                    _FakeBailingLayer(_FakeBailingKDA()),
                    _FakeBailingLayer(_FakeBailingMLA()),
                    _FakeBailingLayer(_FakeBailingKDA()),
                    _FakeBailingLayer(_FakeBailingMLA()),
                ]
            )
        )

        assert runtime.patch_model(model) == 4

        layers = model.model.layers
        assert all(
            isinstance(layers[idx].attention, KDAPagedAttentionWrapper)
            for idx in (0, 2)
        )
        assert all(
            isinstance(layers[idx].attention, MLAPagedAttentionWrapper)
            for idx in (1, 3)
        )
        assert [layers[idx].attention._kda_cache_idx for idx in (0, 2)] == [0, 1]
        assert [layers[idx].attention._mla_layer_idx for idx in (1, 3)] == [0, 1]
        assert runtime._cache.num_layers == 2
        assert runtime.state_cache.num_layers == 2

    def test_repatches_wrapped_layers_and_rebinds_caches(self) -> None:
        runtime = _make_bailing_runtime(2)
        runtime.initialize(num_blocks=3)
        model = SimpleNamespace(
            model=SimpleNamespace(
                layers=[
                    _FakeBailingLayer(_FakeBailingKDA()),
                    _FakeBailingLayer(_FakeBailingMLA()),
                ]
            )
        )

        assert runtime.patch_model(model) == 2
        runtime.initialize(num_blocks=4)
        assert runtime.patch_model(model) == 2

        kda = model.model.layers[0].attention
        mla = model.model.layers[1].attention
        assert kda._kda_state_cache is runtime.state_cache
        assert mla._mla_latent_cache is runtime._cache
