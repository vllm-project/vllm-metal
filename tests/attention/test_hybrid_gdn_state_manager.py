# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from vllm_metal.attention.caches.gdn_cache import GDNPagedStateCache
from vllm_metal.attention.context import PagedAttentionContext
from vllm_metal.attention.runtime.hybrid import HybridPagedAttentionRuntime
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
        assert outputs[1] is cache.pending_conv_states[0]
        assert outputs[2] is cache.pending_recurrent_states[0]

    def test_reset_requests_zeroes_slot_and_keeps_mapping(self) -> None:
        cache = _make_cache(num_layers=1, max_seqs=2)
        manager = HybridGDNStateManager(cache)
        slot = manager.assign_step_slots(["resumed"])[0]

        conv = cache.conv_states[0]
        conv[slot] = 3
        cache.conv_states[0] = conv
        recurrent = cache.recurrent_states[0]
        recurrent[slot] = 5
        cache.recurrent_states[0] = recurrent
        mx.eval(cache.conv_states[0], cache.recurrent_states[0])

        manager.reset_requests({"resumed"})
        mx.eval(cache.conv_states[0], cache.recurrent_states[0])

        assert manager.request_slots == {"resumed": slot}
        assert manager.free_slots == ()
        assert np.all(np.array(cache.conv_states[0][slot]) == 0)
        assert np.all(np.array(cache.recurrent_states[0][slot]) == 0)
        assert manager.assign_step_slots(["resumed"]) == [slot]

    def test_reset_requests_does_not_touch_other_live_slot(self) -> None:
        cache = _make_cache(num_layers=1, max_seqs=2)
        manager = HybridGDNStateManager(cache)
        slot_a, slot_b = manager.assign_step_slots(["resumed", "live"])

        conv = cache.conv_states[0]
        conv[slot_a] = 3
        conv[slot_b] = 11
        cache.conv_states[0] = conv
        recurrent = cache.recurrent_states[0]
        recurrent[slot_a] = 5
        recurrent[slot_b] = 13
        cache.recurrent_states[0] = recurrent
        mx.eval(cache.conv_states[0], cache.recurrent_states[0])

        manager.reset_requests({"resumed"})
        mx.eval(cache.conv_states[0], cache.recurrent_states[0])

        assert np.all(np.array(cache.conv_states[0][slot_a]) == 0)
        assert np.all(np.array(cache.recurrent_states[0][slot_a]) == 0)
        np.testing.assert_array_equal(np.array(cache.conv_states[0][slot_b]), 11)
        np.testing.assert_array_equal(np.array(cache.recurrent_states[0][slot_b]), 13)

    def test_reset_requests_ignores_unknown_request(self) -> None:
        cache = _make_cache(num_layers=1, max_seqs=2)
        manager = HybridGDNStateManager(cache)
        manager.assign_step_slots(["req-A"])

        manager.reset_requests({"never-assigned"})

        assert manager.request_slots == {"req-A": 0}

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
            num_layers=2,
            full_attention_interval=2,
            max_num_seqs=2,
            num_kv_heads=1,
            head_dim=4,
            linear_num_v_heads=1,
            linear_key_head_dim=32,
            linear_value_head_dim=4,
            linear_conv_kernel_dim=2,
            linear_conv_dim=4,
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
