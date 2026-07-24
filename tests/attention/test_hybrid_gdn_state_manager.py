# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
import torch
from vllm.v1.kv_cache_interface import MambaSpec

from vllm_metal.attention.caches.gdn_cache import GDNPagedStateCache
from vllm_metal.attention.context import PagedAttentionContext
from vllm_metal.attention.runtime.hybrid import (
    HybridPagedAttentionRuntime,
    _build_linear_layer_spec,
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


def test_linear_spec_bills_float32_recurrent_state() -> None:
    spec = _build_linear_layer_spec(
        conv_kernel_dim=4,
        conv_dim=64,
        num_v_heads=4,
        value_head_dim=16,
        key_head_dim=16,
        torch_dtype=torch.float16,
        block_size=4,
        mamba_cache_mode="align",
    )

    assert spec.dtypes == (torch.float16, torch.float32)
    assert spec.page_size_bytes == (3 * 64 * 2) + (4 * 16 * 16 * 4)


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

    def test_scheduler_block_checkpoint_restores_all_groups_immutably(self) -> None:
        cache = _make_cache(num_layers=2, max_seqs=1)
        manager = HybridGDNStateManager(cache, block_size=4)
        manager.configure_cache_groups(
            num_blocks=32,
            block_size=4,
            mamba_group_layers={0: (0,), 2: (1,)},
        )
        block_tables = ([10], [20], [11])
        slot = manager.assign_step_slots(["seed"])[0]
        for layer_idx, (conv_value, recurrent_value) in enumerate(((7, 9), (8, 10))):
            conv = cache.conv_states[layer_idx]
            conv[slot] = conv_value
            cache.conv_states[layer_idx] = conv
            recurrent = cache.recurrent_states[layer_idx]
            recurrent[slot] = recurrent_value
            cache.recurrent_states[layer_idx] = recurrent
        mx.eval(*cache.updated_state_arrays())

        manager.checkpoint_blocks([("seed", block_tables, 4)])
        assert manager.block_snapshot_ids == (10, 11)

        # Later live-slot mutation must not change the scheduler block payload.
        for layer_idx in range(2):
            conv = cache.conv_states[layer_idx]
            conv[slot] = 100 + layer_idx
            cache.conv_states[layer_idx] = conv
            recurrent = cache.recurrent_states[layer_idx]
            recurrent[slot] = 200 + layer_idx
            cache.recurrent_states[layer_idx] = recurrent
        manager.release_requests({"seed"})

        restored_slot = manager.assign_step_slots(["hit"])[0]
        assert restored_slot == slot
        assert manager.restore_prefix("hit", block_tables, 4) is True
        np.testing.assert_array_equal(np.array(cache.conv_states[0][slot]), 7)
        np.testing.assert_array_equal(np.array(cache.recurrent_states[0][slot]), 9)
        np.testing.assert_array_equal(np.array(cache.conv_states[1][slot]), 8)
        np.testing.assert_array_equal(np.array(cache.recurrent_states[1][slot]), 10)

    def test_checkpoint_applies_pending_conv_and_recurrent_state(self) -> None:
        cache = _make_cache(num_layers=1, max_seqs=1)
        manager = HybridGDNStateManager(cache, block_size=4)
        manager.configure_cache_groups(
            num_blocks=16,
            block_size=4,
            mamba_group_layers={1: (0,)},
        )
        slot = manager.assign_step_slots(["seed"])[0]
        cache.set_pending_conv_state(
            0,
            [slot],
            mx.full((1, 3, 64), 7, dtype=mx.float16),
        )
        cache.set_pending_recurrent_state(
            0,
            [slot],
            mx.full((1, 4, 16, 16), 9, dtype=mx.float32),
        )

        manager.checkpoint_blocks([("seed", ([2], [3]), 4)])
        assert not cache.has_pending_conv_state(0)
        assert not cache.has_pending_recurrent_state(0)
        manager.release_requests({"seed"})
        manager.assign_step_slots(["hit"])
        manager.restore_prefix("hit", ([2], [3]), 4)

        np.testing.assert_array_equal(np.array(cache.conv_states[0][slot]), 7)
        np.testing.assert_array_equal(np.array(cache.recurrent_states[0][slot]), 9)
        assert cache.conv_states[0].dtype == mx.float16
        assert cache.recurrent_states[0].dtype == mx.float32

    def test_scheduler_hit_without_physical_block_state_fails_closed(self) -> None:
        cache = _make_cache(num_layers=1, max_seqs=1)
        manager = HybridGDNStateManager(cache, block_size=4)
        manager.configure_cache_groups(
            num_blocks=16,
            block_size=4,
            mamba_group_layers={1: (0,)},
        )
        slot = manager.assign_step_slots(["hit"])[0]

        with pytest.raises(RuntimeError, match="physical Mamba block 3"):
            manager.restore_prefix("hit", ([2], [3]), 4)

        np.testing.assert_array_equal(np.array(cache.conv_states[0][slot]), 0)
        np.testing.assert_array_equal(np.array(cache.recurrent_states[0][slot]), 0)

    def test_scheduler_physical_block_reuse_invalidates_checkpoint(self) -> None:
        cache = _make_cache(num_layers=1, max_seqs=1)
        manager = HybridGDNStateManager(cache, block_size=4)
        manager.configure_cache_groups(
            num_blocks=16,
            block_size=4,
            mamba_group_layers={0: (0,)},
        )
        manager.assign_step_slots(["seed"])
        manager.checkpoint_blocks([("seed", ([5],), 4)])
        checkpoint_bytes = manager.block_snapshot_stats["bytes"]
        assert checkpoint_bytes > 0
        manager.invalidate_blocks([5])
        manager.release_requests({"seed"})
        manager.assign_step_slots(["hit"])

        with pytest.raises(RuntimeError, match="physical Mamba block 5"):
            manager.restore_prefix("hit", ([5],), 4)

        assert manager.block_snapshot_stats == {
            "count": 0,
            "bytes": 0,
            "peak_count": 1,
            "peak_bytes": checkpoint_bytes,
            "stores": 1,
            "replacements": 0,
            "invalidations": 1,
            "hits": 0,
            "misses": 1,
        }

    def test_checkpoint_replacement_keeps_snapshot_bytes_bounded(self) -> None:
        cache = _make_cache(num_layers=1, max_seqs=1)
        manager = HybridGDNStateManager(cache, block_size=4)
        manager.configure_cache_groups(
            num_blocks=16,
            block_size=4,
            mamba_group_layers={0: (0,)},
        )
        manager.assign_step_slots(["seed"])

        manager.checkpoint_blocks([("seed", ([5],), 4)])
        first_stats = manager.block_snapshot_stats
        manager.checkpoint_blocks([("seed", ([5],), 4)])

        assert manager.block_snapshot_ids == (5,)
        assert manager.block_snapshot_stats == {
            "count": 1,
            "bytes": first_stats["bytes"],
            "peak_count": 1,
            "peak_bytes": first_stats["bytes"],
            "stores": 2,
            "replacements": 1,
            "invalidations": 0,
            "hits": 0,
            "misses": 0,
        }

    def test_partial_block_state_is_not_published(self) -> None:
        cache = _make_cache(num_layers=1, max_seqs=1)
        manager = HybridGDNStateManager(cache, block_size=4)
        manager.configure_cache_groups(
            num_blocks=16,
            block_size=4,
            mamba_group_layers={0: (0,)},
        )
        manager.assign_step_slots(["partial"])

        manager.checkpoint_blocks([("partial", ([5],), 3)])

        assert manager.block_snapshot_ids == ()
        assert manager.block_snapshot_stats == {
            "count": 0,
            "bytes": 0,
            "peak_count": 0,
            "peak_bytes": 0,
            "stores": 0,
            "replacements": 0,
            "invalidations": 0,
            "hits": 0,
            "misses": 0,
        }

    def test_divergent_suffixes_restore_same_authoritative_prefix(self) -> None:
        cache = _make_cache(num_layers=1, max_seqs=1)
        manager = HybridGDNStateManager(cache, block_size=4)
        manager.configure_cache_groups(
            num_blocks=32,
            block_size=4,
            mamba_group_layers={0: (0,)},
        )
        prefix_tables = ([6],)
        slot = manager.assign_step_slots(["seed"])[0]
        conv = cache.conv_states[0]
        conv[slot] = 5
        cache.conv_states[0] = conv
        recurrent = cache.recurrent_states[0]
        recurrent[slot] = 6
        cache.recurrent_states[0] = recurrent
        manager.checkpoint_blocks([("seed", prefix_tables, 4)])
        manager.release_requests({"seed"})

        manager.assign_step_slots(["suffix-A"])
        manager.restore_prefix("suffix-A", prefix_tables, 4)
        conv = cache.conv_states[0]
        conv[slot] = 17
        cache.conv_states[0] = conv
        recurrent = cache.recurrent_states[0]
        recurrent[slot] = 19
        cache.recurrent_states[0] = recurrent
        manager.checkpoint_blocks([("suffix-A", ([6, 7],), 5)])
        manager.release_requests({"suffix-A"})

        manager.assign_step_slots(["suffix-B"])
        manager.restore_prefix("suffix-B", prefix_tables, 4)
        np.testing.assert_array_equal(np.array(cache.conv_states[0][slot]), 5)
        np.testing.assert_array_equal(np.array(cache.recurrent_states[0][slot]), 6)


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

    def test_configure_cache_groups_maps_every_linear_layer(self) -> None:
        runtime = HybridPagedAttentionRuntime(
            num_layers=4,
            full_attention_interval=4,
            max_num_seqs=1,
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
        runtime.initialize(num_blocks=16)
        mamba_spec = MambaSpec(
            shapes=((1,),),
            dtypes=(torch.float32,),
            block_size=4,
            mamba_cache_mode="align",
        )
        runtime.configure_cache_groups(
            SimpleNamespace(
                num_blocks=16,
                kv_cache_groups=[
                    SimpleNamespace(
                        kv_cache_spec=mamba_spec,
                        layer_names=["layers.0.linear_attn"],
                    ),
                    SimpleNamespace(
                        kv_cache_spec=mamba_spec,
                        layer_names=["layers.1.linear_attn"],
                    ),
                    SimpleNamespace(
                        kv_cache_spec=object(),
                        layer_names=["layers.3.self_attn"],
                    ),
                    SimpleNamespace(
                        kv_cache_spec=mamba_spec,
                        layer_names=["layers.2.linear_attn"],
                    ),
                ],
            )
        )
        ctx = _make_context()
        runtime.populate_step_context(req_ids=["req"], ctx=ctx)
        runtime.checkpoint_blocks(
            [("req", ([3], [4], [5], [6]), 4)]
        )

        assert runtime.gdn_state_manager.block_snapshot_ids == (3, 4, 6)
