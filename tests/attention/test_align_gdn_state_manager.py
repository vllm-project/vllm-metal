# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import mlx.core as mx
import numpy as np

from tests.stub_runner import make_gdn_hybrid_plan
from vllm_metal.attention.caches.gdn_cache import GDNPagedStateCache
from vllm_metal.attention.context import PagedAttentionContext
from vllm_metal.attention.runtime.hybrid import HybridPagedAttentionRuntime
from vllm_metal.attention.state import AlignGDNStateManager

BLOCK = 4


def _make_cache(
    *,
    num_layers: int = 2,
    num_blocks: int = 8,
    initial_blocks: int | None = None,
) -> GDNPagedStateCache:
    return GDNPagedStateCache(
        num_layers=num_layers,
        max_seqs=num_blocks,
        conv_kernel_dim=2,
        conv_dim=4,
        num_v_heads=1,
        value_head_dim=4,
        key_head_dim=32,
        initial_seqs=num_blocks if initial_blocks is None else initial_blocks,
        dtype=mx.float32,
    )


def _fill_slab(cache: GDNPagedStateCache, layer: int, slab: int, value: float) -> None:
    conv = cache.conv_states[layer]
    conv[slab] = value
    cache.store_conv_state(layer, conv)
    rec = cache.recurrent_states[layer]
    rec[slab] = value
    cache.store_recurrent_state(layer, rec)
    mx.eval(cache.conv_states[layer], cache.recurrent_states[layer])


def _slab(cache: GDNPagedStateCache, layer: int, slab: int) -> tuple:
    mx.eval(cache.conv_states[layer], cache.recurrent_states[layer])
    return (
        np.array(cache.conv_states[layer][slab]),
        np.array(cache.recurrent_states[layer][slab]),
    )


class TestAlignGDNStateManager:
    def _populate(self, manager, req_ids, tables, positions, kv_block_ids=None):
        ctx = PagedAttentionContext(slot_mapping=[])
        manager.populate_step_context(
            req_ids=req_ids,
            ctx=ctx,
            state_block_ids=tables,
            step_positions=positions,
            kv_block_ids=kv_block_ids,
        )
        return ctx

    def test_fresh_request_zeroes_its_state_block(self) -> None:
        cache = _make_cache()
        manager = AlignGDNStateManager(cache, BLOCK)
        # A previous life leaves bytes in block 3's slab.
        self._populate(manager, ["req-old"], [[[3]]], [(3, 1)])
        _fill_slab(cache, 0, manager.slot_for(3), 7.0)

        ctx = self._populate(
            manager, ["req-A"], [[[3]]], [(0, 2)]
        )  # fresh, 2 tokens into block 3

        assert ctx.gdn_group_slot_mappings == ([manager.slot_for(3)],)
        assert ctx.gdn_slot_mapping is None  # align sets only group mappings
        conv, rec = _slab(cache, 0, manager.slot_for(3))
        assert np.all(conv == 0) and np.all(rec == 0)

    def test_boundary_crossing_copies_forward_and_keeps_checkpoint(self) -> None:
        cache = _make_cache()
        manager = AlignGDNStateManager(cache, BLOCK)
        # Request's state after 4 tokens in block 2.
        self._populate(manager, ["req-A"], [[[2]]], [(3, 1)])
        _fill_slab(cache, 0, manager.slot_for(2), 5.0)
        _fill_slab(cache, 1, manager.slot_for(2), 5.0)

        # num_computed=4 (block boundary), decoding 1 token → lands in block idx 1
        ctx = self._populate(manager, ["req-A"], [[[2, 6]]], [(4, 1)])

        assert ctx.gdn_group_slot_mappings == ([manager.slot_for(6)],)
        for layer in (0, 1):
            conv_src, rec_src = _slab(cache, layer, manager.slot_for(2))
            conv_dst, rec_dst = _slab(cache, layer, manager.slot_for(6))
            np.testing.assert_array_equal(conv_src, 5.0)  # checkpoint intact
            np.testing.assert_array_equal(rec_src, 5.0)
            np.testing.assert_array_equal(conv_dst, 5.0)  # state moved forward
            np.testing.assert_array_equal(rec_dst, 5.0)

        # Slabs belong to scheduler blocks, not requests: releasing the
        # request must leave the checkpoint readable for later restores.
        manager.release_requests({"req-A"})
        conv_src, _ = _slab(cache, 0, manager.slot_for(2))
        np.testing.assert_array_equal(conv_src, 5.0)

    def test_restore_reads_checkpointed_hit_block(self) -> None:
        cache = _make_cache()
        manager = AlignGDNStateManager(cache, BLOCK)
        # Request advances 2 → 6, leaving a checkpoint at block 2.
        self._populate(manager, ["req-A"], [[[2]]], [(3, 1)])
        _fill_slab(cache, 0, manager.slot_for(2), 5.0)
        self._populate(manager, ["req-A"], [[[2, 6]]], [(4, 1)])
        manager.release_requests({"req-A"})

        # Prefix hit: a new request resumes at num_computed=4 from block 2's
        # checkpoint and decodes into block 9 — no re-zeroing may occur.
        ctx = self._populate(manager, ["req-B"], [[[2, 9]]], [(4, 1)])

        assert ctx.gdn_group_slot_mappings == ([manager.slot_for(9)],)
        conv_dst, _ = _slab(cache, 0, manager.slot_for(9))
        np.testing.assert_array_equal(conv_dst, 5.0)

    def test_striped_groups_share_one_pool_without_colliding(self) -> None:
        # Two layers from different groups sharing one physical pool, the
        # layout kv_cache_tensors.shared_by produces: each group's motion
        # must touch only its own block rows of the shared array.
        cache = _make_cache(num_layers=2)
        cache.set_layer_layout([0, 1], [0, 0])
        assert cache.num_state_pools == 1
        manager = AlignGDNStateManager(cache, BLOCK)
        self._populate(manager, ["req-A"], [[[2], [3]]], [(3, 1)])
        _fill_slab(cache, 0, manager.slot_for(2), 5.0)
        _fill_slab(cache, 1, manager.slot_for(3), 8.0)

        # One request; group 0 crosses 2→6, group 1 crosses 3→7.
        ctx = self._populate(manager, ["req-A"], [[[2, 6], [3, 7]]], [(4, 1)])

        assert ctx.gdn_group_slot_mappings == (
            [manager.slot_for(6)],
            [manager.slot_for(7)],
        )
        assert manager.slot_for(6) != manager.slot_for(7)
        conv, _ = _slab(cache, 0, manager.slot_for(6))
        np.testing.assert_array_equal(conv, 5.0)  # group 0 moved its rows
        conv, _ = _slab(cache, 1, manager.slot_for(7))
        np.testing.assert_array_equal(conv, 8.0)  # group 1 moved its rows
        conv, _ = _slab(cache, 0, manager.slot_for(3))  # group 1 checkpoint
        np.testing.assert_array_equal(conv, 8.0)

    def test_pool_grows_by_distinct_mamba_blocks_not_id_span(self) -> None:
        cache = _make_cache(num_blocks=8, initial_blocks=0)
        manager = AlignGDNStateManager(cache, BLOCK)

        # Mamba ids scattered across the shared pool's id space (full
        # attention blocks own the gaps): one row per distinct mamba block,
        # never one row per id below the high-water mark.
        ctx = self._populate(
            manager,
            ["req-A", "req-B"],
            [[[5], [900]], [[300], [901]]],
            [(0, 2), (0, 2)],
        )

        assert ctx.gdn_group_slot_mappings == (
            [manager.slot_for(5), manager.slot_for(300)],
            [manager.slot_for(900), manager.slot_for(901)],
        )
        assert manager.occupied_slots == 4
        assert cache.allocated_seqs == 4  # not 902
        conv, rec = _slab(cache, 0, manager.slot_for(900))
        assert np.all(conv == 0) and np.all(rec == 0)

    def test_slot_reclaimed_when_block_moves_to_kv_group(self) -> None:
        cache = _make_cache(num_blocks=8, initial_blocks=0)
        manager = AlignGDNStateManager(cache, BLOCK)
        self._populate(manager, ["req-A"], [[[2, 6]]], [(4, 1)])
        slot_of_block_6 = manager.slot_for(6)
        _fill_slab(cache, 0, manager.slot_for(2), 5.0)
        manager.release_requests({"req-A"})

        # The scheduler freed block 6 and reallocated it to a full-attention
        # group; block 2 stays a cached checkpoint. Only 6's slot may go.
        self._populate(manager, ["req-B"], [[[2]]], [(3, 1)], kv_block_ids={6, 100})

        assert manager.slot_for(6) is None
        assert manager.free_slots == (slot_of_block_6,)
        conv, _ = _slab(cache, 0, manager.slot_for(2))
        np.testing.assert_array_equal(conv, 5.0)  # checkpoint intact

        # A later mamba block recycles the reclaimed slot; a fresh request
        # zero-inits it, so the previous life's bytes are unobservable.
        cache = _make_cache(num_blocks=8, initial_blocks=0)
        manager = AlignGDNStateManager(cache, BLOCK)
        self._populate(manager, ["req-A"], [[[2, 6]]], [(4, 1)])
        slot_of_block_6 = manager.slot_for(6)
        _fill_slab(cache, 0, slot_of_block_6, 5.0)
        self._populate(manager, ["req-B"], [[[2]]], [(3, 1)], kv_block_ids={6})
        self._populate(manager, ["req-C"], [[[9]]], [(0, 1)])
        recycled = manager.slot_for(9)
        assert recycled == slot_of_block_6 and manager.occupied_slots == 2
        conv, rec = _slab(cache, 0, recycled)
        assert np.all(conv == 0) and np.all(rec == 0)

    def test_cow_pairs_translated_and_foreign_pairs_skipped(self) -> None:
        cache = _make_cache(num_blocks=16, initial_blocks=0)
        manager = AlignGDNStateManager(cache, BLOCK)
        self._populate(manager, ["req-A"], [[[2]]], [(3, 1)])
        _fill_slab(cache, 0, manager.slot_for(2), 5.0)
        _fill_slab(cache, 1, manager.slot_for(2), 5.0)
        allocated_before = cache.allocated_seqs

        # One mamba pair plus one full-attention pair sharing the step: only
        # the mamba pair holds state, and the foreign pair must not grow the
        # pool toward the shared id-span worst case.
        manager.apply_block_copies([(2, 7), (100, 200)])

        assert manager.slot_for(7) is not None
        assert manager.slot_for(100) is None and manager.slot_for(200) is None
        assert cache.allocated_seqs == allocated_before + 1
        for layer in (0, 1):
            conv, rec = _slab(cache, layer, manager.slot_for(7))
            np.testing.assert_array_equal(conv, 5.0)
            np.testing.assert_array_equal(rec, 5.0)

    def test_cow_reuses_slots_retired_in_the_same_step(self) -> None:
        # Retirement must run before the CoW allocation: capacity never
        # shrinks, so allocating first would grow the pool to cover the
        # dst that this step's about-to-be-freed slot could have served.
        cache = _make_cache(num_blocks=8, initial_blocks=0)
        manager = AlignGDNStateManager(cache, BLOCK)
        self._populate(manager, ["req-A"], [[[2]]], [(3, 1)])
        self._populate(manager, ["req-B"], [[[3]]], [(3, 1)])
        assert manager.occupied_slots == 2
        freed_slot = manager.slot_for(3)

        # Block 3 flips to a KV group this step; mamba CoW 2→7 arrives
        # with the same step's KV ids. The dst must reuse the freed slot
        # instead of growing the pool.
        manager.apply_block_copies([(2, 7)], kv_block_ids={3})

        assert manager.slot_for(3) is None
        assert manager.slot_for(7) == freed_slot
        assert manager.occupied_slots == 2
        assert cache.allocated_seqs == 2  # no growth

    def test_pending_state_drains_before_retirement(self) -> None:
        # A deferred compact update parked on a slot must be written back
        # before that slot's block retires inside the same CoW step — the
        # drain settles writes into rows about to be handed out again.
        cache = _make_cache(num_blocks=8, initial_blocks=0)
        manager = AlignGDNStateManager(cache, BLOCK)
        self._populate(manager, ["req-A"], [[[2]]], [(3, 1)])
        self._populate(manager, ["req-B"], [[[3]]], [(3, 1)])
        _fill_slab(cache, 0, manager.slot_for(3), 5.0)
        slot2 = manager.slot_for(2)
        update = mx.full(
            (1,) + cache.conv_states[0].shape[1:],
            9.0,
            dtype=cache.conv_states[0].dtype,
        )
        cache.set_pending_conv_state(0, [slot2], update)

        # Block 2 retires via KV ids while CoW 3→7 reuses its slot.
        manager.apply_block_copies([(3, 7)], kv_block_ids={2})

        assert not cache.has_pending_conv_state(0)  # drained, not dropped
        assert manager.slot_for(2) is None
        assert manager.slot_for(7) == slot2
        conv, _ = _slab(cache, 0, manager.slot_for(7))
        np.testing.assert_array_equal(conv, 5.0)  # src bytes won the slot

    def test_shared_pool_cow_after_retirement_keeps_aliased_content(self) -> None:
        # One physical pool shared by two groups: CoW + retirement in the
        # same step must land dst rows in the shared array (visible
        # through either layer index) and leave the other group's
        # still-mapped rows untouched.
        cache = _make_cache(num_layers=2, num_blocks=8, initial_blocks=0)
        cache.set_layer_layout([0, 1], [0, 0])
        assert cache.num_state_pools == 1
        manager = AlignGDNStateManager(cache, BLOCK)
        self._populate(manager, ["req-A"], [[[2, 6], [3, 4]]], [(4, 1)])
        freed_slot = manager.slot_for(3)
        _fill_slab(cache, 0, manager.slot_for(2), 5.0)
        _fill_slab(cache, 1, manager.slot_for(4), 8.0)

        # Group 1's block 3 flips to a KV group; group 0's CoW 2→7
        # arrives with the same step's KV ids.
        manager.apply_block_copies([(2, 7)], kv_block_ids={3})

        assert manager.slot_for(3) is None
        assert cache.allocated_seqs == 4  # no growth beyond the 4 mapped
        assert manager.slot_for(7) == freed_slot
        conv_l0, _ = _slab(cache, 0, manager.slot_for(7))
        conv_l1, _ = _slab(cache, 1, manager.slot_for(7))
        np.testing.assert_array_equal(conv_l0, 5.0)
        np.testing.assert_array_equal(conv_l1, 5.0)  # alias sees the copy
        conv_survivor, _ = _slab(cache, 1, manager.slot_for(4))
        np.testing.assert_array_equal(conv_survivor, 8.0)  # untouched


class TestHybridAlignRuntime:
    def _make_runtime(self) -> HybridPagedAttentionRuntime:
        return HybridPagedAttentionRuntime(
            hybrid_plan=make_gdn_hybrid_plan(
                4,
                range(1, 4, 2),
                conv_kernel_dim=2,
                conv_dim=4,
                num_v_heads=1,
                value_head_dim=4,
                key_head_dim=32,
            ),
            max_num_seqs=2,
            num_kv_heads=1,
            head_dim=4,
            block_size=BLOCK,
            dtype=mx.float32,
            mamba_cache_mode="align",
        )

    def test_adopts_shared_layout_before_materializing_state(self) -> None:
        runtime = self._make_runtime()
        runtime.initialize(num_blocks=8)

        assert runtime.state_cache.allocated_seqs == 0
        runtime.adopt_scheduler_group(
            0,
            BLOCK,
            state_group_indices=(1, 2),
            layer_group_ordinals=[0, 1],
            layer_pool_ordinals=[0, 0],
        )
        runtime.state_cache.ensure_capacity(2)

        assert runtime.state_cache.num_state_pools == 1
        assert runtime.state_cache.conv_states[0] is runtime.state_cache.conv_states[1]
        assert (
            runtime.state_cache.recurrent_states[0]
            is runtime.state_cache.recurrent_states[1]
        )
