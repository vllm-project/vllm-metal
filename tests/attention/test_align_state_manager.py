# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import mlx.core as mx
import numpy as np
import pytest

from tests.stub_runner import make_gdn_hybrid_plan
from vllm_metal.attention.caches.gdn_cache import GDNPagedStateCache
from vllm_metal.attention.context import PagedAttentionContext
from vllm_metal.attention.runtime.hybrid import HybridPagedAttentionRuntime
from vllm_metal.attention.state import AlignStateManager
from vllm_metal.state_budget import StateCacheStep

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


class TestAlignStateManager:
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
        manager = AlignStateManager(cache, BLOCK)
        # A previous life leaves bytes in block 3's slab.
        self._populate(manager, ["req-old"], [[[3]]], [(3, 1)])
        _fill_slab(cache, 0, manager.slot_for(3), 7.0)

        ctx = self._populate(
            manager, ["req-A"], [[[3]]], [(0, 2)]
        )  # fresh, 2 tokens into block 3

        assert ctx.state_group_slot_mappings == ([manager.slot_for(3)],)
        assert ctx.state_slot_mapping is None  # align sets only group mappings
        conv, rec = _slab(cache, 0, manager.slot_for(3))
        assert np.all(conv == 0) and np.all(rec == 0)

    def test_boundary_crossing_copies_forward_and_keeps_checkpoint(self) -> None:
        cache = _make_cache()
        manager = AlignStateManager(cache, BLOCK)
        # Request's state after 4 tokens in block 2.
        self._populate(manager, ["req-A"], [[[2]]], [(3, 1)])
        _fill_slab(cache, 0, manager.slot_for(2), 5.0)
        _fill_slab(cache, 1, manager.slot_for(2), 5.0)

        # num_computed=4 (block boundary), decoding 1 token → lands in block idx 1
        ctx = self._populate(manager, ["req-A"], [[[2, 6]]], [(4, 1)])

        assert ctx.state_group_slot_mappings == ([manager.slot_for(6)],)
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
        manager = AlignStateManager(cache, BLOCK)
        # Request advances 2 → 6, leaving a checkpoint at block 2.
        self._populate(manager, ["req-A"], [[[2]]], [(3, 1)])
        _fill_slab(cache, 0, manager.slot_for(2), 5.0)
        self._populate(manager, ["req-A"], [[[2, 6]]], [(4, 1)])
        manager.release_requests({"req-A"})

        # Prefix hit: a new request resumes at num_computed=4 from block 2's
        # checkpoint and decodes into block 9 — no re-zeroing may occur.
        ctx = self._populate(manager, ["req-B"], [[[2, 9]]], [(4, 1)])

        assert ctx.state_group_slot_mappings == ([manager.slot_for(9)],)
        conv_dst, _ = _slab(cache, 0, manager.slot_for(9))
        np.testing.assert_array_equal(conv_dst, 5.0)

    def test_striped_groups_share_one_pool_without_colliding(self) -> None:
        # Two layers from different groups sharing one physical pool, the
        # layout kv_cache_tensors.shared_by produces: each group's motion
        # must touch only its own block rows of the shared array.
        cache = _make_cache(num_layers=2)
        cache.set_layer_layout([0, 1], [0, 0])
        assert cache.num_state_pools == 1
        manager = AlignStateManager(cache, BLOCK)
        self._populate(manager, ["req-A"], [[[2], [3]]], [(3, 1)])
        _fill_slab(cache, 0, manager.slot_for(2), 5.0)
        _fill_slab(cache, 1, manager.slot_for(3), 8.0)

        # One request; group 0 crosses 2→6, group 1 crosses 3→7.
        ctx = self._populate(manager, ["req-A"], [[[2, 6], [3, 7]]], [(4, 1)])

        assert ctx.state_group_slot_mappings == (
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
        manager = AlignStateManager(cache, BLOCK)

        # Mamba ids scattered across the shared pool's id space (full
        # attention blocks own the gaps): one row per distinct mamba block,
        # never one row per id below the high-water mark.
        ctx = self._populate(
            manager,
            ["req-A", "req-B"],
            [[[5], [900]], [[300], [901]]],
            [(0, 2), (0, 2)],
        )

        assert ctx.state_group_slot_mappings == (
            [manager.slot_for(5), manager.slot_for(300)],
            [manager.slot_for(900), manager.slot_for(901)],
        )
        assert manager.occupied_slots == 4
        assert cache.allocated_seqs == 4  # not 902
        conv, rec = _slab(cache, 0, manager.slot_for(900))
        assert np.all(conv == 0) and np.all(rec == 0)

    def test_slot_reclaimed_when_block_moves_to_kv_group(self) -> None:
        cache = _make_cache(num_blocks=8, initial_blocks=0)
        manager = AlignStateManager(cache, BLOCK)
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
        manager = AlignStateManager(cache, BLOCK)
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
        manager = AlignStateManager(cache, BLOCK)
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
        manager = AlignStateManager(cache, BLOCK)
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
        manager = AlignStateManager(cache, BLOCK)
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
        manager = AlignStateManager(cache, BLOCK)
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
    def _make_runtime(self, capacity=None) -> HybridPagedAttentionRuntime:
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
            state_slot_capacity=capacity,
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

    def test_bounded_pool_preallocates_only_after_adopting_shared_layout(self) -> None:
        runtime = self._make_runtime(capacity=3)
        runtime.initialize(num_blocks=200)
        assert runtime.state_cache.max_seqs == 3
        assert runtime.state_cache.allocated_seqs == 0
        assert runtime.num_blocks() == 200

        runtime.adopt_scheduler_group(
            0,
            BLOCK,
            state_group_indices=(1, 2),
            layer_group_ordinals=[0, 1],
            layer_pool_ordinals=[0, 0],
        )
        cache = runtime.state_cache
        assert cache.num_state_pools == 1
        assert cache.allocated_seqs == 3
        assert cache.conv_states[0] is cache.conv_states[1]
        expected_bytes = cache.conv_states[0].nbytes + cache.recurrent_states[0].nbytes
        assert runtime.state_cache_telemetry()["stable_bytes"] == expected_bytes

    def test_runner_consumes_cleanup_only_catalog_and_requires_metadata(self) -> None:
        from vllm_metal.v1.model_runner import MetalModelRunner

        runtime = self._make_runtime(capacity=2)
        runtime.initialize(num_blocks=8)
        runtime.adopt_scheduler_group(
            0,
            BLOCK,
            state_group_indices=(1, 2),
            layer_group_ordinals=[0, 1],
            layer_pool_ordinals=[0, 0],
        )
        runner = SimpleNamespace(
            _paged_attention_runtime=runtime, _decode_pipeline=Mock()
        )
        with pytest.raises(RuntimeError, match="requires scheduler state catalog"):
            MetalModelRunner._prepare_state_cache_step(runner, SimpleNamespace())

        runtime.prepare_state_cache_step(StateCacheStep(1, ((5, 1), (6, 1))))
        TestAlignStateManager()._populate(
            runtime.state_manager, ["a"], [[[5], [6]]], [(0, 1)]
        )
        assert runtime.state_manager.occupied_slots == 2
        MetalModelRunner._prepare_state_cache_step(
            runner,
            SimpleNamespace(
                total_num_scheduled_tokens=0,
                metal_state_cache=StateCacheStep(2, ()),
            ),
        )
        runner._decode_pipeline.begin_step.assert_not_called()
        assert runtime.state_manager.occupied_slots == 0
        assert runtime.state_cache_telemetry()["retired_slots"] == 2

    def test_retirement_preserves_pending_tokens_and_future_delivery_order(
        self, monkeypatch
    ) -> None:
        # Exercise the real pipeline owner without allocating GPU arrays.
        # State-slot reuse must not resolve an independent token output early:
        # the engine consumes step 1's future only after step 2 is submitted.
        from vllm_metal.v1.decode_pipeline import (
            PENDING_TOKEN_PLACEHOLDER,
            DecodePipeline,
            PendingBackfillEntry,
            PendingSampleStep,
            PipelineGateDecision,
        )
        from vllm_metal.v1.model_runner import MetalModelRunner

        monkeypatch.setattr(mx, "eval", lambda *args: None)
        pipeline = DecodePipeline(build_output=lambda batch: batch, validate=Mock())
        runtime = Mock(spec=HybridPagedAttentionRuntime)
        runtime.state_slot_capacity = 2
        runner = SimpleNamespace(
            _paged_attention_runtime=runtime, _decode_pipeline=pipeline
        )
        first_output = Mock()
        first_state = SimpleNamespace(token_ids=[PENDING_TOKEN_PLACEHOLDER])
        requests = {"finished": first_state}
        first_step = PendingSampleStep(
            tokens=SimpleNamespace(tolist=lambda: [42]),
            entries=(
                PendingBackfillEntry(
                    req_id="finished",
                    state=first_state,
                    row=0,
                    token_index=0,
                    output_idx=0,
                ),
            ),
            batch=first_output,
            scheduler_output=SimpleNamespace(),
        )
        pipeline.begin_step(PipelineGateDecision(True, "eligible"))
        first_future = pipeline.submit(first_step)
        pipeline.begin_step(PipelineGateDecision(True, "eligible"))
        del requests["finished"]  # pending entry owns the direct state reference

        MetalModelRunner._prepare_state_cache_step(
            runner, SimpleNamespace(metal_state_cache=StateCacheStep(2, ()))
        )

        assert pipeline.step_eligible
        assert pipeline.has_pending
        assert pipeline._pending is first_step
        assert pipeline._resolved is None
        assert first_state.token_ids == [PENDING_TOKEN_PLACEHOLDER]
        runtime.prepare_state_cache_step.assert_called_once()

        second_output = object()
        second_step = PendingSampleStep(
            tokens=SimpleNamespace(tolist=lambda: []),
            entries=(),
            batch=second_output,
            scheduler_output=SimpleNamespace(),
        )
        next_future = pipeline.submit(second_step)
        assert first_state.token_ids == [42]
        first_output.set_output.assert_called_once_with(0, [42])
        assert first_future.get_output() is first_output
        assert next_future.get_output() is second_output


class TestBoundedAlignStateManager:
    _populate = TestAlignStateManager._populate

    def _make_manager(self, capacity=2, shared=False):
        cache = _make_cache(num_blocks=capacity)
        if shared:
            cache.set_layer_layout([0, 1], [0, 0])
        manager = AlignStateManager(cache, BLOCK, state_slot_capacity=capacity)
        return cache, manager

    def test_catalog_only_advertises_blocks_and_does_not_create_restore_sources(self):
        cache, manager = self._make_manager()
        manager.prepare_state_cache_step(StateCacheStep(1, ((100, 1), (200, 1))))
        assert manager.resident_blocks == 2
        assert manager.occupied_slots == 0
        with pytest.raises(RuntimeError, match="missing state source"):
            self._populate(manager, ["restore"], [[[100, 200]]], [(4, 1)])
        assert manager.occupied_slots == 0
        assert cache.allocated_seqs == 2

    def test_targets_must_be_in_catalog(self):
        _, manager = self._make_manager()
        with pytest.raises(RuntimeError, match="absent from scheduler catalog"):
            self._populate(manager, ["a"], [[[100]]], [(0, 1)])
        manager.prepare_state_cache_step(StateCacheStep(1, ((100, 1),)))
        with pytest.raises(RuntimeError, match="absent from scheduler catalog"):
            self._populate(manager, ["a"], [[[200]]], [(0, 1)])
        assert manager.occupied_slots == 0

    def test_crossing_preserves_live_source_and_capacity(self):
        cache, manager = self._make_manager()
        manager.prepare_state_cache_step(StateCacheStep(1, ((100, 1),)))
        self._populate(manager, ["a"], [[[100]]], [(0, 4)])
        _fill_slab(cache, 0, manager.slot_for(100), 7.0)
        manager.prepare_state_cache_step(StateCacheStep(2, ((100, 1), (200, 1))))
        self._populate(manager, ["a"], [[[100, 200]]], [(4, 1)])
        for block in (100, 200):
            conv, rec = _slab(cache, 0, manager.slot_for(block))
            np.testing.assert_array_equal(conv, 7.0)
            np.testing.assert_array_equal(rec, 7.0)
        assert cache.allocated_seqs == 2

    def test_full_pool_retirement_reuses_slot_without_growth(self):
        cache, manager = self._make_manager()
        manager.prepare_state_cache_step(StateCacheStep(1, ((5, 1), (6, 1))))
        self._populate(manager, ["a", "b"], [[[5]], [[6]]], [(0, 1), (0, 1)])
        freed_slot = manager.slot_for(5)
        _fill_slab(cache, 0, freed_slot, 13.0)
        manager.prepare_state_cache_step(StateCacheStep(2, ((6, 1), (900, 1))))
        self._populate(manager, ["new"], [[[900]]], [(0, 1)])
        assert manager.slot_for(5) is None
        assert manager.slot_for(900) == freed_slot
        assert manager.retired_slots == 1
        assert manager.occupied_slots == cache.allocated_seqs == 2
        conv, rec = _slab(cache, 0, freed_slot)
        np.testing.assert_array_equal(conv, 0.0)
        np.testing.assert_array_equal(rec, 0.0)

    def test_generation_change_cannot_restore_previous_lifes_bytes(self):
        cache, manager = self._make_manager()
        manager.prepare_state_cache_step(StateCacheStep(1, ((5, 1),)))
        self._populate(manager, ["old"], [[[5]]], [(0, 1)])
        _fill_slab(cache, 0, manager.slot_for(5), 17.0)
        manager.prepare_state_cache_step(StateCacheStep(2, ((5, 2),)))
        with pytest.raises(RuntimeError, match="missing state source"):
            self._populate(manager, ["wrong-restore"], [[[5]]], [(1, 1)])
        self._populate(manager, ["new"], [[[5]]], [(0, 1)])
        conv, rec = _slab(cache, 0, manager.slot_for(5))
        np.testing.assert_array_equal(conv, 0.0)
        np.testing.assert_array_equal(rec, 0.0)

    @pytest.mark.parametrize("sequence", [0, 1, 3])
    def test_catalog_sequence_is_strictly_ordered(self, sequence):
        _, manager = self._make_manager()
        manager.prepare_state_cache_step(StateCacheStep(1, ((5, 1),)))
        with pytest.raises(RuntimeError, match="out-of-order"):
            manager.prepare_state_cache_step(StateCacheStep(sequence, ()))
        assert manager.step_sequence == 1
        assert manager.resident_blocks == 1

    def test_stale_generation_and_resurrection_are_rejected(self):
        _, manager = self._make_manager()
        manager.prepare_state_cache_step(StateCacheStep(1, ((5, 2),)))
        with pytest.raises(RuntimeError, match="stale state generation"):
            manager.prepare_state_cache_step(StateCacheStep(2, ((5, 1),)))
        manager.prepare_state_cache_step(StateCacheStep(2, ()))
        with pytest.raises(RuntimeError, match="stale state generation"):
            manager.prepare_state_cache_step(StateCacheStep(3, ((5, 2),)))
        manager.prepare_state_cache_step(StateCacheStep(3, ((5, 3),)))

    def test_oversized_catalog_is_rejected_before_retirement(self):
        _, manager = self._make_manager()
        manager.prepare_state_cache_step(StateCacheStep(1, ((5, 1),)))
        self._populate(manager, ["a"], [[[5]]], [(0, 1)])
        slot = manager.slot_for(5)
        with pytest.raises(RuntimeError, match="exceeds capacity"):
            manager.prepare_state_cache_step(
                StateCacheStep(2, ((6, 1), (7, 1), (8, 1)))
            )
        assert manager.slot_for(5) == slot
        assert manager.step_sequence == 1
        assert manager.retired_slots == 0

    def test_pending_updates_finish_before_reuse_and_keep_pool_aliases(
        self, monkeypatch
    ):
        cache, manager = self._make_manager(shared=True)
        manager.prepare_state_cache_step(StateCacheStep(1, ((5, 1), (6, 1))))
        self._populate(manager, ["a"], [[[5], [6]]], [(0, 1)])
        retired_slot = manager.slot_for(5)
        survivor_slot = manager.slot_for(6)
        _fill_slab(cache, 1, survivor_slot, 11.0)
        update = mx.full((1,) + cache.recurrent_states[0].shape[1:], 9.0)
        cache.set_pending_recurrent_state(0, [retired_slot], update)
        events = []
        original_sync = mx.synchronize

        def synchronize():
            assert not cache.has_pending_recurrent_state(0)
            assert manager.slot_for(5) == retired_slot
            events.append("fence-before-retirement")
            original_sync()

        monkeypatch.setattr(mx, "synchronize", synchronize)
        manager.prepare_state_cache_step(StateCacheStep(2, ((6, 1), (7, 1))))
        manager.apply_block_copies([(6, 7)])
        assert manager.slot_for(7) == retired_slot
        assert events == ["fence-before-retirement"]
        assert cache.recurrent_states[0] is cache.recurrent_states[1]
        for layer in (0, 1):
            conv, rec = _slab(cache, layer, manager.slot_for(7))
            np.testing.assert_array_equal(conv, 11.0)
            np.testing.assert_array_equal(rec, 11.0)
        assert cache.allocated_seqs == 2

    def test_cow_requires_valid_source_and_catalog_destination(self):
        _, manager = self._make_manager()
        manager.prepare_state_cache_step(StateCacheStep(1, ((5, 1), (6, 1))))
        with pytest.raises(RuntimeError, match="missing state source"):
            manager.apply_block_copies([(5, 6)])
        self._populate(manager, ["a"], [[[5]]], [(0, 1)])
        with pytest.raises(RuntimeError, match="absent from scheduler catalog"):
            manager.apply_block_copies([(5, 7)])
        manager.apply_block_copies([(100, 200)])
        assert manager.slot_for(100) is None
        assert manager.slot_for(200) is None
