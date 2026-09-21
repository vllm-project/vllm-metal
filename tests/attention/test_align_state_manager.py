# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import mlx.core as mx
import numpy as np

from tests.stub_runner import (
    initialize_hybrid_runtime,
    make_gdn_hybrid_plan,
    make_state_cache,
)
from vllm_metal.attention.caches.state_cache import PagedStateCache
from vllm_metal.attention.context import PagedAttentionContext
from vllm_metal.attention.runtime.hybrid import HybridPagedAttentionRuntime
from vllm_metal.attention.state import AlignStateManager

BLOCK = 4


def _make_cache(
    *,
    num_layers: int = 2,
    num_blocks: int = 8,
) -> PagedStateCache:
    return make_state_cache(
        num_layers=num_layers,
        max_seqs=num_blocks,
        conv_kernel_dim=2,
        conv_dim=4,
        num_v_heads=1,
        value_head_dim=4,
        key_head_dim=32,
        dtype=mx.float32,
    )


def _fill_slab(cache: PagedStateCache, layer: int, slab: int, value: float) -> None:
    conv = cache.conv_states[layer]
    conv[slab] = value
    cache.store_conv_state(layer, conv)
    rec = cache.recurrent_states[layer]
    rec[slab] = value
    cache.store_recurrent_state(layer, rec)
    mx.eval(cache.conv_states[layer], cache.recurrent_states[layer])


def _slab(cache: PagedStateCache, layer: int, slab: int) -> tuple:
    mx.eval(cache.conv_states[layer], cache.recurrent_states[layer])
    return (
        np.array(cache.conv_states[layer][slab]),
        np.array(cache.recurrent_states[layer][slab]),
    )


class TestAlignStateManager:
    def _populate(self, manager, req_ids, tables, positions):
        ctx = PagedAttentionContext(slot_mapping=[])
        manager.populate_step_context(
            req_ids=req_ids,
            ctx=ctx,
            state_block_ids=tables,
            step_positions=positions,
        )
        return ctx

    def test_fresh_request_zeroes_its_state_block(self) -> None:
        cache = _make_cache()
        manager = AlignStateManager(cache, BLOCK)
        _fill_slab(cache, 0, 3, 7.0)  # stale bytes from a previous block life

        ctx = self._populate(
            manager, ["req-A"], [[[3]]], [(0, 2)]
        )  # fresh, 2 tokens into block 3

        assert ctx.state_group_slot_mappings == ([3],)
        assert ctx.state_slot_mapping is None  # align sets only group mappings
        conv, rec = _slab(cache, 0, 3)
        assert np.all(conv == 0) and np.all(rec == 0)

    def test_none_mode_keeps_compact_updates_between_decode_steps(self) -> None:
        cache = _make_cache()
        manager = AlignStateManager(cache, 4096, mamba_cache_mode="none")
        self._populate(manager, ["a"], [[[3]]], [(0, 1)])
        cache.set_pending_recurrent_state(
            0, [3], mx.full((1, 1, 4, 32), 9, dtype=mx.float32)
        )

        manager.materialize_pending_state()
        self._populate(manager, ["a"], [[[3]]], [(1, 1)])

        view = cache.recurrent_state_for_decode(0, [3])
        assert view.uses_compact_state
        np.testing.assert_array_equal(np.array(view.state), 9)
        np.testing.assert_array_equal(np.array(cache.recurrent_states[0][3]), 0)

        manager.release_requests({"a"})
        manager.materialize_pending_state()
        assert not cache.has_pending_recurrent_state(0)
        np.testing.assert_array_equal(np.array(cache.recurrent_states[0][3]), 9)

    def test_state_motion_flushes_only_intersecting_pending_updates(self) -> None:
        cache = _make_cache()
        for layer, slot, value in [(0, 3, 7), (1, 6, 9)]:
            cache.set_pending_conv_state(
                layer, [slot], mx.full((1, 1, 4), value, dtype=mx.float32)
            )
        cache.copy_slots([3], [4], [0])
        assert cache.has_pending_conv_state(1)
        np.testing.assert_array_equal(np.array(cache.conv_states[0][4]), 7)

        cache.set_pending_conv_state(0, [3], mx.full((1, 1, 4), 11, dtype=mx.float32))
        cache.zero_slots([3], [0])
        assert not cache.has_pending_conv_state(0)
        assert cache.has_pending_conv_state(1)
        cache.apply_pending_states()
        mx.eval(*cache.updated_state_arrays())
        np.testing.assert_array_equal(np.array(cache.conv_states[0][3]), 0)
        np.testing.assert_array_equal(np.array(cache.conv_states[0][4]), 7)
        np.testing.assert_array_equal(np.array(cache.conv_states[1][6]), 9)

    def test_boundary_crossing_copies_forward_and_keeps_checkpoint(self) -> None:
        cache = _make_cache()
        manager = AlignStateManager(cache, BLOCK)
        _fill_slab(cache, 0, 2, 5.0)  # request's state after 4 tokens in block 2
        _fill_slab(cache, 1, 2, 5.0)

        # num_computed=4 (block boundary), decoding 1 token → lands in block idx 1
        ctx = self._populate(manager, ["req-A"], [[[2, 6]]], [(4, 1)])

        assert ctx.state_group_slot_mappings == ([6],)
        for layer in (0, 1):
            conv_src, rec_src = _slab(cache, layer, 2)
            conv_dst, rec_dst = _slab(cache, layer, 6)
            np.testing.assert_array_equal(conv_src, 5.0)  # checkpoint intact
            np.testing.assert_array_equal(rec_src, 5.0)
            np.testing.assert_array_equal(conv_dst, 5.0)  # state moved forward
            np.testing.assert_array_equal(rec_dst, 5.0)

        # Slabs belong to scheduler blocks, not requests: releasing the
        # request must leave the checkpoint readable for later restores.
        manager.release_requests({"req-A"})
        conv_src, _ = _slab(cache, 0, 2)
        np.testing.assert_array_equal(conv_src, 5.0)


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
            dtype=mx.float32,
            mamba_cache_mode="align",
        )

    def test_storage_matches_upstream_allocation(self) -> None:
        runtime = self._make_runtime()
        config = initialize_hybrid_runtime(runtime, 8, mamba_cache_mode="align")
        assert runtime.state_cache.conv_states[0].shape[0] == config.num_blocks
        assert runtime.storage.nbytes == config.kv_cache_tensors[0].size
        # One allocation supplies attention and state, with no extra state budget.
        assert runtime.kv_cache._storage is runtime.storage

    def test_initialize_wires_state_manager_delegation(self) -> None:
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
            dtype=mx.float32,
        )
        initialize_hybrid_runtime(runtime, 2)

        ctx = PagedAttentionContext(slot_mapping=[])
        runtime.populate_step_context(
            req_ids=["req-A"],
            ctx=ctx,
            state_block_ids=[[[1]]],
            step_positions=[(0, 1)],
        )

        assert ctx.state_group_slot_mappings == ([1],)

        cache = runtime.state_cache
        slot = ctx.state_group_slot_mappings[0][0]
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
        assert runtime.state_manager.needs_materialize is False


def test_scheduler_copy_and_zero_preserve_unrelated_compact_updates():
    runtime = TestHybridAlignRuntime()._make_runtime()
    initialize_hybrid_runtime(runtime, 8, mamba_cache_mode="align")
    cache = runtime.state_cache
    cache.set_pending_recurrent_state(
        0, [3], mx.full((1, 1, 4, 32), 9, dtype=mx.float32)
    )

    runtime.zero_blocks([1])
    runtime.copy_blocks([(1, 2)])
    assert cache.has_pending_recurrent_state(0)
    runtime.copy_blocks([(3, 4)])
    assert not cache.has_pending_recurrent_state(0)
    mx.eval(*runtime.storage.buffers)
    np.testing.assert_array_equal(np.array(cache.recurrent_states[0][4]), 9)
