# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import mlx.core as mx
import numpy as np
import torch

from vllm_metal.attention.caches.gdn_cache import GDNPagedStateCache
from vllm_metal.attention.context import PagedAttentionContext
from vllm_metal.attention.runtime.hybrid import (
    HybridPagedAttentionRuntime,
    _build_linear_layer_spec,
)
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
    def _populate(self, manager, req_ids, tables, positions):
        # The production batch packs decode requests first. Mirror that
        # ordering so multi-token decode can be distinguished from chunked
        # prefill without changing the ordinary prefill path.
        num_decode_requests = 0
        for num_computed, _ in positions:
            if num_computed <= 0:
                break
            num_decode_requests += 1
        ctx = PagedAttentionContext(
            slot_mapping=[], num_decode_requests=num_decode_requests
        )
        manager.populate_step_context(
            req_ids=req_ids,
            ctx=ctx,
            state_block_ids=tables,
            step_positions=positions,
        )
        return ctx

    def test_fresh_request_zeroes_its_state_block(self) -> None:
        cache = _make_cache()
        manager = AlignGDNStateManager(cache, BLOCK)
        _fill_slab(cache, 0, 3, 7.0)  # stale bytes from a previous block life

        ctx = self._populate(
            manager, ["req-A"], [[[3]]], [(0, 2)]
        )  # fresh, 2 tokens into block 3

        assert ctx.gdn_group_slot_mappings == ([3],)
        assert ctx.gdn_slot_mapping is None  # align sets only group mappings
        conv, rec = _slab(cache, 0, 3)
        assert np.all(conv == 0) and np.all(rec == 0)

    def test_boundary_crossing_copies_forward_and_keeps_checkpoint(self) -> None:
        cache = _make_cache()
        manager = AlignGDNStateManager(cache, BLOCK)
        _fill_slab(cache, 0, 2, 5.0)  # request's state after 4 tokens in block 2
        _fill_slab(cache, 1, 2, 5.0)

        # num_computed=4 (block boundary), decoding 1 token → lands in block idx 1
        ctx = self._populate(manager, ["req-A"], [[[2, 6]]], [(4, 1)])

        assert ctx.gdn_group_slot_mappings == ([6],)
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

    def test_striped_groups_share_one_pool_without_colliding(self) -> None:
        # Two layers from different groups sharing one physical pool, the
        # layout kv_cache_tensors.shared_by produces: each group's motion
        # must touch only its own block rows of the shared array.
        cache = _make_cache(num_layers=2)
        cache.set_layer_layout([0, 1], [0, 0])
        assert cache.num_state_pools == 1
        manager = AlignGDNStateManager(cache, BLOCK)
        _fill_slab(cache, 0, 2, 5.0)
        _fill_slab(cache, 1, 3, 8.0)

        # One request; group 0 crosses 2→6, group 1 crosses 3→7.
        ctx = self._populate(manager, ["req-A"], [[[2, 6], [3, 7]]], [(4, 1)])

        assert ctx.gdn_group_slot_mappings == ([6], [7])
        conv, _ = _slab(cache, 0, 6)
        np.testing.assert_array_equal(conv, 5.0)  # group 0 moved its rows
        conv, _ = _slab(cache, 1, 7)
        np.testing.assert_array_equal(conv, 8.0)  # group 1 moved its rows
        conv, _ = _slab(cache, 0, 3)  # group 1's checkpoint intact in the pool
        np.testing.assert_array_equal(conv, 8.0)

    def test_speculative_window_plans_one_state_per_input_token(self) -> None:
        cache = _make_cache(num_blocks=12)
        manager = AlignGDNStateManager(cache, BLOCK, num_speculative_blocks=2)
        _fill_slab(cache, 0, 3, 5.0)
        _fill_slab(cache, 1, 3, 5.0)

        # Five tokens are already computed, so block-table column 1 is the
        # current recurrent state. The two following columns are the scheduler's
        # speculative Mamba blocks for a [confirmed, draft1, draft2] window.
        ctx = self._populate(
            manager,
            ["req-A"],
            [[[2, 3, 6, 7]]],
            [(5, 3)],
        )

        assert ctx.gdn_group_slot_mappings == ([7],)
        assert ctx.gdn_group_state_chains == ([[3, 3, 6, 7]],)

    def test_speculative_commit_promotes_selected_checkpoint(self) -> None:
        cache = _make_cache(num_blocks=12)
        manager = AlignGDNStateManager(cache, BLOCK, num_speculative_blocks=2)
        tables = [[[2, 3, 6, 7]]]
        positions = [(5, 3)]
        self._populate(manager, ["req-A"], tables, positions)

        # Simulate the state-producing GDN path: slot 3 after confirmed token,
        # slot 6 after draft one, slot 7 after draft two.
        for layer in (0, 1):
            _fill_slab(cache, layer, 3, 10.0)
            _fill_slab(cache, layer, 6, 20.0)
            _fill_slab(cache, layer, 7, 30.0)

        manager.commit_speculative_state(
            req_ids=["req-A"],
            state_block_ids=tables,
            step_positions=positions,
            num_sampled_tokens=[2],
        )

        # Two emitted tokens means one draft accepted. The selected snapshot is
        # slot 6, but the committed sequence is still in block-table column 1,
        # so it is promoted back to scheduler block 3.
        for layer in (0, 1):
            conv, rec = _slab(cache, layer, 3)
            np.testing.assert_array_equal(conv, 20.0)
            np.testing.assert_array_equal(rec, 20.0)

    def test_speculative_commit_crosses_block_boundary(self) -> None:
        cache = _make_cache(num_blocks=12)
        manager = AlignGDNStateManager(cache, BLOCK, num_speculative_blocks=2)
        tables = [[[2, 3, 6, 7]]]
        positions = [(7, 3)]
        self._populate(manager, ["req-A"], tables, positions)

        for layer in (0, 1):
            _fill_slab(cache, layer, 3, 10.0)
            _fill_slab(cache, layer, 6, 20.0)
            _fill_slab(cache, layer, 7, 30.0)

        manager.commit_speculative_state(
            req_ids=["req-A"],
            state_block_ids=tables,
            step_positions=positions,
            num_sampled_tokens=[3],
        )

        # Three emitted tokens selects state slot 7. The accepted sequence has
        # crossed into positional block-table column 2, so slot 7 is promoted
        # into scheduler block 6.
        for layer in (0, 1):
            conv, rec = _slab(cache, layer, 6)
            np.testing.assert_array_equal(conv, 30.0)
            np.testing.assert_array_equal(rec, 30.0)

    def test_linear_spec_reports_scheduler_speculative_blocks(self) -> None:
        spec = _build_linear_layer_spec(
            conv_kernel_dim=2,
            conv_dim=4,
            num_v_heads=1,
            value_head_dim=4,
            key_head_dim=32,
            torch_dtype=torch.float16,
            mamba_block_size=BLOCK,
            mamba_cache_mode="align",
            num_speculative_blocks=3,
        )
        assert spec.num_speculative_blocks == 3
        assert spec.max_num_blocks_per_req is not None

    def test_pool_materializes_lazily_by_high_water_block_id(self) -> None:
        cache = _make_cache(num_blocks=8, initial_blocks=2)
        manager = AlignGDNStateManager(cache, BLOCK)

        # Fresh request lands in block 5, beyond the initially materialized
        # slabs; the pool grows to cover it (geometric, capped at max_seqs).
        ctx = self._populate(manager, ["req-A"], [[[5]]], [(0, 2)])

        assert ctx.gdn_group_slot_mappings == ([5],)
        assert cache.allocated_seqs == 6
        conv, rec = _slab(cache, 0, 5)
        assert np.all(conv == 0) and np.all(rec == 0)


class TestHybridAlignRuntime:
    def _make_runtime(self) -> HybridPagedAttentionRuntime:
        return HybridPagedAttentionRuntime(
            num_layers=4,
            full_attention_interval=2,
            max_num_seqs=2,
            num_kv_heads=1,
            head_dim=4,
            linear_num_v_heads=1,
            linear_key_head_dim=32,
            linear_value_head_dim=4,
            linear_conv_kernel_dim=2,
            linear_conv_dim=4,
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
