# SPDX-License-Identifier: Apache-2.0
"""ShortConv pool and scheduler checkpoint lifecycle regressions."""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from tests.stub_runner import make_state_cache
from vllm_metal.attention.caches.state_cache import PagedStateCache
from vllm_metal.attention.context import PagedAttentionContext
from vllm_metal.attention.state import AlignStateManager


def _cache(
    *,
    num_layers: int = 2,
) -> PagedStateCache:
    return make_state_cache(
        num_layers=num_layers,
        max_seqs=8,
        conv_kernel_dim=3,
        conv_dim=4,
        dtype=mx.bfloat16,
    )


def _write(cache: PagedStateCache, layer: int, slot: int, value: int) -> None:
    cache.write_conv_rows(
        layer, mx.full((1, 2, 4), value), mx.array([slot], dtype=mx.int32)
    )


def _values(cache: PagedStateCache, layer: int = 0) -> np.ndarray:
    mx.eval(*cache.updated_state_arrays())
    return np.array(cache.conv_states[layer].astype(mx.float32))


def _populate(manager, req_ids, tables, positions) -> PagedAttentionContext:
    ctx = PagedAttentionContext(slot_mapping=[])
    manager.populate_step_context(
        req_ids=req_ids,
        ctx=ctx,
        state_block_ids=tables,
        step_positions=positions,
    )
    return ctx


def test_partial_prefix_cow_preserves_producer_and_consumer_checkpoint() -> None:
    cache = _cache(num_layers=1)
    _write(cache, 0, 2, 5)  # state after six tokens, inside scheduler block 1
    manager = AlignStateManager(cache, block_size=4)

    # A producer keeps its append-only running table; scheduler moves the
    # partial cache entry to block 5 before another token overwrites block 2.
    cache.copy_slots([2], [5], [0])
    _populate(manager, ["producer"], [[[0, 2]]], [(6, 1)])
    _write(cache, 0, 2, 7)

    # A new hit is redirected to private block 6. Copy precedes state planning
    # because both computed and scheduled positions are in that same block.
    cache.copy_slots([5], [6], [0])
    ctx = _populate(manager, ["consumer"], [[[0, 6]]], [(6, 2)])
    assert ctx.state_group_slot_mappings == ([6],)
    values = _values(cache)
    np.testing.assert_array_equal(values[2], 7)
    np.testing.assert_array_equal(values[5], 5)
    np.testing.assert_array_equal(values[6], 5)

    _write(cache, 0, 6, 11)
    values = _values(cache)
    np.testing.assert_array_equal(values[5], 5)
    np.testing.assert_array_equal(values[6], 11)


def test_none_mode_recycled_request_slot_is_reset_in_every_layer() -> None:
    cache = _cache()
    manager = AlignStateManager(cache, 4096)
    _populate(manager, ["a", "b"], [[[0]], [[1]]], [(0, 1), (0, 1)])
    for layer in range(cache.num_layers):
        _write(cache, layer, 0, 5 + layer)
        _write(cache, layer, 1, 9 + layer)

    manager.release_requests({"a"})
    _populate(manager, ["b", "c"], [[[1]], [[0]]], [(1, 1), (0, 1)])
    for layer in range(cache.num_layers):
        values = _values(cache, layer)
        np.testing.assert_array_equal(values[0], 0)
        np.testing.assert_array_equal(values[1], 9 + layer)
