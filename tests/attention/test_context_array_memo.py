# SPDX-License-Identifier: Apache-2.0
"""Per-forward memoized int-array conversions on PagedAttentionContext.

The attention and GDN/state wrappers consume the same Python metadata lists
on every layer of a forward pass; the memoized helpers convert each list
once and share the array across layers.  These tests pin the memo semantics:
identity keys for per-forward context lists, value keys for derived slot id
lists, correct dtypes, and per-forward isolation.
"""

import mlx.core as mx

from vllm_metal.attention.context import (
    PagedAttentionContext,
    get_context,
    memoized_block_table_arrays,
    memoized_context_lens,
    memoized_cu_seqlens,
    memoized_slot_ids,
    memoized_slot_mapping,
    prepare_grouped,
)


def _two_request_context() -> PagedAttentionContext:
    """One decode + one prefill request through the real prepare path."""
    prepare_grouped(
        decode_requests=[([[5, 6]], 4)],
        prefill_requests=[([[7, 8, 9]], 3, 0)],
        block_sizes=(16,),
    )
    prepared = get_context()
    assert prepared is not None
    return prepared


def test_slot_mapping_memoized_by_context():
    ctx = PagedAttentionContext(slot_mapping=[3, 17, 5])
    first = memoized_slot_mapping(ctx)
    second = memoized_slot_mapping(ctx)
    assert first is second
    assert first.dtype == mx.int64
    assert first.tolist() == [3, 17, 5]


def test_cu_seqlens_keyed_by_list_identity():
    ctx = PagedAttentionContext(slot_mapping=[])
    boundaries = [0, 1, 4]
    first = memoized_cu_seqlens(ctx, boundaries)
    # Same list object → same array (the per-layer reuse case).
    assert memoized_cu_seqlens(ctx, boundaries) is first
    assert first.dtype == mx.int32
    assert first.tolist() == [0, 1, 4]
    # A fresh, equal-valued list is a distinct conversion.
    second = memoized_cu_seqlens(ctx, [0, 1, 4])
    assert second is not first
    assert second.tolist() == first.tolist()


def test_slot_ids_keyed_by_value():
    ctx = PagedAttentionContext(slot_mapping=[])
    first = memoized_slot_ids(ctx, [2, 0, 1])
    # Equal-valued lists share the conversion even when the list objects
    # differ — the state manager derives fresh lists per layer.
    assert memoized_slot_ids(ctx, [2, 0, 1]) is first
    assert first.dtype == mx.int32
    assert first.tolist() == [2, 0, 1]
    # Different values are a different entry.
    other = memoized_slot_ids(ctx, [2, 1, 0])
    assert other is not first
    assert other.tolist() == [2, 1, 0]


def test_context_lens_memoized():
    ctx = PagedAttentionContext(slot_mapping=[], context_lens=[5, 3])
    first = memoized_context_lens(ctx)
    assert memoized_context_lens(ctx) is first
    assert first.dtype == mx.uint32
    assert first.tolist() == [5, 3]


def test_block_table_arrays_keyed_by_list_identity():
    ctx = PagedAttentionContext(slot_mapping=[])
    tables = [[5, 6], [7]]
    first = memoized_block_table_arrays(ctx, tables)
    assert memoized_block_table_arrays(ctx, tables) is first
    assert len(first) == 2
    assert first[0].dtype == mx.int32
    assert first[0].tolist() == [5, 6]
    assert first[1].tolist() == [7]
    # Rebuilding the tables re-converts (new list object).
    rebuilt = memoized_block_table_arrays(ctx, [[5, 6], [7]])
    assert rebuilt is not first
    assert rebuilt[0].tolist() == first[0].tolist()


def test_memo_isolated_per_context():
    first_ctx = PagedAttentionContext(slot_mapping=[1])
    second_ctx = PagedAttentionContext(slot_mapping=[1])
    first = memoized_slot_mapping(first_ctx)
    second = memoized_slot_mapping(second_ctx)
    assert first is not second
    # Both contexts end up with equal-valued arrays.
    assert first.tolist() == second.tolist()


def test_prepare_grouped_context_feeds_memos():
    ctx = _two_request_context()
    boundaries = memoized_cu_seqlens(ctx, ctx.cu_seqlens)
    assert boundaries.tolist() == [0, 1, 4]
    lens = memoized_context_lens(ctx)
    assert lens.tolist() == [5, 3]
    tables = memoized_block_table_arrays(ctx, ctx.block_tables)
    assert [row.tolist() for row in tables] == [[5, 6], [7, 8, 9]]
    slots = memoized_slot_mapping(ctx)
    # Decode row first: pos 4 of block 5 → 5*16 + 4. Then the three prefill
    # tokens in block 7: 7*16 + {0, 1, 2}.
    assert slots.tolist() == [84, 112, 113, 114]
