# SPDX-License-Identifier: Apache-2.0
"""Unit tests for hybrid block-size translation in Metal paged attention.

Verifies that _pick_kernel_block_size and _build_block_tables correctly
translate large vLLM block sizes (e.g. 544 for hybrid models) into
kernel-compatible block sizes (8, 16, 32).
"""

from __future__ import annotations

import pytest

from vllm_metal.attention.context import (
    clear_context,
    get_context,
    prepare_grouped,
    prepare_unified,
)
from vllm_metal.attention.impls.sdpa import (
    _KERNEL_BLOCK_SIZES,
    _build_block_tables,
    _kernel_metadata,
    _pick_kernel_block_size,
)


class TestPickKernelBlockSize:
    """Tests for _pick_kernel_block_size."""

    def test_returns_exact_match(self):
        for bs in _KERNEL_BLOCK_SIZES:
            assert _pick_kernel_block_size(bs) == bs

    def test_picks_largest_divisor(self):
        # 544 % 32 == 0, so should pick 32 (not 16 or 8)
        assert _pick_kernel_block_size(544) == 32

    def test_picks_16_when_32_does_not_divide(self):
        # 48 % 32 != 0, but 48 % 16 == 0
        assert _pick_kernel_block_size(48) == 16

    def test_picks_8_as_fallback(self):
        # 24 % 32 != 0, 24 % 16 != 0, but 24 % 8 == 0
        assert _pick_kernel_block_size(24) == 8

    def test_raises_on_indivisible(self):
        with pytest.raises(ValueError, match="not divisible"):
            _pick_kernel_block_size(7)


class TestBuildBlockTables:
    """Tests for _build_block_tables."""

    def test_no_translation_for_supported_sizes(self):
        bt, kbs = _build_block_tables([[0, 1], [2]], 16)
        assert kbs == 16
        assert bt.tolist() == [[0, 1], [2, 0]]

    def test_translation_single_block(self):
        # 544 -> 32, ratio=17
        bt, kbs = _build_block_tables([[0], [1]], 544)
        assert kbs == 32
        ratio = 544 // 32  # 17
        # block 0 -> [0, 1, ..., 16]
        assert bt[0].tolist() == list(range(0, ratio))
        # block 1 -> [17, 18, ..., 33]
        assert bt[1].tolist() == list(range(ratio, 2 * ratio))

    def test_translation_multi_block(self):
        bt, kbs = _build_block_tables([[0, 2]], 544)
        ratio = 544 // 32
        expected = list(range(0, ratio)) + list(range(2 * ratio, 3 * ratio))
        assert bt[0].tolist() == expected

    def test_translation_with_padding(self):
        # Unequal block table lengths — shorter rows are zero-padded before
        # expansion, so padding block_id=0 expands to [0, 1, …, ratio-1].
        # The kernel never reads these entries (bounded by context_len).
        bt, kbs = _build_block_tables([[0, 1], [2]], 544)
        ratio = 544 // 32
        assert bt.shape[0] == 2
        assert bt.shape[1] == 2 * ratio
        # Second row: block 2 expanded, then padded block 0 expanded
        row1 = bt[1].tolist()
        assert row1[:ratio] == list(range(2 * ratio, 3 * ratio))
        assert row1[ratio:] == list(range(0, ratio))

    def test_output_shape(self):
        bt, kbs = _build_block_tables([[0, 1, 2]], 544)
        ratio = 544 // 32
        assert bt.shape == (1, 3 * ratio)

    def test_empty_block_tables(self):
        bt, kbs = _build_block_tables([], 16)
        assert bt.shape == (0, 0)
        assert kbs == 16

    def test_empty_block_tables_hybrid(self):
        bt, kbs = _build_block_tables([], 544)
        assert bt.shape == (0, 0)
        assert kbs == 544


class TestKernelMetadataMemo:
    """Per-forward memo of kernel-format metadata (_kernel_metadata).

    The paged context lives for exactly one forward pass and its list
    metadata is identical for every layer of a KV group, so the converted
    mx arrays are built once per (group, cache block size) and reused by
    the remaining layers instead of being rebuilt per layer.
    """

    @pytest.fixture(autouse=True)
    def _clean_context(self):
        yield
        clear_context()

    def _fresh_ctx(self, block_ids, seq_len, num_tokens):
        prepare_unified([(block_ids, seq_len, num_tokens)], [], 16)
        return get_context()

    def test_returns_same_objects_within_one_context(self):
        ctx = self._fresh_ctx([0, 1, 2], 40, 4)
        first = _kernel_metadata(ctx, 0, ctx.slot_mapping, ctx.block_tables, 16)
        second = _kernel_metadata(ctx, 0, ctx.slot_mapping, ctx.block_tables, 16)
        assert second is first

    def test_matches_direct_conversion(self):
        ctx = self._fresh_ctx([0, 1, 2], 40, 4)
        meta = _kernel_metadata(ctx, 0, ctx.slot_mapping, ctx.block_tables, 16)
        direct_bt, direct_bs = _build_block_tables(ctx.block_tables, 16)
        assert meta.block_tables.tolist() == direct_bt.tolist()
        assert meta.block_size == direct_bs
        assert meta.slot_mapping.tolist() == ctx.slot_mapping
        assert meta.seq_lens.tolist() == ctx.context_lens
        assert meta.cu_seqlens_q.tolist() == ctx.cu_seqlens
        assert meta.max_seq_len == max(ctx.context_lens)

    def test_new_context_is_not_served_stale_arrays(self):
        ctx1 = self._fresh_ctx([0, 1, 2], 40, 4)
        stale = _kernel_metadata(ctx1, 0, ctx1.slot_mapping, ctx1.block_tables, 16)
        ctx2 = self._fresh_ctx([7, 8], 16, 1)
        fresh = _kernel_metadata(ctx2, 0, ctx2.slot_mapping, ctx2.block_tables, 16)
        assert fresh is not stale
        assert fresh.block_tables.tolist() == [[7, 8]]

    def test_cache_keys_by_block_size(self):
        ctx = self._fresh_ctx([0, 1, 2], 32, 2)
        m16 = _kernel_metadata(ctx, 0, ctx.slot_mapping, ctx.block_tables, 16)
        m32 = _kernel_metadata(ctx, 0, ctx.slot_mapping, ctx.block_tables, 32)
        assert m32 is not m16
        assert m32.block_size == 32
        assert _kernel_metadata(ctx, 0, ctx.slot_mapping, ctx.block_tables, 16) is m16

    def test_cache_keys_by_group_with_equal_block_size(self):
        # Two scheduler groups with the SAME cache block size: only the
        # group index separates their memo entries.  Guards the key against
        # ever being reduced to cache_block_size alone, which would serve
        # group 0's tables to group 1 on hybrid models.
        prepare_grouped([(([10, 11], [77, 78]), 24)], [], (16, 16))
        ctx = get_context()
        g0 = ctx.kv_groups[0]
        g1 = ctx.kv_groups[1]
        m0 = _kernel_metadata(ctx, 0, g0.slot_mapping, g0.block_tables, 16)
        m1 = _kernel_metadata(ctx, 1, g1.slot_mapping, g1.block_tables, 16)
        assert m1 is not m0
        assert m0.block_tables.tolist() == [[10, 11]]
        assert m1.block_tables.tolist() == [[77, 78]]
