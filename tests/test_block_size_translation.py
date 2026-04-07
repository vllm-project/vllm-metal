# SPDX-License-Identifier: Apache-2.0
"""Unit tests for hybrid block-size translation in Metal paged attention.

Verifies that _pick_kernel_block_size and _build_block_tables correctly
translate large vLLM block sizes (e.g. 544 for hybrid models) into
kernel-compatible block sizes (8, 16, 32).
"""

from __future__ import annotations

import pytest

from vllm_metal.metal_kernel_backend.attention_sdpa import (
    _KERNEL_BLOCK_SIZES,
    _build_block_tables,
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
