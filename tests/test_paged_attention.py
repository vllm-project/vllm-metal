# SPDX-License-Identifier: Apache-2.0
"""Tests for paged attention shared utilities — OffsetCache, prepare functions.

Run with:
    python -m pytest tests/test_paged_attention.py -v -s
"""

from __future__ import annotations

from vllm_metal.paged_attention_common import (
    OffsetCache,
    clear_context,
    get_context,
    prepare_decode,
    prepare_prefill,
)


class TestOffsetCache:
    def test_offset_property(self):
        c = OffsetCache(42)
        assert c.offset == 42

    def test_make_mask_single_token(self):
        c = OffsetCache(10)
        assert c.make_mask(1) is None

    def test_make_mask_multi_token(self):
        c = OffsetCache(0)
        assert c.make_mask(5) == "causal"


class TestPrepare:
    def teardown_method(self):
        clear_context()

    def test_prepare_prefill_slot_mapping(self):
        # Arrange
        block_ids = [10, 11]

        # Act
        prepare_prefill(block_ids, num_tokens=5, block_size=4)
        ctx = get_context()

        # Assert — block 10: slots 40,41,42,43; block 11: slot 44
        assert ctx is not None
        assert ctx.is_prefill
        assert ctx.slot_mapping == [40, 41, 42, 43, 44]

    def test_prepare_decode(self):
        # Arrange
        requests = [([5, 6], 7)]

        # Act
        prepare_decode(requests, block_size=4)
        ctx = get_context()

        # Assert — new_pos=7, block_ids[7//4]=block_ids[1]=6, slot=6*4+(7%4)=27
        assert ctx is not None
        assert not ctx.is_prefill
        assert ctx.slot_mapping == [27]
        assert ctx.context_lens == [8]
        assert ctx.offsets == [7]
