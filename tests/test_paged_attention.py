# SPDX-License-Identifier: Apache-2.0
"""Tests for paged attention shared utilities — OffsetCache, prepare functions.

Run with:
    python -m pytest tests/test_paged_attention.py -v -s
"""

from __future__ import annotations

from vllm_metal.paged_attention_common import (
    OffsetCache,
    clear_context,
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
    def test_prepare_prefill_slot_mapping(self):
        prepare_prefill([10, 11], num_tokens=5, block_size=4)
        from vllm_metal.paged_attention_common import get_context

        ctx = get_context()
        assert ctx is not None
        assert ctx.is_prefill
        # block 10: slots 40,41,42,43; block 11: slot 44
        assert ctx.slot_mapping == [40, 41, 42, 43, 44]
        clear_context()

    def test_prepare_decode(self):
        prepare_decode([([5, 6], 7)], block_size=4)
        from vllm_metal.paged_attention_common import get_context

        ctx = get_context()
        assert ctx is not None
        assert not ctx.is_prefill
        # new_pos=7, block_ids[7//4]=block_ids[1]=6, slot=6*4+(7%4)=24+3=27
        assert ctx.slot_mapping == [27]
        assert ctx.context_lens == [8]
        assert ctx.offsets == [7]
        clear_context()
