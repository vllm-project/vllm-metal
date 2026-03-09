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
    prepare_prefill_packed,
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

    def test_prepare_prefill_packed_slot_mapping(self):
        # Two requests: 3 tokens in block 10, 2 tokens in block 20
        requests = [([10], 3), ([20], 2)]
        prepare_prefill_packed(requests, block_size=4)
        ctx = get_context()

        assert ctx is not None
        assert ctx.is_prefill
        # Request 0: block 10, slots 40,41,42
        # Request 1: block 20, slots 80,81
        assert ctx.slot_mapping == [40, 41, 42, 80, 81]
        assert ctx.cu_seq_lens == [0, 3, 5]

    def test_prepare_prefill_packed_single_request(self):
        # Single request should still produce valid cu_seq_lens
        requests = [([5, 6], 5)]
        prepare_prefill_packed(requests, block_size=4)
        ctx = get_context()

        assert ctx is not None
        assert ctx.cu_seq_lens == [0, 5]
        # block 5: slots 20,21,22,23; block 6: slot 24
        assert ctx.slot_mapping == [20, 21, 22, 23, 24]

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


class TestPackedCausalMask:
    """Tests for the block-diagonal causal mask used in packed prefill."""

    def test_single_sequence(self):
        from vllm_metal.metal_kernel_backend.paged_attention import (
            _build_packed_causal_mask,
        )

        mask = _build_packed_causal_mask([0, 3], total_len=3)
        # Standard causal: lower-triangular (0) with upper-triangular (-inf)
        assert mask.shape == (1, 1, 3, 3)
        m = mask[0, 0]
        # Diagonal and below should be 0
        assert m[0, 0].item() == 0.0
        assert m[1, 0].item() == 0.0
        assert m[1, 1].item() == 0.0
        # Above diagonal should be -inf
        assert m[0, 1].item() == float("-inf")
        assert m[0, 2].item() == float("-inf")

    def test_two_sequences_isolation(self):
        from vllm_metal.metal_kernel_backend.paged_attention import (
            _build_packed_causal_mask,
        )

        # Two sequences: [0,2) and [2,5)
        mask = _build_packed_causal_mask([0, 2, 5], total_len=5)
        m = mask[0, 0]
        # Seq 0 tokens should not attend to seq 1 tokens
        assert m[0, 2].item() == float("-inf")
        assert m[0, 3].item() == float("-inf")
        assert m[1, 2].item() == float("-inf")
        # Seq 1 tokens should not attend to seq 0 tokens
        assert m[2, 0].item() == float("-inf")
        assert m[2, 1].item() == float("-inf")
        assert m[3, 0].item() == float("-inf")
        # Within seq 1: causal
        assert m[2, 2].item() == 0.0
        assert m[3, 2].item() == 0.0
        assert m[3, 3].item() == 0.0
        assert m[2, 3].item() == float("-inf")
