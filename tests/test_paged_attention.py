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

    def test_prepare_prefill_single_request(self):
        # Single request via prepare_prefill_packed
        prepare_prefill_packed([([10, 11], 5)], block_size=4)
        ctx = get_context()

        # block 10: slots 40,41,42,43; block 11: slot 44
        assert ctx is not None
        assert ctx.is_prefill
        assert ctx.slot_mapping == [40, 41, 42, 43, 44]
        assert ctx.block_tables == [[10, 11]]
        assert ctx.context_lens == [5]
        assert ctx.cu_seqlens == [0, 5]

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
        assert ctx.cu_seqlens == [0, 3, 5]
        assert ctx.block_tables == [[10], [20]]
        assert ctx.context_lens == [3, 2]

    def test_prepare_prefill_packed_single_request(self):
        # Single request through packed path should produce valid metadata
        requests = [([5, 6], 5)]
        prepare_prefill_packed(requests, block_size=4)
        ctx = get_context()

        assert ctx is not None
        assert ctx.cu_seqlens == [0, 5]
        # block 5: slots 20,21,22,23; block 6: slot 24
        assert ctx.slot_mapping == [20, 21, 22, 23, 24]
        assert ctx.block_tables == [[5, 6]]
        assert ctx.context_lens == [5]

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


class TestPackedRoPE:
    """Tests for per-request RoPE position reset in packed prefill."""

    def test_positions_reset_per_request(self):
        """Each packed request's RoPE should start from position 0."""
        import mlx.core as mx

        from vllm_metal.metal_kernel_backend.packed_prefill_compat import (
            apply_packed_rope,
        )

        # Minimal RoPE stub: returns input + offset so we can verify offsets
        class FakeRoPE:
            def rope(self, x, offset=0):
                return x + offset

        module = FakeRoPE()
        # Two requests packed: 3 tokens + 2 tokens
        # Shape: (1, heads=1, total_len=5, head_dim=2)
        q = mx.zeros((1, 1, 5, 2))
        k = mx.zeros((1, 1, 5, 2))
        cu_seqlens = [0, 3, 5]

        q_out, k_out = apply_packed_rope(module, q, k, cu_seqlens)

        # All values should be 0 (offset=0 for every request)
        assert q_out.shape == (1, 1, 5, 2)
        assert mx.allclose(q_out, mx.zeros_like(q_out)).item()
        assert mx.allclose(k_out, mx.zeros_like(k_out)).item()


class TestBatchSplitting:
    """Tests for the packed-prefill batch splitting logic."""

    @staticmethod
    def _split_batches(
        entries: list[tuple[int, int]],
        max_tokens: int,
    ) -> list[list[tuple[int, int]]]:
        """Reproduce the batch splitting algorithm from _run_packed_prefill.

        entries: list of (index, num_tokens) for simplicity.
        """
        batches: list[list[tuple[int, int]]] = [[]]
        batch_tokens = 0
        for entry in entries:
            entry_tokens = entry[1]
            if batch_tokens + entry_tokens > max_tokens and batches[-1]:
                batches.append([])
                batch_tokens = 0
            batches[-1].append(entry)
            batch_tokens += entry_tokens
        return batches

    def test_all_fit_single_batch(self):
        entries = [(0, 100), (1, 200), (2, 300)]
        batches = self._split_batches(entries, max_tokens=4096)
        assert len(batches) == 1
        assert batches[0] == entries

    def test_split_into_two_batches(self):
        entries = [(0, 3000), (1, 2000)]
        batches = self._split_batches(entries, max_tokens=4096)
        assert len(batches) == 2
        assert batches[0] == [(0, 3000)]
        assert batches[1] == [(1, 2000)]

    def test_single_large_request_not_dropped(self):
        # A request exceeding the cap should still go into its own batch
        entries = [(0, 5000)]
        batches = self._split_batches(entries, max_tokens=4096)
        assert len(batches) == 1
        assert batches[0] == [(0, 5000)]

    def test_preserves_all_entries(self):
        entries = [(i, 1000) for i in range(10)]
        batches = self._split_batches(entries, max_tokens=4096)
        flat = [e for batch in batches for e in batch]
        assert flat == entries
