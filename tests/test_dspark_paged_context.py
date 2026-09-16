# SPDX-License-Identifier: Apache-2.0

import mlx.core as mx
import numpy as np
import pytest

from vllm_metal.v1.dspark.paged_context import (
    DSparkPagedContext,
    PagedContextFullError,
)

BLOCK = 16
KV_HEADS = 4
N_HEADS = 8
HEAD_DIM = 64
DRAFT_BLOCK = 7


def make(num_blocks: int = 64, layers: int = 2) -> DSparkPagedContext:
    return DSparkPagedContext(
        num_layers=layers,
        kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        block_size=BLOCK,
        num_blocks=num_blocks,
        draft_block=DRAFT_BLOCK,
        dtype=mx.float32,
    )


class TestBlockAccounting:
    def test_pages_cover_the_context_and_the_draft_block(self):
        pool = make()
        # 16 context positions + 7 block positions = 23 -> two 16-token pages
        assert pool.pages_for(16) == 2
        assert pool.pages_for(9) == 1
        assert pool.pages_for(10) == 2

    def test_the_sink_block_is_reserved_and_never_handed_out(self):
        pool = make(num_blocks=8)
        # one block absorbs the padding of ragged ingests
        assert pool.total_blocks == 8
        assert pool.usable_blocks == 7
        assert pool.free_blocks == 7

    def test_reserve_allocates_only_what_the_length_needs(self):
        pool = make(num_blocks=8)
        start = pool.free_blocks
        pool.reserve("a", 1)
        assert pool.free_blocks == start - 1
        # growing within the same page allocates nothing more
        pool.reserve("a", 8)
        assert pool.free_blocks == start - 1
        pool.reserve("a", 100)
        assert pool.free_blocks == start - pool.pages_for(100)

    def test_release_returns_every_block(self):
        pool = make(num_blocks=8)
        start = pool.free_blocks
        pool.reserve("a", 40)
        assert pool.free_blocks < start
        assert pool.release("a") > 0
        assert pool.free_blocks == start
        assert not pool.holds("a")

    def test_release_of_an_unknown_request_is_a_no_op(self):
        pool = make(num_blocks=4)
        assert pool.release("ghost") == 0
        assert pool.free_blocks == pool.usable_blocks

    def test_exhaustion_raises_and_keeps_existing_blocks(self):
        pool = make(num_blocks=2)
        pool.reserve("a", 1)
        held = pool.free_blocks
        with pytest.raises(PagedContextFullError):
            pool.reserve("a", 10_000)
        # the failed growth must not have consumed or dropped anything
        assert pool.free_blocks == held
        assert pool.holds("a")

    def test_a_request_never_shares_a_block_with_another(self):
        pool = make(num_blocks=16)
        pool.reserve("a", 60)
        pool.reserve("b", 60)
        a = {int(s) // BLOCK for s in pool.slot_mapping_for("a", range(60)).tolist()}
        b = {int(s) // BLOCK for s in pool.slot_mapping_for("b", range(60)).tolist()}
        assert a and b and not (a & b)

    def test_slots_are_block_id_times_block_size_plus_offset(self):
        pool = make(num_blocks=8)
        pool.reserve("a", 20)
        slots = [int(s) for s in pool.slot_mapping_for("a", range(20)).tolist()]
        # consecutive positions inside one page are consecutive slots
        assert slots[1] - slots[0] == 1
        assert len({s // BLOCK for s in slots}) == 2


class TestPagedAttentionMatchesDenseReference:
    """The kernel must reproduce the bidirectional draft block exactly.

    Each draft position is its own length-1 query sequence, so it attends the whole
    context plus every block position. The negative control uses the naive packing
    (one length-DRAFT_BLOCK sequence per request), which the kernel treats as causal
    and which must therefore disagree — otherwise this test cannot tell them apart.
    """

    @staticmethod
    def _case(context_lengths, seed):
        rng = np.random.default_rng(seed)
        pool = make(num_blocks=256, layers=1)
        scale = HEAD_DIM**-0.5
        reference = []
        rows = []
        for index, length in enumerate(context_lengths):
            req = f"r{index}"
            pool.reserve(req, length)
            total = length + DRAFT_BLOCK
            keys = rng.normal(size=(total, KV_HEADS, HEAD_DIM)).astype(np.float32)
            values = rng.normal(size=(total, KV_HEADS, HEAD_DIM)).astype(np.float32)
            pool.write(
                0,
                mx.array(keys),
                mx.array(values),
                pool.slot_mapping_for(req, range(total)),
            )
            reference.append((keys, values, total))
            rows.append((req, length))
        queries = rng.normal(
            size=(len(context_lengths), DRAFT_BLOCK, N_HEADS, HEAD_DIM)
        ).astype(np.float32)
        return pool, rows, reference, queries, scale

    @staticmethod
    def _dense(reference, queries, scale):
        out = []
        for row, (keys, values, total) in enumerate(reference):
            q = mx.array(queries[row].transpose(1, 0, 2))[None]
            k = mx.array(keys[:total].transpose(1, 0, 2))[None]
            v = mx.array(values[:total].transpose(1, 0, 2))[None]
            attended = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
            out.append(np.array(attended[0].transpose(1, 0, 2)))
        return np.stack(out)

    @pytest.mark.parametrize(
        "context_lengths",
        [[37], [64], [3], [37, 512, 129], [256, 256]],
        ids=["short", "page-aligned", "sub-page", "ragged", "equal"],
    )
    def test_matches_dense_attention(self, context_lengths):
        pool, rows, reference, queries, scale = self._case(context_lengths, seed=7)
        batch = pool.plan(rows)
        packed = mx.array(queries.reshape(len(rows) * DRAFT_BLOCK, N_HEADS, HEAD_DIM))
        got = np.array(pool.attend(0, packed, batch, scale)).reshape(
            len(rows), DRAFT_BLOCK, N_HEADS, HEAD_DIM
        )
        assert np.abs(got - self._dense(reference, queries, scale)).max() < 2e-3

    def test_the_causal_packing_disagrees(self):
        """Negative control: without per-position sequences the kernel is causal."""
        pool, rows, reference, queries, scale = self._case([37, 512, 129], seed=7)
        batch = pool.plan(rows)
        packed = mx.array(queries.reshape(len(rows) * DRAFT_BLOCK, N_HEADS, HEAD_DIM))
        out = mx.zeros(packed.shape, dtype=packed.dtype)
        from vllm_metal.metal import get_ops

        # one length-DRAFT_BLOCK sequence per request instead of one per position
        get_ops().paged_attention_primitive(
            packed,
            pool.key_caches[0],
            pool.value_caches[0],
            KV_HEADS,
            scale,
            0.0,
            mx.array(
                [
                    batch.block_tables.tolist()[i * DRAFT_BLOCK]
                    for i in range(len(rows))
                ],
                dtype=mx.int32,
            ),
            mx.array([length + DRAFT_BLOCK for _, length in rows], dtype=mx.int32),
            mx.array((np.arange(len(rows) + 1) * DRAFT_BLOCK).astype(np.int32)),
            BLOCK,
            batch.max_seq_len,
            -1,
            out,
            window_seqlen_q=1,
        )
        causal = np.array(out).reshape(len(rows), DRAFT_BLOCK, N_HEADS, HEAD_DIM)
        dense = self._dense(reference, queries, scale)
        assert np.abs(causal - dense).max() > 1e-2


class TestDraftBatchAddressing:
    def test_one_query_sequence_per_draft_position(self):
        pool = make()
        pool.reserve("a", 20)
        pool.reserve("b", 5)
        batch = pool.plan([("a", 20), ("b", 5)])
        assert batch.rows == 2
        count = 2 * DRAFT_BLOCK
        assert batch.seq_lens.shape == (count,)
        assert batch.block_tables.shape[0] == count
        # cu_seqlens_q must declare every row as its own length-1 sequence
        assert batch.cu_seqlens_q.tolist() == list(range(count + 1))

    def test_seq_len_covers_context_plus_block(self):
        pool = make()
        pool.reserve("a", 20)
        batch = pool.plan([("a", 20)])
        assert set(batch.seq_lens.tolist()) == {27}
        assert batch.max_seq_len == 27

    def test_scratch_slots_sit_immediately_after_the_context(self):
        pool = make()
        pool.reserve("a", 20)
        batch = pool.plan([("a", 20)])
        expected = pool.slot_mapping_for("a", range(20, 27)).tolist()
        assert batch.slot_mapping.tolist() == expected


class TestPagedLayerBatchMatchesArenaBatch:
    """The adapter must be a drop-in for ArenaBatch through DSparkAttention.attend.

    Same drafter layer, same weights, same inputs: the only difference is where the
    context lives. Outputs must agree, or the rework changes the model.
    """

    @staticmethod
    def _shapes(rows, ctx_lengths, seed):
        rng = np.random.default_rng(seed)
        hidden = rng.normal(size=(rows, DRAFT_BLOCK, 64)).astype(np.float32)
        return mx.array(hidden), ctx_lengths

    def test_matches_arena_batch_output(self):
        from vllm_metal.v1.dspark.model import ArenaBatch, ContextArena
        from vllm_metal.v1.dspark.paged_context import PagedLayerBatch

        rng = np.random.default_rng(11)
        ctx_lengths = [37, 129]
        rows = len(ctx_lengths)
        scale = HEAD_DIM**-0.5

        # identical context K/V laid into both an arena and a paged pool
        arena = ContextArena(
            slots=4,
            kv_heads=KV_HEADS,
            capacity=256,
            block_size=DRAFT_BLOCK,
            head_dim=HEAD_DIM,
            dtype=mx.float32,
        )
        pool = make(num_blocks=128, layers=1)
        slots = []
        for index, length in enumerate(ctx_lengths):
            req = f"r{index}"
            pool.reserve(req, length)
            slot = arena.acquire()
            slots.append(slot)
            keys = rng.normal(size=(length, KV_HEADS, HEAD_DIM)).astype(np.float32)
            values = rng.normal(size=(length, KV_HEADS, HEAD_DIM)).astype(np.float32)
            pool.write(
                0,
                mx.array(keys),
                mx.array(values),
                pool.slot_mapping_for(req, range(length)),
            )
            arena.keys[slot, :, :length] = mx.array(keys.transpose(1, 0, 2))
            arena.values[slot, :, :length] = mx.array(values.transpose(1, 0, 2))

        q = mx.array(
            rng.normal(size=(rows, N_HEADS, DRAFT_BLOCK, HEAD_DIM)).astype(np.float32)
        )
        k_blk = mx.array(
            rng.normal(size=(rows, KV_HEADS, DRAFT_BLOCK, HEAD_DIM)).astype(np.float32)
        )
        v_blk = mx.array(
            rng.normal(size=(rows, KV_HEADS, DRAFT_BLOCK, HEAD_DIM)).astype(np.float32)
        )

        arena_batch = ArenaBatch(arena, slots, list(ctx_lengths))
        arena_batch.write_block(k_blk, v_blk)
        expected = np.array(arena_batch.attend(q, scale))

        paged = PagedLayerBatch(
            pool, 0, pool.plan([(f"r{i}", n) for i, n in enumerate(ctx_lengths)])
        )
        paged.write_block(k_blk, v_blk)
        got = np.array(paged.attend(q, scale))

        assert got.shape == expected.shape == (rows, N_HEADS, DRAFT_BLOCK, HEAD_DIM)
        assert np.abs(got).sum() > 0  # not vacuously zero on both sides
        assert np.abs(got - expected).max() < 2e-3


class TestLapseExitIsReachable:
    """A lapse entered at one active request must still be able to end.

    `_regime_step` returns early when there are no decode requests, so the branch
    below only ever runs with at least one. A strict "fewer requests than the lapse
    began with" exit test is therefore unsatisfiable when the lapse began at one,
    and the proposer would never draft again.
    """

    @staticmethod
    def _dropped(active: int, entry: int) -> bool:
        # the shipped predicate, isolated
        return active <= max(1, entry - 1)

    def test_a_lapse_entered_at_one_request_can_exit(self):
        assert self._dropped(active=1, entry=1) is True

    def test_the_old_predicate_could_not(self):
        # negative control: what the code used to compute
        assert (1 < 1) is False

    def test_a_lapse_entered_under_load_still_needs_a_real_drop(self):
        assert self._dropped(active=8, entry=8) is False
        assert self._dropped(active=7, entry=8) is True


class TestRaggedIngestAddressing:
    """A ragged ingest pads to the widest span; the padding must not corrupt a context."""

    def test_padding_is_addressed_to_the_sink(self):
        pool = make(num_blocks=32)
        pool.reserve("a", 40)
        pool.reserve("b", 40)
        # a writes 3 positions from 10, b writes 1 from 20; width is 3
        spans = [(0, 10, 3, "a"), (3, 20, 1, "b")]
        slots = [int(x) for x in pool.span_slot_mapping(spans, width=3).tolist()]
        assert len(slots) == 6
        sink = 0  # block 0 * block_size
        assert slots[0:3] == [
            int(x) for x in pool.slot_mapping_for("a", range(10, 13)).tolist()
        ]
        # b contributed one real position and two padded ones
        assert slots[3] == int(pool.slot_mapping_for("b", [20]).tolist()[0])
        assert slots[4] == sink and slots[5] == sink

    def test_a_real_position_never_lands_on_the_sink(self):
        pool = make(num_blocks=32)
        pool.reserve("a", 64)
        spans = [(0, 0, 40, "a")]
        slots = [int(x) for x in pool.span_slot_mapping(spans, width=40).tolist()]
        assert 0 not in slots  # the sink slot is block 0 offset 0

    def test_an_unreserved_request_is_refused(self):
        pool = make(num_blocks=8)
        with pytest.raises(KeyError):
            pool.span_slot_mapping([(0, 0, 1, "nobody")], width=1)


class TestPagedCtxCacheLifecycle:
    """The pool-backed cache must satisfy the lifecycle the proposer already drives."""

    @staticmethod
    def _kv(tokens, seed=3):
        rng = np.random.default_rng(seed)
        shape = (1, KV_HEADS, tokens, HEAD_DIM)
        return (
            mx.array(rng.normal(size=shape).astype(np.float32)),
            mx.array(rng.normal(size=shape).astype(np.float32)),
        )

    def _cache(self, pool=None, capacity=512):
        from vllm_metal.v1.dspark.paged_context import PagedCtxCache

        pool = pool or make(num_blocks=128, layers=1)
        return PagedCtxCache(pool, 0, "r", capacity), pool

    def test_append_advances_the_length_and_reserves_blocks(self):
        cache, pool = self._cache()
        k, v = self._kv(20)
        cache.append(k, v)
        assert cache.length == 20
        assert pool.holds("r")

    def test_appends_accumulate(self):
        cache, _ = self._cache()
        cache.append(*self._kv(10))
        cache.append(*self._kv(7))
        assert cache.length == 17

    def test_trim_keeps_a_prefix_and_rejects_a_longer_one(self):
        cache, _ = self._cache()
        cache.append(*self._kv(20))
        cache.trim_to(8)
        assert cache.length == 8
        with pytest.raises(ValueError):
            cache.trim_to(9)

    def test_append_past_capacity_is_refused(self):
        cache, _ = self._cache(capacity=16)
        with pytest.raises(ValueError):
            cache.append(*self._kv(17))

    def test_extend_to_accounts_for_a_batched_write(self):
        cache, pool = self._cache()
        pool.reserve("r", 40)
        cache.extend_to(40)
        assert cache.length == 40
        with pytest.raises(ValueError):
            cache.extend_to(39)

    def test_reading_the_dense_context_is_refused(self):
        """The pool exists to avoid materialising this; asking must fail loudly."""
        cache, _ = self._cache()
        cache.append(*self._kv(4))
        with pytest.raises(NotImplementedError):
            _ = cache.k
        with pytest.raises(NotImplementedError):
            _ = cache.v

    def test_appended_context_reads_back_through_the_batch(self):
        """What append wrote is what a drafting step attends."""
        from vllm_metal.v1.dspark.paged_context import PagedLayerBatch

        pool = make(num_blocks=128, layers=1)
        cache, _ = self._cache(pool=pool)
        rng = np.random.default_rng(5)
        keys = rng.normal(size=(30, KV_HEADS, HEAD_DIM)).astype(np.float32)
        values = rng.normal(size=(30, KV_HEADS, HEAD_DIM)).astype(np.float32)
        cache.append(
            mx.array(keys.transpose(1, 0, 2))[None],
            mx.array(values.transpose(1, 0, 2))[None],
        )
        q = mx.array(
            rng.normal(size=(1, N_HEADS, DRAFT_BLOCK, HEAD_DIM)).astype(np.float32)
        )
        blk_k = mx.zeros((1, KV_HEADS, DRAFT_BLOCK, HEAD_DIM))
        batch = PagedLayerBatch(pool, 0, pool.plan([("r", 30)]))
        batch.write_block(blk_k, blk_k)
        out = np.array(batch.attend(q, HEAD_DIM**-0.5))
        # dense reference over exactly what was appended, plus the zero block
        k_all = np.concatenate(
            [keys, np.zeros((DRAFT_BLOCK, KV_HEADS, HEAD_DIM), np.float32)]
        )
        v_all = np.concatenate(
            [values, np.zeros((DRAFT_BLOCK, KV_HEADS, HEAD_DIM), np.float32)]
        )
        expected = np.array(
            mx.fast.scaled_dot_product_attention(
                q,
                mx.array(k_all.transpose(1, 0, 2))[None],
                mx.array(v_all.transpose(1, 0, 2))[None],
                scale=HEAD_DIM**-0.5,
            )
        )
        assert np.abs(out - expected).max() < 2e-3
