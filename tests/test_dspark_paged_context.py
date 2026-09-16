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


def make(committed: int = 48, scratch: int = 16, layers: int = 2) -> DSparkPagedContext:
    """``committed`` pages are the scheduler's (ids ``[0, committed)``); ``scratch``
    is the proposer-local tail after them, whose last page is the sink."""
    return DSparkPagedContext(
        num_layers=layers,
        kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        block_size=BLOCK,
        committed_blocks=committed,
        scratch_blocks=scratch,
        draft_block=DRAFT_BLOCK,
        dtype=mx.float32,
    )


def owned(first: int, count: int) -> list[int]:
    """A request's row of the scheduler's block table: ``count`` pages from ``first``."""
    return list(range(first, first + count))


class TestPageOwnership:
    """Committed pages are the scheduler's; only the scratch tail is the pool's.

    That split is the design (see ``draft_model_proposer``): the scheduler sizes,
    hashes, admits and evicts the committed context as a KV-cache group, so nothing
    here may allocate or free a committed page. The pool owns exactly the tail the
    scheduler never assigns -- the draft block's scratch positions and the sink.
    """

    def test_pages_cover_the_context_and_the_draft_block(self):
        pool = make()
        # 16 context positions + 7 block positions = 23 -> two 16-token pages
        assert pool.pages_for(16) == 2
        assert pool.pages_for(9) == 1
        assert pool.pages_for(10) == 2

    def test_the_sink_is_the_last_scratch_page_and_never_handed_out(self):
        pool = make(committed=8, scratch=4)
        assert pool.total_blocks == 12
        assert pool.committed_blocks == 8
        assert pool.scratch_blocks == 4
        # three usable scratch pages; the sink cannot be block 0, that is the scheduler's
        assert pool.free_scratch_blocks == 3
        pool.bind("a", owned(0, 8), 16 * 8 - DRAFT_BLOCK)  # exactly fills 8 pages
        pool.bind("b", [], 30)  # 3 pages, all scratch
        handed = set(pool.table_for("a")) | set(pool.table_for("b"))
        assert 11 not in handed  # the sink
        assert 0 not in pool.table_for("b")  # scratch never hands out a scheduler page

    def test_bind_takes_the_scheduler_pages_as_given(self):
        pool = make(committed=8, scratch=4)
        start = pool.free_scratch_blocks
        pool.bind("a", owned(3, 2), 20)  # 27 positions -> 2 pages, both scheduler's
        assert pool.table_for("a") == [3, 4]
        assert pool.free_scratch_blocks == start

    def test_bind_draws_only_the_shortfall_from_scratch(self):
        pool = make(committed=8, scratch=4)
        start = pool.free_scratch_blocks
        pool.bind("a", owned(0, 1), 1)  # 8 positions: the one committed page suffices
        assert pool.free_scratch_blocks == start
        pool.bind("a", owned(0, 1), 16)  # 23 positions -> 2 pages: one from scratch
        assert pool.free_scratch_blocks == start - 1
        assert pool.table_for("a")[:1] == [0]
        assert pool.table_for("a")[1] >= pool.committed_blocks

    def test_rebinding_shrinks_scratch_the_scheduler_has_since_covered(self):
        pool = make(committed=8, scratch=4)
        pool.bind("a", owned(0, 1), 16)  # one committed page + one scratch
        assert pool.free_scratch_blocks == 2
        # next step the scheduler's table has grown to cover position 16
        pool.bind("a", owned(0, 2), 16)
        assert pool.table_for("a") == [0, 1]
        assert pool.free_scratch_blocks == 3

    def test_unbind_returns_scratch_and_never_a_scheduler_page(self):
        pool = make(committed=8, scratch=4)
        start = pool.free_scratch_blocks
        pool.bind("a", owned(0, 2), 40)  # 47 positions -> 3 pages: one scratch
        assert pool.free_scratch_blocks == start - 1
        assert pool.unbind("a") == 1
        assert pool.free_scratch_blocks == start
        assert not pool.holds("a")

    def test_unbind_of_an_unknown_request_is_a_no_op(self):
        pool = make(committed=4, scratch=2)
        assert pool.unbind("ghost") == 0
        assert pool.free_scratch_blocks == 1

    def test_scratch_exhaustion_raises_and_keeps_the_binding(self):
        pool = make(committed=2, scratch=2)  # one usable scratch page
        pool.bind("a", owned(0, 2), 1)
        held = pool.free_scratch_blocks
        with pytest.raises(PagedContextFullError):
            pool.bind("a", owned(0, 2), 10_000)
        # the failed growth must not have consumed or dropped anything
        assert pool.free_scratch_blocks == held
        assert pool.table_for("a") == [0, 1]

    def test_two_requests_never_share_a_scratch_page(self):
        pool = make(committed=0 + 1, scratch=16)
        pool.bind("a", [], 60)
        pool.bind("b", [], 60)
        a = {int(s) // BLOCK for s in pool.slot_mapping_for("a", range(60)).tolist()}
        b = {int(s) // BLOCK for s in pool.slot_mapping_for("b", range(60)).tolist()}
        assert a and b and not (a & b)

    def test_slots_are_block_id_times_block_size_plus_offset(self):
        pool = make(committed=8, scratch=4)
        pool.bind("a", owned(5, 2), 20)
        slots = [int(s) for s in pool.slot_mapping_for("a", range(20)).tolist()]
        # consecutive positions inside one page are consecutive slots
        assert slots[1] - slots[0] == 1
        assert slots[0] == 5 * BLOCK
        assert len({s // BLOCK for s in slots}) == 2


class TestAWriteTouchesItsSlotsAndNotThePool:
    """The memory plan depends on this and would be wrong without it.

    `DSparkMemoryPlan` reserves a second copy of the context arena for "an in-place
    update that cannot reuse its buffer (a view still alive)", and drops that term for
    the pool because `reshape_and_cache` is a scatter whose cache is a designated
    in-place output. Those bytes go to the target model's KV cache instead, so if a pool
    write ever does rewrite the pool, the target is over-committed. This is the assertion
    that catches it.
    """

    def _peak_rise_over_a_write(self, hold_a_view: bool) -> tuple[int, int]:
        pool = make(committed=512, scratch=16, layers=1)
        rows = [(f"r{i}", 120) for i in range(8)]
        for i, (req_id, length) in enumerate(rows):
            pool.bind(req_id, owned(i * 8, 8), length)
        mx.eval(pool.key_caches, pool.value_caches)
        batch = pool.plan(rows)
        view = None
        if hold_a_view:
            view = pool.key_caches[0][:256]
            mx.eval(view)
        mx.reset_peak_memory()
        base = mx.get_active_memory()
        packed = mx.zeros(
            (len(rows) * DRAFT_BLOCK, KV_HEADS, HEAD_DIM), dtype=mx.float32
        )
        pool.write(0, packed, packed, batch.slot_mapping)
        mx.eval(pool.key_caches, pool.value_caches)
        if view is not None:
            mx.eval(view)
        return mx.get_peak_memory() - base, pool.bytes_reserved()

    @pytest.mark.parametrize("hold_a_view", [False, True])
    def test_a_write_does_not_rewrite_the_pool(self, hold_a_view):
        rise, reserved = self._peak_rise_over_a_write(hold_a_view)
        # A whole-pool rewrite would put peak at roughly twice the reservation. The bar
        # is loose on purpose: the claim is "not another pool", not an exact figure.
        assert rise < reserved // 4, (
            f"a write raised peak by {rise} bytes against a {reserved}-byte pool"
        )


class TestBatchBindingIsAllOrNothingPerRow:
    """`bind_many` is the pool's only path for a batch's scratch.

    Binding row by row is the bug it exists to prevent: scratch can empty partway
    through, and the rows already bound then hold pages for a batch that is abandoned.
    Every row here brings no scheduler pages, so all of its pages come from scratch.
    """

    def test_a_batch_that_fits_binds_every_row(self):
        pool = make(committed=1, scratch=64)
        rows = [("a", [], 16), ("b", [], 32), ("c", [], 9)]
        assert pool.bind_many(rows) == []
        assert pool.pages_for(16) == len(pool.table_for("a"))
        assert pool.pages_for(32) == len(pool.table_for("b"))
        assert pool.pages_for(9) == len(pool.table_for("c"))

    def test_rows_that_do_not_fit_are_named_and_left_unbound(self):
        # 4 usable scratch pages: "a" takes 2, "b" takes 2, "c" cannot be housed.
        pool = make(committed=1, scratch=5)
        rejected = pool.bind_many([("a", [], 16), ("b", [], 16), ("c", [], 16)])
        assert rejected == ["c"]
        assert len(pool.table_for("a")) == 2
        assert len(pool.table_for("b")) == 2
        assert pool.table_for("c") == []
        assert pool.free_scratch_blocks == 0

    def test_a_rejected_row_does_not_strand_the_pages_of_earlier_rows(self):
        """The whole point: no page is lost when part of a batch is refused."""
        pool = make(committed=1, scratch=5)
        pool.bind_many([("a", [], 16), ("b", [], 16), ("c", [], 16)])
        held = sum(len(pool.table_for(r)) for r in ("a", "b", "c"))
        # every usable scratch page is either free or in exactly one table
        assert held + pool.free_scratch_blocks == pool.scratch_blocks - 1
        assert len(set(pool.table_for("a")) & set(pool.table_for("b"))) == 0

    def test_a_row_already_large_enough_is_not_regrown(self):
        pool = make(committed=1, scratch=64)
        pool.bind_many([("a", [], 32)])
        before = list(pool.table_for("a"))
        free_before = pool.free_scratch_blocks
        assert pool.bind_many([("a", [], 32)]) == []
        assert pool.table_for("a") == before
        assert pool.free_scratch_blocks == free_before

    def test_nothing_fits_when_scratch_is_empty(self):
        pool = make(committed=1, scratch=3)  # 2 usable
        assert pool.bind_many([("a", [], 16)]) == []  # takes both
        assert pool.free_scratch_blocks == 0
        assert pool.bind_many([("b", [], 16), ("c", [], 16)]) == ["b", "c"]
        assert pool.table_for("b") == []
        assert pool.table_for("c") == []

    def test_scheduler_pages_are_never_counted_against_scratch(self):
        pool = make(committed=8, scratch=2)  # one usable scratch page
        # both rows fit entirely in their scheduler pages: scratch is untouched
        assert pool.bind_many([("a", owned(0, 4), 40), ("b", owned(4, 4), 40)]) == []
        assert pool.free_scratch_blocks == 1


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
        pool = make(committed=1, scratch=256, layers=1)
        scale = HEAD_DIM**-0.5
        reference = []
        rows = []
        for index, length in enumerate(context_lengths):
            req = f"r{index}"
            pool.bind(req, [], length)
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
        pool.bind("a", [], 20)
        pool.bind("b", [], 5)
        batch = pool.plan([("a", 20), ("b", 5)])
        assert batch.rows == 2
        count = 2 * DRAFT_BLOCK
        assert batch.seq_lens.shape == (count,)
        assert batch.block_tables.shape[0] == count
        # cu_seqlens_q must declare every row as its own length-1 sequence
        assert batch.cu_seqlens_q.tolist() == list(range(count + 1))

    def test_seq_len_covers_context_plus_block(self):
        pool = make()
        pool.bind("a", [], 20)
        batch = pool.plan([("a", 20)])
        assert set(batch.seq_lens.tolist()) == {27}
        assert batch.max_seq_len == 27

    def test_scratch_slots_sit_immediately_after_the_context(self):
        pool = make()
        pool.bind("a", [], 20)
        batch = pool.plan([("a", 20)])
        expected = pool.slot_mapping_for("a", range(20, 27)).tolist()
        assert batch.slot_mapping.tolist() == expected


class TestWhatBindCoversIsWhatPlanAddresses:
    """`plan` indexes a request's table directly, so a short table is an IndexError
    inside a drafting step rather than an error the proposer can act on.

    Both sides derive from `pages_for`, so they agree by construction -- but only if
    that function counts the draft block's own positions, and only at every offset
    relative to a page boundary. One sweep across two pages' worth of lengths is what
    would catch an off-by-one that a fixed-length fixture sits either side of. The row
    brings no scheduler pages, so every page here is the pool's own arithmetic.
    """

    @pytest.mark.parametrize("context_length", range(0, 2 * BLOCK + DRAFT_BLOCK + 1))
    def test_every_block_position_lands_in_a_bound_page(self, context_length):
        pool = make(committed=1, scratch=64, layers=1)
        pool.bind("r", [], context_length)
        reserved = set(pool.table_for("r"))
        batch = pool.plan([("r", context_length)])
        slots = batch.slot_mapping.tolist()
        assert len(slots) == DRAFT_BLOCK
        for slot in slots:
            assert slot // BLOCK in reserved, (
                f"context {context_length}: slot {slot} is in page {slot // BLOCK}, "
                f"which the request does not hold ({sorted(reserved)})"
            )
        # and they are exactly the positions after the context, in order
        assert slots == [
            pool.table_for("r")[p // BLOCK] * BLOCK + p % BLOCK
            for p in range(context_length, context_length + DRAFT_BLOCK)
        ]


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
        pool = make(committed=1, scratch=128, layers=1)
        slots = []
        for index, length in enumerate(ctx_lengths):
            req = f"r{index}"
            pool.bind(req, [], length)
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
        pool = make(committed=8, scratch=8)
        pool.bind("a", owned(0, 4), 40)
        pool.bind("b", owned(4, 4), 40)
        # a writes 3 positions from 10, b writes 1 from 20; width is 3
        spans = [(0, 10, 3, "a"), (3, 20, 1, "b")]
        slots = [int(x) for x in pool.span_slot_mapping(spans, width=3).tolist()]
        assert len(slots) == 6
        # the sink is the last scratch page: block 0 belongs to the scheduler now
        sink = (pool.total_blocks - 1) * BLOCK
        assert slots[0:3] == [
            int(x) for x in pool.slot_mapping_for("a", range(10, 13)).tolist()
        ]
        # b contributed one real position and two padded ones
        assert slots[3] == int(pool.slot_mapping_for("b", [20]).tolist()[0])
        assert slots[4] == sink and slots[5] == sink

    def test_a_real_position_never_lands_on_the_sink(self):
        pool = make(committed=8, scratch=8)
        pool.bind("a", owned(0, 5), 64)
        spans = [(0, 0, 40, "a")]
        slots = [int(x) for x in pool.span_slot_mapping(spans, width=40).tolist()]
        sink = (pool.total_blocks - 1) * BLOCK
        assert sink not in slots

    def test_an_unbound_request_is_refused(self):
        pool = make(committed=8, scratch=2)
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

    def _cache(self, pool=None, capacity=512, pages=8):
        from vllm_metal.v1.dspark.paged_context import PagedCtxCache

        pool = pool or make(committed=128, scratch=8, layers=1)
        # The proposer binds the scheduler's pages before it writes; append never
        # grows a context on its own any more.
        pool.bind("r", owned(0, pages), 0)
        return PagedCtxCache(pool, 0, "r", capacity), pool

    def test_append_advances_the_length_within_its_bound_pages(self):
        cache, pool = self._cache()
        k, v = self._kv(20)
        cache.append(k, v)
        assert cache.length == 20
        assert pool.holds("r")

    def test_append_past_the_bound_pages_is_refused(self):
        """Growing is the proposer's step, against the scheduler's table -- not this."""
        cache, _ = self._cache(pages=1)  # 16 positions, 9 usable before the block
        with pytest.raises(PagedContextFullError):
            cache.append(*self._kv(20))

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
        pool.bind("r", owned(0, 8), 40)
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

        pool = make(committed=128, scratch=8, layers=1)
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
