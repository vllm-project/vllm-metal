# SPDX-License-Identifier: Apache-2.0
"""The DSpark drafter's context K/V in a paged pool, served by vllm-metal's kernel.

The drafter cross-attends a ``block_size``-position draft block over the request's
committed context. For every committed token position the proposer stores, per drafter
layer, one K and one V vector projected from the fused target residuals at that position.
That is structurally a KV cache entry and a pure function of the token prefix, so it
belongs in the same paged pool the rest of vllm-metal uses, reached through block tables,
rather than in a private per-request buffer.

The draft block is *bidirectional*: every block position attends the whole context and
every other block position. vllm-metal's paged kernel is hard-causal with no mask input,
which looks like it rules the kernel out. It does not. The kernel derives its mask per
query sequence::

    effective_context_len = context_len - q_len + q_pos_in_seq + 1

so a query sequence of length one attends exactly ``[0, context_len)``. Declaring each
draft-block position as its own length-1 query sequence over a shared block-table row
therefore gives every position the full context plus the entire block, which is the
bidirectional semantics, with no kernel change. ``tools/dspark_paged_probe.py`` checks
this against a dense reference and against the naive length-``block`` packing, which is
causal and differs by six orders of magnitude.

Who owns the blocks. The committed context is a scheduler-owned KV-cache group:
``cache_policy._draft_layer_specs`` registers one ``FullAttentionSpec`` per drafter
layer, so the scheduler sizes it from the KV budget, allocates it per request and
evicts it, exactly as it does the target's own groups -- the same arrangement
``draft_model_proposer`` uses for a separate draft model. A request's committed pages
arrive as ``RequestState.block_ids[group]`` and are never grown or freed here. Only the
draft block's scratch positions -- written past a request's committed length and never
verified, so no scheduler group is ever "ahead" of them -- come from a small
proposer-local tail the scheduler never assigns, sized by
``cache_policy.draft_scratch_reserve_blocks``. One scratch block is the padding sink a
ragged ingest points its padded columns at.

One ``paged_attention_primitive`` dispatch per layer covers the whole drafting batch,
instead of one ``scaled_dot_product_attention`` call per row per layer.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from vllm_metal.attention.caches.kv_cache import MetalPagedKVCache
from vllm_metal.metal import get_ops

from .paging import KERNEL_HEAD_SIZES

__all__ = [
    "KERNEL_HEAD_SIZES",
    "DSparkPagedContext",
    "DraftBatch",
    "PagedContextFullError",
    "PagedCtxCache",
    "PagedLayerBatch",
]


class PagedContextFullError(RuntimeError):
    """The proposer-local scratch tail cannot house a request's draft block.

    The committed context is the scheduler's to allocate and cannot run out here; the
    scratch tail is sized for every active request drafting at once, so this is
    reachable only when that sizing and the scheduler's admission disagree. Raised
    rather than silently evicting: the caller turns it into target-only generation for
    the request, which is the documented fallback.
    """


@dataclass(frozen=True, slots=True)
class DraftBatch:
    """One drafting step's rows, addressed in the paged pool.

    ``block_tables`` and ``seq_lens`` carry one row per *query position*, not per
    request: each of a request's ``block_size`` draft positions is its own length-1
    query sequence, and they share the request's blocks.
    """

    block_tables: mx.array  # [rows * block_size, max_pages] int32
    seq_lens: mx.array  # [rows * block_size] int32
    cu_seqlens_q: mx.array  # [rows * block_size + 1] int32
    slot_mapping: mx.array  # [rows * block_size] int64, where the block K/V is written
    max_seq_len: int
    rows: int


class DSparkPagedContext:
    """A paged K/V pool for the drafter's per-request context, one cache per layer.

    Layout matches the target's: ``[num_blocks, block_size, kv_heads, head_dim]`` per
    layer, addressed by block tables, written with ``reshape_and_cache`` and read with
    ``paged_attention_primitive`` — the same two primitives the target attention uses.
    """

    __slots__ = (
        "_block_size",
        "_draft_block",
        "_free",
        "_head_dim",
        "_kv_heads",
        "_sink",
        "_tables",
        "key_caches",
        "value_caches",
        "_cache",
        "_committed_blocks",
        "_scratch_blocks",
        "_total_blocks",
        "_scratch",
    )

    def __init__(
        self,
        *,
        num_layers: int,
        kv_heads: int,
        head_dim: int,
        block_size: int,
        committed_blocks: int,
        scratch_blocks: int,
        draft_block: int,
        dtype: mx.Dtype,
    ) -> None:
        if (
            min(
                num_layers,
                kv_heads,
                head_dim,
                block_size,
                committed_blocks,
                draft_block,
            )
            <= 0
        ):
            raise ValueError("DSparkPagedContext needs positive geometry")
        if scratch_blocks < 1:
            raise ValueError("the scratch tail needs at least one block for the sink")
        if block_size not in (8, 16, 32):
            raise ValueError(
                f"paged attention supports block sizes 8, 16 and 32, not {block_size}"
            )
        if head_dim not in KERNEL_HEAD_SIZES:
            raise ValueError(
                f"paged attention is not instantiated for head size {head_dim}; "
                f"supported: {', '.join(str(h) for h in KERNEL_HEAD_SIZES)}"
            )
        # The same paged cache class the draft-model proposer's runtime allocates
        # (``SDPAPagedAttentionRuntime.initialize``): scheduler-visible committed
        # pages first, then the proposer-local scratch tail the scheduler never
        # assigns. ``write`` rebinds entries in place, so share the cache's lists.
        self._cache = MetalPagedKVCache(
            num_layers=num_layers,
            num_kv_heads=kv_heads,
            head_dim=head_dim,
            num_blocks=committed_blocks + scratch_blocks,
            block_size=block_size,
            dtype=dtype,
        )
        self.key_caches = self._cache.key_caches
        self.value_caches = self._cache.value_caches
        self._block_size = block_size
        self._draft_block = draft_block
        self._kv_heads = kv_heads
        self._head_dim = head_dim
        self._committed_blocks = committed_blocks
        self._scratch_blocks = scratch_blocks
        self._total_blocks = committed_blocks + scratch_blocks
        # The last scratch block is the sink: a ragged ingest pads its rows to the
        # widest span, and those padded positions need a destination that no request
        # can read back. It cannot be block 0 any more -- that page is the scheduler's.
        self._sink = self._total_blocks - 1
        self._free = list(reversed(range(committed_blocks, self._sink)))
        # Per request: the scheduler's committed pages followed by this request's
        # scratch pages, in context order; and the scratch pages alone, to hand back.
        self._tables: dict[str, list[int]] = {}
        self._scratch: dict[str, list[int]] = {}

    # ---- pool accounting -------------------------------------------------

    @property
    def free_scratch_blocks(self) -> int:
        return len(self._free)

    @property
    def committed_blocks(self) -> int:
        """Scheduler-owned pages: ids ``[0, committed_blocks)``."""
        return self._committed_blocks

    @property
    def scratch_blocks(self) -> int:
        """Proposer-local pages after them, one of which is the sink."""
        return self._scratch_blocks

    @property
    def total_blocks(self) -> int:
        return self._total_blocks

    def bytes_reserved(self) -> int:
        cache = self.key_caches[0]
        return 2 * len(self.key_caches) * cache.size * cache.dtype.size

    def request_bytes(self, req_id: str) -> int:
        """Bytes of pool the request's blocks occupy across every layer."""
        table = self._tables.get(req_id)
        if not table:
            return 0
        cache = self.key_caches[0]
        per_block = cache.size // self._total_blocks * cache.dtype.size
        return 2 * len(self.key_caches) * len(table) * per_block

    def pages_for(self, context_length: int) -> int:
        """Pages a context of this length needs, including the draft block's scratch."""
        total = context_length + self._draft_block
        return (total + self._block_size - 1) // self._block_size

    # ---- per-request block tables ---------------------------------------

    def bind(
        self, req_id: str, committed_block_ids: list[int], context_length: int
    ) -> None:
        """Address ``req_id`` for this step: the scheduler's pages plus scratch.

        ``committed_block_ids`` is the request's row of the scheduler's block table for
        the drafter's group, taken as given: never grown or freed here. The tail beyond
        it -- what ``context_length`` plus the draft block needs and the scheduler did
        not allocate -- is drawn from the local scratch pool, grown or shrunk to exactly
        this step's need, mirroring ``DraftModelProposer._ensure_blocks``.

        Raises :class:`PagedContextFullError` when the scratch tail cannot supply the
        shortfall, leaving the request's binding unchanged so the caller can fall back
        to target-only.
        """
        needed = self.pages_for(context_length)
        scratch_needed = max(0, needed - len(committed_block_ids))
        scratch = self._scratch.setdefault(req_id, [])
        if len(scratch) > scratch_needed:
            self._free.extend(scratch[scratch_needed:])
            del scratch[scratch_needed:]
        short = scratch_needed - len(scratch)
        if short > len(self._free):
            raise PagedContextFullError(
                f"DSpark scratch tail exhausted: {req_id} needs {short} more block(s), "
                f"{len(self._free)} free of {self._scratch_blocks - 1}"
            )
        for _ in range(short):
            scratch.append(self._free.pop())
        if not scratch:
            self._scratch.pop(req_id, None)
        self._tables[req_id] = list(committed_block_ids) + scratch

    def bind_many(self, rows: list[tuple[str, list[int], int]]) -> list[str]:
        """Bind every row, or leave unbound exactly the rows scratch cannot house.

        ``rows`` is ``(req_id, committed_block_ids, context_length)`` per row. Returns
        the request ids that did not fit, in the order given; every other row is bound.

        Binding row by row is what a caller must not do: scratch can empty partway
        through, leaving earlier rows bound while the batch built around all of them is
        abandoned. This plans every row's shortfall against the free list before it
        touches a single binding, and grants in arrival order so a large latecomer
        cannot starve rows already being served.
        """
        shortfalls: list[tuple[str, list[int], int, int]] = []
        for req_id, committed, context_length in rows:
            needed = max(0, self.pages_for(context_length) - len(committed))
            held = len(self._scratch.get(req_id, ()))
            shortfalls.append((req_id, committed, context_length, needed - held))
        available = len(self._free)
        rejected: list[str] = []
        for req_id, _, _, short in shortfalls:
            if short > available:
                rejected.append(req_id)
                continue
            available -= max(0, short)
        for req_id, committed, context_length, _ in shortfalls:
            if req_id not in rejected:
                self.bind(req_id, committed, context_length)
        return rejected

    def unbind(self, req_id: str) -> int:
        """Drop ``req_id``'s binding and return its scratch pages. Returns how many.

        The committed pages are the scheduler's to reclaim; only scratch comes back.
        """
        self._tables.pop(req_id, None)
        scratch = self._scratch.pop(req_id, None)
        if not scratch:
            return 0
        self._free.extend(reversed(scratch))
        return len(scratch)

    def holds(self, req_id: str) -> bool:
        return req_id in self._tables

    def table_for(self, req_id: str) -> list[int]:
        """The blocks a request holds, in context order. Empty if it holds none.

        A copy: the pool's own table is not the caller's to edit.
        """
        return list(self._tables.get(req_id, ()))

    def slot_mapping_for(self, req_id: str, positions: range | list[int]) -> mx.array:
        """Flat cache slots for a request's absolute context positions.

        A slot is ``block_id * block_size + offset``, the same address
        ``reshape_and_cache`` expects for the target's cache.
        """
        table = self._tables.get(req_id)
        if table is None:
            raise KeyError(f"no draft context bound for {req_id}")
        slots = []
        for position in positions:
            page, offset = divmod(position, self._block_size)
            if page >= len(table):
                raise PagedContextFullError(
                    f"position {position} of {req_id} is past its bound pages"
                )
            slots.append(table[page] * self._block_size + offset)
        # int64: reshape_and_cache reads 64-bit slots (attention/impls/sdpa.py:212)
        return mx.array(slots, dtype=mx.int64)

    def span_slot_mapping(
        self, spans: list[tuple[int, int, int, str]], width: int
    ) -> mx.array:
        """Slots for a padded ``[rows, width]`` ingest grid, flattened row-major.

        ``spans`` are ``(start_row, start_pos, count, req_id)``. A row shorter than
        ``width`` is padded by the caller repeating its last feature; those columns are
        addressed to the sink block, so the padding is written and never read.
        """
        sink = self._sink * self._block_size
        slots: list[int] = []
        for _, start_pos, count, req_id in spans:
            table = self._tables.get(req_id)
            if table is None:
                raise KeyError(f"no draft context bound for {req_id}")
            for column in range(width):
                if column >= count:
                    slots.append(sink)
                    continue
                page, offset = divmod(start_pos + column, self._block_size)
                if page >= len(table):
                    raise PagedContextFullError(
                        f"position {start_pos + column} of {req_id} is past its pages"
                    )
                slots.append(table[page] * self._block_size + offset)
        return mx.array(slots, dtype=mx.int64)

    # ---- write and read, through vllm-metal's own primitives -------------

    def write(
        self, layer: int, keys: mx.array, values: mx.array, slot_mapping: mx.array
    ) -> None:
        """Scatter ``[tokens, kv_heads, head_dim]`` K/V into the pool in one dispatch.

        The caches are rebound to the returned arrays because the primitive aliases its
        inputs in place and carries the graph provenance the subsequent read needs.
        """
        new_k, new_v = get_ops().reshape_and_cache(
            keys.astype(self.key_caches[layer].dtype),
            values.astype(self.value_caches[layer].dtype),
            self.key_caches[layer],
            self.value_caches[layer],
            slot_mapping,
        )
        self.key_caches[layer] = new_k
        self.value_caches[layer] = new_v

    def plan(self, rows: list[tuple[str, int]]) -> DraftBatch:
        """Address one drafting step: ``rows`` is ``(req_id, context_length)`` per row.

        Every request contributes ``draft_block`` query positions, each declared as its
        own length-1 sequence so it attends the whole context and the whole block.
        """
        if not rows:
            raise ValueError("a drafting step needs at least one row")
        block = self._draft_block
        width = max(len(self._tables[req_id]) for req_id, _ in rows)
        tables: list[list[int]] = []
        lengths: list[int] = []
        scratch: list[int] = []
        for req_id, context_length in rows:
            table = self._tables[req_id]
            padded = table + [0] * (width - len(table))
            total = context_length + block
            for position in range(context_length, total):
                page, offset = divmod(position, self._block_size)
                scratch.append(table[page] * self._block_size + offset)
            # one query sequence per block position, all sharing this request's blocks
            tables.extend([padded] * block)
            lengths.extend([total] * block)
        count = len(rows) * block
        return DraftBatch(
            block_tables=mx.array(tables, dtype=mx.int32),
            seq_lens=mx.array(lengths, dtype=mx.int32),
            cu_seqlens_q=mx.arange(count + 1, dtype=mx.int32),
            slot_mapping=mx.array(scratch, dtype=mx.int64),
            max_seq_len=max(lengths),
            rows=len(rows),
        )

    def attend(
        self, layer: int, queries: mx.array, batch: DraftBatch, scale: float
    ) -> mx.array:
        """One paged-attention dispatch for the whole drafting batch of this layer.

        ``queries`` is ``[rows * draft_block, n_heads, head_dim]`` in row-major order,
        matching :meth:`plan`. Returns the attention output in the same shape.
        """
        out = mx.zeros(queries.shape, dtype=queries.dtype)
        get_ops().paged_attention_primitive(
            queries,
            self.key_caches[layer],
            self.value_caches[layer],
            self._kv_heads,
            scale,
            0.0,  # softcap: the drafter's attention is uncapped
            batch.block_tables,
            batch.seq_lens,
            batch.cu_seqlens_q,
            self._block_size,
            batch.max_seq_len,
            -1,  # no sliding window: the standalone checkpoint needs full attention
            out,
            window_seqlen_q=1,  # per-token decode: one query row per threadgroup
        )
        return out


class PagedLayerBatch:
    """One layer's view of a drafting step, shaped like :class:`ArenaBatch`.

    ``DSparkAttention.attend`` drives the context through two calls, ``write_block``
    then ``attend``, and works in ``[rows, heads, block, head_dim]``. The paged pool
    works in packed varlen ``[rows * block, heads, head_dim]``, so this adapter owns the
    transposes and leaves the attention module unchanged apart from accepting it.
    """

    __slots__ = ("_batch", "_layer", "_pool")

    def __init__(self, pool: DSparkPagedContext, layer: int, batch: DraftBatch) -> None:
        self._pool = pool
        self._layer = layer
        self._batch = batch

    @staticmethod
    def _pack(tensor: mx.array) -> mx.array:
        """[rows, heads, block, dim] -> [rows * block, heads, dim], row-major."""
        rows, heads, block, dim = tensor.shape
        return mx.contiguous(
            tensor.transpose(0, 2, 1, 3).reshape(rows * block, heads, dim)
        )

    def write_block(self, k_blk: mx.array, v_blk: mx.array) -> None:
        """Place each row's draft-block K/V at the scratch positions after its context."""
        self._pool.write(
            self._layer,
            self._pack(k_blk),
            self._pack(v_blk),
            self._batch.slot_mapping,
        )

    def attend(self, q: mx.array, scale: float) -> mx.array:
        """One paged dispatch for every row and block position of this layer."""
        rows, heads, block, dim = q.shape
        out = self._pool.attend(self._layer, self._pack(q), self._batch, scale)
        return out.reshape(rows, block, heads, dim).transpose(0, 2, 1, 3)


class PagedCtxCache:
    """One layer's view of a request's context, stored in the paged pool.

    Satisfies the parts of :class:`~vllm_metal.v1.dspark.model.CtxCache` the proposer's
    lifecycle uses — ``length``, ``append``, ``trim_to``, ``extend_to``,
    ``allocated_bytes`` — so admission, rollback, recompute and release keep working
    unchanged when the context moves from a private arena into the pool.

    ``k``/``v`` are deliberately absent. They exist on the arena cache so a
    single-request step can concatenate the context and attend it densely; a paged
    context is never materialised that way, and every drafting step goes through
    :class:`PagedLayerBatch`. Reaching for them is a bug, so it raises rather than
    quietly rebuilding the dense tensor the pool exists to avoid.
    """

    __slots__ = ("_layer", "_length", "_pool", "_req_id", "capacity")

    def __init__(
        self, pool: DSparkPagedContext, layer: int, req_id: str, capacity: int
    ) -> None:
        self._pool = pool
        self._layer = layer
        self._req_id = req_id
        self._length = 0
        self.capacity = capacity

    @property
    def request_id(self) -> str:
        return self._req_id

    @property
    def length(self) -> int:
        return self._length

    @property
    def k(self) -> mx.array:
        raise NotImplementedError(
            "a paged draft context is read through PagedLayerBatch, not materialised"
        )

    @property
    def v(self) -> mx.array:
        raise NotImplementedError(
            "a paged draft context is read through PagedLayerBatch, not materialised"
        )

    @property
    def allocated_bytes(self) -> int:
        """This layer's share of the blocks the request currently holds."""
        return self._pool.request_bytes(self._req_id) // max(
            1, len(self._pool.key_caches)
        )

    def append(self, k: mx.array, v: mx.array) -> None:
        """Write ``[1, kv_heads, tokens, dim]`` at the positions after the committed end."""
        if k.ndim != 4 or v.ndim != 4 or k.shape[:3] != v.shape[:3]:
            raise ValueError(
                "context K/V must cover matching batch, head and token axes"
            )
        count = k.shape[2]
        end = self._length + count
        if end > self.capacity:
            raise ValueError("DSpark context append exceeds reserved capacity")
        if self._pool.pages_for(end) > len(self._pool.table_for(self._req_id)):
            raise PagedContextFullError(
                f"{self._req_id}: appending to {end} outruns its bound pages; the "
                "proposer binds the scheduler's pages before it writes"
            )
        slots = self._pool.slot_mapping_for(self._req_id, range(self._length, end))
        self._pool.write(
            self._layer,
            mx.contiguous(k[0].transpose(1, 0, 2)),
            mx.contiguous(v[0].transpose(1, 0, 2)),
            slots,
        )
        self._length = end

    def trim_to(self, length: int) -> None:
        if not 0 <= length <= self._length:
            raise ValueError("context trim must retain an existing prefix")
        self._length = length

    def extend_to(self, length: int) -> None:
        """Account for positions written through :meth:`DSparkPagedContext.write`."""
        if not self._length <= length <= self.capacity:
            raise ValueError("context extension must stay within reserved capacity")
        self._length = length
