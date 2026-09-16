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

Two consequences follow from using the kernel rather than a per-row Python loop:

- One ``paged_attention_primitive`` dispatch per layer covers the whole drafting batch,
  instead of one ``scaled_dot_product_attention`` call per row per layer.
- Blocks are allocated as a context grows, so a server no longer reserves
  ``slots x max_model_len`` of drafter context up front.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from vllm_metal.metal import get_ops

__all__ = [
    "DSparkPagedContext",
    "DraftBatch",
    "PagedContextFullError",
    "PagedLayerBatch",
]


class PagedContextFullError(RuntimeError):
    """The drafter's block pool cannot satisfy an allocation.

    Raised rather than silently evicting: the proposer's caller turns this into
    target-only generation for the request, which is the documented fallback.
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
        "_num_blocks",
        "_tables",
        "key_caches",
        "value_caches",
    )

    def __init__(
        self,
        *,
        num_layers: int,
        kv_heads: int,
        head_dim: int,
        block_size: int,
        num_blocks: int,
        draft_block: int,
        dtype: mx.Dtype,
    ) -> None:
        if (
            min(num_layers, kv_heads, head_dim, block_size, num_blocks, draft_block)
            <= 0
        ):
            raise ValueError("DSparkPagedContext needs positive geometry")
        if block_size not in (8, 16, 32):
            raise ValueError(
                f"paged attention supports block sizes 8, 16 and 32, not {block_size}"
            )
        shape = (num_blocks, block_size, kv_heads, head_dim)
        self.key_caches = [mx.zeros(shape, dtype=dtype) for _ in range(num_layers)]
        self.value_caches = [mx.zeros(shape, dtype=dtype) for _ in range(num_layers)]
        self._block_size = block_size
        self._draft_block = draft_block
        self._kv_heads = kv_heads
        self._head_dim = head_dim
        self._num_blocks = num_blocks
        self._free = list(reversed(range(num_blocks)))
        self._tables: dict[str, list[int]] = {}

    # ---- pool accounting -------------------------------------------------

    @property
    def free_blocks(self) -> int:
        return len(self._free)

    @property
    def total_blocks(self) -> int:
        return self._num_blocks

    def bytes_reserved(self) -> int:
        cache = self.key_caches[0]
        return 2 * len(self.key_caches) * cache.size * cache.dtype.size

    def pages_for(self, context_length: int) -> int:
        """Pages a context of this length needs, including the draft block's scratch."""
        total = context_length + self._draft_block
        return (total + self._block_size - 1) // self._block_size

    # ---- per-request block tables ---------------------------------------

    def reserve(self, req_id: str, context_length: int) -> None:
        """Grow ``req_id``'s block table so it covers ``context_length`` plus the block.

        Allocates only the pages the length actually needs. Raises
        :class:`PagedContextFullError` when the pool is exhausted, leaving the request's
        existing blocks untouched so the caller can fall back to target-only.
        """
        needed = self.pages_for(context_length)
        table = self._tables.setdefault(req_id, [])
        if needed <= len(table):
            return
        short = needed - len(table)
        if short > len(self._free):
            raise PagedContextFullError(
                f"DSpark context pool exhausted: {req_id} needs {short} more blocks, "
                f"{len(self._free)} free of {self._num_blocks}"
            )
        for _ in range(short):
            table.append(self._free.pop())

    def release(self, req_id: str) -> int:
        """Return ``req_id``'s blocks to the pool. Returns how many were freed."""
        table = self._tables.pop(req_id, None)
        if not table:
            return 0
        self._free.extend(reversed(table))
        return len(table)

    def holds(self, req_id: str) -> bool:
        return req_id in self._tables

    def slot_mapping_for(self, req_id: str, positions: range | list[int]) -> mx.array:
        """Flat cache slots for a request's absolute context positions.

        A slot is ``block_id * block_size + offset``, the same address
        ``reshape_and_cache`` expects for the target's cache.
        """
        table = self._tables.get(req_id)
        if table is None:
            raise KeyError(f"no draft context blocks reserved for {req_id}")
        slots = []
        for position in positions:
            page, offset = divmod(position, self._block_size)
            if page >= len(table):
                raise PagedContextFullError(
                    f"position {position} of {req_id} is past its reserved blocks"
                )
            slots.append(table[page] * self._block_size + offset)
        # int64: reshape_and_cache reads 64-bit slots (attention/impls/sdpa.py:212)
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
