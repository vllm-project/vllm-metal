"""Does vllm-metal's paged attention reproduce the DSpark drafter's block attention?

The DSpark draft block is bidirectional: every one of the `block` positions attends the
request's whole context AND every other block position. vllm-metal's paged kernel is
hard-causal with no mask input, which looks like it rules the kernel out.

It does not. The mask is derived per query sequence:

    effective_context_len = context_len - q_len + q_pos_in_seq + 1     (pagedattention.metal:913)

so a query sequence of length 1 (q_len == 1, q_pos_in_seq == 0) attends exactly
[0, context_len). Declaring each block position as its own length-1 query sequence over a
shared block table therefore gives every position the full [0, L + block) range -- the
bidirectional semantics -- with no kernel change and one dispatch per layer for the whole
batch, instead of today's one SDPA call per row.

This probe checks that claim numerically against the reference MLX path before any
production code is written. Run it directly; it prints a verdict and exits nonzero on
mismatch.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from vllm_metal.metal import get_ops

BLOCK_SIZE = 16  # kernel page size (8/16/32 supported)
KV_HEADS = 8
N_HEADS = 32  # GQA: 4 query heads per kv head
HEAD_DIM = 128
DRAFT_BLOCK = 7  # DSpark block_size


def reference_attention(
    q: mx.array, k: mx.array, v: mx.array, scale: float
) -> mx.array:
    """What the drafter computes today: one SDPA call per row, full context + block.

    q [1, heads, block, dim]; k/v [1, kv_heads, length, dim]. No mask: every block
    position attends every context and block position.
    """
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)


def paged_attention(
    q: mx.array,
    k_cache: mx.array,
    v_cache: mx.array,
    block_tables: mx.array,
    seq_lens: mx.array,
    cu_seqlens_q: mx.array,
    scale: float,
    max_seq_len: int,
) -> mx.array:
    """One primitive call for every block position of every row of the batch.

    `q` is packed varlen [total_q, heads, dim] where total_q = rows * DRAFT_BLOCK, and
    cu_seqlens_q declares each of those rows as its own length-1 sequence.
    """
    out = mx.zeros(q.shape, dtype=q.dtype)
    get_ops().paged_attention_primitive(
        q,
        k_cache,
        v_cache,
        KV_HEADS,
        scale,
        0.0,  # softcap
        block_tables,
        seq_lens,
        cu_seqlens_q,
        BLOCK_SIZE,
        max_seq_len,
        -1,  # sliding_window disabled
        out,
        window_seqlen_q=1,  # per-token decode: one query row per threadgroup
    )
    return out


def build_case(context_lengths: list[int], seed: int):
    """Lay one request per context length into a paged pool, plus its draft block."""
    rng = np.random.default_rng(seed)
    scale = HEAD_DIM**-0.5
    rows = len(context_lengths)

    # Each row needs ceil((L + DRAFT_BLOCK) / BLOCK_SIZE) pages.
    pages_per_row = [
        (length + DRAFT_BLOCK + BLOCK_SIZE - 1) // BLOCK_SIZE
        for length in context_lengths
    ]
    total_pages = sum(pages_per_row)
    k_cache = np.zeros((total_pages, BLOCK_SIZE, KV_HEADS, HEAD_DIM), dtype=np.float32)
    v_cache = np.zeros_like(k_cache)

    table_rows, references, next_page = [], [], 0
    for length, pages in zip(context_lengths, pages_per_row, strict=True):
        page_ids = list(range(next_page, next_page + pages))
        next_page += pages
        table_rows.append(page_ids)

        total = length + DRAFT_BLOCK  # context, then the block's own K/V
        keys = rng.normal(size=(total, KV_HEADS, HEAD_DIM)).astype(np.float32)
        values = rng.normal(size=(total, KV_HEADS, HEAD_DIM)).astype(np.float32)
        for position in range(total):
            page = page_ids[position // BLOCK_SIZE]
            k_cache[page, position % BLOCK_SIZE] = keys[position]
            v_cache[page, position % BLOCK_SIZE] = values[position]
        references.append((keys, values, total))

    width = max(len(r) for r in table_rows)
    block_tables = np.zeros((rows, width), dtype=np.int32)
    for i, page_ids in enumerate(table_rows):
        block_tables[i, : len(page_ids)] = page_ids

    queries = rng.normal(size=(rows, DRAFT_BLOCK, N_HEADS, HEAD_DIM)).astype(np.float32)
    return scale, k_cache, v_cache, block_tables, references, queries


def run(context_lengths: list[int], seed: int) -> float:
    scale, k_cache, v_cache, block_tables, references, queries = build_case(
        context_lengths, seed
    )
    rows = len(context_lengths)

    # ---- reference: today's per-row SDPA over the full [context | block] ----
    expected = []
    for row, (keys, values, total) in enumerate(references):
        q = mx.array(queries[row].transpose(1, 0, 2))[None]  # [1, heads, block, dim]
        k = mx.array(keys[:total].transpose(1, 0, 2))[None]  # [1, kv_heads, total, dim]
        v = mx.array(values[:total].transpose(1, 0, 2))[None]
        out = reference_attention(q, k, v, scale)  # [1, heads, block, dim]
        expected.append(np.array(out[0].transpose(1, 0, 2)))  # [block, heads, dim]
    reference = np.stack(expected)  # [rows, block, heads, dim]

    # ---- paged: every block position is its own length-1 query sequence ----
    total_q = rows * DRAFT_BLOCK
    q_packed = mx.array(queries.reshape(total_q, N_HEADS, HEAD_DIM))
    # one sequence per query row, each pointing at its request's blocks
    seq_lens = mx.array(
        np.repeat([total for _, _, total in references], DRAFT_BLOCK).astype(np.int32)
    )
    tables = mx.array(np.repeat(block_tables, DRAFT_BLOCK, axis=0))
    cu_seqlens_q = mx.array(np.arange(total_q + 1, dtype=np.int32))
    got = paged_attention(
        q_packed,
        mx.array(k_cache),
        mx.array(v_cache),
        tables,
        seq_lens,
        cu_seqlens_q,
        scale,
        int(max(total for _, _, total in references)),
    )
    mx.eval(got)
    actual = np.array(got).reshape(rows, DRAFT_BLOCK, N_HEADS, HEAD_DIM)

    return float(np.abs(actual - reference).max())


def negative_control(context_lengths: list[int], seed: int) -> float:
    """The naive packing: all block positions as ONE length-`block` sequence.

    Then q_len == DRAFT_BLOCK and effective_context_len = context_len - block + r + 1, so
    row r sees only block positions <= r: causal, not bidirectional. This MUST disagree
    with the reference, otherwise the probe cannot tell the two apart and proves nothing.
    """
    scale, k_cache, v_cache, block_tables, references, queries = build_case(
        context_lengths, seed
    )
    rows = len(context_lengths)
    expected = []
    for row, (keys, values, total) in enumerate(references):
        q = mx.array(queries[row].transpose(1, 0, 2))[None]
        k = mx.array(keys[:total].transpose(1, 0, 2))[None]
        v = mx.array(values[:total].transpose(1, 0, 2))[None]
        expected.append(
            np.array(reference_attention(q, k, v, scale)[0].transpose(1, 0, 2))
        )
    reference = np.stack(expected)

    total_q = rows * DRAFT_BLOCK
    q_packed = mx.array(queries.reshape(total_q, N_HEADS, HEAD_DIM))
    seq_lens = mx.array(np.array([t for _, _, t in references], dtype=np.int32))
    tables = mx.array(block_tables)
    cu = mx.array((np.arange(rows + 1) * DRAFT_BLOCK).astype(np.int32))
    got = paged_attention(
        q_packed,
        mx.array(k_cache),
        mx.array(v_cache),
        tables,
        seq_lens,
        cu,
        scale,
        int(max(t for _, _, t in references)),
    )
    mx.eval(got)
    actual = np.array(got).reshape(rows, DRAFT_BLOCK, N_HEADS, HEAD_DIM)
    return float(np.abs(actual - reference).max())


def main() -> int:
    cases = [
        ("single row, short context", [37], 0),
        ("single row, page-aligned", [64], 1),
        ("single row, long context", [1301], 2),
        ("ragged batch", [37, 512, 129, 1301], 3),
        ("batch of equal lengths", [256, 256, 256], 4),
        ("context shorter than a page", [3], 5),
    ]
    worst = 0.0
    failures = 0
    for name, lengths, seed in cases:
        try:
            delta = run(lengths, seed)
        except Exception as error:  # noqa: BLE001 - probe reports, never raises
            print(f"  {name:32s} ERROR  {type(error).__name__}: {error}")
            failures += 1
            continue
        worst = max(worst, delta)
        verdict = "ok" if delta < 2e-3 else "MISMATCH"
        if delta >= 2e-3:
            failures += 1
        print(f"  {name:32s} max|paged - reference| = {delta:.3e}   {verdict}")
    print()
    if failures:
        print(
            f"VERDICT: {failures} of {len(cases)} cases failed — the packing is wrong"
        )
        return 1
    control = negative_control([37, 512, 129, 1301], 3)
    print(
        f"  {'negative control (causal packing)':32s} max|paged - reference| = {control:.3e}"
        f"   {'discriminates' if control > 1e-2 else 'PROBE IS BLIND'}"
    )
    if control <= 1e-2:
        print(
            "\nVERDICT: the probe cannot distinguish causal from bidirectional — it proves nothing"
        )
        return 1
    print(
        f"\nVERDICT: paged attention reproduces the bidirectional draft block "
        f"(worst {worst:.3e}) with no kernel change; the causal packing it is "
        f"distinguished from differs by {control:.3e}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
