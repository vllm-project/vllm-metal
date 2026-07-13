# SPDX-License-Identifier: Apache-2.0
"""Small draft ingests must ride the decode path, not the tiled prefill.

Regression tests for #482 Problem 2: the proposer's steady-state ingest
(K+1 committed tokens per accepted round) was submitted as a prefill
segment, routing the forward to the tiled prefill kernel, which reads the
whole context regardless of query size (~70 ms vs ~11 ms per pass at 8k).
"""

import mlx.core as mx

from vllm_metal.attention.context import get_context
from vllm_metal.v1.draft_model_proposer import (
    _DECODE_INGEST_MAX_TOKENS,
    DraftModelProposer,
    _DraftPlan,
)

BLOCK_SIZE = 16
VOCAB = 32


class _ContextCapture:
    """Stub draft model that records the paged-attention context it saw."""

    def __init__(self) -> None:
        self.ctx = None

    def __call__(self, input_ids: mx.array, cache=None) -> mx.array:
        self.ctx = get_context()
        return mx.zeros((1, input_ids.shape[1], VOCAB), dtype=mx.float32)


def _proposer(model) -> DraftModelProposer:
    return DraftModelProposer(
        model=model,
        block_size=BLOCK_SIZE,
        num_blocks=64,
        num_layers=1,
        controller=None,
        extract_logits=lambda logits: logits,
    )


def _plan(n_ingest: int, draft_seq_len: int) -> _DraftPlan:
    committed = draft_seq_len + n_ingest
    n_blocks = (committed + BLOCK_SIZE - 1) // BLOCK_SIZE + 1
    return _DraftPlan(
        req_id="r1",
        block_ids=list(range(n_blocks)),
        committed_len=committed,
        draft_seq_len=draft_seq_len,
        ingest_tokens=list(range(100, 100 + n_ingest)),
    )


def test_steady_ingest_prepares_decode_rows() -> None:
    model = _ContextCapture()
    plan = _plan(n_ingest=4, draft_seq_len=8192)

    _proposer(model)._ingest_and_draft_first([plan], [])

    ctx = model.ctx
    assert ctx.num_decode_requests == 1
    # One single-query segment per ingested token, like verify rows.
    assert ctx.cu_seqlens == [0, 1, 2, 3, 4]
    # Rows sit at the committed suffix positions with full-context lens.
    assert ctx.offsets == [8192, 8193, 8194, 8195]
    assert ctx.context_lens == [8193, 8194, 8195, 8196]


def test_large_ingest_keeps_prefill_path() -> None:
    model = _ContextCapture()
    n = _DECODE_INGEST_MAX_TOKENS + 1
    plan = _plan(n_ingest=n, draft_seq_len=0)

    _proposer(model)._ingest_and_draft_first([plan], [])

    ctx = model.ctx
    assert ctx.num_decode_requests == 0
    # A single prefill segment covering all tokens.
    assert ctx.cu_seqlens == [0, n]
    assert ctx.offsets == [0]


def test_threshold_boundary_takes_decode_path() -> None:
    model = _ContextCapture()
    plan = _plan(n_ingest=_DECODE_INGEST_MAX_TOKENS, draft_seq_len=64)

    _proposer(model)._ingest_and_draft_first([plan], [])

    assert model.ctx.num_decode_requests == 1
    assert len(model.ctx.cu_seqlens) == _DECODE_INGEST_MAX_TOKENS + 1


def test_mixed_plans_route_by_largest_ingest() -> None:
    """One oversized plan sends the whole batch down the prefill path —
    per-forward routing must stay uniform because the context is global."""
    model = _ContextCapture()
    small = _plan(n_ingest=2, draft_seq_len=32)
    big = _plan(n_ingest=_DECODE_INGEST_MAX_TOKENS + 8, draft_seq_len=0)

    _proposer(model)._ingest_and_draft_first([small, big], [])

    ctx = model.ctx
    assert ctx.num_decode_requests == 0
    assert len(ctx.cu_seqlens) == 3  # two prefill segments
