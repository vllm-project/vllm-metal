# SPDX-License-Identifier: Apache-2.0
"""Tests for the draft-model proposer's draft KV block pool.

A stub draft model returns logits of the right shape, so the block allocator and
the release path run through ``propose`` without loading any weights.
"""

from __future__ import annotations

import mlx.core as mx
import pytest
from vllm.sampling_params import SamplingParams

from vllm_metal.attention.context import OffsetCache, get_context
from vllm_metal.v1.draft_model_proposer import DraftModelProposer
from vllm_metal.v1.model_runner import RequestState
from vllm_metal.v1.proposer import ProposeContext
from vllm_metal.v1.spec_decode import SpeculativeDecodeController

BLOCK_SIZE = 16
NUM_BLOCKS = 2
VOCAB_SIZE = 8
# Long enough that one request plus its drafted position spans the whole pool.
PROMPT_LEN = 20


class _StubDraftModel:
    """mlx_lm-shaped draft model: logits per input token, recorded block tables."""

    def __init__(self) -> None:
        self.block_tables: list[list[list[int]]] = []

    def __call__(self, input_ids: mx.array, *, cache: list[OffsetCache]) -> mx.array:
        ctx = get_context()
        assert ctx is not None
        self.block_tables.append([list(block_ids) for block_ids in ctx.block_tables])
        return mx.zeros((1, int(input_ids.shape[1]), VOCAB_SIZE), dtype=mx.float32)


def _proposer(model: _StubDraftModel) -> DraftModelProposer:
    return DraftModelProposer(
        model=model,
        block_size=BLOCK_SIZE,
        num_blocks=NUM_BLOCKS,
        num_layers=1,
        controller=SpeculativeDecodeController(),
        extract_logits=lambda output: output,
    )


def _request_state() -> RequestState:
    return RequestState(
        token_ids=list(range(PROMPT_LEN)),
        prompt_len=PROMPT_LEN,
        cache=[],
        sampling_params=SamplingParams(temperature=0.0),
    )


def _context(
    req_id: str,
    state: RequestState,
    request_states: dict[str, RequestState],
) -> ProposeContext:
    return ProposeContext(
        target_hidden_states=None,
        decode_reqs=[(req_id, state)],
        decode_segments=[],
        decode_token_ids=[[state.token_ids[-1]]],
        prefill_reqs=[],
        prefill_token_ids=[],
        prefill_result_modes=[],
        request_states=request_states,
        cu_seqlens=[],
        num_decode_segments=1,
        num_speculative_tokens=1,
        logitsprocs=None,
        finished_req_ids=set(),
    )


def _drafting_blocks(model: _StubDraftModel, forward_index: int) -> set[int]:
    (block_ids,) = model.block_tables[forward_index]
    return set(block_ids)


def test_release_requests_returns_draft_blocks_to_the_free_pool() -> None:
    model = _StubDraftModel()
    proposer = _proposer(model)
    waiting, resumed = _request_state(), _request_state()
    # A preempted request keeps its RequestState, so the finished-request sweep
    # inside propose() never reclaims its blocks.
    request_states = {"waiting": waiting, "resumed": resumed}

    assert proposer.propose(_context("waiting", waiting, request_states)) is not None
    pool = _drafting_blocks(model, 0)
    assert len(pool) == NUM_BLOCKS

    proposer.release_requests({"waiting"})

    drafts = proposer.propose(_context("resumed", resumed, request_states))

    assert drafts is not None
    assert list(drafts.req_ids) == ["resumed"]
    assert _drafting_blocks(model, 1) == pool


def test_draft_blocks_stay_pinned_without_release() -> None:
    model = _StubDraftModel()
    proposer = _proposer(model)
    waiting, resumed = _request_state(), _request_state()
    request_states = {"waiting": waiting, "resumed": resumed}

    assert proposer.propose(_context("waiting", waiting, request_states)) is not None

    with pytest.raises(RuntimeError) as excinfo:
        proposer.propose(_context("resumed", resumed, request_states))

    assert "'resumed'" in str(excinfo.value)
    # Allocation fails before any draft forward runs.
    assert len(model.block_tables) == 1
