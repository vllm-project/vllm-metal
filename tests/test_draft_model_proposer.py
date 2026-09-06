# SPDX-License-Identifier: Apache-2.0
"""Tests for the draft-model proposer's committed/scratch block split.

A stub draft model returns logits of the right shape, so ingest, drafting,
and the release path run through ``propose`` without loading any weights.
The committed portion of the block table is scheduler-assigned (as it would
be by a real KVCacheManager for the draft's registered KV-cache group, see
``cache_policy.ModelCachePolicy._draft_layer_specs``); these tests simulate
that assignment directly on ``RequestState.block_ids`` rather than driving a
real scheduler.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import mlx.core as mx
import pytest
from vllm.sampling_params import SamplingParams

from vllm_metal.attention.context import OffsetCache, get_context
from vllm_metal.v1 import draft_model_proposer
from vllm_metal.v1.draft_model_proposer import DraftModelProposer
from vllm_metal.v1.model_runner import PrefillRequest, RequestState
from vllm_metal.v1.proposer import ProposeContext
from vllm_metal.v1.spec_decode import SpeculativeDecodeController

BLOCK_SIZE = 16
COMMITTED_GROUP_INDEX = 0
COMMITTED_NUM_BLOCKS = 2
SCRATCH_RESERVE_BLOCKS = 2
VOCAB_SIZE = 64
PROMPT_LEN = 20


@pytest.mark.parametrize("revision", [None, "release-tag", "a" * 40])
def test_draft_load_preserves_revision(monkeypatch, revision):
    dims, model = object(), object()
    config = SimpleNamespace(
        draft_model_config=SimpleNamespace(model="org/draft", revision=revision)
    )
    monkeypatch.setattr(draft_model_proposer, "resolve_draft_dims", lambda *_: dims)
    resolve_path = Mock(return_value="org/draft")
    load = Mock(return_value=(model, None))
    monkeypatch.setattr(draft_model_proposer, "get_model_download_path", resolve_path)
    monkeypatch.setattr(draft_model_proposer, "mlx_lm_load", load)

    assert draft_model_proposer._load_draft_model(config, None) == (model, dims)
    resolve_path.assert_called_once_with("org/draft", revision=revision)
    load.assert_called_once_with("org/draft", revision=revision)


class _StubDraftModel:
    """mlx_lm-shaped draft model: logits per input token, recorded block tables."""

    def __init__(self) -> None:
        self.block_tables: list[list[list[int]]] = []
        self.input_lens: list[int] = []

    def __call__(self, input_ids: mx.array, *, cache: list[OffsetCache]) -> mx.array:
        ctx = get_context()
        assert ctx is not None
        self.block_tables.append([list(block_ids) for block_ids in ctx.block_tables])
        self.input_lens.append(int(input_ids.shape[1]))
        return mx.zeros((1, int(input_ids.shape[1]), VOCAB_SIZE), dtype=mx.float32)


class _PositionEncodingDraftModel(_StubDraftModel):
    """Logits keyed on each row's true KV position, read from the paged
    context (per-segment RoPE offsets + cu_seqlens) -- the same position
    signal the real attention gets -- so chunk boundaries and cross-plan
    packing order cannot leak into the logits.  Argmax at the row whose
    true position is p is p % VOCAB_SIZE.
    """

    def __call__(self, input_ids: mx.array, *, cache: list[OffsetCache]) -> mx.array:
        n = int(input_ids.shape[1])
        ctx = get_context()
        assert ctx is not None
        assert ctx.cu_seqlens is not None
        self.block_tables.append([list(block_ids) for block_ids in ctx.block_tables])
        self.input_lens.append(n)
        positions = [
            position
            for segment_index, (start, end) in enumerate(
                zip(ctx.cu_seqlens[:-1], ctx.cu_seqlens[1:], strict=True)
            )
            for position in range(
                ctx.offsets[segment_index],
                ctx.offsets[segment_index] + end - start,
            )
        ]
        assert len(positions) == n
        return mx.eye(VOCAB_SIZE)[mx.array(positions) % VOCAB_SIZE][None]


def _proposer(
    model: _StubDraftModel,
    *,
    committed_num_blocks: int = COMMITTED_NUM_BLOCKS,
    scratch_reserve_blocks: int = SCRATCH_RESERVE_BLOCKS,
) -> DraftModelProposer:
    proposer = DraftModelProposer(
        model=model,
        block_size=BLOCK_SIZE,
        committed_num_blocks=committed_num_blocks,
        scratch_reserve_blocks=scratch_reserve_blocks,
        num_layers=1,
        controller=SpeculativeDecodeController(),
        extract_logits=lambda output: output,
    )
    proposer.adopt_committed_group(COMMITTED_GROUP_INDEX)
    return proposer


def _request_state(
    *,
    committed_block_ids: list[int],
    num_computed_tokens: int = 0,
    sampling_params: SamplingParams | None = None,
    token_ids: list[int] | None = None,
) -> RequestState:
    if token_ids is None:
        token_ids = list(range(PROMPT_LEN))
    return RequestState(
        token_ids=list(token_ids),
        prompt_len=len(token_ids),
        sampling_params=sampling_params or SamplingParams(temperature=0.0),
        block_ids=[list(committed_block_ids)],
        num_computed_tokens=num_computed_tokens,
    )


def _context(
    req_id: str,
    state: RequestState,
    request_states: dict[str, RequestState],
    *,
    num_speculative_tokens: int = 1,
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
        num_speculative_tokens=num_speculative_tokens,
        finished_req_ids=set(),
    )


def _prefills_context(
    prefills: list[tuple[str, list[int]]],
    *,
    num_speculative_tokens: int = 1,
) -> ProposeContext:
    """Context whose requests are final-chunk prefills (greedy, drafting)."""
    prefill_reqs = [
        PrefillRequest(
            req_id=req_id,
            token_ids=list(token_ids),
            sampling_params=SamplingParams(temperature=0.0),
            block_ids=[[0]],
            generator=None,
            prompt_len=None,
            start_pos=0,
            full_prompt_token_ids=None,
        )
        for req_id, token_ids in prefills
    ]
    return ProposeContext(
        target_hidden_states=None,
        decode_reqs=[],
        decode_segments=[],
        decode_token_ids=[],
        prefill_reqs=prefill_reqs,
        prefill_token_ids=[42] * len(prefill_reqs),
        prefill_result_modes=["final"] * len(prefill_reqs),
        request_states={
            req_id: _request_state(committed_block_ids=[0]) for req_id, _ in prefills
        },
        cu_seqlens=[],
        num_decode_segments=0,
        num_speculative_tokens=num_speculative_tokens,
        finished_req_ids=set(),
    )


def _drafting_blocks(model: _StubDraftModel, forward_index: int) -> set[int]:
    (block_ids,) = model.block_tables[forward_index]
    return set(block_ids)


def test_propose_before_adopt_committed_group_raises() -> None:
    model = _StubDraftModel()
    proposer = DraftModelProposer(
        model=model,
        block_size=BLOCK_SIZE,
        committed_num_blocks=COMMITTED_NUM_BLOCKS,
        scratch_reserve_blocks=SCRATCH_RESERVE_BLOCKS,
        num_layers=1,
        controller=SpeculativeDecodeController(),
        extract_logits=lambda output: output,
    )
    state = _request_state(committed_block_ids=[0, 1])
    with pytest.raises(RuntimeError, match="adopt_committed_group"):
        proposer.propose(_context("r1", state, {"r1": state}))


def test_committed_blocks_come_from_scheduler_assignment() -> None:
    """The committed portion of the block table is exactly what the
    scheduler assigned on RequestState.block_ids, not a proposer-owned pool."""
    model = _StubDraftModel()
    proposer = _proposer(model)
    state = _request_state(committed_block_ids=[1, 0])  # order matters
    drafts = proposer.propose(_context("r1", state, {"r1": state}))

    assert drafts is not None
    used = model.block_tables[0][0]
    assert used[:2] == [1, 0]


def test_lookahead_tail_draws_from_scratch_pool_beyond_committed() -> None:
    """A request whose committed blocks exactly cover its prompt still needs
    a scratch block for the speculative lookahead position(s)."""
    model = _StubDraftModel()
    proposer = _proposer(model, committed_num_blocks=1)
    state = _request_state(committed_block_ids=[0])
    drafts = proposer.propose(
        _context("r1", state, {"r1": state}, num_speculative_tokens=4)
    )

    assert drafts is not None
    used = model.block_tables[0][0]
    assert used[0] == 0  # committed block, scheduler-assigned
    assert used[1] >= 1  # scratch block, drawn from the offset range


def test_cache_hit_seeds_draft_seq_len_from_scheduler_boundary() -> None:
    """A resubmitted/shared prefix reported via num_computed_tokens must not
    be re-ingested -- this is #482's core fix."""
    model = _StubDraftModel()
    proposer = _proposer(model)
    state = _request_state(
        committed_block_ids=[0, 1], num_computed_tokens=PROMPT_LEN - 1
    )
    drafts = proposer.propose(_context("r1", state, {"r1": state}))

    assert drafts is not None
    assert model.input_lens[0] == 1


def test_no_cache_hit_ingests_the_whole_committed_range() -> None:
    model = _StubDraftModel()
    proposer = _proposer(model)
    state = _request_state(committed_block_ids=[0, 1], num_computed_tokens=0)
    drafts = proposer.propose(_context("r1", state, {"r1": state}))

    assert drafts is not None
    assert model.input_lens[0] == PROMPT_LEN


def test_non_greedy_request_ingests_but_never_drafts() -> None:
    """Non-greedy requests must still keep the committed group's KV in sync
    (the scheduler advances num_computed_tokens for them regardless), but
    must never receive draft tokens."""
    model = _StubDraftModel()
    proposer = _proposer(model)
    state = _request_state(
        committed_block_ids=[0, 1], sampling_params=SamplingParams(temperature=1.0)
    )
    drafts = proposer.propose(_context("r1", state, {"r1": state}))

    assert drafts is None  # nothing to draft for
    assert len(model.block_tables) == 1  # but ingest still ran


def test_intermediate_prefill_chunk_ingests_without_drafting() -> None:
    """An intermediate (not-yet-final) prefill chunk must ingest its
    scheduled slice so the committed group's KV stays in sync, but must not
    produce draft tokens (mirrors non-greedy: keep pace, never draft)."""
    model = _StubDraftModel()
    proposer = _proposer(model)
    prefill = PrefillRequest(
        req_id="r1",
        token_ids=list(range(16)),
        sampling_params=SamplingParams(temperature=0.0),
        block_ids=[[0]],
        generator=None,
        prompt_len=None,
        start_pos=0,
        full_prompt_token_ids=None,
    )
    state = _request_state(committed_block_ids=[0])
    ctx = ProposeContext(
        target_hidden_states=None,
        decode_reqs=[],
        decode_segments=[],
        decode_token_ids=[],
        prefill_reqs=[prefill],
        prefill_token_ids=[],
        prefill_result_modes=["intermediate"],
        request_states={"r1": state},
        cu_seqlens=[],
        num_decode_segments=0,
        num_speculative_tokens=1,
        finished_req_ids=set(),
    )
    drafts = proposer.propose(ctx)

    assert drafts is None
    assert len(model.block_tables) == 1


def test_release_requests_returns_scratch_blocks_to_the_free_pool() -> None:
    model = _StubDraftModel()
    proposer = _proposer(model, committed_num_blocks=1)
    waiting = _request_state(committed_block_ids=[0])
    resumed = _request_state(committed_block_ids=[0])
    request_states = {"waiting": waiting, "resumed": resumed}

    assert (
        proposer.propose(
            _context("waiting", waiting, request_states, num_speculative_tokens=4)
        )
        is not None
    )
    pool = _drafting_blocks(model, 0)

    proposer.release_requests({"waiting"})

    drafts = proposer.propose(
        _context("resumed", resumed, request_states, num_speculative_tokens=4)
    )

    assert drafts is not None
    assert list(drafts.req_ids) == ["resumed"]
    assert _drafting_blocks(model, 1) == pool


def test_scratch_pool_exhaustion_raises() -> None:
    model = _StubDraftModel()
    proposer = _proposer(model, committed_num_blocks=1, scratch_reserve_blocks=1)
    waiting = _request_state(committed_block_ids=[0])
    resumed = _request_state(committed_block_ids=[0])
    request_states = {"waiting": waiting, "resumed": resumed}

    assert (
        proposer.propose(
            _context("waiting", waiting, request_states, num_speculative_tokens=4)
        )
        is not None
    )

    calls_before = len(model.block_tables)
    with pytest.raises(RuntimeError, match="scratch pool exhausted"):
        proposer.propose(
            _context("resumed", resumed, request_states, num_speculative_tokens=4)
        )
    assert len(model.block_tables) == calls_before


# -- Speculative-KV reuse (#482 direction 2) --------------------------------


def _draft_two_rounds(
    model: _StubDraftModel,
    proposer: DraftModelProposer,
    *,
    second_round_tokens: list[int],
    second_round_blocks: list[int],
    num_speculative_tokens: int = 3,
    first_round_len: int = PROMPT_LEN,
    first_round_blocks: list[int] | None = None,
) -> None:
    """Round 1 drafts from a fresh prompt; round 2 re-proposes after the
    verifier committed ``second_round_tokens`` (drafts + bonus).

    The stub model argmaxes all-zeros logits to token 0, so every drafted
    token is 0: "accepted" round-2 tokens are 0s, "rejected" ones are not.

    Returns the index into ``model.input_lens`` of round 2's ingest forward
    (round 1 contributes one ingest plus K-1 single-token draft steps).
    """
    if first_round_blocks is None:
        first_round_blocks = [0, 1]
    state1 = _request_state(
        committed_block_ids=first_round_blocks,
        token_ids=list(range(first_round_len)),
    )
    assert (
        proposer.propose(
            _context(
                "r1",
                state1,
                {"r1": state1},
                num_speculative_tokens=num_speculative_tokens,
            )
        )
        is not None
    )
    round2_ingest_index = len(model.input_lens)
    state2 = _request_state(
        committed_block_ids=second_round_blocks,
        token_ids=list(range(first_round_len)) + second_round_tokens,
    )
    assert (
        proposer.propose(
            _context(
                "r1",
                state2,
                {"r1": state2},
                num_speculative_tokens=num_speculative_tokens,
            )
        )
        is not None
    )
    return round2_ingest_index


def test_full_acceptance_skips_reingest_of_drafted_kv() -> None:
    """On full acceptance the K-1 drafts whose KV the previous round's
    lookahead steps already wrote are not re-ingested: the steady-state
    K+1-token ingest shrinks to 2 rows (#482 direction 2)."""
    model = _StubDraftModel()
    proposer = _proposer(model)

    ingest = _draft_two_rounds(
        model,
        proposer,
        second_round_tokens=[0, 0, 0, 5],  # drafts 0,0,0 accepted + bonus
        second_round_blocks=[0, 1],
    )

    assert model.input_lens[0] == PROMPT_LEN  # round 1: full prompt
    assert model.input_lens[ingest] == 2  # round 2: only d_K's slot + bonus


def test_rejected_draft_reingests_the_full_committed_range() -> None:
    """A rejected first draft means the committed token at that position
    differs from what the speculative KV was written with -- nothing may be
    skipped."""
    model = _StubDraftModel()
    proposer = _proposer(model)

    ingest = _draft_two_rounds(
        model,
        proposer,
        second_round_tokens=[7, 0, 0, 5],  # first draft rejected
        second_round_blocks=[0, 1],
    )

    assert model.input_lens[ingest] == 4  # all K+1 newly committed tokens


def test_scratch_block_spec_kv_is_not_reused() -> None:
    """Speculative KV that landed in a scratch block (lookahead crossed a
    block boundary) must not be skipped past: the scheduler allocated a
    different physical block for those positions, so the KV is not where
    the new block table looks."""
    model = _StubDraftModel()
    proposer = _proposer(model)

    # committed_len 31 with 16-token blocks: lookahead positions 31, 32
    # land in committed block 1 and scratch block 2 respectively.
    ingest = _draft_two_rounds(
        model,
        proposer,
        first_round_len=31,
        first_round_blocks=[0, 1],
        second_round_tokens=[0, 0, 0, 5],
        second_round_blocks=[0, 1, 5],  # scheduler's own block 5, not scratch 2
    )

    # Position 31 (committed block, matching token) is skipped; position 32
    # (scratch write) is not: 35 - 32 = 3 tokens re-ingested.
    assert model.input_lens[ingest] == 3


def test_reallocated_block_spec_kv_is_not_reused() -> None:
    """A scheduler re-allocation that swaps the committed block table's
    physical blocks (tokens still matching) must stop the walk: the
    speculative KV is not where the new table looks."""
    model = _StubDraftModel()
    proposer = _proposer(model)

    ingest = _draft_two_rounds(
        model,
        proposer,
        second_round_tokens=[0, 0, 0, 5],  # full acceptance: tokens match
        second_round_blocks=[4, 7],  # but the scheduler remapped the blocks
    )

    # No position may be skipped: all 4 newly committed tokens re-ingest.
    assert model.input_lens[ingest] == 4


def test_partial_acceptance_never_produces_an_empty_ingest() -> None:
    """The skip is capped one short of committed_len: the last committed
    token's row must always run because its logits predict this round's
    first draft token."""
    model = _StubDraftModel()
    proposer = _proposer(model)

    ingest = _draft_two_rounds(
        model,
        proposer,
        second_round_tokens=[0, 9],  # one draft accepted + bonus
        second_round_blocks=[0, 1],
    )

    assert model.input_lens[ingest] == 1


def test_spec_kv_ledger_cleared_on_release() -> None:
    """After release_requests, a request reusing the id cannot skip off a
    stale ledger -- its KV was never written."""
    model = _StubDraftModel()
    proposer = _proposer(model)

    state1 = _request_state(committed_block_ids=[0, 1])
    assert (
        proposer.propose(
            _context("r1", state1, {"r1": state1}, num_speculative_tokens=3)
        )
        is not None
    )
    proposer.release_requests({"r1"})

    state2 = _request_state(
        committed_block_ids=[0, 1],
        num_computed_tokens=PROMPT_LEN,
        token_ids=list(range(PROMPT_LEN)) + [0, 0, 0, 5],
    )
    ingest = len(model.input_lens)
    assert (
        proposer.propose(
            _context("r1", state2, {"r1": state2}, num_speculative_tokens=3)
        )
        is not None
    )

    assert model.input_lens[ingest] == 4  # no skip: ledger gone


# -- Chunked cold ingest (#482 direction 3) ----------------------------------


def test_chunked_ingest_forwards_per_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A >16-token cold ingest runs one forward per chunk: a 20-token prompt
    plus the sampled token runs 16 then 5, and the predicted token comes from
    the sampled token's row (position 20), not a chunk boundary."""
    monkeypatch.setenv("VLLM_METAL_SPEC_INGEST_CHUNK", "16")
    model = _PositionEncodingDraftModel()
    proposer = _proposer(model)

    drafts = proposer.propose(_prefills_context([("r1", list(range(20)))]))

    assert model.input_lens == [16, 5]
    assert drafts is not None
    assert drafts.draft_token_ids == [[20 % VOCAB_SIZE]]


def test_ingest_chunk_exact_multiple_two_equal_rounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 31-token prompt plus its sampled token splits 16 / 16; the final
    row of the second round (position 31) predicts."""
    monkeypatch.setenv("VLLM_METAL_SPEC_INGEST_CHUNK", "16")
    model = _PositionEncodingDraftModel()
    proposer = _proposer(model)

    drafts = proposer.propose(_prefills_context([("r1", list(range(31)))]))

    assert model.input_lens == [16, 16]
    assert drafts is not None
    assert drafts.draft_token_ids == [[31 % VOCAB_SIZE]]


def test_ingest_chunk_larger_than_prompt_single_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A chunk bigger than the ingest keeps the single-forward behavior."""
    monkeypatch.setenv("VLLM_METAL_SPEC_INGEST_CHUNK", "1024")
    model = _PositionEncodingDraftModel()
    proposer = _proposer(model)

    drafts = proposer.propose(_prefills_context([("r1", list(range(20)))]))

    assert model.input_lens == [21]
    assert drafts is not None
    assert drafts.draft_token_ids == [[20 % VOCAB_SIZE]]


def test_ingest_chunk_size_zero_single_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Chunk "0" restores the pre-chunk single-forward behavior."""
    monkeypatch.setenv("VLLM_METAL_SPEC_INGEST_CHUNK", "0")
    model = _PositionEncodingDraftModel()
    proposer = _proposer(model)

    drafts = proposer.propose(_prefills_context([("r1", list(range(20)))]))

    assert model.input_lens == [21]
    assert drafts is not None
    assert drafts.draft_token_ids == [[20 % VOCAB_SIZE]]


def test_small_ingest_ignores_chunk_knob(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ingests at or below the decode threshold keep the single decode-path
    forward whatever the chunk size -- the knob only governs cold ingests."""
    monkeypatch.setenv("VLLM_METAL_SPEC_INGEST_CHUNK", "1")
    model = _PositionEncodingDraftModel()
    proposer = _proposer(model)

    drafts = proposer.propose(_prefills_context([("r1", list(range(15)))]))

    assert model.input_lens == [16]
    assert drafts is not None
    assert drafts.draft_token_ids == [[15 % VOCAB_SIZE]]


def test_mixed_chunk_lengths_share_rounds(monkeypatch: pytest.MonkeyPatch) -> None:
    """21- and 26-token ingests with a 16-token chunk: round 0 carries both
    plans' first 16 tokens, round 1 the two tails; each plan predicts from
    its own sampled-token row (positions 20 and 25)."""
    monkeypatch.setenv("VLLM_METAL_SPEC_INGEST_CHUNK", "16")
    model = _PositionEncodingDraftModel()
    proposer = _proposer(model)

    drafts = proposer.propose(
        _prefills_context([("r1", list(range(20))), ("r2", list(range(25)))])
    )

    assert model.input_lens == [32, 15]
    assert drafts is not None
    assert drafts.req_ids == ["r1", "r2"]
    assert drafts.draft_token_ids == [[20 % VOCAB_SIZE], [25 % VOCAB_SIZE]]


# -- Sliding-window / hybrid draft rejection ---------------------------------


class _FakeModelConfig:
    """Minimal mock of ``vllm.config.ModelConfig`` for config-only checks."""

    def __init__(
        self,
        *,
        sliding_window: int | None = None,
        layer_types: list[str] | None = None,
    ) -> None:
        self._sliding_window = sliding_window
        # hf_text_config is where layer_types and sliding_window live
        self.hf_text_config = type("_HFConfig", (), {})()
        if layer_types is not None:
            self.hf_text_config.layer_types = layer_types

    def get_sliding_window(self) -> int | None:
        return self._sliding_window

    def get_total_num_hidden_layers(self) -> int:
        return 4

    def get_num_kv_heads(self, parallel_config: object) -> int:
        return 2

    def get_head_size(self) -> int:
        return 64


def test_sliding_window_draft_rejected() -> None:
    """A draft model with sliding_window must be rejected at config time."""
    from vllm_metal.v1.draft_model_proposer import _require_full_attention_draft

    cfg = _FakeModelConfig(sliding_window=4096)
    with pytest.raises(NotImplementedError, match="sliding_window=4096"):
        _require_full_attention_draft(cfg)


def test_hybrid_layer_types_draft_rejected() -> None:
    """A draft model with mixed layer types (e.g. Gemma4) must be rejected."""
    from vllm_metal.v1.draft_model_proposer import _require_full_attention_draft

    cfg = _FakeModelConfig(
        layer_types=["sliding_attention", "full_attention"] * 2,
    )
    with pytest.raises(NotImplementedError, match="non-full-attention layer types"):
        _require_full_attention_draft(cfg)


def test_full_attention_draft_accepted() -> None:
    """A plain full-attention draft model must pass the check."""
    from vllm_metal.v1.draft_model_proposer import _require_full_attention_draft

    # No sliding window, no layer_types at all (standard transformer)
    cfg_plain = _FakeModelConfig()
    _require_full_attention_draft(cfg_plain)  # should not raise

    # Explicit full_attention layer_types only
    cfg_explicit = _FakeModelConfig(
        layer_types=["full_attention"] * 4,
    )
    _require_full_attention_draft(cfg_explicit)  # should not raise
