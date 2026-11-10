# SPDX-License-Identifier: Apache-2.0
"""Draft lookahead against vLLM's real block allocator and prefix cache."""

from dataclasses import replace

import pytest
import torch
from vllm import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.request import Request

from tests.stub_runner import adopt_draft_group, make_stub_runner
from tests.test_draft_model_proposer import (
    BLOCK_SIZE,
    _context,
    _PositionEncodingDraftModel,
    _prefills_context,
    _proposer,
    _request_state,
)
from vllm_metal.attention.context import get_context


def _manager(*, num_blocks=16, max_model_len=128, caching=False):
    init_none_hash(sha256)
    # Ordinary target and draft full-attention layers share one scheduler
    # group. Their physical caches are separate, indexed by these same IDs.
    config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=["layers.0.self_attn", "draft_layers.0.self_attn"],
                kv_cache_spec=FullAttentionSpec(
                    block_size=BLOCK_SIZE,
                    num_kv_heads=1,
                    head_size=64,
                    dtype=torch.bfloat16,
                ),
            )
        ],
        kv_cache_layout="LBNHC",
    )
    return KVCacheManager(
        config,
        max_model_len=max_model_len,
        scheduler_block_size=BLOCK_SIZE,
        hash_block_size=BLOCK_SIZE,
        enable_caching=caching,
    )


def _request(req_id, prompt):
    return Request(
        req_id,
        list(prompt),
        SamplingParams(temperature=0, max_tokens=32),
        None,
        block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
    )


class _WriteTrackingModel(_PositionEncodingDraftModel):
    def __init__(self):
        super().__init__()
        self.writes = []

    def __call__(self, input_ids, *, cache):
        ctx = get_context()
        assert ctx is not None and ctx.cu_seqlens is not None
        for i, (start, end) in enumerate(
            zip(ctx.cu_seqlens[:-1], ctx.cu_seqlens[1:], strict=True)
        ):
            for pos in range(ctx.offsets[i], ctx.offsets[i] + end - start):
                self.writes.append((ctx.block_tables[i][pos // BLOCK_SIZE], pos))
        return super().__call__(input_ids, cache=cache)


def _prefill(manager, request, *, k, start=0):
    ctx = _prefills_context(
        [(request.request_id, request.prompt_token_ids[start:])],
        start_pos=start,
        num_speculative_tokens=k,
    )
    block_ids = list(manager.get_blocks(request.request_id).get_block_ids())
    prefill = ctx.prefill_reqs[0]._replace(block_ids=block_ids)
    state = ctx.request_states[request.request_id]
    state.block_ids = block_ids
    state.token_ids = [*request.prompt_token_ids, 42]
    return replace(ctx, prefill_reqs=[prefill])


@pytest.mark.parametrize("prompt_len", [15, 16, 17, 29, 30, 31, 32])
@pytest.mark.parametrize("k", [0, 1, 3, 17])
@pytest.mark.parametrize("caching", [False, True])
def test_scheduler_reservation_covers_exact_draft_write_span(prompt_len, k, caching):
    manager = _manager(caching=caching)
    request = _request("r", range(prompt_len))
    assert manager.allocate_slots(request, prompt_len, num_lookahead_tokens=k)
    free_before = manager.block_pool.get_num_free_blocks()
    blocks = manager.get_blocks("r").get_block_ids()[0]
    model = _WriteTrackingModel()
    proposer = _proposer(model, max_model_len=128)

    drafts = proposer.propose(_prefill(manager, request, k=k))

    assert (drafts is not None) == (k > 0)
    # A final prefill ingests the sampled token only when drafting; producing
    # K tokens then writes K-1 more positions. No K+1 allocation is needed.
    assert [pos for _, pos in model.writes] == list(range(prompt_len + k))
    assert all(block == blocks[pos // BLOCK_SIZE] for block, pos in model.writes)
    assert manager.block_pool.get_num_free_blocks() == free_before
    assert len(blocks) == (prompt_len + k + BLOCK_SIZE - 1) // BLOCK_SIZE


def test_lookahead_pages_are_not_prefix_hits():
    manager = _manager(caching=True)
    request = _request("first", range(16))
    assert manager.allocate_slots(request, 16, num_lookahead_tokens=17)
    model = _WriteTrackingModel()
    proposer = _proposer(model)
    assert proposer.propose(_prefill(manager, request, k=17)) is not None
    blocks = manager.get_blocks("first").blocks[0]
    assert blocks[0].block_hash is not None
    assert all(block.block_hash is None for block in blocks[1:])
    manager.free(request)
    proposer.release_requests({"first"})

    # The entire second block was written speculatively. Even when a new
    # prompt contains those same tokens, only the committed first block hits.
    prompt = [*range(16), 42, *range(16, 31)]
    resumed = _request("second", prompt)
    cached, num_computed, _ = manager.get_computed_blocks(resumed)
    assert num_computed == 16
    assert manager.allocate_slots(
        resumed,
        len(prompt) - num_computed,
        num_new_computed_tokens=num_computed,
        new_computed_blocks=cached,
        num_lookahead_tokens=3,
    )
    model.writes.clear()
    assert proposer.propose(_prefill(manager, resumed, k=3, start=16)) is not None
    assert [pos for _, pos in model.writes] == list(range(16, len(prompt) + 3))


@pytest.mark.parametrize("event", ["finished", "preempted"])
def test_exhaustion_and_lifecycle_return_lookahead_to_scheduler(event):
    manager = _manager(num_blocks=5)
    free_before = manager.block_pool.get_num_free_blocks()
    first = _request("first", range(31))
    waiting = _request("waiting", range(31))
    assert manager.allocate_slots(first, 31, num_lookahead_tokens=3)
    first_blocks = manager.get_blocks("first").get_block_ids()[0]
    assert len(first_blocks) == 3
    assert manager.allocate_slots(waiting, 31, num_lookahead_tokens=3) is None

    model = _WriteTrackingModel()
    proposer = _proposer(model)
    assert proposer.propose(_prefill(manager, first, k=3)) is not None
    assert proposer._spec_kv_writes["first"]
    runner = make_stub_runner(_drafter=proposer)
    runner._reconcile_request_lifecycle(
        {"first"} if event == "finished" else set(),
        preempted_req_ids={"first"} if event == "preempted" else set(),
    )
    assert "first" not in proposer._draft_seq_lens
    assert "first" not in proposer._spec_kv_writes
    # Proposer cleanup must not free scheduler allocations itself.
    assert manager.get_blocks("first").get_block_ids()[0] == first_blocks
    manager.free(first)
    assert manager.block_pool.get_num_free_blocks() == free_before
    assert manager.allocate_slots(waiting, 31, num_lookahead_tokens=3)
    assert proposer.propose(_prefill(manager, waiting, k=3)) is not None
    manager.free(waiting)
    assert manager.block_pool.get_num_free_blocks() == free_before


def test_reused_request_id_reingests_after_runner_cleanup():
    model = _WriteTrackingModel()
    proposer = _proposer(model)
    old = _request_state(scheduler_block_ids=[1, 2, 3], token_ids=list(range(31)))
    assert proposer.propose(_context("r", old, {"r": old}, num_speculative_tokens=3))
    runner = make_stub_runner(_drafter=proposer, _request_states={"r": old})
    runner._reconcile_request_lifecycle({"r"})

    new = _request_state(scheduler_block_ids=[1], token_ids=[7, 8])
    model.writes.clear()
    assert proposer.propose(_context("r", new, {"r": new}, num_speculative_tokens=3))
    assert [pos for _, pos in model.writes] == [0, 1, 2, 3]


@pytest.mark.parametrize("prompt_len", [29, 30, 31, 32])
def test_scheduler_context_limit_caps_draft_writes(prompt_len):
    manager = _manager(max_model_len=32)
    request = _request("r", range(prompt_len))
    assert manager.allocate_slots(request, prompt_len, num_lookahead_tokens=3)
    model = _WriteTrackingModel()
    proposer = _proposer(model, max_model_len=4096)
    adopt_draft_group(proposer, 0, 32)

    drafts = proposer.propose(_prefill(manager, request, k=3))

    assert (drafts is not None) == (prompt_len == 29)
    assert max(pos for _, pos in model.writes) < 32
    assert len(manager.get_blocks("r").get_block_ids()[0]) == 2


def test_chunked_prefill_uses_reserved_blocks_without_drafting_future_prompt():
    manager = _manager()
    request = _request("r", range(32))
    model = _WriteTrackingModel()
    proposer = _proposer(model)
    for start in (0, 16):
        assert manager.allocate_slots(request, 16, num_lookahead_tokens=3)
        ctx = _prefill(manager, request, k=3, start=start)
        if start == 0:
            ctx = replace(
                ctx,
                prefill_reqs=[ctx.prefill_reqs[0]._replace(token_ids=list(range(16)))],
                prefill_result_modes=["intermediate"],
                prefill_token_ids=[],
            )
        model.writes.clear()
        drafts = proposer.propose(ctx)
        assert (drafts is not None) == (start == 16)
        assert [pos for _, pos in model.writes] == list(
            range(start, 16 if start == 0 else 35)
        )
        request.num_computed_tokens += 16
