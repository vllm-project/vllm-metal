# SPDX-License-Identifier: Apache-2.0
"""Context-cap fallback diagnostics must not change drafting or ingest."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock
from weakref import ref

import mlx.core as mx
import pytest
from vllm import SamplingParams

from tests.stub_draft_model import StubDraftModel
from tests.stub_runner import make_stub_runner
from tests.test_draft_model_proposer import (
    _context,
    _prefills_context,
    _proposer,
    _request_state,
)
from vllm_metal.v1 import draft_model_proposer as dmp
from vllm_metal.v1.cache_policy import ModelCachePolicy
from vllm_metal.v1.spec_decode import SpeculativeDecodeController


@pytest.fixture
def info(monkeypatch):
    log = Mock()
    monkeypatch.setattr(dmp.logger, "info", log)
    return log


@pytest.mark.parametrize("prefill", [False, True])
def test_smaller_final_target_limit_logs_once_and_bounds_ingest(prefill, info):
    model = StubDraftModel()
    proposer = _proposer(model, max_model_len=4096, min_speculative_tokens=3)
    runner = SimpleNamespace(
        _drafter=proposer,
        _draft_dims=SimpleNamespace(num_layers=1),
        model_config=SimpleNamespace(max_model_len=32),
    )
    config = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(layer_names=["draft_layers.0.self_attn"])]
    )
    ModelCachePolicy(runner, Mock())._adopt_draft_scheduler_group(config)
    if prefill:
        ctx = _prefills_context([("r", list(range(29)))], num_speculative_tokens=3)
        state = ctx.request_states["r"]
        state.block_ids = ctx.prefill_reqs[0].block_ids
    else:
        state = _request_state(scheduler_block_ids=[0, 1], token_ids=list(range(30)))
        ctx = _context("r", state, {"r": state}, num_speculative_tokens=3)

    # Input length 29 + K=3 fits exactly; the next sampled token crosses
    # the final target limit even though the draft checkpoint permits it.
    drafts = proposer.propose(ctx)
    assert drafts is not None and len(drafts.draft_token_ids[0]) == 3
    info.assert_not_called()
    assert proposer.get_stats()["num_context_limit_fallback_requests"] == 0
    for token in (9, 10, 11):
        state.token_ids.append(token)
        assert (
            proposer.propose(
                _context("r", state, {"r": state}, num_speculative_tokens=3)
            )
            is None
        )
    info.assert_called_once()
    assert proposer.get_stats()["num_context_limit_fallback_requests"] == 1
    message = info.call_args.args[0] % info.call_args.args[1:]
    assert "request 'r'" in message
    assert "input_tokens=30, min_draft_tokens=3, max_model_len=32" in message
    assert "target-only" in message
    assert proposer._draft_seq_lens["r"] == 32
    # Two lookahead forwards, then only the two committed rows within the cap.
    assert model.input_lens == [30, 1, 1, 1, 1]


@pytest.mark.parametrize("deferred", [False, True])
def test_dynamic_width_can_resume_after_temporary_context_skip(deferred, info):
    proposer = _proposer(
        StubDraftModel(),
        max_model_len=32,
        min_speculative_tokens=1,
        allow_deferred_zero_k_ingest=deferred,
    )
    state = _request_state(scheduler_block_ids=[0, 1, 2], token_ids=list(range(31)))
    ctx = _context("r", state, {"r": state}, num_speculative_tokens=3)
    assert proposer.propose(ctx) is None
    info.assert_not_called()
    assert proposer.get_stats()["num_context_limit_fallback_requests"] == 0

    state.token_ids.append(9)
    drafts = proposer.propose(replace(ctx, num_speculative_tokens=1))
    assert drafts is not None and len(drafts.draft_token_ids[0]) == 1
    info.assert_not_called()
    assert proposer.get_stats()["num_context_limit_fallback_requests"] == 0

    state.token_ids.append(10)
    assert proposer.propose(replace(ctx, num_speculative_tokens=0)) is None
    info.assert_not_called()
    assert proposer.get_stats()["num_context_limit_fallback_requests"] == 0
    # Now even K=1 cannot fit. Report it when drafting is requested again.
    assert proposer.propose(replace(ctx, num_speculative_tokens=1)) is None
    assert proposer.propose(replace(ctx, num_speculative_tokens=3)) is None
    info.assert_called_once()
    assert proposer.get_stats()["num_context_limit_fallback_requests"] == 1


def test_initial_prefill_fallback_is_per_request_and_keeps_other_rows_drafting(info):
    model = StubDraftModel()
    proposer = _proposer(model, max_model_len=32, min_speculative_tokens=3)
    proposer.adopt_scheduler_group(0, 4096)
    ctx = _prefills_context(
        [
            ("fits", list(range(8))),
            ("capped", list(range(31))),
            ("beyond", list(range(35))),
        ],
        num_speculative_tokens=3,
    )
    drafts = proposer.propose(ctx)
    assert drafts is not None and drafts.req_ids == ["fits"]
    assert model.input_lens == [9 + 31 + 32, 1, 1]
    assert {call.args[1] for call in info.call_args_list} == {"capped", "beyond"}
    assert info.call_count == 2
    assert proposer.get_stats()["num_context_limit_fallback_requests"] == 2
    assert proposer.propose(ctx) is None
    assert info.call_count == 2
    assert proposer.get_stats()["num_context_limit_fallback_requests"] == 2


@pytest.mark.parametrize("mode", ["non_greedy", "no_sample", "intermediate"])
def test_other_eligibility_gates_do_not_log_context_fallback(mode, info):
    proposer = _proposer(StubDraftModel(), max_model_len=32, min_speculative_tokens=3)
    state = _request_state(scheduler_block_ids=[0, 1], token_ids=list(range(31)))
    ctx = _context("r", state, {"r": state}, num_speculative_tokens=3)
    if mode == "non_greedy":
        state.sampling_params = SamplingParams(temperature=1.0)
    elif mode == "no_sample":
        ctx = replace(ctx, decode_token_ids=[[]])
    else:
        ctx = _prefills_context(
            [("r", list(range(31)))],
            result_mode="intermediate",
            num_speculative_tokens=3,
        )
    assert proposer.propose(ctx) is None
    info.assert_not_called()
    assert proposer.get_stats()["num_context_limit_fallback_requests"] == 0


@pytest.mark.parametrize("finish", ["explicit", "pruned"])
def test_log_survives_recompute_but_not_request_id_reuse(finish, info):
    proposer = _proposer(StubDraftModel(), max_model_len=32, min_speculative_tokens=3)
    state = _request_state(scheduler_block_ids=[0, 1], token_ids=list(range(31)))
    ctx = _context("r", state, {"r": state}, num_speculative_tokens=3)
    assert proposer.propose(ctx) is None
    # Preemption/resume invalidates KV, but must not repeat the diagnostic.
    proposer.release_requests({"r"})
    assert proposer.propose(ctx) is None
    info.assert_called_once()
    assert proposer.get_stats()["num_context_limit_fallback_requests"] == 1

    proposer.release_requests({"r"})
    if finish == "explicit":
        ctx = replace(ctx, finished_req_ids={"r"})
    else:
        assert (
            proposer.propose(
                replace(
                    ctx,
                    request_states={},
                    decode_reqs=[],
                    decode_token_ids=[],
                    num_speculative_tokens=0,
                )
            )
            is None
        )
    assert proposer.propose(ctx) is None
    assert info.call_count == 2
    assert proposer.get_stats()["num_context_limit_fallback_requests"] == 2


@pytest.mark.parametrize("keep_old_state", [False, True])
def test_request_id_reuse_after_cleanup_without_a_proposal(keep_old_state, info):
    proposer = _proposer(StubDraftModel(), max_model_len=32, min_speculative_tokens=3)
    old = _request_state(scheduler_block_ids=[0, 1], token_ids=list(range(31)))
    assert (
        proposer.propose(_context("r", old, {"r": old}, num_speculative_tokens=3))
        is None
    )
    runner = make_stub_runner(_drafter=proposer, _request_states={"r": old})
    # A zero-token cleanup step releases the old request without calling
    # propose(), so its finished ID will not reach the next ProposeContext.
    runner._reconcile_request_lifecycle({"r"})
    if not keep_old_state:
        old_ref = ref(old)
        del old
        assert old_ref() is None  # Diagnostics must not retain completed tokens.
    new = _request_state(scheduler_block_ids=[0, 1], token_ids=list(range(31)))
    assert (
        proposer.propose(_context("r", new, {"r": new}, num_speculative_tokens=3))
        is None
    )
    assert info.call_count == 2
    assert proposer.get_stats()["num_context_limit_fallback_requests"] == 2


@pytest.mark.parametrize(
    "schedule,max_num_seqs,k,logs",
    [
        (None, 4, 3, True),
        ([(1, 1, 3), (2, 2, 1), (3, 4, 0)], 4, 3, False),
        ([(1, 1, 3), (2, 2, 1)], 1, 3, True),
        ([(1, 4, 7)], 4, 3, True),
        ([(1, 1, 3), (2, 4, 0)], 4, 3, True),
        ([(1, 4, 0)], 4, 0, False),
    ],
)
def test_build_uses_reachable_scheduler_widths(
    schedule, max_num_seqs, k, logs, monkeypatch, info
):
    model = StubDraftModel()
    monkeypatch.setattr(
        dmp, "_load_draft_model", lambda *_: (model, dmp.DraftDims(1, 1, 64))
    )
    monkeypatch.setattr(dmp, "SDPAPagedAttentionRuntime", Mock())
    config = SimpleNamespace(
        draft_model_config=SimpleNamespace(model="draft"),
        num_speculative_tokens=3,
        num_speculative_tokens_per_batch_size=schedule,
    )
    proposer = dmp.DraftModelProposer.build(
        speculative_config=config,
        parallel_config=None,
        controller=SpeculativeDecodeController(),
        model_adapter=SimpleNamespace(
            supports_selective_logits=lambda model: False,
            extract_logits=lambda value: value,
        ),
        max_model_len=4096,
        max_num_seqs=max_num_seqs,
        block_size=16,
        allow_deferred_zero_k_ingest=False,
    )
    proposer.bind_paged_cache(num_blocks=3, block_size=16, dtype=mx.float32)
    proposer.adopt_scheduler_group(0, 32)
    info.reset_mock()  # Ignore the separate model-load message.
    state = _request_state(scheduler_block_ids=[0, 1], token_ids=list(range(31)))
    assert (
        proposer.propose(_context("r", state, {"r": state}, num_speculative_tokens=k))
        is None
    )
    assert info.call_count == int(logs)
    assert proposer.get_stats()["num_context_limit_fallback_requests"] == int(logs)
