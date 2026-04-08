# SPDX-License-Identifier: Apache-2.0
"""Tests for paged KV prefix caching in the unified model runner path.

Verifies that when `num_computed_tokens > 0` (prefix cache hit), the model
runner correctly creates RequestState with full prompt and tracks the full
sequence length for subsequent decode steps.

Run with:
    python -m pytest tests/test_paged_prefix_caching.py -v -s
"""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mlx.core as mx
from vllm.sampling_params import SamplingParams

import vllm_metal.paged_attention_common as pac
import vllm_metal.v1.model_runner as mr
from tests.stub_runner import make_stub_runner


def _make_paged_runner(num_layers: int = 2) -> mr.MetalModelRunner:
    """Build a minimal MetalModelRunner with paged KV wired up."""
    return make_stub_runner(
        model_args={"vocab_size": 32000},
        model=MagicMock(),
        _paged_attention_backend=MagicMock(),
        _paged_block_size=4,
        _gdn_req_to_slot={},
        _gdn_free_slots=[],
        _rust_state_manager=None,
        num_layers=num_layers,
    )


def _greedy_sp() -> SamplingParams:
    return SamplingParams(temperature=0.0)


def _make_scheduler_output(
    new_reqs: list[SimpleNamespace],
    num_scheduled: dict[str, int] | None = None,
) -> SimpleNamespace:
    """Minimal SchedulerOutput stub."""
    if num_scheduled is None:
        num_scheduled = {}
        for r in new_reqs:
            computed = r.num_computed_tokens
            total = len(r.prompt_token_ids)
            num_scheduled[r.req_id] = total - computed

    return SimpleNamespace(
        scheduled_new_reqs=new_reqs,
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[],
            new_block_ids=[],
            resumed_req_ids=set(),
            num_computed_tokens=[],
        ),
        num_scheduled_tokens=num_scheduled,
        total_num_scheduled_tokens=sum(num_scheduled.values()),
        finished_req_ids=set(),
        preempted_req_ids=set(),
        grammar_bitmask=None,
    )


def _make_new_req(
    req_id: str,
    prompt_token_ids: list[int],
    num_computed_tokens: int = 0,
    block_ids: list[int] | None = None,
) -> SimpleNamespace:
    if block_ids is None:
        num_blocks = (len(prompt_token_ids) + 3) // 4 + 1
        block_ids = list(range(num_blocks))
    return SimpleNamespace(
        req_id=req_id,
        prompt_token_ids=prompt_token_ids,
        sampling_params=_greedy_sp(),
        block_ids=(block_ids,),
        num_computed_tokens=num_computed_tokens,
    )


class TestPagedPrefixCacheHit:
    """Verify model runner handles num_computed_tokens > 0 on new requests."""

    def test_request_state_has_full_prompt(self):
        """When prefix cache hits, RequestState.token_ids must contain the
        full prompt (not just the suffix slice) plus the sampled token."""
        runner = _make_paged_runner()
        prompt = [10, 20, 30, 40, 50, 60, 70, 80]
        num_computed = 4  # first 4 tokens cached

        # Model returns dummy logits; greedy picks token 0
        vocab = 100
        logits = mx.zeros((1, len(prompt) - num_computed + 0, vocab))
        runner.model.return_value = MagicMock(logits=logits)

        # Patch _extract_logits and greedy sample to return deterministic token
        fake_token = 99
        with (
            patch.object(
                mr.MetalModelRunner,
                "_extract_logits",
                return_value=logits,
            ),
            patch(
                "vllm_metal.v1.sampling_batch._mlx_greedy_sample",
                return_value=mx.array(fake_token),
            ),
            patch(
                "vllm_metal.paged_attention_common.prepare_unified",
            ),
            patch(
                "vllm_metal.paged_attention_common.clear_context",
            ),
        ):
            new_req = _make_new_req("req-1", prompt, num_computed_tokens=num_computed)
            sched_out = _make_scheduler_output([new_req])
            result = runner.execute_model(sched_out)
            if result is None:
                runner.sample_tokens(None)

        state = runner._request_states.get("req-1")
        assert state is not None
        # token_ids = full prompt + sampled token
        assert state.token_ids == prompt + [fake_token]
        assert state.prompt_len == len(prompt)
        assert state.generated_tokens == 1

    def test_seq_lens_tracking_includes_prefix(self):
        """_paged_request_seq_lens must be start_pos + suffix_len, not
        just suffix_len."""
        runner = _make_paged_runner()
        prompt = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        num_computed = 6

        vocab = 100
        suffix_len = len(prompt) - num_computed
        logits = mx.zeros((1, suffix_len, vocab))
        runner.model.return_value = MagicMock(logits=logits)

        with (
            patch.object(
                mr.MetalModelRunner,
                "_extract_logits",
                return_value=logits,
            ),
            patch(
                "vllm_metal.v1.sampling_batch._mlx_greedy_sample",
                return_value=mx.array(0),
            ),
            patch(
                "vllm_metal.paged_attention_common.prepare_unified",
            ),
            patch(
                "vllm_metal.paged_attention_common.clear_context",
            ),
        ):
            new_req = _make_new_req("req-1", prompt, num_computed_tokens=num_computed)
            sched_out = _make_scheduler_output([new_req])
            result = runner.execute_model(sched_out)
            if result is None:
                runner.sample_tokens(None)

        # Must be full sequence length, not just suffix
        assert runner._paged_request_seq_lens["req-1"] == len(prompt)

    def test_no_assert_on_start_pos_gt_zero(self):
        """Prefix cache hit (start_pos > 0) must not crash."""
        runner = _make_paged_runner()
        prompt = [1, 2, 3, 4, 5, 6]
        num_computed = 4

        vocab = 100
        logits = mx.zeros((1, len(prompt) - num_computed, vocab))
        runner.model.return_value = MagicMock(logits=logits)

        with (
            patch.object(
                mr.MetalModelRunner,
                "_extract_logits",
                return_value=logits,
            ),
            patch(
                "vllm_metal.v1.sampling_batch._mlx_greedy_sample",
                return_value=mx.array(0),
            ),
            patch(
                "vllm_metal.paged_attention_common.prepare_unified",
            ),
            patch(
                "vllm_metal.paged_attention_common.clear_context",
            ),
        ):
            new_req = _make_new_req("req-1", prompt, num_computed_tokens=num_computed)
            sched_out = _make_scheduler_output([new_req])
            # This would raise AssertionError before the fix
            result = runner.execute_model(sched_out)
            if result is None:
                runner.sample_tokens(None)


class TestSamplingMetadataWithPenalties:
    """Verify advanced sampling uses full prompt on prefix cache hits."""

    def test_sampling_metadata_uses_full_prompt_with_penalties(self):
        """When repetition_penalty is set, the SamplingBatch must
        receive the full prompt, not just the suffix slice."""
        runner = _make_paged_runner()
        prompt = [10, 20, 30, 40, 50, 60, 70, 80]
        num_computed = 4

        vocab = 100
        suffix_len = len(prompt) - num_computed
        logits = mx.zeros((1, suffix_len, vocab))
        runner.model.return_value = MagicMock(logits=logits)

        # Use repetition_penalty to force the advanced sampling path
        sp = SamplingParams(temperature=0.8, repetition_penalty=1.2)

        captured_batches: list = []

        def spy_sample(logits_2d, batch, sampler, device):
            captured_batches.append(batch)
            return [99]

        with (
            patch.object(
                mr.MetalModelRunner,
                "_extract_logits",
                return_value=logits,
            ),
            patch(
                "vllm_metal.v1.sampling_batch.sample_from_logits",
                spy_sample,
            ),
            patch(
                "vllm_metal.paged_attention_common.prepare_unified",
            ),
            patch(
                "vllm_metal.paged_attention_common.clear_context",
            ),
        ):
            runner._sampler = MagicMock()

            new_req = _make_new_req("req-1", prompt, num_computed_tokens=num_computed)
            new_req.sampling_params = sp
            sched_out = _make_scheduler_output([new_req])
            result = runner.execute_model(sched_out)
            if result is None:
                runner.sample_tokens(None)

        # SamplingBatch should have been constructed with the full
        # prompt as prompt_token_ids, not just the suffix.
        assert len(captured_batches) >= 1
        prompt_token_ids_passed = captured_batches[-1].prompt_token_id_lists[0]
        assert prompt_token_ids_passed == prompt


def _make_cached_scheduler_output(
    req_ids: list[str],
    num_computed_tokens: list[int],
    num_scheduled: dict[str, int],
    new_block_ids: list | None = None,
) -> SimpleNamespace:
    """Minimal SchedulerOutput with cached requests only."""
    if new_block_ids is None:
        new_block_ids = [None] * len(req_ids)
    return SimpleNamespace(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=req_ids,
            new_block_ids=new_block_ids,
            resumed_req_ids=set(),
            num_computed_tokens=num_computed_tokens,
        ),
        num_scheduled_tokens=num_scheduled,
        total_num_scheduled_tokens=sum(num_scheduled.values()),
        finished_req_ids=set(),
        preempted_req_ids=set(),
        grammar_bitmask=None,
    )


class TestCachedRequestBlockUpdates:
    def test_resumed_request_replaces_blocks_and_resets_prefill_state(self):
        runner = _make_paged_runner()
        runner._request_states["req-1"] = mr.RequestState(
            token_ids=[10, 20, 30, 40, 50],
            prompt_len=4,
            cache=[],
            sampling_params=_greedy_sp(),
            generator=None,
            generated_tokens=1,
            block_ids=[0, 1],
        )
        runner._paged_request_seq_lens["req-1"] = 5

        cached_reqs = SimpleNamespace(
            req_ids=["req-1"],
            new_block_ids=[([7, 8],)],
            resumed_req_ids={"req-1"},
            num_computed_tokens=[4],
        )

        runner._update_cached_request_blocks(cached_reqs)

        state = runner._request_states["req-1"]
        assert state.block_ids == [7, 8]
        assert state.generated_tokens == 0
        assert "req-1" not in runner._paged_request_seq_lens


class TestMixedDecodeAndPrefixHitPrefill:
    """Verify a decode request and a prefix-hit prefill in the same unified step."""

    def test_decode_and_prefix_hit_prefill_produce_correct_state(self):
        runner = _make_paged_runner()
        prompt_a = [10, 20, 30]
        runner._request_states["req-A"] = mr.RequestState(
            token_ids=prompt_a + [99],
            prompt_len=len(prompt_a),
            cache=[],
            sampling_params=_greedy_sp(),
            generator=None,
            generated_tokens=1,
            block_ids=[0, 1],
        )
        runner._paged_request_seq_lens["req-A"] = len(prompt_a)

        prompt_b = [1, 2, 3, 4, 5, 6]
        num_computed_b = 4
        suffix_len_b = len(prompt_b) - num_computed_b
        logits = mx.zeros((1, 1 + suffix_len_b, 100))
        runner.model.return_value = MagicMock(logits=logits)

        decode_token = 55
        prefill_token = 77
        # Decode is processed before prefill in execute_model; side_effect order matches.
        greedy_tokens = [mx.array([decode_token]), mx.array([prefill_token])]

        new_req_b = _make_new_req("req-B", prompt_b, num_computed_tokens=num_computed_b)
        sched_out = SimpleNamespace(
            scheduled_new_reqs=[new_req_b],
            scheduled_cached_reqs=SimpleNamespace(
                req_ids=["req-A"],
                new_block_ids=[None],
                resumed_req_ids=set(),
                num_computed_tokens=[len(prompt_a)],
            ),
            num_scheduled_tokens={"req-A": 1, "req-B": suffix_len_b},
            total_num_scheduled_tokens=1 + suffix_len_b,
            finished_req_ids=set(),
            preempted_req_ids=set(),
            grammar_bitmask=None,
        )

        with (
            patch.object(mr.MetalModelRunner, "_extract_logits", return_value=logits),
            patch(
                "vllm_metal.v1.sampling_batch._mlx_greedy_sample",
                side_effect=greedy_tokens,
            ),
            patch("vllm_metal.v1.model_runner.prepare_unified"),
            patch("vllm_metal.v1.model_runner.clear_context"),
        ):
            result = runner.execute_model(sched_out)
            if result is None:
                runner.sample_tokens(None)

        state_a = runner._request_states["req-A"]
        assert state_a.token_ids[-1] == decode_token
        assert state_a.generated_tokens == 2

        state_b = runner._request_states.get("req-B")
        assert state_b is not None
        assert state_b.token_ids == prompt_b + [prefill_token]
        assert state_b.prompt_len == len(prompt_b)
        assert state_b.generated_tokens == 1
        assert runner._paged_request_seq_lens.get("req-B") == len(prompt_b)


class TestCachedRequestContinuation:
    """Verify the cached/intermediate-chunk path works with prefix offsets."""

    def test_cached_final_chunk_with_offset(self):
        """Final chunk (computed + scheduled == prompt_len) with offset:
        token is kept and state transitions to decode phase."""
        runner = _make_paged_runner()
        prompt = list(range(1, 13))  # 12 tokens
        block_ids = list(range(4))

        runner._request_states["req-1"] = mr.RequestState(
            token_ids=list(prompt),
            prompt_len=len(prompt),
            cache=[],
            sampling_params=SamplingParams(temperature=0.0),
            generator=None,
            generated_tokens=0,
            block_ids=block_ids,
        )
        runner._paged_request_seq_lens["req-1"] = 6

        # Final chunk: computed=6, scheduled=6 → 6+6=12=prompt_len
        vocab = 100
        logits = mx.zeros((1, 6, vocab))
        runner.model.return_value = MagicMock(logits=logits)

        fake_token = 42
        with (
            patch.object(
                mr.MetalModelRunner,
                "_extract_logits",
                return_value=logits,
            ),
            patch(
                "vllm_metal.v1.sampling_batch._mlx_greedy_sample",
                return_value=mx.array(fake_token),
            ),
            patch("vllm_metal.v1.model_runner.prepare_unified"),
            patch("vllm_metal.v1.model_runner.clear_context"),
        ):
            sched_out = _make_cached_scheduler_output(
                req_ids=["req-1"],
                num_computed_tokens=[6],
                num_scheduled={"req-1": 6},
            )
            result = runner.execute_model(sched_out)
            if result is None:
                runner.sample_tokens(None)

        state = runner._request_states["req-1"]
        assert state.token_ids == prompt + [fake_token]
        assert state.generated_tokens == len(state.token_ids) - state.prompt_len
        assert runner._paged_request_seq_lens["req-1"] == len(prompt)

    def test_true_intermediate_chunk_with_prefix_cache_hit(self):
        """Intermediate chunk (computed + scheduled < prompt_len) with
        num_computed > 0: sampled token must be discarded, generated_tokens
        stays 0, and _paged_request_seq_lens tracks partial progress."""
        runner = _make_paged_runner()
        prompt = list(range(1, 13))  # 12 tokens
        block_ids = list(range(4))

        # Prefix cache hit: 4 tokens already computed
        runner._request_states["req-1"] = mr.RequestState(
            token_ids=list(prompt),
            prompt_len=len(prompt),
            cache=[],
            sampling_params=SamplingParams(temperature=0.0),
            generator=None,
            generated_tokens=0,
            block_ids=block_ids,
        )
        runner._paged_request_seq_lens["req-1"] = 4

        # Intermediate chunk: computed=4, scheduled=4 → 4+4=8 < 12
        vocab = 100
        chunk_len = 4
        logits = mx.zeros((1, chunk_len, vocab))
        runner.model.return_value = MagicMock(logits=logits)

        with (
            patch.object(
                mr.MetalModelRunner,
                "_extract_logits",
                return_value=logits,
            ),
            patch(
                "vllm_metal.v1.sampling_batch._mlx_greedy_sample",
                return_value=mx.array(0),
            ),
            patch("vllm_metal.v1.model_runner.prepare_unified"),
            patch("vllm_metal.v1.model_runner.clear_context"),
        ):
            sched_out = _make_cached_scheduler_output(
                req_ids=["req-1"],
                num_computed_tokens=[4],
                num_scheduled={"req-1": 4},
            )
            result = runner.execute_model(sched_out)
            if result is None:
                runner.sample_tokens(None)

        state = runner._request_states["req-1"]
        # Still prefilling — no token appended, generated_tokens stays 0
        assert state.generated_tokens == 0
        assert state.token_ids == prompt
        # seq_lens = start_pos(4) + chunk_len(4) = 8
        assert runner._paged_request_seq_lens["req-1"] == 8


def _make_paged_ctx_spy(
    captured: list,
) -> Callable[[pac.PagedAttentionContext], None]:
    def spy(ctx: pac.PagedAttentionContext) -> None:
        captured.append(ctx)
        pac._thread_local.paged_ctx = ctx

    return spy


class TestPrepareUnifiedSlotMapping:
    """Verify prepare_unified is called with correct slot mapping and RoPE offsets.

    All other tests in this file patch prepare_unified out.  These tests let it
    run for real and spy on set_context to confirm the runner passes the right
    block_ids, num_tokens, and start_pos arguments so that slot mapping and RoPE
    offsets are exercised end-to-end.
    """

    def test_fresh_prefill_slot_mapping_and_rope_offset(self):
        """start_pos == 0: slots cover positions 0..N-1, offset is 0."""
        runner = _make_paged_runner()
        prompt = [10, 20, 30, 40]
        block_ids = [0]  # block_size=4, block 0 covers positions 0-3
        logits = mx.zeros((1, len(prompt), 100))
        runner.model.return_value = MagicMock(logits=logits)

        captured: list[pac.PagedAttentionContext] = []

        new_req = _make_new_req(
            "req-1", prompt, num_computed_tokens=0, block_ids=block_ids
        )
        sched_out = _make_scheduler_output([new_req])

        with (
            patch.object(mr.MetalModelRunner, "_extract_logits", return_value=logits),
            patch(
                "vllm_metal.v1.sampling_batch._mlx_greedy_sample",
                return_value=mx.array([0]),
            ),
            patch.object(pac, "set_context", side_effect=_make_paged_ctx_spy(captured)),
        ):
            result = runner.execute_model(sched_out)
            if result is None:
                runner.sample_tokens(None)

        assert len(captured) == 1
        ctx = captured[0]
        assert ctx.slot_mapping == [0, 1, 2, 3]
        assert ctx.offsets == [0]
        assert ctx.context_lens == [4]

    def test_prefix_hit_slot_mapping_starts_at_start_pos(self):
        """start_pos == 2: slots cover positions 2-3, RoPE offset is 2."""
        runner = _make_paged_runner()
        prompt = [10, 20, 30, 40]
        num_computed = 2
        block_ids = [0]  # block_size=4, block 0 covers positions 0-3
        suffix_len = len(prompt) - num_computed
        logits = mx.zeros((1, suffix_len, 100))
        runner.model.return_value = MagicMock(logits=logits)

        captured: list[pac.PagedAttentionContext] = []

        new_req = _make_new_req(
            "req-1", prompt, num_computed_tokens=num_computed, block_ids=block_ids
        )
        sched_out = _make_scheduler_output([new_req])

        with (
            patch.object(mr.MetalModelRunner, "_extract_logits", return_value=logits),
            patch(
                "vllm_metal.v1.sampling_batch._mlx_greedy_sample",
                return_value=mx.array([0]),
            ),
            patch.object(pac, "set_context", side_effect=_make_paged_ctx_spy(captured)),
        ):
            result = runner.execute_model(sched_out)
            if result is None:
                runner.sample_tokens(None)

        assert len(captured) == 1
        ctx = captured[0]
        assert ctx.slot_mapping == [2, 3]  # positions 2-3 in block 0
        assert ctx.offsets == [2]
        assert ctx.context_lens == [4]  # start_pos + num_tokens = 2 + 2
