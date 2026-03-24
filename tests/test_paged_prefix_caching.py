# SPDX-License-Identifier: Apache-2.0
"""Tests for paged KV prefix caching in the unified model runner path.

Verifies that when `num_computed_tokens > 0` (prefix cache hit), the model
runner correctly creates RequestState with full prompt and tracks the full
sequence length for subsequent decode steps.

Run with:
    python -m pytest tests/test_paged_prefix_caching.py -v -s
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mlx.core as mx
import torch
from vllm.sampling_params import SamplingParams

import vllm_metal.v1.model_runner as mr


def _make_paged_runner(num_layers: int = 2) -> mr.MetalModelRunner:
    """Build a minimal MetalModelRunner with paged KV wired up."""
    runner = mr.MetalModelRunner.__new__(mr.MetalModelRunner)
    runner.model = MagicMock()
    runner._is_stt = False
    runner._paged_attention_backend = MagicMock()  # non-None enables paged path
    runner._paged_block_size = 4
    runner._paged_request_seq_lens = {}
    runner._request_states = {}
    runner._rust_state_manager = None
    runner.num_layers = num_layers
    runner.device = torch.device("cpu")
    runner._sampler = None
    runner._pending_output = None
    runner.use_async_scheduling = True
    return runner


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
                "vllm_metal.v1.model_runner._mlx_greedy_sample",
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
            runner.execute_model(sched_out)

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
                "vllm_metal.v1.model_runner._mlx_greedy_sample",
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
            runner.execute_model(sched_out)

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
                "vllm_metal.v1.model_runner._mlx_greedy_sample",
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
            runner.execute_model(sched_out)


class TestSamplingMetadataWithPenalties:
    """Verify advanced sampling uses full prompt on prefix cache hits."""

    def test_sampling_metadata_uses_full_prompt_with_penalties(self):
        """When repetition_penalty is set, _make_sampling_metadata must
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

        captured_metadata: list = []

        def spy_make(self_, *args, **kwargs):
            captured_metadata.append(args)
            return MagicMock()

        fake_sampler_output = MagicMock()
        fake_sampler_output.sampled_token_ids = torch.tensor([[99]])

        with (
            patch.object(
                mr.MetalModelRunner,
                "_extract_logits",
                return_value=logits,
            ),
            patch.object(
                mr.MetalModelRunner,
                "_make_sampling_metadata",
                spy_make,
            ),
            patch(
                "vllm_metal.paged_attention_common.prepare_unified",
            ),
            patch(
                "vllm_metal.paged_attention_common.clear_context",
            ),
        ):
            runner._sampler = MagicMock()
            runner._sampler.forward.return_value = fake_sampler_output

            new_req = _make_new_req("req-1", prompt, num_computed_tokens=num_computed)
            new_req.sampling_params = sp
            sched_out = _make_scheduler_output([new_req])
            runner.execute_model(sched_out)

        # _make_sampling_metadata should have been called with the full
        # prompt as prompt_token_ids, not just the suffix.
        assert len(captured_metadata) >= 1
        # args[1] is prompt_token_ids_list
        prompt_token_ids_passed = captured_metadata[-1][1][0]
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


class TestCachedRequestContinuation:
    """Verify the cached/intermediate-chunk path works with prefix offsets."""

    def test_cached_intermediate_chunk_with_offset(self):
        """A cached request continuing prefill with computed > 0 must
        produce correct state and seq_lens tracking."""
        runner = _make_paged_runner()
        prompt = list(range(1, 13))  # 12 tokens
        block_ids = list(range(4))

        # Simulate: first chunk already processed tokens 0-5 (6 tokens)
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

        # Second chunk: computed=6, scheduled=6 → tokens[6:12], complete
        vocab = 100
        suffix_len = 6
        logits = mx.zeros((1, suffix_len, vocab))
        runner.model.return_value = MagicMock(logits=logits)

        fake_token = 42
        with (
            patch.object(
                mr.MetalModelRunner,
                "_extract_logits",
                return_value=logits,
            ),
            patch(
                "vllm_metal.v1.model_runner._mlx_greedy_sample",
                return_value=mx.array(fake_token),
            ),
            patch(
                "vllm_metal.paged_attention_common.prepare_unified",
            ),
            patch(
                "vllm_metal.paged_attention_common.clear_context",
            ),
        ):
            sched_out = _make_cached_scheduler_output(
                req_ids=["req-1"],
                num_computed_tokens=[6],
                num_scheduled={"req-1": 6},
            )
            runner.execute_model(sched_out)

        state = runner._request_states["req-1"]
        # Should have full prompt + sampled token
        assert state.token_ids == prompt + [fake_token]
        assert state.generated_tokens == len(state.token_ids) - state.prompt_len
        # seq_lens must reflect full sequence
        assert runner._paged_request_seq_lens["req-1"] == len(prompt)
