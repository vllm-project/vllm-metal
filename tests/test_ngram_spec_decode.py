# SPDX-License-Identifier: Apache-2.0
"""Tests for the Metal n-gram speculative-decode proposer.

The proposer wraps vLLM's pure-Python/Numba n-gram kernel, so these tests need no
model or engine: a ``SimpleNamespace`` ``vllm_config`` exercises the upstream
constructor (which reads only scalar config) and a hand-built ``ProposeContext``
drives ``propose``. They lock in the request filtering (greedy-only, skip
intermediate prefills) and the array marshalling into the upstream kernel.
"""

from __future__ import annotations

import random
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from vllm.sampling_params import SamplingParams
from vllm.v1.outputs import DraftTokenIds
from vllm.v1.spec_decode.ngram_proposer import NgramProposer as VllmNgramProposer

from vllm_metal.v1 import ngram_proposer as ngram_mod
from vllm_metal.v1.ngram_proposer import NgramProposer
from vllm_metal.v1.proposer import ProposeContext
from vllm_metal.v1.spec_decode import PagedDecodeSegment, SpeculativeDecodeController


def _vllm_config(
    prompt_lookup_min: int,
    prompt_lookup_max: int,
    num_speculative_tokens: int,
    max_model_len: int,
    max_num_seqs: int,
) -> SimpleNamespace:
    return SimpleNamespace(
        speculative_config=SimpleNamespace(
            prompt_lookup_min=prompt_lookup_min,
            prompt_lookup_max=prompt_lookup_max,
            num_speculative_tokens=num_speculative_tokens,
        ),
        model_config=SimpleNamespace(max_model_len=max_model_len),
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
    )


def _proposer(
    *,
    prompt_lookup_min: int = 2,
    prompt_lookup_max: int = 3,
    num_speculative_tokens: int = 3,
    max_model_len: int = 512,
    max_num_seqs: int = 4,
) -> NgramProposer:
    return NgramProposer(
        vllm_config=_vllm_config(
            prompt_lookup_min,
            prompt_lookup_max,
            num_speculative_tokens,
            max_model_len,
            max_num_seqs,
        ),
        controller=SpeculativeDecodeController(),
    )


def _request_state(
    token_ids: list[int], *, temperature: float = 0.0
) -> SimpleNamespace:
    return SimpleNamespace(
        token_ids=list(token_ids),
        sampling_params=SamplingParams(temperature=temperature),
        generated_tokens=1,
    )


def _context(
    *,
    decode_reqs: list[tuple[str, SimpleNamespace]] | None = None,
    decode_token_ids: list[list[int]] | None = None,
    decode_segments: list[PagedDecodeSegment] | None = None,
    prefill_reqs: list[SimpleNamespace] | None = None,
    prefill_result_modes: list[str] | None = None,
    request_states: dict[str, SimpleNamespace] | None = None,
    num_speculative_tokens: int = 3,
    finished_req_ids: set[str] | None = None,
    draft_caps: dict[str, int] | None = None,
) -> ProposeContext:
    decode_reqs = decode_reqs or []
    prefill_reqs = prefill_reqs or []
    if decode_token_ids is None:
        decode_token_ids = [[state.token_ids[-1]] for _, state in decode_reqs]
    if prefill_result_modes is None:
        prefill_result_modes = ["new_final"] * len(prefill_reqs)
    if request_states is None:
        request_states = dict(decode_reqs)
    return ProposeContext(
        target_hidden_states=None,
        decode_reqs=decode_reqs,
        decode_segments=decode_segments or [],
        decode_token_ids=decode_token_ids,
        prefill_reqs=prefill_reqs,
        prefill_token_ids=[0] * len(prefill_reqs),
        prefill_result_modes=prefill_result_modes,
        request_states=request_states,
        cu_seqlens=[],
        num_decode_segments=len(decode_reqs),
        num_speculative_tokens=num_speculative_tokens,
        finished_req_ids=finished_req_ids or set(),
        draft_caps=draft_caps,
    )


def _verified_segment(req_id: str, *, has_draft: bool = True) -> PagedDecodeSegment:
    """A decode_segment for req_id: a real (if trivial) verify row if
    has_draft, a plain no-draft row otherwise."""
    draft_token_ids = (0,) if has_draft else ()
    num_query_tokens = len(draft_token_ids) + 1
    return PagedDecodeSegment(
        req_id=req_id,
        input_token_ids=tuple(range(num_query_tokens)),
        start_row=0,
        num_query_tokens=num_query_tokens,
        draft_token_ids=draft_token_ids,
        cache_start_pos=0,
        block_ids=(0,),
    )


class TestNgramProposerProtocol:
    def test_never_needs_target_hidden_states(self) -> None:
        proposer = _proposer()
        assert proposer.needs_target_hidden_states([], has_final_prefill=False) is False
        assert proposer.needs_target_hidden_states([], has_final_prefill=True) is False


class TestNgramProposePropose:
    def test_matches_repetitive_suffix_and_drafts_continuation(self) -> None:
        # Suffix [1, 2] recurs earlier; the tokens that followed it were 3, 1, 2.
        proposer = _proposer(prompt_lookup_min=2, prompt_lookup_max=3)
        state = _request_state([1, 2, 3, 1, 2, 3, 1, 2])
        ctx = _context(decode_reqs=[("r0", state)])

        drafts = proposer.propose(ctx)

        assert drafts is not None
        assert drafts.req_ids == ["r0"]
        assert drafts.draft_token_ids == [[3, 1, 2]]

    def test_uses_scheduler_selected_num_speculative_tokens(self) -> None:
        proposer = _proposer(prompt_lookup_min=2, prompt_lookup_max=3)
        state = _request_state([1, 2, 3, 1, 2, 3, 1, 2])
        ctx = _context(
            decode_reqs=[("r0", state)],
            num_speculative_tokens=1,
        )

        drafts = proposer.propose(ctx)

        assert drafts is not None
        assert drafts.req_ids == ["r0"]
        assert drafts.draft_token_ids == [[3]]

    def test_scheduler_selected_zero_tokens_returns_none(self) -> None:
        proposer = _proposer(prompt_lookup_min=2, prompt_lookup_max=3)
        state = _request_state([1, 2, 3, 1, 2, 3, 1, 2])
        ctx = _context(
            decode_reqs=[("r0", state)],
            num_speculative_tokens=0,
        )

        assert proposer.propose(ctx) is None

    def test_no_match_returns_none(self) -> None:
        # Non-repeating context shorter than the n-gram window: no draft.
        proposer = _proposer(prompt_lookup_min=2, prompt_lookup_max=3)
        state = _request_state([7, 8, 9])
        ctx = _context(decode_reqs=[("r0", state)])

        assert proposer.propose(ctx) is None

    def test_cooldown_resume_syncs_all_appended_history(self) -> None:
        """Tokens committed during cooldown must reach the next real lookup."""
        proposer = _proposer()
        state = _request_state([1, 2, 3, 4, 5])
        for token in range(10, 10 + ngram_mod._MAX_CONSECUTIVE_MISSES):
            state.token_ids.append(token)
            assert proposer.propose(_context(decode_reqs=[("r0", state)])) is None
        assert proposer._cooldown["r0"] == ngram_mod._COOLDOWN_STEPS
        row = proposer._row_of["r0"]
        synced = int(proposer._row_len[row])

        for _ in range(ngram_mod._COOLDOWN_STEPS):
            state.token_ids.extend([1, 2])
            assert proposer.propose(_context(decode_reqs=[("r0", state)])) is None
            assert proposer._row_len[row] == synced

        state.token_ids.extend([3, 4])
        ctx = _context(decode_reqs=[("r0", state)])
        drafts = _proposed(proposer.propose(ctx))
        assert drafts == {"r0": [5, 10, 11]}
        length = len(state.token_ids)
        assert proposer._row_len[row] == length
        assert proposer._token_ids_cpu[row, :length].tolist() == state.token_ids

    @pytest.mark.parametrize("positional_cap,max_tokens", [(1, None), (1, 5), (3, 3)])
    def test_draft_caps_take_the_tighter_bound(
        self, positional_cap, max_tokens
    ) -> None:
        # The match yields three tokens. With one output already committed,
        # max_tokens=5/3 leaves a draft budget of 3/1, respectively.
        proposer = _proposer(prompt_lookup_min=2, prompt_lookup_max=3)
        state = _request_state([1, 2, 3, 1, 2, 3, 1, 2])
        state.sampling_params.max_tokens = max_tokens
        ctx = _context(decode_reqs=[("r0", state)], draft_caps={"r0": positional_cap})

        drafts = proposer.propose(ctx)

        assert drafts is not None
        assert drafts.req_ids == ["r0"]
        assert drafts.draft_token_ids == [[3]]

    def test_draft_cap_to_empty_is_not_miss_evidence(self) -> None:
        # A draft trimmed away by a positional cap says nothing about
        # repetition exploitability: no miss streak, no pending record, and
        # the request drafts again once the cap lifts.
        proposer = _proposer(prompt_lookup_min=2, prompt_lookup_max=3)
        state = _request_state([1, 2, 3, 1, 2, 3, 1, 2])
        capped = _context(decode_reqs=[("r0", state)], draft_caps={"r0": 0})

        for _ in range(2 * ngram_mod._MAX_CONSECUTIVE_MISSES):
            assert proposer.propose(capped) is None

        assert "r0" not in proposer._cooldown
        assert proposer._miss_streak.get("r0", 0) == 0
        assert "r0" not in proposer._pending

        uncapped = _context(decode_reqs=[("r0", state)])
        drafts = proposer.propose(uncapped)
        assert drafts is not None
        assert drafts.draft_token_ids == [[3, 1, 2]]

    @pytest.mark.parametrize("generated_tokens,expected_draft", [(2, [3]), (3, [])])
    def test_remaining_output_tokens_bound_the_draft(
        self, generated_tokens, expected_draft
    ) -> None:
        # max_tokens=4 with 2 generated leaves room for one draft plus the
        # bonus token; with 3 generated there is no room. Each case starts
        # fresh so a prior unverified proposal cannot leave pending state.
        proposer = _proposer(prompt_lookup_min=2, prompt_lookup_max=3)
        state = _request_state([1, 2, 3, 1, 2, 3, 1, 2])
        state.sampling_params = SamplingParams(temperature=0.0, max_tokens=4)
        state.generated_tokens = generated_tokens

        drafts = proposer.propose(_context(decode_reqs=[("r0", state)]))
        if expected_draft:
            assert drafts is not None
            assert drafts.req_ids == ["r0"]
            assert drafts.draft_token_ids == [expected_draft]
        else:
            assert drafts is None
            assert "r0" not in proposer._pending
        assert proposer._miss_streak.get("r0", 0) == 0

    def test_empty_context_returns_none(self) -> None:
        assert _proposer().propose(_context()) is None

    def test_skips_request_without_sampled_tokens(self) -> None:
        proposer = _proposer(prompt_lookup_min=2)
        state = _request_state([1, 2, 3, 1, 2, 3, 1, 2])
        # An empty sampled-ids entry marks a row that did not decode this step.
        ctx = _context(decode_reqs=[("r0", state)], decode_token_ids=[[]])

        assert proposer.propose(ctx) is None

    @pytest.mark.parametrize("temperature,max_tokens", [(0.8, 16), (0.0, 1)])
    def test_skips_ineligible_request(self, temperature, max_tokens) -> None:
        proposer = _proposer(prompt_lookup_min=2)
        state = _request_state([1, 2, 3, 1, 2, 3, 1, 2], temperature=temperature)
        state.sampling_params.max_tokens = max_tokens
        ctx = _context(decode_reqs=[("r0", state)])

        assert proposer.propose(ctx) is None

    @pytest.mark.parametrize("max_tokens,expected_draft", [(16, [6, 4, 5]), (2, [])])
    def test_finalized_prefill_respects_output_budget(
        self, max_tokens, expected_draft
    ) -> None:
        proposer = _proposer(prompt_lookup_min=2)
        state = _request_state([4, 5, 6, 4, 5, 6, 4, 5])
        # The first output is already committed. max_tokens=2 still passes
        # eligibility but leaves no room for a draft plus a verifier token.
        state.sampling_params.max_tokens = max_tokens
        prefill = SimpleNamespace(req_id="p0")
        ctx = _context(
            prefill_reqs=[prefill],
            prefill_result_modes=["new_final"],
            request_states={"p0": state},
        )

        drafts = proposer.propose(ctx)

        if expected_draft:
            assert drafts is not None
            assert drafts.req_ids == ["p0"]
            assert drafts.draft_token_ids == [expected_draft]
        else:
            assert drafts is None

    def test_skips_intermediate_prefill(self) -> None:
        proposer = _proposer(prompt_lookup_min=2)
        state = _request_state([4, 5, 6, 4, 5, 6, 4, 5])
        prefill = SimpleNamespace(req_id="p0")
        ctx = _context(
            prefill_reqs=[prefill],
            prefill_result_modes=["intermediate"],
            request_states={"p0": state},
        )

        assert proposer.propose(ctx) is None

    def test_mixed_batch_drops_unmatched_keeps_matched(self) -> None:
        proposer = _proposer(prompt_lookup_min=2, prompt_lookup_max=3)
        matched = _request_state([1, 2, 3, 1, 2, 3, 1, 2])
        unmatched = _request_state([7, 8, 9])
        ctx = _context(decode_reqs=[("r0", matched), ("r1", unmatched)])

        drafts = proposer.propose(ctx)

        # Only the matched request appears; the kernel returned [] for r1.
        assert drafts is not None
        assert drafts.req_ids == ["r0"]
        assert drafts.draft_token_ids == [[3, 1, 2]]

    def test_mixed_output_budgets_preserve_request_alignment(self) -> None:
        proposer = _proposer()
        short = _request_state([1, 2, 3, 1, 2, 3, 1, 2])
        last = _request_state([4, 5, 6, 4, 5, 6, 4, 5])
        full = _request_state([7, 8, 9, 7, 8, 9, 7, 8])
        short.sampling_params.max_tokens = 3  # Two output slots remain.
        last.sampling_params.max_tokens = 2  # Only one output slot remains.
        # Drop the middle row; distinct histories expose any shifted IDs.
        ctx = _context(decode_reqs=[("r0", short), ("r1", last), ("r2", full)])

        drafts = proposer.propose(ctx)

        assert drafts is not None
        assert drafts.req_ids == ["r0", "r2"]
        assert drafts.draft_token_ids == [[3], [9, 7, 8]]

    def test_drops_prefill_already_seen_as_decode(self) -> None:
        # A request present in both decode and prefill lists must draft once.
        proposer = _proposer(prompt_lookup_min=2)
        state = _request_state([1, 2, 3, 1, 2, 3, 1, 2])
        prefill = SimpleNamespace(req_id="r0")
        ctx = _context(
            decode_reqs=[("r0", state)],
            prefill_reqs=[prefill],
            prefill_result_modes=["new_final"],
            request_states={"r0": state},
        )

        drafts = proposer.propose(ctx)

        assert drafts is not None
        assert drafts.req_ids == ["r0"]


class TestNgramMissThrottle:
    """The kernel scans a request's whole history every step whether or not
    it finds anything, so a request that never matches should stop being
    handed to the kernel after enough consecutive misses -- these tests
    mock the wrapped upstream call directly so they exercise only the
    throttle bookkeeping, independent of the installed vLLM's kernel
    signature."""

    def test_throttles_after_max_consecutive_misses(self) -> None:
        proposer = _proposer()
        state = _request_state([7, 8, 9])
        ctx = _context(decode_reqs=[("r0", state)])

        with patch.object(
            proposer._ngram, "propose", return_value=[[]]
        ) as mock_propose:
            for _ in range(ngram_mod._MAX_CONSECUTIVE_MISSES):
                assert proposer.propose(ctx) is None
            calls_before_throttle = mock_propose.call_count

            # One more miss would be the (N+1)th in a row: by now the request
            # should be on cooldown and skipped before ever reaching the kernel.
            assert proposer.propose(ctx) is None
            assert mock_propose.call_count == calls_before_throttle

    def test_capped_streak_returns_to_cooldown_after_one_more_miss(self) -> None:
        """A retry that misses again right after cooldown must go straight
        back into cooldown, not get another full _MAX_CONSECUTIVE_MISSES-long
        grace period -- the streak caps at the threshold instead of clearing
        when cooldown is set."""
        proposer = _proposer()
        state = _request_state([7, 8, 9])
        ctx = _context(decode_reqs=[("r0", state)])

        with patch.object(
            proposer._ngram, "propose", return_value=[[]]
        ) as mock_propose:
            for _ in range(ngram_mod._MAX_CONSECUTIVE_MISSES):
                proposer.propose(ctx)
            assert "r0" in proposer._cooldown

            for _ in range(ngram_mod._COOLDOWN_STEPS):
                proposer.propose(ctx)
            calls_at_retry = mock_propose.call_count

            # The retry itself misses again: this single miss must be enough
            # to re-cooldown, not the first of another eight.
            proposer.propose(ctx)
            assert mock_propose.call_count == calls_at_retry + 1
            assert "r0" in proposer._cooldown

            calls_after_recooldown = mock_propose.call_count
            proposer.propose(ctx)
            assert mock_propose.call_count == calls_after_recooldown

    def test_real_acceptance_resets_the_streak(self) -> None:
        """A lookup match only counts once the *next* step's committed
        history shows the target actually accepted it -- not the moment the
        kernel proposes it."""
        proposer = _proposer(prompt_lookup_min=2, prompt_lookup_max=3)
        state = _request_state([7, 8, 9])
        ctx = _context(
            decode_reqs=[("r0", state)],
            decode_segments=[_verified_segment("r0")],
        )

        with patch.object(proposer._ngram, "propose") as mock_propose:
            mock_propose.return_value = [[]]
            for _ in range(ngram_mod._MAX_CONSECUTIVE_MISSES - 1):
                proposer.propose(ctx)
            assert "r0" not in proposer._cooldown

            # The kernel finds something...
            mock_propose.return_value = [[99]]
            drafts = proposer.propose(ctx)
            assert drafts is not None
            # ...and the engine genuinely accepts it (appends exactly what
            # was proposed to committed history).
            state.token_ids.append(99)

            # Resolving that acceptance must fully clear the streak, not
            # just discount it by one.
            mock_propose.return_value = [[]]
            proposer.propose(ctx)
            assert proposer._miss_streak.get("r0", 0) <= 1

    def test_lookup_match_without_acceptance_still_throttles(self) -> None:
        """The kernel finding *something* every step isn't evidence the
        request benefits -- if the target never actually accepts any of it,
        this must throttle exactly like true misses would."""
        proposer = _proposer(prompt_lookup_min=2, prompt_lookup_max=3)
        state = _request_state([1, 2, 3, 1, 2])
        ctx = _context(
            decode_reqs=[("r0", state)],
            decode_segments=[_verified_segment("r0")],
        )

        # One resolved miss per call from the second call on (the first has
        # nothing pending yet to resolve), so the streak needs one extra
        # call beyond _MAX_CONSECUTIVE_MISSES to actually cross the
        # threshold and land in cooldown.
        with patch.object(proposer._ngram, "propose", return_value=[[99]]):
            for _ in range(ngram_mod._MAX_CONSECUTIVE_MISSES + 1):
                proposer.propose(ctx)
                # The target never agrees: the engine appends some other
                # token instead of the proposed 99. (Once throttled, propose
                # stops even calling the kernel, so this is a no-op then.)
                state.token_ids.append(123)

        assert "r0" in proposer._cooldown

    def test_short_history_is_not_counted_as_a_miss(self) -> None:
        """Below prompt_lookup_min, the kernel structurally cannot match --
        that's not evidence against repetition, just not enough history yet,
        and must never accumulate toward throttling."""
        proposer = _proposer(prompt_lookup_min=4, prompt_lookup_max=5)
        state = _request_state([1, 2])  # shorter than prompt_lookup_min
        ctx = _context(decode_reqs=[("r0", state)])

        with patch.object(proposer._ngram, "propose", return_value=[[]]):
            for _ in range(2 * ngram_mod._MAX_CONSECUTIVE_MISSES):
                assert proposer.propose(ctx) is None

        assert "r0" not in proposer._cooldown
        assert proposer._miss_streak.get("r0", 0) == 0

    def test_cooldown_expires_and_retries(self) -> None:
        proposer = _proposer()
        state = _request_state([7, 8, 9])
        ctx = _context(decode_reqs=[("r0", state)])

        with patch.object(
            proposer._ngram, "propose", return_value=[[]]
        ) as mock_propose:
            for _ in range(ngram_mod._MAX_CONSECUTIVE_MISSES):
                proposer.propose(ctx)
            calls_at_throttle = mock_propose.call_count

            # Every call during cooldown must be skipped before the kernel.
            for _ in range(ngram_mod._COOLDOWN_STEPS - 1):
                proposer.propose(ctx)
            assert mock_propose.call_count == calls_at_throttle

            # The cooldown-th call is the last skipped one; the call after
            # that must reach the kernel again.
            proposer.propose(ctx)
            assert mock_propose.call_count == calls_at_throttle
            proposer.propose(ctx)
            assert mock_propose.call_count == calls_at_throttle + 1

    def test_prune_finished_clears_throttle_state(self) -> None:
        proposer = _proposer()
        state = _request_state([7, 8, 9])
        ctx = _context(decode_reqs=[("r0", state)])

        with patch.object(proposer._ngram, "propose", return_value=[[]]):
            for _ in range(ngram_mod._MAX_CONSECUTIVE_MISSES):
                proposer.propose(ctx)
        assert "r0" in proposer._cooldown

        # r0 finishes: its bookkeeping must not survive to be misread by a
        # later, unrelated request that happens to reuse the same id.
        proposer._prune_finished({"r0"})
        assert "r0" not in proposer._cooldown
        assert "r0" not in proposer._miss_streak

        with patch.object(
            proposer._ngram, "propose", return_value=[[1]]
        ) as mock_propose:
            drafts = proposer.propose(ctx)
        assert drafts is not None
        mock_propose.assert_called_once()

    def test_reused_request_id_does_not_inherit_old_throttle_state(self) -> None:
        """vLLM can hand a finished request's id straight to a brand-new
        request in the same scheduler step. The new request is present in
        request_states under that id from the moment it appears, so pruning
        on absence from request_states would never catch this -- pruning
        must key off the scheduler's own finished_req_ids instead."""
        proposer = _proposer()
        old_state = _request_state([7, 8, 9])
        ctx = _context(decode_reqs=[("r0", old_state)])

        with patch.object(proposer._ngram, "propose", return_value=[[]]):
            for _ in range(ngram_mod._MAX_CONSECUTIVE_MISSES):
                proposer.propose(ctx)
        assert "r0" in proposer._cooldown

        # Same step: r0 finishes and a brand-new request reuses the id, so
        # it is simultaneously in finished_req_ids and in request_states.
        new_state = _request_state([1, 2, 3])
        new_ctx = _context(
            decode_reqs=[("r0", new_state)],
            finished_req_ids={"r0"},
        )

        with patch.object(
            proposer._ngram, "propose", return_value=[[42]]
        ) as mock_propose:
            drafts = proposer.propose(new_ctx)

        # The new request must be evaluated fresh, not skipped as if it
        # were still the old, throttled one.
        assert drafts is not None
        assert drafts.req_ids == ["r0"]
        mock_propose.assert_called_once()

    def test_prune_and_resolve_run_even_when_speculative_tokens_disabled(
        self,
    ) -> None:
        """A step with drafting disabled (num_speculative_tokens <= 0) must
        still prune finished ids and resolve pending drafts -- that
        bookkeeping can't be gated behind the same early return that skips
        drafting, or a request id reused during a disabled step would carry
        stale throttle state into a later step where drafting resumes."""
        proposer = _proposer()
        old_state = _request_state([7, 8, 9])
        ctx = _context(decode_reqs=[("r0", old_state)])

        with patch.object(proposer._ngram, "propose", return_value=[[]]):
            for _ in range(ngram_mod._MAX_CONSECUTIVE_MISSES):
                proposer.propose(ctx)
        assert "r0" in proposer._cooldown

        # Same id reused by a new request, but this step has speculative
        # decoding disabled.
        new_state = _request_state([1, 2, 3])
        disabled_ctx = _context(
            decode_reqs=[("r0", new_state)],
            finished_req_ids={"r0"},
            num_speculative_tokens=0,
        )
        assert proposer.propose(disabled_ctx) is None  # K=0, nothing drafted
        assert "r0" not in proposer._cooldown  # cleanup must still happen
        assert "r0" not in proposer._miss_streak
        assert "r0" not in proposer._row_of

        # A later, enabled step must evaluate the new request fresh.
        enabled_ctx = _context(decode_reqs=[("r0", new_state)])
        with patch.object(
            proposer._ngram, "propose", return_value=[[42]]
        ) as mock_propose:
            drafts = proposer.propose(enabled_ctx)
        assert drafts is not None
        mock_propose.assert_called_once()

    def test_pending_not_resolved_for_unscheduled_live_request(self) -> None:
        """A request can be alive in request_states without being scheduled
        for decode this step (still mid-prefill, paused, preempted). A
        pending draft for it must not be scored as a miss just because it
        wasn't verified this particular step."""
        proposer = _proposer(prompt_lookup_min=2, prompt_lookup_max=3)
        state = _request_state([1, 2, 3])
        ctx = _context(decode_reqs=[("r0", state)])

        with patch.object(proposer._ngram, "propose", return_value=[[99]]):
            drafts = proposer.propose(ctx)
        assert drafts is not None
        assert "r0" in proposer._pending

        # Next step: r0 is still alive but not scheduled for decode.
        unscheduled_ctx = _context(
            decode_reqs=[],
            request_states={"r0": state},
        )
        proposer.propose(unscheduled_ctx)

        # The pending draft must survive untouched, not be scored as a miss.
        assert "r0" in proposer._pending
        assert proposer._miss_streak.get("r0", 0) == 0

    def test_pending_not_resolved_when_segment_carries_no_draft(self) -> None:
        """A request can be in decode_reqs with an empty decode_segment
        draft_token_ids -- no draft was scheduled for it at all, or the
        scheduler padded/dropped a proposed draft on batch admission (see
        SpeculativeDecodeController.active_spec_decode_tokens). Either way,
        an older pending draft for it must not be scored just because the
        request itself shows up in decode_reqs again."""
        proposer = _proposer(prompt_lookup_min=2, prompt_lookup_max=3)
        state = _request_state([1, 2, 3])
        ctx = _context(
            decode_reqs=[("r0", state)],
            decode_segments=[_verified_segment("r0")],
        )

        with patch.object(proposer._ngram, "propose", side_effect=[[[99]], [[]]]):
            drafts = proposer.propose(ctx)
            assert drafts is not None
            assert proposer._pending["r0"] == (3, (99,))

            # Next step: r0 is still in decode_reqs, but this step's
            # segment carries no draft at all.
            state.token_ids.append(123)
            plain_ctx = _context(
                decode_reqs=[("r0", state)],
                decode_segments=[_verified_segment("r0", has_draft=False)],
            )
            proposer.propose(plain_ctx)

        # The pending draft from the verified step must survive untouched.
        assert proposer._pending["r0"] == (3, (99,))

    def test_release_frees_rows_but_preserves_throttle_and_pending(self) -> None:
        proposer = _proposer(max_num_seqs=2)
        ctx = _context(decode_reqs=[("miss", _request_state([7, 8, 9]))])
        for _ in range(ngram_mod._MAX_CONSECUTIVE_MISSES):
            proposer.propose(ctx)
        matched = _request_state([1, 2, 3, 1, 2, 3, 1, 2])
        proposer.propose(_context(decode_reqs=[("match", matched)]))
        before = (
            dict(proposer._miss_streak),
            dict(proposer._cooldown),
            dict(proposer._pending),
        )
        assert all(before)

        # Reconciliation can release an ID at both preemption and resume.
        for _ in range(2):
            proposer.release_requests({"miss", "match", "unknown"})
        assert (proposer._miss_streak, proposer._cooldown, proposer._pending) == before
        assert not proposer._row_of
        assert sorted(proposer._free_rows) == [0, 1]
        assert all(source is None for source in proposer._row_source)

        with patch.object(proposer._ngram, "propose") as mock_propose:
            assert proposer.propose(ctx) is None
        mock_propose.assert_not_called()  # Resume must keep the cooldown.


def _full_copy_drafts(
    kernel: VllmNgramProposer,
    k: int,
    histories: dict[str, list[int]],
    max_model_len: int,
) -> dict[str, list[int]]:
    """Copy full histories into a fresh batch and call the upstream proposer."""
    req_ids = list(histories)
    token_ids_cpu = np.zeros((len(req_ids), max_model_len), dtype=np.int32)
    for i, req_id in enumerate(req_ids):
        token_ids_cpu[i, : len(histories[req_id])] = histories[req_id]
    num_tokens_no_spec = np.array(
        [len(histories[req_id]) for req_id in req_ids], dtype=np.int32
    )
    drafts = kernel.propose(k, [[0]] * len(req_ids), num_tokens_no_spec, token_ids_cpu)
    return {req_id: d for req_id, d in zip(req_ids, drafts, strict=True) if d}


def _proposed(drafts: DraftTokenIds | None) -> dict[str, list[int]]:
    if drafts is None:
        return {}
    return dict(zip(drafts.req_ids, drafts.draft_token_ids, strict=True))


class TestNgramIncrementalHistory:
    """Persistent histories must match fresh copies without recopying old tokens."""

    @pytest.mark.parametrize("seed,k", [(0, 3), (1, 5)])
    def test_appended_histories_match_full_copy_across_reordered_batches(
        self, monkeypatch: pytest.MonkeyPatch, seed: int, k: int
    ) -> None:
        accepted_counts = (0, 1, k, 2) * 4
        monkeypatch.setattr(
            ngram_mod, "_MAX_CONSECUTIVE_MISSES", len(accepted_counts) + 1
        )
        max_model_len, max_num_seqs = 128, 4
        rng = random.Random(seed)
        proposer = _proposer(
            num_speculative_tokens=k,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
        )
        reference = VllmNgramProposer(
            _vllm_config(2, 3, k, max_model_len, max_num_seqs)
        )
        states: dict[str, SimpleNamespace] = {}
        for i in range(max_num_seqs):
            pattern = rng.choices(range(6), k=6)
            states[f"r{i}"] = _request_state(pattern * 3 + pattern[:2])
        previous_drafts: dict[str, list[int]] = {}

        for accepted_count in accepted_counts:
            scheduled = rng.sample(list(states.items()), rng.randint(1, max_num_seqs))
            for req_id, state in scheduled:
                state.token_ids.extend(previous_drafts.get(req_id, [])[:accepted_count])
                state.token_ids.append(rng.randrange(6))
            histories = {req_id: list(state.token_ids) for req_id, state in scheduled}
            ctx = _context(
                decode_reqs=scheduled, request_states=states, num_speculative_tokens=k
            )

            actual = _proposed(proposer.propose(ctx))
            expected = _full_copy_drafts(reference, k, histories, max_model_len)

            assert actual == expected
            assert list(actual) == list(expected)
            assert {req_id: state.token_ids for req_id, state in scheduled} == histories
            previous_drafts.update(
                {req_id: actual.get(req_id, []) for req_id, _ in scheduled}
            )

    def test_only_new_tokens_are_copied(self) -> None:
        proposer = _proposer()
        state = _request_state([1, 2, 3, 1, 2, 3, 1, 2])
        proposer.propose(_context(decode_reqs=[("r0", state)]))
        row = proposer._row_of["r0"]
        # Existing tokens and unused capacity must stay untouched.
        proposer._token_ids_cpu[row, 0] = 777
        proposer._token_ids_cpu[row, 10:] = 888
        state.token_ids.extend([3, 1])

        proposer.propose(_context(decode_reqs=[("r0", state)]))

        assert proposer._token_ids_cpu[row, 0] == 777
        assert proposer._token_ids_cpu[row, 8:10].tolist() == [3, 1]
        assert proposer._row_len[row] == 10
        assert np.all(proposer._token_ids_cpu[row, 10:] == 888)

    @pytest.mark.parametrize(
        "indices,expected,expected_sampled,expected_lengths",
        [
            ([3], {"r3": [2, 3, 1]}, [[0]], [8]),
            ([3, 1], {"r3": [2, 3, 1], "r1": [2, 1, 1]}, [[0], [], [0]], [8, 0, 8]),
        ],
        ids=["single-row", "reordered-sparse-rows"],
    )
    def test_active_rows_share_storage_and_preserve_request_order(
        self,
        indices: list[int],
        expected: dict[str, list[int]],
        expected_sampled: list[list[int]],
        expected_lengths: list[int],
    ) -> None:
        proposer = _proposer()
        states = [(f"r{i}", _request_state([i, 1, 2, i, 1, 2, i, 1])) for i in range(4)]
        proposer.propose(_context(decode_reqs=states))
        scheduled = [states[index] for index in indices]

        with patch.object(
            proposer._ngram, "propose", wraps=proposer._ngram.propose
        ) as propose:
            actual = _proposed(proposer.propose(_context(decode_reqs=scheduled)))

        assert actual == expected
        assert list(actual) == list(expected)
        _, sampled, lengths, tokens = propose.call_args.args
        assert sampled == expected_sampled
        assert lengths.tolist() == expected_lengths
        assert tokens.shape == (len(expected_lengths), 512)
        assert np.shares_memory(tokens, proposer._token_ids_cpu)

    def test_released_request_rebuilds_history_on_resume(self) -> None:
        proposer = _proposer()
        state = _request_state([1, 2, 3, 1, 2, 3, 1, 2])
        proposer.propose(_context(decode_reqs=[("r0", state)]))
        proposer.release_requests({"r0"})
        state.token_ids = state.token_ids[:-1]

        actual = _proposed(proposer.propose(_context(decode_reqs=[("r0", state)])))

        assert actual == {"r0": [2, 3, 1]}
        row = proposer._row_of["r0"]
        assert proposer._token_ids_cpu[row, :7].tolist() == state.token_ids

    def test_streaming_continuation_replaces_history_without_release(self) -> None:
        proposer = _proposer()
        state = _request_state([1, 2, 3, 1, 2, 3, 1, 9])
        assert proposer.propose(_context(decode_reqs=[("r0", state)])) is None
        # Streaming drops the last sampled output before adding the next input.
        continued = _request_state(state.token_ids[:-1] + [2])

        actual = _proposed(proposer.propose(_context(decode_reqs=[("r0", continued)])))

        assert actual == {"r0": [3, 1, 2]}

    @pytest.mark.parametrize("new_id", ["r0", "r1"])
    def test_finished_row_is_reused_without_stale_tail(self, new_id: str) -> None:
        proposer = _proposer()
        long_state = _request_state([9, 8, 9, 8, 9, 8, 9, 8, 7, 7, 7, 7])
        proposer.propose(_context(decode_reqs=[("r0", long_state)]))
        row = proposer._row_of["r0"]
        # r0 finishes; a new request takes its row with a shorter history.
        short_state = _request_state([1, 2, 1, 2])

        drafts = proposer.propose(
            _context(decode_reqs=[(new_id, short_state)], finished_req_ids={"r0"})
        )

        assert proposer._row_of == {new_id: row}
        assert _proposed(drafts) == {new_id: [1, 2]}

    def test_rows_reclaimed_when_live_requests_exceed_capacity(self) -> None:
        proposer = _proposer(max_num_seqs=2)
        states = {f"r{i}": _request_state([i, 1, 2, i, 1, 2, i, 1]) for i in range(3)}
        # Three live requests, never more than two drafting in one step.
        for pair in (("r0", "r1"), ("r2", "r0"), ("r1", "r2"), ("r0", "r1")):
            retained = {r: proposer._row_of[r] for r in pair if r in proposer._row_of}
            ctx = _context(decode_reqs=[(r, states[r]) for r in pair])
            assert _proposed(proposer.propose(ctx)) == {
                r: [2, int(r[1]), 1] for r in pair
            }
            # A newcomer must not evict an owner later in drafting order.
            assert all(proposer._row_of[r] == row for r, row in retained.items())
        assert len(proposer._row_of) == 2

    @pytest.mark.parametrize("length,expected", [(7, {"r0": [2]}), (8, {}), (9, {})])
    def test_history_near_max_model_len_is_bounded(
        self, length: int, expected: dict[str, list[int]]
    ) -> None:
        proposer = _proposer(max_model_len=8)
        state = _request_state(([1, 2, 3] * 4)[:length])

        drafts = proposer.propose(_context(decode_reqs=[("r0", state)]))

        assert _proposed(drafts) == expected
        assert proposer._row_len[proposer._row_of["r0"]] == min(length, 8)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
