# SPDX-License-Identifier: Apache-2.0
"""Tests for the Metal n-gram speculative-decode proposer.

The proposer wraps vLLM's pure-Python/Numba n-gram kernel, so these tests need no
model or engine: a ``SimpleNamespace`` ``vllm_config`` exercises the upstream
constructor (which reads only scalar config) and a hand-built ``ProposeContext``
drives ``propose``. They lock in the request filtering (greedy-only, skip
intermediate prefills) and the array marshalling into the upstream kernel.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from vllm.sampling_params import SamplingParams

from vllm_metal.v1 import ngram_proposer as ngram_mod
from vllm_metal.v1.ngram_proposer import NgramProposer
from vllm_metal.v1.proposer import ProposeContext
from vllm_metal.v1.spec_decode import SpeculativeDecodeController


def _proposer(
    *,
    prompt_lookup_min: int = 2,
    prompt_lookup_max: int = 3,
    num_speculative_tokens: int = 3,
    max_model_len: int = 512,
) -> NgramProposer:
    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            prompt_lookup_min=prompt_lookup_min,
            prompt_lookup_max=prompt_lookup_max,
            num_speculative_tokens=num_speculative_tokens,
        ),
        model_config=SimpleNamespace(max_model_len=max_model_len),
        scheduler_config=SimpleNamespace(max_num_seqs=4),
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
    )
    return NgramProposer(
        vllm_config=vllm_config,
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
    prefill_reqs: list[SimpleNamespace] | None = None,
    prefill_result_modes: list[str] | None = None,
    request_states: dict[str, SimpleNamespace] | None = None,
    num_speculative_tokens: int = 3,
    finished_req_ids: set[str] | None = None,
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
        decode_segments=[],
        decode_token_ids=decode_token_ids,
        prefill_reqs=prefill_reqs,
        prefill_token_ids=[0] * len(prefill_reqs),
        prefill_result_modes=prefill_result_modes,
        request_states=request_states,
        cu_seqlens=[],
        num_decode_segments=len(decode_reqs),
        num_speculative_tokens=num_speculative_tokens,
        logitsprocs=None,
        finished_req_ids=finished_req_ids or set(),
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

    def test_empty_context_returns_none(self) -> None:
        assert _proposer().propose(_context()) is None

    def test_skips_request_without_sampled_tokens(self) -> None:
        proposer = _proposer(prompt_lookup_min=2)
        state = _request_state([1, 2, 3, 1, 2, 3, 1, 2])
        # An empty sampled-ids entry marks a row that did not decode this step.
        ctx = _context(decode_reqs=[("r0", state)], decode_token_ids=[[]])

        assert proposer.propose(ctx) is None

    def test_skips_non_greedy_request(self) -> None:
        proposer = _proposer(prompt_lookup_min=2)
        state = _request_state([1, 2, 3, 1, 2, 3, 1, 2], temperature=0.8)
        ctx = _context(decode_reqs=[("r0", state)])

        assert proposer.propose(ctx) is None

    def test_finalized_prefill_participates(self) -> None:
        proposer = _proposer(prompt_lookup_min=2)
        state = _request_state([4, 5, 6, 4, 5, 6, 4, 5])
        prefill = SimpleNamespace(req_id="p0")
        ctx = _context(
            prefill_reqs=[prefill],
            prefill_result_modes=["new_final"],
            request_states={"p0": state},
        )

        drafts = proposer.propose(ctx)

        assert drafts is not None
        assert drafts.req_ids == ["p0"]
        assert drafts.draft_token_ids == [[6, 4, 5]]

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
        ctx = _context(decode_reqs=[("r0", state)])

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
        ctx = _context(decode_reqs=[("r0", state)])

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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
