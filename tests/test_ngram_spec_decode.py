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

import pytest
from vllm.sampling_params import SamplingParams

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
        logitsprocs=None,
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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
