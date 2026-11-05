# SPDX-License-Identifier: Apache-2.0
"""Tests for the Metal suffix-decoding proposer.

The proposer wraps vLLM's ``SuffixDecodingProposer``, whose engine-side loop
runs entirely off an InputBatch-shaped stub, so most tests drive
``propose`` against a recording fake of that upstream class (no
``arctic_inference`` install needed) and lock in the marshalling: stub
fields, per-request deltas, non-drafting rows kept alive, preemption
rollback, width/cap/length trimming, prefill rows, and bookkeeping
lifecycle. ``TestSuffixRealTree`` exercises the real arctic suffix tree
(lifecycle across preemption and finish, verify-span round trip) when
the package is importable.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from vllm.sampling_params import SamplingParams

from tests.test_ngram_spec_decode import _context
from vllm_metal.v1 import suffix_proposer as suffix_mod
from vllm_metal.v1.spec_decode import SpeculativeDecodeController
from vllm_metal.v1.suffix_proposer import SuffixProposer


class _FakeVllmSuffix:
    """Records upstream-facing calls; drafts mirror the fed deltas."""

    def __init__(self, num_speculative_tokens: int = 3) -> None:
        self.num_speculative_tokens = num_speculative_tokens
        self.calls: list[tuple[int, object, list[list[int]]]] = []
        self.extra_draft: dict[int, list[int]] = {}
        self.suffix_cache = _FakeSuffixCache()

    def propose(self, num_speculative_tokens, input_batch, sampled_token_ids):
        # Upstream asserts the configured width.
        assert num_speculative_tokens == self.num_speculative_tokens
        self.calls.append(
            (num_speculative_tokens, input_batch, [list(s) for s in sampled_token_ids])
        )
        drafts: list[list[int]] = []
        for i, sampled in enumerate(sampled_token_ids):
            if not sampled:
                drafts.append([])
                continue
            drafts.append(self.extra_draft.get(i, sampled))
        return drafts


class _FakeSuffixCache:
    """Records the archive-time history feed a real cache would receive."""

    def __init__(self) -> None:
        self.added: list[tuple[str, list[int]]] = []
        self.stopped: list[str] = []
        self.active_requests: set[str] = set()

    def add_active_response(self, req_id: str, sampled_ids: list[int]) -> None:
        self.added.append((req_id, list(sampled_ids)))

    def stop_request(self, req_id: str) -> None:
        self.active_requests.discard(req_id)
        self.stopped.append(req_id)


def _request_state(
    token_ids: list[int],
    *,
    prompt_len: int,
    temperature: float = 0.0,
) -> SimpleNamespace:
    return SimpleNamespace(
        token_ids=list(token_ids),
        prompt_len=prompt_len,
        sampling_params=SamplingParams(temperature=temperature),
        generated_tokens=1,
        block_ids=[[0]],
    )


def _real_proposer(*, num_speculative_tokens: int = 4) -> SuffixProposer:
    pytest.importorskip(
        "arctic_inference.suffix_decoding", reason="arctic_inference not installed"
    )
    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            method="suffix",
            num_speculative_tokens=num_speculative_tokens,
            suffix_decoding_max_tree_depth=24,
            suffix_decoding_max_spec_factor=1.0,
            suffix_decoding_min_token_prob=0.0,
            suffix_decoding_max_cached_requests=10000,
        ),
        model_config=SimpleNamespace(max_model_len=512),
        scheduler_config=SimpleNamespace(max_num_seqs=4),
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
    )
    return SuffixProposer.build(
        vllm_config=vllm_config, controller=SpeculativeDecodeController()
    )


def _proposer(
    fake: _FakeVllmSuffix, *, num_speculative_tokens: int = 3
) -> SuffixProposer:
    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            method="suffix",
            num_speculative_tokens=num_speculative_tokens,
            suffix_decoding_max_tree_depth=24,
            suffix_decoding_max_spec_factor=1.0,
            suffix_decoding_min_token_prob=0.1,
            suffix_decoding_max_cached_requests=10000,
        ),
        model_config=SimpleNamespace(max_model_len=512),
        scheduler_config=SimpleNamespace(max_num_seqs=4),
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
    )
    fake.num_speculative_tokens = num_speculative_tokens
    with patch.object(suffix_mod, "_build_vllm_suffix", return_value=fake):
        return SuffixProposer.build(
            vllm_config=vllm_config,
            controller=SpeculativeDecodeController(),
        )


class TestSuffixMarshalling:
    def test_stub_carries_prompt_split_and_history(self) -> None:
        fake = _FakeVllmSuffix()
        proposer = _proposer(fake)
        state = _request_state([5, 6, 7, 8, 9, 10, 11], prompt_len=4)

        drafts = proposer.propose(_context(decode_reqs=[("r0", state)]))

        assert drafts is not None
        num_spec, stub, sampled = fake.calls[0]
        assert num_spec == 3  # the configured width, satisfying upstream's assert
        assert stub.req_ids == ["r0"]
        assert stub.req_id_to_index == {"r0": 0}
        assert stub.num_tokens_no_spec.tolist() == [7]
        assert stub.num_prompt_tokens.tolist() == [4]
        assert stub.token_ids_cpu[0, :7].tolist() == [5, 6, 7, 8, 9, 10, 11]
        # First sighting: the delta is everything past the prompt.
        assert sampled == [[9, 10, 11]]
        # The fake echoes the delta back as the draft.
        assert drafts.draft_token_ids == [[9, 10, 11]]

    def test_second_step_feeds_only_the_delta(self) -> None:
        fake = _FakeVllmSuffix()
        proposer = _proposer(fake)
        state0 = _request_state([1, 2, 3, 4, 5], prompt_len=3)
        proposer.propose(_context(decode_reqs=[("r0", state0)]))

        state1 = _request_state([1, 2, 3, 4, 5, 6, 7], prompt_len=3)
        proposer.propose(_context(decode_reqs=[("r0", state1)]))

        _, _, sampled = fake.calls[1]
        assert sampled == [[6, 7]]

    def test_nondrafting_rows_stay_in_the_stub_but_get_no_draft(self) -> None:
        fake = _FakeVllmSuffix()
        proposer = _proposer(fake)
        greedy = _request_state([1, 2, 3, 4], prompt_len=2)
        sampled_req = _request_state([5, 6, 7, 8], prompt_len=2, temperature=1.0)

        drafts = proposer.propose(
            _context(decode_reqs=[("greedy", greedy), ("sampled", sampled_req)])
        )

        _, stub, sampled = fake.calls[0]
        # Both scheduled requests are rows (upstream archives any active
        # tree missing from the batch), but only the greedy row carries a
        # delta; the sampled row never starts a tree.
        assert stub.req_ids == ["greedy", "sampled"]
        assert sampled == [[3, 4], []]
        assert drafts is not None
        assert drafts.req_ids == ["greedy"]
        assert drafts.draft_token_ids == [[3, 4]]

    def test_preemption_rollback_clamps_the_delta(self) -> None:
        fake = _FakeVllmSuffix()
        proposer = _proposer(fake)
        state0 = _request_state([1, 2, 3, 4, 5, 6], prompt_len=2)
        proposer.propose(_context(decode_reqs=[("r0", state0)]))

        # Recomputed shorter than what was already fed (without a release
        # in between): the anchor resets and the delta is empty rather than
        # negative.
        state1 = _request_state([1, 2, 3, 4], prompt_len=2)
        drafts = proposer.propose(_context(decode_reqs=[("r0", state1)]))
        _, _, sampled = fake.calls[1]
        assert sampled == [[]]
        assert drafts is None

    def test_draft_caps_trim(self) -> None:
        fake = _FakeVllmSuffix()
        proposer = _proposer(fake)
        state = _request_state([1, 2, 3, 4, 5], prompt_len=2)

        ctx = replace(_context(decode_reqs=[("r0", state)]), draft_caps={"r0": 1})
        drafts = proposer.propose(ctx)

        assert drafts is not None
        assert len(drafts.draft_token_ids[0]) == 1

    def test_finished_ids_and_release_clear_bookkeeping(self) -> None:
        fake = _FakeVllmSuffix()
        fake.suffix_cache.active_requests.update({"r0", "r1"})
        proposer = _proposer(fake)
        state0 = _request_state([1, 2, 3, 4, 5], prompt_len=2)
        proposer.propose(_context(decode_reqs=[("r0", state0)]))
        assert proposer._last_len == {"r0": 5}

        proposer.propose(_context(decode_reqs=[], finished_req_ids={"r0"}))
        assert "r0" not in proposer._last_len

        proposer.propose(_context(decode_reqs=[("r1", state0)]))
        proposer.release_requests({"r1"})
        assert "r1" not in proposer._last_len
        assert fake.suffix_cache.stopped == ["r0", "r1"]
        assert not fake.suffix_cache.active_requests

    def test_disabled_or_empty_drafting_returns_none(self) -> None:
        fake = _FakeVllmSuffix()
        proposer = _proposer(fake)
        state = _request_state([1, 2, 3], prompt_len=1)

        assert (
            proposer.propose(
                _context(decode_reqs=[("r0", state)], num_speculative_tokens=0)
            )
            is None
        )
        # Nothing eligible (non-greedy only): no upstream call at all.
        sampled_req = _request_state([1, 2, 3], prompt_len=1, temperature=1.0)
        assert proposer.propose(_context(decode_reqs=[("s", sampled_req)])) is None
        assert fake.calls == []

    def test_empty_draft_rows_are_dropped(self) -> None:
        fake = _FakeVllmSuffix()
        fake.extra_draft = {0: []}  # the tree has nothing for this suffix
        proposer = _proposer(fake)
        state = _request_state([9, 9, 9], prompt_len=1)

        assert proposer.propose(_context(decode_reqs=[("r0", state)])) is None

    def test_output_capped_row_still_feeds_its_delta(self) -> None:
        fake = _FakeVllmSuffix()
        proposer = _proposer(fake)
        capped = _request_state([1, 2, 3, 4], prompt_len=2)
        other = _request_state([9, 9, 9], prompt_len=1)
        proposer.propose(_context(decode_reqs=[("c", capped), ("o", other)]))
        assert proposer._last_len == {"c": 4, "o": 3}

        # The capped request reaches its output budget: it is no longer
        # draft-eligible, but its row is still in the batch and its live
        # tree must still learn the tokens committed since the last feed.
        capped.token_ids += [5, 6]
        capped.sampling_params = SamplingParams(temperature=0.0, max_tokens=2)
        capped.generated_tokens = 2
        other.token_ids += [7]

        drafts = proposer.propose(_context(decode_reqs=[("c", capped), ("o", other)]))
        _, stub, sampled = fake.calls[-1]
        assert stub.req_ids == ["c", "o"]
        assert sampled == [[5, 6], [7]]
        assert drafts is not None
        assert drafts.req_ids == ["o"]  # the capped row drafts nothing
        assert drafts.draft_token_ids == [[7]]
        assert proposer._last_len == {"c": 6, "o": 4}

    def test_finish_feeds_final_segment_before_archiving(self) -> None:
        fake = _FakeVllmSuffix()
        cache = _FakeSuffixCache()
        fake.suffix_cache = cache
        cache.active_requests.add("r0")
        proposer = _proposer(fake)
        state = _request_state([1, 2, 3, 4], prompt_len=2)
        proposer.propose(_context(decode_reqs=[("r0", state)]))

        # The final verify commits two more tokens, then the request
        # finishes without ever drafting again.
        state.token_ids += [5, 6]
        proposer.release_requests({"r0"})

        assert cache.added == [("r0", [5, 6])]
        assert cache.stopped == ["r0"]
        assert proposer._last_len == {}
        assert proposer._states == {}

    def test_reused_request_id_feeds_from_its_own_state(self) -> None:
        fake = _FakeVllmSuffix()
        cache = _FakeSuffixCache()
        fake.suffix_cache = cache
        cache.active_requests.add("r0")
        proposer = _proposer(fake)
        state = _request_state([1, 2, 3, 4], prompt_len=2)
        proposer.propose(_context(decode_reqs=[("r0", state)]))

        # The engine hands the finished id straight to a new request whose
        # history is one token longer: the tail feed must use the held
        # (finished) state, not request_states' replacement under the id.
        state.token_ids += [5, 6]
        reused = _request_state([1, 2, 3], prompt_len=2)
        proposer.propose(
            _context(
                decode_reqs=[("r0", reused)],
                request_states={"r0": reused},
                finished_req_ids={"r0"},
            )
        )
        assert cache.added == [("r0", [5, 6])]


@pytest.mark.suffix
class TestSuffixRealTree:
    def test_repetitive_history_drafts_continuation(self) -> None:
        pytest.importorskip(
            "arctic_inference.suffix_decoding", reason="arctic_inference not installed"
        )
        vllm_config = SimpleNamespace(
            speculative_config=SimpleNamespace(
                method="suffix",
                num_speculative_tokens=4,
                suffix_decoding_max_tree_depth=24,
                suffix_decoding_max_spec_factor=1.0,
                suffix_decoding_min_token_prob=0.0,
                suffix_decoding_max_cached_requests=10000,
            ),
            model_config=SimpleNamespace(max_model_len=512),
            scheduler_config=SimpleNamespace(max_num_seqs=4),
            parallel_config=SimpleNamespace(tensor_parallel_size=1),
        )
        proposer = SuffixProposer.build(
            vllm_config=vllm_config,
            controller=SpeculativeDecodeController(),
        )

        # The history contains [10, 11, 12] -> [13, 14] (-> 10, the repeat)
        # and the request has just re-emitted [10, 11, 12]: the tree should
        # speculate the earlier continuation, as deep as the matched length
        # allows under max_spec_factor=1.0.
        state = _request_state([1, 2, 10, 11, 12, 13, 14, 10, 11, 12], prompt_len=7)
        drafts = proposer.propose(_context(decode_reqs=[("r0", state)]))

        assert drafts is not None
        assert drafts.draft_token_ids == [[13, 14, 10]]

        # A request whose suffix never appeared drafts nothing.
        fresh = _request_state([31, 32, 33, 34], prompt_len=2)
        assert proposer.propose(_context(decode_reqs=[("r1", fresh)])) is None


class TestSuffixBatchShape:
    def test_prefill_rows_are_in_the_stub_and_final_chunk_drafts(self) -> None:
        fake = _FakeVllmSuffix()
        proposer = _proposer(fake)
        decoding = _request_state([1, 2, 3, 4], prompt_len=2)
        final = _request_state([7, 8, 9, 10, 11], prompt_len=4)
        chunk = _request_state([20, 21, 22], prompt_len=6)
        prefills = [
            SimpleNamespace(req_id="final", start_pos=0, token_ids=[7, 8, 9, 10]),
            SimpleNamespace(req_id="chunk", start_pos=0, token_ids=[20, 21, 22]),
        ]

        drafts = proposer.propose(
            _context(
                decode_reqs=[("dec", decoding)],
                prefill_reqs=prefills,
                prefill_result_modes=["new_final", "intermediate"],
                request_states={"dec": decoding, "final": final, "chunk": chunk},
            )
        )

        _, stub, sampled = fake.calls[0]
        assert stub.req_ids == ["dec", "final", "chunk"]
        # The final chunk feeds its first sampled token; the intermediate
        # chunk is present (keeps any live tree) with no delta.
        assert sampled == [[3, 4], [11], []]
        assert drafts is not None
        assert drafts.req_ids == ["dec", "final"]

    def test_scheduled_width_below_configured_truncates(self) -> None:
        fake = _FakeVllmSuffix()
        fake.extra_draft = {0: [1, 2, 3, 4]}
        proposer = _proposer(fake, num_speculative_tokens=4)
        state = _request_state([1, 2, 3, 4, 5], prompt_len=2)

        drafts = proposer.propose(
            _context(decode_reqs=[("r0", state)], num_speculative_tokens=2)
        )

        assert fake.calls[0][0] == 4  # configured width to upstream
        assert drafts is not None
        assert drafts.draft_token_ids == [[1, 2]]

    def test_remaining_output_tokens_bound_the_draft(self) -> None:
        fake = _FakeVllmSuffix()
        fake.extra_draft = {0: [1, 2, 3]}
        proposer = _proposer(fake)
        state = _request_state([1, 2, 3, 4, 5], prompt_len=2)
        state.sampling_params = SamplingParams(temperature=0.0, max_tokens=4)
        state.generated_tokens = 2

        drafts = proposer.propose(_context(decode_reqs=[("r0", state)]))
        assert drafts is not None
        assert drafts.draft_token_ids == [[1]]

        state.generated_tokens = 3
        assert proposer.propose(_context(decode_reqs=[("r0", state)])) is None

    def test_absent_request_is_refed_from_the_prompt(self) -> None:
        fake = _FakeVllmSuffix()
        proposer = _proposer(fake)
        other = _request_state([9, 9, 9], prompt_len=1)
        state = _request_state([1, 2, 3, 4, 5], prompt_len=2)
        proposer.propose(_context(decode_reqs=[("r0", state), ("o", other)]))
        # r0 missing for a step: upstream archives its tree.
        proposer.propose(_context(decode_reqs=[("o", other)]))
        assert "r0" not in proposer._last_len

        state = _request_state([1, 2, 3, 4, 5, 6], prompt_len=2)
        proposer.propose(_context(decode_reqs=[("r0", state), ("o", other)]))
        assert fake.calls[-1][2][0] == [3, 4, 5, 6]


@pytest.mark.suffix
class TestSuffixRealLifecycle:
    _PROMPT = [1, 2, 3, 4, 5, 6, 7, 8]
    _RESPONSE = [20, 21, 22, 23, 24, 25]

    def test_preempted_then_resumed_request_keeps_its_response(self) -> None:
        proposer = _real_proposer()
        cache = proposer._vllm.suffix_cache
        history = self._PROMPT + self._RESPONSE
        proposer.propose(
            _context(decode_reqs=[("b", _request_state(history, prompt_len=8))])
        )

        # Preempted: released (archived) and absent while waiting.
        proposer.release_requests({"b"})
        assert "b" not in cache.active_requests
        assert "b" in cache.cached_requests

        # Resumed: the recompute's final prefill chunk samples [20] again.
        resumed = _request_state(history + [20], prompt_len=8)
        drafts = proposer.propose(
            _context(
                prefill_reqs=[
                    SimpleNamespace(req_id="b", start_pos=0, token_ids=history)
                ],
                request_states={"b": resumed},
            )
        )
        assert drafts is not None
        assert drafts.draft_token_ids[0][0] == 21

    def test_finish_archives_immediately(self) -> None:
        proposer = _real_proposer()
        cache = proposer._vllm.suffix_cache
        state = _request_state(self._PROMPT + [1, 2], prompt_len=8)
        proposer.propose(_context(decode_reqs=[("c", state)]))
        assert "c" in cache.active_requests

        proposer.release_requests({"c"})

        assert "c" not in cache.active_requests
        assert "c" in cache.cached_requests
        assert "c" not in proposer._last_len

    def test_final_committed_token_reaches_archived_response(self) -> None:
        proposer = _real_proposer()
        state = _request_state(self._PROMPT + [20, 21], prompt_len=8)
        state.sampling_params = SamplingParams(temperature=0.0, max_tokens=3)
        state.generated_tokens = 2
        proposer.propose(_context(decode_reqs=[("r0", state)]))

        # The final verify commits 22, the budget is exhausted, and the
        # request finishes -- it never drafts again. Regression: the last
        # committed token used to miss the tree, so a later request
        # re-emitting [20, 21] could not speculate 22.
        state.token_ids.append(22)
        state.generated_tokens = 3
        proposer.propose(_context(decode_reqs=[], finished_req_ids={"r0"}))

        nxt = _request_state(self._PROMPT + [20, 21], prompt_len=8)
        drafts = proposer.propose(_context(decode_reqs=[("r1", nxt)]))
        assert drafts is not None
        assert drafts.draft_token_ids == [[22]]

    def test_final_multi_token_commit_reaches_archived_response(self) -> None:
        proposer = _real_proposer()
        state = _request_state(self._PROMPT + [20, 21], prompt_len=8)
        proposer.propose(_context(decode_reqs=[("m", state)]))

        # The last verify round accepts a deep draft plus the bonus token
        # before the request finishes.
        state.token_ids += [22, 23, 24]
        proposer.release_requests({"m"})

        nxt = _request_state(self._PROMPT + [20, 21, 22], prompt_len=8)
        drafts = proposer.propose(_context(decode_reqs=[("n", nxt)]))
        assert drafts is not None
        assert drafts.draft_token_ids == [[23, 24]]

    @pytest.mark.parametrize("finish_path", ["release", "finished"])
    def test_context_limit_preserves_the_final_committed_token(
        self, finish_path: str
    ) -> None:
        proposer = _real_proposer()
        prompt = list(range(1000, 1509))
        state = _request_state(prompt + [20, 21], prompt_len=len(prompt))
        state.sampling_params = SamplingParams(temperature=0.0, max_tokens=3)
        state.generated_tokens = 2
        other = _request_state([90, 91], prompt_len=1)
        batch = [("producer", state), ("other", other)]
        proposer.propose(_context(decode_reqs=batch))

        # Another drafting row keeps the upstream loop running, but upstream
        # skips this row before ingestion once its history reaches 512.
        state.token_ids.append(22)
        state.generated_tokens = 3
        proposer.propose(_context(decode_reqs=batch))
        if finish_path == "release":
            proposer.release_requests({"producer", "other"})
        else:
            proposer.propose(_context(finished_req_ids={"producer", "other"}))

        consumer = _request_state([90, 20, 21], prompt_len=1)
        drafts = proposer.propose(_context(decode_reqs=[("consumer", consumer)]))
        assert drafts is not None
        assert drafts.draft_token_ids == [[22]]

    def test_absent_request_archives_tokens_committed_while_drafting_disabled(
        self,
    ) -> None:
        proposer = _real_proposer()
        state = _request_state(self._PROMPT + [20, 21], prompt_len=8)
        proposer.propose(_context(decode_reqs=[("producer", state)]))
        state.token_ids.append(22)
        proposer.propose(
            _context(decode_reqs=[("producer", state)], num_speculative_tokens=0)
        )

        # Requests can be absent without finishing or being preempted. Flush
        # pending history before upstream archives the missing active tree.
        other = _request_state([90, 91], prompt_len=1)
        proposer.propose(_context(decode_reqs=[("other", other)]))
        consumer = _request_state([90, 20, 21], prompt_len=1)
        drafts = proposer.propose(_context(decode_reqs=[("consumer", consumer)]))
        assert drafts is not None
        assert drafts.draft_token_ids == [[22]]

    def test_reused_request_id_does_not_inherit_the_old_tree(self) -> None:
        proposer = _real_proposer()
        cache = proposer._vllm.suffix_cache
        state = _request_state(self._PROMPT + [20, 21], prompt_len=8)
        proposer.propose(_context(decode_reqs=[("same", state)]))
        state.token_ids += [22, 23]
        proposer.release_requests({"same"})
        assert "same" in cache.cached_requests

        # The engine hands the finished id to a brand-new request: the old
        # archived tree is evicted (upstream keys archives by id) and the
        # drafts must come from the new request's own content alone.
        fresh = _request_state([30, 31, 32, 33, 30, 31], prompt_len=2)
        drafts = proposer.propose(_context(decode_reqs=[("same", fresh)]))
        assert drafts is not None
        assert drafts.draft_token_ids == [[32, 33]]
        assert proposer._last_len == {"same": 6}

    def test_final_prefill_chunk_drafts(self) -> None:
        proposer = _real_proposer()
        state = _request_state(self._PROMPT + [1], prompt_len=8)
        drafts = proposer.propose(
            _context(
                prefill_reqs=[
                    SimpleNamespace(req_id="a", start_pos=0, token_ids=self._PROMPT)
                ],
                request_states={"a": state},
            )
        )
        assert drafts is not None
        assert drafts.draft_token_ids[0][0] == 2

    def test_mixed_batch_variable_drafts_round_trip_through_verify(self) -> None:
        mx = pytest.importorskip("mlx.core")
        proposer = _real_proposer()
        controller = SpeculativeDecodeController()
        long_req = _request_state([1, 2, 10, 11, 12, 13, 14, 10, 11, 12], prompt_len=7)
        short_req = _request_state([5, 6, 7, 30, 5], prompt_len=4)
        sampled = _request_state([40, 41, 42], prompt_len=2, temperature=1.0)
        decode_reqs = [("long", long_req), ("short", short_req), ("s", sampled)]

        drafts = proposer.propose(_context(decode_reqs=decode_reqs))
        assert drafts is not None
        by_id = dict(zip(drafts.req_ids, drafts.draft_token_ids, strict=True))
        assert "s" not in by_id
        assert by_id["long"] == [13, 14, 10]
        assert len(by_id["short"]) != len(by_id["long"])

        # Next step: verify through the shared greedy span path. The target
        # accepts the first long draft then diverges; accepts all of short.
        greedy_reqs = decode_reqs[:2]
        segments = controller.build_decode_segments(
            greedy_reqs, by_id, {"long": 9, "short": 4}
        )
        vocab = 64
        rows = sum(seg.num_query_tokens for seg in segments)
        targets = [13, 50, 0, 0] + [*by_id["short"], 60]
        assert len(targets) == rows
        logits = mx.zeros((1, rows, vocab))
        logits[0, mx.arange(rows), mx.array(targets)] = 1.0
        out = controller.verify_greedy(logits, greedy_reqs, segments)
        assert out == [[13, 50], [*by_id["short"], 60]]

        # Commit the verified tokens; the next propose feeds exactly them.
        long_req.token_ids += out[0]
        short_req.token_ids += out[1]
        proposer.propose(_context(decode_reqs=greedy_reqs))
        assert proposer._last_len == {
            "long": len(long_req.token_ids),
            "short": len(short_req.token_ids),
        }
