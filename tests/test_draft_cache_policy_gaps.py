# SPDX-License-Identifier: Apache-2.0
"""Tests that the scheduler-managed draft KV cache honors the scheduler's
cache policy -- the gaps a proposer-private draft cache cannot see.

One property per gap named when the proposer-private draft cache was
rejected (see #500):

- ``cache_salt``: a resubmitted prompt under a different salt must not
  reuse draft KV.
- ``skip_reading_prefix_cache``: a request that opts out of cache reads
  must not reuse draft KV.
- Allocation limit: the proposer-local scratch reserve is sized to cover
  every concurrently active request drafting ``num_speculative_tokens``
  positions, so the draft pool cannot exhaust near the request limit.

The first two run end-to-end in a spawned child process (``spawn`` start
method -- Metal is not fork-safe) with Qwen3-0.6B draft==target, recording
each request's first draft plan (``draft_seq_len`` / ingest length) via a
monkeypatched proposer. The third is stub-level, no weights.

A fourth test (``test_chunked_cold_ingest_token_identity_e2e``) runs two
spawned children -- ``VLLM_METAL_SPEC_INGEST_CHUNK=16`` (chunked cold
ingest) vs ``0`` (single forward) -- and asserts the generated tokens are
identical, i.e. the chunk boundaries are lossless (#482 direction 3).
"""

from __future__ import annotations

import multiprocessing as mp
import os

import pytest
from vllm.utils.math_utils import cdiv

from tests.test_draft_model_proposer import (
    BLOCK_SIZE,
    _proposer,
    _StubDraftModel,
)
from tests.test_paged_deterministic import (
    DEFAULT_PAGED_MEMORY_FRACTION,
    MODEL_NAME,
)

K = 3
GEN = 8

SALT_PROMPT = (
    "Once upon a time in a far away kingdom there was a great king "
    "named Aragorn who ruled the land of Gondor with wisdom and "
    "grace, and his people loved him dearly because "
)
SKIP_PROMPT = (
    "The engineer examined the trace carefully, noting where the latency "
    "spiked and which subsystem held the lock the longest before yielding. "
)


def _setenv_default(key: str, default: str) -> None:
    if os.environ.get(key) is None:
        os.environ[key] = default


def _install_plan_logger(dmp, plans: list) -> None:
    """Record every draft plan as (draft_seq_len, n_ingest) into ``plans``."""
    if hasattr(dmp.DraftModelProposer, "_make_plan"):
        _orig_make_plan = dmp.DraftModelProposer._make_plan

        def _logged_make_plan(self, req_id, state, num_speculative_tokens):
            plan = _orig_make_plan(self, req_id, state, num_speculative_tokens)
            if plan is not None:
                plans.append((plan.draft_seq_len, len(plan.ingest_tokens)))
            return plan

        dmp.DraftModelProposer._make_plan = _logged_make_plan
    else:
        _orig_decode = dmp.DraftModelProposer._make_decode_plan
        _orig_prefill = dmp.DraftModelProposer._make_prefill_plan

        def _logged_decode(self, req_id, state, k, drafting_req_ids):
            plan = _orig_decode(self, req_id, state, k, drafting_req_ids)
            if plan is not None:
                plans.append((plan.draft_seq_len, len(plan.ingest_tokens)))
            return plan

        def _logged_prefill(self, prefill, result_mode, k, drafting_req_ids):
            plan = _orig_prefill(self, prefill, result_mode, k, drafting_req_ids)
            if plan is not None:
                plans.append((plan.draft_seq_len, len(plan.ingest_tokens)))
            return plan

        dmp.DraftModelProposer._make_decode_plan = _logged_decode
        dmp.DraftModelProposer._make_prefill_plan = _logged_prefill


def _run_policy_e2e() -> None:
    """Body of the e2e test -- runs in a spawned child process."""
    _setenv_default("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    _setenv_default("VLLM_METAL_MEMORY_FRACTION", DEFAULT_PAGED_MEMORY_FRACTION)

    from vllm import LLM, SamplingParams

    from vllm_metal.v1 import draft_model_proposer as dmp

    plans: list[tuple[int, int]] = []
    _install_plan_logger(dmp, plans)

    llm = LLM(
        model=MODEL_NAME,
        max_model_len=1024,
        max_num_seqs=1,
        enable_prefix_caching=True,
        async_scheduling=False,
        speculative_config={
            "method": "draft_model",
            "model": MODEL_NAME,
            "num_speculative_tokens": K,
        },
    )
    sp = SamplingParams(temperature=0, max_tokens=GEN)

    def first_plan(prompt, sampling_params) -> tuple[int, int] | None:
        plans.clear()
        llm.generate([prompt], sampling_params)
        return plans[0] if plans else None

    # cache_salt: same prompt, different salt must not reuse draft KV.
    # max_tokens=2 keeps the warm generate short while still running
    # propose(), which writes the draft group's KV.
    warm = SamplingParams(temperature=0, max_tokens=2)
    first_plan({"prompt": SALT_PROMPT, "cache_salt": "a"}, warm)
    hit_a = first_plan({"prompt": SALT_PROMPT, "cache_salt": "a"}, sp)
    miss_b = first_plan({"prompt": SALT_PROMPT, "cache_salt": "b"}, sp)
    if hit_a is None or hit_a[0] == 0:
        raise AssertionError(f"same cache_salt should reuse draft KV, got {hit_a}")
    if miss_b is None or miss_b[0] != 0:
        raise AssertionError(
            f"different cache_salt must not reuse draft KV, got {miss_b}"
        )
    tokens = list(
        llm.generate([{"prompt": SALT_PROMPT, "cache_salt": "a"}], sp)[0]
        .outputs[0]
        .token_ids
    )
    if tokens != list(
        llm.generate([{"prompt": SALT_PROMPT, "cache_salt": "b"}], sp)[0]
        .outputs[0]
        .token_ids
    ):
        raise AssertionError("cache_salt changed the generated tokens")

    # skip_reading_prefix_cache: opting out of cache reads must not reuse
    # draft KV, and must not change the generated tokens.
    first_plan(SKIP_PROMPT, warm)
    hit = first_plan(SKIP_PROMPT, sp)
    skip = first_plan(
        SKIP_PROMPT,
        SamplingParams(temperature=0, max_tokens=GEN, skip_reading_prefix_cache=True),
    )
    if hit is None or hit[0] == 0:
        raise AssertionError(f"default read should reuse draft KV, got {hit}")
    if skip is None or skip[0] != 0:
        raise AssertionError(
            f"skip_reading_prefix_cache must not reuse draft KV, got {skip}"
        )
    tokens = list(llm.generate([SKIP_PROMPT], sp)[0].outputs[0].token_ids)
    skipped = list(
        llm.generate(
            [SKIP_PROMPT],
            SamplingParams(
                temperature=0, max_tokens=GEN, skip_reading_prefix_cache=True
            ),
        )[0]
        .outputs[0]
        .token_ids
    )
    if tokens != skipped:
        raise AssertionError("skip_reading_prefix_cache changed the generated tokens")


@pytest.mark.slow
def test_draft_cache_honors_scheduler_cache_policy_e2e() -> None:
    ctx = mp.get_context("spawn")
    proc = ctx.Process(target=_run_policy_e2e)
    proc.start()
    proc.join()
    if proc.exitcode != 0:
        raise AssertionError(
            "Draft cache-policy e2e test failed in spawned child "
            f"(exit code: {proc.exitcode})"
        )


def _run_chunked_identity_e2e(chunk: str, result_q: mp.Queue) -> None:
    """Body of the chunked-ingest e2e -- runs in a spawned child process.

    Generates once on a fresh prompt with ``VLLM_METAL_SPEC_INGEST_CHUNK``
    set to ``chunk`` and reports (first-plan ingest length, token ids) back
    through ``result_q``.  The fresh prompt's first propose ingests the whole
    prompt (>16 tokens), so a chunk of 16 splits the cold ingest into
    multiple forwards while a chunk of 0 keeps the single-forward behavior.
    """
    _setenv_default("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    _setenv_default("VLLM_METAL_MEMORY_FRACTION", DEFAULT_PAGED_MEMORY_FRACTION)
    os.environ["VLLM_METAL_SPEC_INGEST_CHUNK"] = chunk

    from vllm import LLM, SamplingParams

    from vllm_metal.v1 import draft_model_proposer as dmp

    plans: list[tuple[int, int]] = []
    _install_plan_logger(dmp, plans)

    llm = LLM(
        model=MODEL_NAME,
        max_model_len=1024,
        max_num_seqs=1,
        enable_prefix_caching=True,
        async_scheduling=False,
        speculative_config={
            "method": "draft_model",
            "model": MODEL_NAME,
            "num_speculative_tokens": K,
        },
    )
    sp = SamplingParams(temperature=0, max_tokens=GEN)
    tokens = list(llm.generate([SALT_PROMPT], sp)[0].outputs[0].token_ids)
    first_plan = plans[0] if plans else (-1, -1)
    result_q.put(("ok", first_plan[1], tokens))


@pytest.mark.slow
def test_chunked_cold_ingest_token_identity_e2e() -> None:
    """Chunked cold ingest (chunk=16) is lossless: the generated tokens are
    identical to the single-forward control (chunk=0), and both runs took the
    chunked path (first-plan ingest > the 16-token decode threshold)."""
    ctx = mp.get_context("spawn")
    result_q = ctx.Queue()
    procs = [
        ctx.Process(target=_run_chunked_identity_e2e, args=(chunk, result_q))
        for chunk in ("16", "0")
    ]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join()
    if any(proc.exitcode != 0 for proc in procs):
        raise AssertionError(
            "Chunked-ingest e2e failed in a spawned child "
            f"(exit codes: {[proc.exitcode for proc in procs]})"
        )

    chunked, control = (result_q.get(), result_q.get())
    for label, (status, ingest, _) in (("chunked", chunked), ("control", control)):
        if status != "ok":
            raise AssertionError(f"{label} child skipped")
        if ingest <= 16:
            raise AssertionError(
                f"{label} first plan ingested {ingest} tokens -- expected a "
                ">16-token cold full-prompt ingest on the chunked path"
            )
    if chunked[2] != control[2]:
        raise AssertionError(
            "chunked cold ingest changed the generated tokens:\n"
            f"chunk=16: {chunked[2]}\nchunk=0:  {control[2]}"
        )


def test_scratch_reserve_covers_max_concurrency() -> None:
    """The reserve formula fits every concurrent drafter at full concurrency.

    Each request's committed allocation covers its committed length (one
    block for 16 tokens here); the lookahead to ``committed_len + K - 1``
    then needs exactly ``cdiv(K, block_size)`` scratch block(s) per request.
    ``max_num_seqs`` requests must fit in ``max_num_seqs * cdiv(K, BLOCK_SIZE)``
    scratch blocks, and one block short must exhaust.
    """
    num_speculative_tokens = 4
    max_num_seqs = 3
    reserve = max_num_seqs * cdiv(num_speculative_tokens, BLOCK_SIZE)

    from vllm.sampling_params import SamplingParams

    from vllm_metal.v1.model_runner import RequestState
    from vllm_metal.v1.proposer import ProposeContext

    def _state() -> RequestState:
        return RequestState(
            token_ids=list(range(16)),
            prompt_len=16,
            sampling_params=SamplingParams(temperature=0.0),
            block_ids=[[0]],
            num_computed_tokens=0,
        )

    def _context_for(states: dict[str, RequestState]) -> ProposeContext:
        return ProposeContext(
            target_hidden_states=None,
            decode_reqs=list(states.items()),
            decode_segments=[],
            decode_token_ids=[[state.token_ids[-1]] for state in states.values()],
            prefill_reqs=[],
            prefill_token_ids=[],
            prefill_result_modes=[],
            request_states=states,
            cu_seqlens=[],
            num_decode_segments=1,
            num_speculative_tokens=num_speculative_tokens,
            finished_req_ids=set(),
        )

    states = {f"r{i}": _state() for i in range(max_num_seqs)}

    model = _StubDraftModel()
    proposer = _proposer(model, committed_num_blocks=1, scratch_reserve_blocks=reserve)
    drafts = proposer.propose(_context_for(states))
    assert drafts is not None

    model = _StubDraftModel()
    starved = _proposer(
        model, committed_num_blocks=1, scratch_reserve_blocks=reserve - 1
    )
    with pytest.raises(RuntimeError, match="scratch pool exhausted"):
        starved.propose(_context_for(states))
