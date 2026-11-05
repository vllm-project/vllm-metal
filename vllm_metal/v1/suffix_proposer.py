# SPDX-License-Identifier: Apache-2.0
"""Suffix-decoding (model-free trie) speculative decoding proposer for the
Metal paged path.

Suffix decoding (Snowflake, https://arxiv.org/pdf/2411.04975) drafts by
walking a suffix tree built over the prompts and previously generated
responses of the serving process — no draft model, no draft KV cache, and
unlike n-gram matching it learns *across* requests: an agentic loop that
re-emits near-identical tool-call scaffolding hits trees warmed by earlier
responses. vLLM ships the logic in
:class:`vllm.v1.spec_decode.suffix_decoding.SuffixDecodingProposer`, whose
engine-side propose is driven entirely off an ``InputBatch``'s token
bookkeeping; this wrapper mirrors :class:`NgramProposer`'s job: translate
the per-step :class:`ProposeContext` into that bookkeeping and hand back
:class:`DraftTokenIds`.

The suffix tree itself lives in the ``arctic_inference`` package (a pure
C++/nanobind core with no CUDA in it). PyPI ships an sdist only, so on
macOS it builds from source -- the recipe is in
``docs/speculative_decoding.md``. A missing install is rejected by vLLM's
own ``SpeculativeConfig`` validation before this module is reached.

Metal-specific decisions on top of upstream's loop:

- **Every scheduled request stays in the stub batch, not just drafting
  ones** (decode and prefill rows, as upstream's GPU runner passes its
  whole InputBatch). Upstream archives an active tree the first step its
  id is missing from the batch. Rows that never drafted never start a
  tree, so keeping them costs nothing; what the rows protect is a request
  that has been drafting but cannot draft *this* step (an intermediate
  prefill chunk while recomputing, a step it sampled nothing): without its
  row its live tree would be archived and rebuilt from the prompt.
- **Deltas, not raw sampled ids.** Upstream feeds its engine's
  last-step sampled ids into ``add_active_response``; the committed
  history in ``RequestState.token_ids`` already contains exactly those
  tokens, so the wrapper remembers each request's last-fed length and
  feeds the slice in between. When upstream holds no active tree for the
  id (first sighting, or archived while preempted) the slice starts at
  ``prompt_len``, so a restarted tree relearns the whole response.
  Ingestion is decoupled from drafting eligibility: a row that cannot
  draft but holds a live tree still feeds its delta, and the state
  reference is held so the final committed segment reaches the tree when
  the request is archived (``release_requests`` /
  ``finished_req_ids``) instead of on upstream's next propose call.

Draft lengths are dynamic per request (the tree speculates as deep as the
matched suffix and the probability floor allow), bounded by this step's
scheduled width; the verify half is the established greedy span path
shared with n-gram. As with vLLM's default execution, floating-point and
batch-shape effects can change output tokens relative to ordinary decode;
this adapter does not install batch-invariant model kernels.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
from vllm.logger import init_logger
from vllm.v1.outputs import DraftTokenIds

from vllm_metal.v1.proposer import NoOwnKVProposer

if TYPE_CHECKING:
    from vllm.config import VllmConfig

    from vllm_metal.v1.model_runner import RequestState
    from vllm_metal.v1.proposer import ProposeContext
    from vllm_metal.v1.spec_decode import SpeculativeDecodeController

logger = init_logger(__name__)


def _build_vllm_suffix(vllm_config: VllmConfig):
    """Construct upstream's proposer (kept as a seam for tests to patch)."""
    from vllm.v1.spec_decode.suffix_decoding import SuffixDecodingProposer

    return SuffixDecodingProposer(vllm_config)


class SuffixProposer(NoOwnKVProposer):
    """:class:`vllm_metal.v1.proposer.MetalProposer` backed by suffix trees."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        controller: SpeculativeDecodeController,
    ) -> None:
        self._controller = controller
        # req_id -> committed-history length at this request's last feeding.
        # The delta since then is what the suffix tree should learn from.
        self._last_len: dict[str, int] = {}
        # req_id -> the RequestState whose token_ids that length anchors.
        # Held so the final committed segment can still be fed when the
        # request is archived: by then the engine may have dropped it from
        # request_states, or even reused the id for a new request.
        self._states: dict[str, RequestState] = {}
        self._vllm = _build_vllm_suffix(vllm_config)
        spec = vllm_config.speculative_config
        assert spec is not None
        # Upstream's propose asserts this exact value; drafts are then
        # truncated to the scheduler's per-step width (ctx).
        self._num_speculative_tokens = spec.num_speculative_tokens
        self._max_tree_depth = spec.suffix_decoding_max_tree_depth

        # Same pre-allocated history buffer trick as the n-gram wrapper:
        # upstream reads rows [:num_prompt_tokens] and [start:num_tokens].
        max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        max_model_len = vllm_config.model_config.max_model_len
        self._max_model_len = max_model_len
        self._token_ids_cpu = np.zeros((max_num_seqs, max_model_len), dtype=np.int32)
        logger.info(
            "Suffix speculative decoding enabled "
            "(num_speculative_tokens=%d, max_tree_depth=%d, "
            "max_spec_factor=%g, min_token_prob=%g, max_cached_requests=%d, "
            "token_ids_cpu=(%d, %d) (%.2f MiB))",
            spec.num_speculative_tokens,
            spec.suffix_decoding_max_tree_depth,
            spec.suffix_decoding_max_spec_factor,
            spec.suffix_decoding_min_token_prob,
            spec.suffix_decoding_max_cached_requests,
            max_num_seqs,
            max_model_len,
            self._token_ids_cpu.nbytes / (1024 * 1024),
        )

    # -- construction --------------------------------------------------------

    @classmethod
    def build(
        cls,
        *,
        vllm_config: VllmConfig,
        controller: SpeculativeDecodeController,
    ) -> SuffixProposer:
        return cls(vllm_config=vllm_config, controller=controller)

    # -- MetalProposer protocol ---------------------------------------------

    def needs_target_hidden_states(self, decode_segments, *, has_final_prefill) -> bool:
        # Suffix trees read token ids only; never the target's hidden states.
        return False

    def release_requests(self, req_ids: set[str]) -> None:
        # Finished, preempted and resumed ids. Archive their trees now rather
        # than on the next propose call that happens to run (an idle server
        # may never make one). A preempted/resumed request loses nothing: its
        # next drafting step re-feeds the response since the prompt boundary.
        for req_id in req_ids:
            self._stop(req_id)

    def _stop(self, req_id: str) -> None:
        anchor = self._last_len.pop(req_id, None)
        state = self._states.pop(req_id, None)
        cache = self._vllm.suffix_cache
        if req_id in cache.active_requests:
            # Feed the final committed segment before archiving. The request
            # leaves after its last drafting step (finished, preempted or
            # evicted), so the tokens committed since then -- the verify
            # round that finished it, or the output-budget-capped tail --
            # would otherwise never reach the tree, and that tail is exactly
            # what later requests want to reuse. The held state is the
            # finished request's own; request_states cannot be used here
            # because a finished id may already be reused by a new request.
            if anchor is not None and state is not None:
                delta = [int(t) for t in state.token_ids[anchor:]]
                if delta:
                    cache.add_active_response(req_id, delta)
            cache.stop_request(req_id)

    def propose(self, ctx: ProposeContext) -> DraftTokenIds | None:
        for req_id in ctx.finished_req_ids:
            self._stop(req_id)

        if ctx.num_speculative_tokens <= 0:
            return None

        drafting = dict(
            self._controller.draft_eligible_requests(
                ctx.decode_reqs,
                ctx.decode_token_ids,
                ctx.prefill_reqs,
                ctx.prefill_result_modes,
                ctx.request_states,
            )
        )
        if not drafting:
            return None

        # Every request in this step's batch gets a row -- decode rows and
        # prefill rows alike, matching upstream's GPU runner, which hands the
        # whole InputBatch over. Upstream stops (archives) any active tree
        # whose id is missing from the batch, so a drafting request that sits
        # in an intermediate prefill chunk (recompute after preemption) or
        # cannot draft this step must still be present. Active trees ingest
        # their deltas even when the row cannot receive drafts.
        rows: list[tuple[str, RequestState | None]] = list(ctx.decode_reqs)
        seen = {req_id for req_id, _ in rows}
        for prefill in ctx.prefill_reqs:
            if prefill.req_id in seen:
                continue
            seen.add(prefill.req_id)
            rows.append((prefill.req_id, ctx.request_states.get(prefill.req_id)))
        num_rows = len(rows)
        token_ids_cpu = self._token_ids_cpu[:num_rows]
        num_tokens_no_spec = np.zeros(num_rows, dtype=np.int32)
        num_prompt_tokens = np.zeros(num_rows, dtype=np.int32)
        req_id_to_index: dict[str, int] = {}
        sampled_token_ids: list[list[int]] = []
        for i, (req_id, state) in enumerate(rows):
            req_id_to_index[req_id] = i
            if state is not None:
                num_tokens_no_spec[i] = len(state.token_ids)
                num_prompt_tokens[i] = state.prompt_len
            draft_state = drafting.get(req_id)
            if draft_state is None:
                # Not drafting and no live tree to feed: upstream skips the
                # row and never reads its token_ids_cpu, so nothing is
                # copied for it.
                if state is None or req_id not in self._last_len:
                    sampled_token_ids.append([])
                    continue
                # A row that is not drafting but holds a live tree (it
                # reached its output budget; this is its last batch
                # appearance) still feeds its committed delta -- without
                # this, its final tokens would only reach the tree through
                # _stop's tail feed, and never at all if the batch row were
                # dropped. Upstream drafts for the row anyway; the draft is
                # discarded below.
                source = state
            else:
                source = draft_state
            history = source.token_ids
            cur = len(history)
            if cur >= self._max_model_len:
                # Upstream skips ingestion at the context limit. Preserve
                # the last ingested position so _stop can flush the tail.
                if req_id in self._last_len:
                    self._states[req_id] = source
                sampled_token_ids.append([])
                continue
            # Not in _last_len means upstream holds no active tree for this
            # id (first sighting, or archived while preempted/absent): it
            # will (re)start the tree from the prompt, so feed the whole
            # response so far, not just the latest delta.
            starting = req_id not in self._last_len
            prev = self._last_len.get(req_id, source.prompt_len)
            if prev > cur:
                # Rollback: the recomputed history is shorter than what was
                # fed; re-anchor and skip the delta this step.
                prev = cur
            self._last_len[req_id] = cur
            self._states[req_id] = source
            sampled_token_ids.append(history[prev:cur])
            # Upstream reads only the prompt (when it starts a tree) and the
            # last max_tree_depth tokens (the match pattern); copy just those
            # instead of the whole history every step.
            if starting:
                prompt_len = source.prompt_len
                token_ids_cpu[i, :prompt_len] = history[:prompt_len]
            tail = max(0, cur - self._max_tree_depth)
            token_ids_cpu[i, tail:cur] = history[tail:cur]

        stub = SimpleNamespace(
            req_ids=[req_id for req_id, _ in rows],
            num_tokens_no_spec=num_tokens_no_spec,
            req_id_to_index=req_id_to_index,
            num_prompt_tokens=num_prompt_tokens,
            token_ids_cpu=token_ids_cpu,
        )
        # Flush missing requests before upstream archives them. Their last
        # step may have committed tokens while drafting was disabled.
        for req_id in self._last_len.keys() - req_id_to_index.keys():
            self._stop(req_id)
        drafts = self._vllm.propose(
            self._num_speculative_tokens, stub, sampled_token_ids
        )

        req_ids: list[str] = []
        draft_token_ids: list[list[int]] = []
        for (req_id, _), draft in zip(rows, drafts, strict=True):
            if req_id not in drafting or not draft:
                continue
            # Upstream drafts up to the configured width (its assert pins
            # that value); this step's scheduled width may be smaller.
            limit = ctx.num_speculative_tokens
            cap = ctx.draft_caps.get(req_id) if ctx.draft_caps is not None else None
            if cap is not None:
                limit = min(limit, cap)
            max_tokens = drafting[req_id].sampling_params.max_tokens
            if max_tokens is not None:
                # Verification also emits a correction or bonus token, so
                # drafts past max_tokens - 1 would be verified and discarded.
                limit = min(limit, max_tokens - drafting[req_id].generated_tokens - 1)
            if limit < 1:
                continue
            draft = draft[:limit]
            req_ids.append(req_id)
            draft_token_ids.append(list(draft))

        if not req_ids:
            return None
        return DraftTokenIds(req_ids=req_ids, draft_token_ids=draft_token_ids)
