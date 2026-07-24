# SPDX-License-Identifier: Apache-2.0
"""N-gram (prompt-lookup) speculative decoding proposer for the Metal paged path.

An :class:`NgramProposer` drafts by matching the longest suffix n-gram of each
request's committed token history against an earlier occurrence and copying the
tokens that followed it (vLLM ``method="ngram"``). Unlike
:class:`vllm_metal.v1.draft_model_proposer.DraftModelProposer` it loads no model
and keeps no KV cache: the matching is the pure-Python + Numba KMP kernel that
vLLM ships in :mod:`vllm.v1.spec_decode.ngram_proposer`, which this class wraps.

The wrapper's only job is to translate the per-step :class:`ProposeContext` into
the runtime draft count and array arguments that upstream's stateless
``propose`` expects (``num_speculative_tokens``, ``sampled_token_ids``,
``num_tokens_no_spec``, ``token_ids_cpu``) and hand the result back as
:class:`DraftTokenIds`. The committed history lives in
``state.token_ids`` (already updated with this step's accepted/sampled tokens by
the time the runner builds the context).

The one piece of per-request bookkeeping this wrapper does keep: a consecutive-
miss streak per request, so a request with no exploitable repetition (free-form
prose, for instance) stops paying the match kernel's per-step scan cost after
a few misses in a row, with periodic retries in case the content turns
repetitive later. See ``_record_miss``/``_on_cooldown``.

A lookup match is not itself evidence the request benefits: the target may
reject every proposed token. ``_resolve_pending`` checks a step's proposal
against the *next* step's committed history (which already reflects what the
target actually accepted) to score it as a real hit or a miss, rather than
trusting the lookup kernel's own optimism.

The verify half is unchanged: drafts are handed back via ``take_draft_token_ids``
and verified next step by ``SpeculativeDecodeController.verify_greedy``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from vllm.logger import init_logger
from vllm.v1.outputs import DraftTokenIds
from vllm.v1.spec_decode.ngram_proposer import NgramProposer as VllmNgramProposer

if TYPE_CHECKING:
    from collections.abc import Sequence

    from vllm.config import VllmConfig

    from vllm_metal.v1.proposer import ProposeContext
    from vllm_metal.v1.spec_decode import (
        PagedDecodeSegment,
        SpeculativeDecodeController,
    )

logger = init_logger(__name__)

# The match kernel scans a request's whole committed history every decode
# step whether or not it finds anything -- an O(history length) Numba scan
# plus a full-history copy into token_ids_cpu, paid regardless of outcome.
# A request whose content has no exploitable repetition (free-form prose,
# for instance) pays that tax every step for nothing. After this many
# consecutive misses, stop attempting a request for _COOLDOWN_STEPS steps
# rather than giving up on it forever -- generation can turn repetitive
# partway through (e.g. a response that starts as prose and then quotes
# earlier context), so a permanent cutoff would miss that.
_MAX_CONSECUTIVE_MISSES = 8
_COOLDOWN_STEPS = 8


class NgramProposer:
    """:class:`vllm_metal.v1.proposer.MetalProposer` backed by vLLM's n-gram kernel."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        controller: SpeculativeDecodeController,
    ) -> None:
        self._controller = controller
        # Per-request consecutive-miss count and, once throttled, remaining
        # cooldown steps before the next retry. Pruned each step against the
        # scheduler's own finished-id set, not against absence from
        # request_states: vLLM can hand a finished id straight back out to a
        # new request in the same step, and that new request repopulates
        # request_states under the same id before this wrapper ever sees it.
        self._miss_streak: dict[str, int] = {}
        self._cooldown: dict[str, int] = {}
        # A lookup match doesn't mean the target accepted any of it. Each
        # non-empty draft is stashed here (position it starts at, tokens
        # proposed) and scored against the *next* step's actual committed
        # history in _resolve_pending, before this step's proposal is
        # scored as a hit or a miss.
        self._pending: dict[str, tuple[int, tuple[int, ...]]] = {}
        # Upstream reads only scalar config (prompt_lookup_min/max,
        # num_speculative_tokens, max_model_len, max_num_seqs) and runs a one-time
        # Numba JIT warmup in its constructor — keep that off the hot path.
        self._ngram = VllmNgramProposer(vllm_config)
        spec = vllm_config.speculative_config
        assert spec is not None
        # Below this many committed tokens, the kernel structurally cannot
        # match (not enough history to form even the shortest n-gram window)
        # -- that's not evidence against repetition, just not enough data
        # yet, so it must not count as a miss. Normalized to a concrete int
        # by SpeculativeConfig.__post_init__ for method="ngram".
        assert spec.prompt_lookup_min is not None
        self._prompt_lookup_min = spec.prompt_lookup_min

        # Pre-allocate the int32 token-id buffer once. Upstream only reads
        # ``token_ids_cpu[i, :num_tokens_no_spec[i]]`` per row, so the buffer
        # just needs to be large enough to hold the longest any request's
        # committed history can ever be, across every simultaneously-scheduled
        # request. Reusing it removes a per-step ``np.zeros`` allocation.
        max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        max_model_len = vllm_config.model_config.max_model_len
        self._token_ids_cpu = np.zeros((max_num_seqs, max_model_len), dtype=np.int32)
        logger.info(
            "N-gram speculative decoding enabled "
            "(prompt_lookup=[%d, %d], num_speculative_tokens=%d, "
            "token_ids_cpu=(%d, %d) (%.2f MiB))",
            spec.prompt_lookup_min,
            spec.prompt_lookup_max,
            spec.num_speculative_tokens,
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
    ) -> NgramProposer:
        return cls(vllm_config=vllm_config, controller=controller)

    # -- MetalProposer protocol ---------------------------------------------

    def needs_target_hidden_states(
        self,
        decode_segments: Sequence[PagedDecodeSegment],
        *,
        has_final_prefill: bool,
    ) -> bool:
        # N-gram matches token ids only; it never reads the target's hidden states.
        return False

    def propose(self, ctx: ProposeContext) -> DraftTokenIds | None:
        # Bookkeeping runs unconditionally, before the num_speculative_tokens
        # check: a step with drafting disabled still needs finished ids
        # pruned and pending drafts resolved, or a request id reused in a
        # disabled step would carry stale throttle state into the next step
        # where drafting is enabled again.
        self._prune_finished(ctx.finished_req_ids)
        self._resolve_pending(ctx)

        if ctx.num_speculative_tokens <= 0:
            return None

        drafting = list(
            self._controller.draft_eligible_requests(
                ctx.decode_reqs,
                ctx.decode_token_ids,
                ctx.prefill_reqs,
                ctx.prefill_result_modes,
                ctx.request_states,
                logitsprocs=ctx.logitsprocs,
            )
        )
        if not drafting:
            return None

        drafting = [
            (req_id, state)
            for req_id, state in drafting
            if not self._on_cooldown(req_id)
        ]
        if not drafting:
            return None

        # Upstream marks a row "active" by a non-empty sampled-ids entry; the
        # match itself reads only token_ids_cpu[i, :num_tokens_no_spec[i]]. We
        # forward exactly the requests we have decided may draft, so every row is
        # active and num_tokens_no_spec is the committed history length.
        num_requests = len(drafting)
        num_tokens_no_spec = np.array(
            [len(state.token_ids) for _, state in drafting], dtype=np.int32
        )
        token_ids_cpu = self._token_ids_cpu[:num_requests]
        token_ids_cpu[:, :] = 0
        for i, (_, state) in enumerate(drafting):
            token_ids_cpu[i, : len(state.token_ids)] = state.token_ids
        sampled_token_ids: list[list[int]] = [[0]] * num_requests

        drafts = self._ngram.propose(
            ctx.num_speculative_tokens,
            sampled_token_ids,
            num_tokens_no_spec,
            token_ids_cpu,
        )

        req_ids: list[str] = []
        draft_token_ids: list[list[int]] = []
        for (req_id, state), draft in zip(drafting, drafts, strict=True):
            if not draft:
                # A too-short history isn't evidence against repetition --
                # there just isn't enough of it yet to check.
                if len(state.token_ids) >= self._prompt_lookup_min:
                    self._record_miss(req_id)
                continue
            # Don't score this as a hit yet -- the target hasn't verified it.
            # _resolve_pending scores it next step against what was actually
            # accepted.
            self._pending[req_id] = (len(state.token_ids), tuple(draft))
            req_ids.append(req_id)
            # Upstream already yields Python ints via ndarray.tolist() — the
            # old ``[int(t) for t in draft]`` was redundant.
            draft_token_ids.append(list(draft))

        if not req_ids:
            return None

        return DraftTokenIds(req_ids=req_ids, draft_token_ids=draft_token_ids)

    # -- miss-streak throttling ----------------------------------------------

    def _on_cooldown(self, req_id: str) -> bool:
        remaining = self._cooldown.get(req_id, 0)
        if remaining <= 0:
            return False
        if remaining == 1:
            del self._cooldown[req_id]
        else:
            self._cooldown[req_id] = remaining - 1
        return True

    def _record_miss(self, req_id: str) -> None:
        # Cap rather than clear: once a request has earned a cooldown, a
        # miss on the very next retry should send it right back into
        # cooldown, not grant another full _MAX_CONSECUTIVE_MISSES-long
        # grace period. Only a genuine hit (_resolve_pending finding real
        # acceptance) fully clears the streak.
        streak = min(self._miss_streak.get(req_id, 0) + 1, _MAX_CONSECUTIVE_MISSES)
        self._miss_streak[req_id] = streak
        if streak >= _MAX_CONSECUTIVE_MISSES:
            self._cooldown[req_id] = _COOLDOWN_STEPS

    def _resolve_pending(self, ctx: ProposeContext) -> None:
        """Score the previous step's proposals against what was actually
        accepted, but only for requests whose draft was actually verified
        this step: present in ``decode_reqs`` *and* the decode_segment for
        that request carried a non-empty draft. Being in ``decode_reqs``
        alone isn't enough -- a request can sit there with an empty
        ``draft_token_ids`` either because no draft was scheduled for it at
        all (a plain decode row) or because the scheduler padded/dropped a
        proposed draft on batch admission (see
        ``SpeculativeDecodeController.active_spec_decode_tokens``). Either
        way, this step's plain-decode token isn't the verification outcome
        of the older pending draft, so it must not be scored against it."""
        if not self._pending:
            return
        verified_req_ids = {
            segment.req_id for segment in ctx.decode_segments if segment.draft_token_ids
        }
        decode_states = dict(ctx.decode_reqs)
        for req_id in list(self._pending):
            if req_id not in verified_req_ids:
                continue  # not verified this step; leave pending for later
            state = decode_states.get(req_id)
            if state is None:
                continue
            position, proposed = self._pending.pop(req_id)
            actual = state.token_ids[position : position + len(proposed)]
            accepted = 0
            # actual may be shorter than proposed if the engine hasn't
            # committed that many new tokens yet -- strict=False by design.
            for proposed_id, actual_id in zip(proposed, actual, strict=False):
                if proposed_id != actual_id:
                    break
                accepted += 1
            if accepted > 0:
                self._miss_streak.pop(req_id, None)
            else:
                self._record_miss(req_id)

    def _prune_finished(self, finished_req_ids: set[str]) -> None:
        if not finished_req_ids:
            return
        for req_id in finished_req_ids:
            self._miss_streak.pop(req_id, None)
            self._cooldown.pop(req_id, None)
            self._pending.pop(req_id, None)
