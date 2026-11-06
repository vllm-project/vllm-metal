# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash MTP drafting (vLLM ``method="mtp"`` on a ``glm5_next`` target).

The drafter is the checkpoint's MTP layer as mlx-vlm's ``Glm5NextMTPDraftModel``,
split from the checkpoint with ``mlx_vlm.split_mtp`` (MLX conversions drop it)
and named by ``--additional-config '{"glm5_next_mtp_drafter": "<path>"}'``. It
shares the target's embedding and LM head.

Each request keeps a request-local mlx-vlm cache for the MTP layer; nothing is
scheduler-owned. An entry at position ``i`` pairs token ``t[i + 1]`` with the
target's hidden state at ``i``:

- prefill: every prompt chunk feeds the cache, the final chunk adds the sampled
  token and seeds the first draft;
- verify: entries appended while drafting are rolled back (mlx-vlm's cache
  transaction, which also restores the indexer's pooled keys); the accepted
  drafts and the bonus token are appended from the target's verify rows, so
  the cache always equals one pass over the committed tokens;
- draft: the seed plus ``k - 1`` recurrent MTP steps give ``k`` tokens.

On a prefix-cache hit the drafter starts at the first uncached token:
acceptance can be lower until it has seen enough context, output is unaffected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import mlx.core as mx
from vllm.logger import init_logger
from vllm.v1.outputs import DraftTokenIds

from vllm_metal.config import GLM5_NEXT_MTP_DRAFTER_KEY
from vllm_metal.v1.proposer import NoOwnKVProposer

if TYPE_CHECKING:
    from collections.abc import Sequence

    from vllm.config import SpeculativeConfig

    from vllm_metal.v1.model_runner import MetalModelRunner
    from vllm_metal.v1.proposer import ProposeContext
    from vllm_metal.v1.spec_decode import (
        PagedDecodeSegment,
        SpeculativeDecodeController,
    )

logger = init_logger(__name__)


def is_glm5_next_mtp(speculative_config: SpeculativeConfig | None) -> bool:
    """Whether the config asks for GLM-5-Next's MTP drafter."""
    if speculative_config is None or speculative_config.method != "mtp":
        return False
    draft_model_config = speculative_config.draft_model_config
    return (
        draft_model_config is not None
        and draft_model_config.hf_config.model_type == "glm5_next_mtp"
    )


@dataclass
class _RequestDraftState:
    cache: list[Any]
    next_position: int
    # Entries appended while drafting, not yet confirmed by the target, and
    # the transaction that rolls them back.
    round_appended: int = 0
    rollback: Any = None
    # Last row of an intermediate prompt chunk, waiting for the next chunk's
    # first token.
    pending_hidden: mx.array | None = None


class Glm5NextMTPProposer(NoOwnKVProposer):
    """:class:`vllm_metal.v1.proposer.MetalProposer` backed by GLM's MTP layer."""

    def __init__(
        self,
        *,
        drafter: Any,
        controller: SpeculativeDecodeController,
    ) -> None:
        self._drafter = drafter
        self._controller = controller
        self._states: dict[str, _RequestDraftState] = {}

    @classmethod
    def build(cls, runner: MetalModelRunner) -> Glm5NextMTPProposer:
        from mlx_vlm.speculative.drafters import (
            load_drafter,
            validate_drafter_compatibility,
        )

        additional = runner.vllm_config.additional_config
        assert isinstance(additional, dict)
        source = str(additional[GLM5_NEXT_MTP_DRAFTER_KEY])
        drafter, kind = load_drafter(source, kind="mtp")
        validate_drafter_compatibility(runner.model, drafter, kind)
        drafter.bind(runner.model)
        if runner.tp_group is not None:
            drafter.shard(runner.tp_group)
            mx.eval(drafter.parameters())
        logger.info("GLM-5-Next MTP drafter loaded from %s", source)
        return cls(drafter=drafter, controller=runner._spec_decode_controller)

    # -- MetalProposer protocol ---------------------------------------------

    def needs_target_hidden_states(
        self,
        decode_segments: Sequence[PagedDecodeSegment],
        *,
        has_final_prefill: bool,
    ) -> bool:
        # Intermediate prompt chunks feed the drafter cache too.
        return True

    def release_requests(self, req_ids: set[str]) -> None:
        for req_id in req_ids:
            self._states.pop(req_id, None)

    def propose(self, ctx: ProposeContext) -> DraftTokenIds | None:
        self.release_requests(ctx.finished_req_ids)
        hidden = ctx.target_hidden_states
        if hidden is None:
            # An image step at pipeline_parallel_size 1 returns no hidden
            # states: its requests stop drafting rather than skip tokens.
            self.release_requests(
                {req_id for req_id, _ in ctx.decode_reqs}
                | {prefill.req_id for prefill in ctx.prefill_reqs}
            )
            return None
        hidden = hidden.reshape(-1, hidden.shape[-1])

        # (req_id, state, last output row) for each request whose cache now
        # ends at its newest committed token.
        seeds: list[tuple[str, _RequestDraftState, mx.array]] = []
        touched: list[_RequestDraftState] = []

        for (req_id, _), segment, sampled in zip(
            ctx.decode_reqs, ctx.decode_segments, ctx.decode_token_ids, strict=True
        ):
            state = self._states.get(req_id)
            if state is None or not sampled:
                continue
            rows = hidden[
                segment.start_row : segment.start_row + segment.num_query_tokens
            ]
            out = self._ingest_verified(state, segment.draft_token_ids, sampled, rows)
            seeds.append((req_id, state, out))
            touched.append(state)

        num_decode = ctx.num_decode_segments
        for i, (prefill, sampled_token, mode) in enumerate(
            zip(
                ctx.prefill_reqs,
                ctx.prefill_token_ids,
                ctx.prefill_result_modes,
                strict=True,
            )
        ):
            state = self._states.get(prefill.req_id)
            if state is None:
                if not self._admit(prefill.req_id, ctx):
                    continue
                state = _RequestDraftState(
                    cache=self._drafter.make_cache(),
                    next_position=prefill.start_pos,
                )
                self._states[prefill.req_id] = state
            rows = hidden[
                ctx.cu_seqlens[num_decode + i] : ctx.cu_seqlens[num_decode + i + 1]
            ]
            final = mode != "intermediate"
            out = self._ingest_prompt_chunk(
                state, prefill.token_ids, rows, int(sampled_token) if final else None
            )
            if out is None:
                continue
            touched.append(state)
            if final:
                seeds.append((prefill.req_id, state, out))

        drafts = self._draft(seeds, ctx) if ctx.num_speculative_tokens > 0 else {}
        if touched or drafts:
            mx.eval(
                *[
                    layer_cache.state
                    for state in touched
                    for layer_cache in state.cache
                ],
                *drafts.values(),
            )
        if not drafts:
            return None
        req_ids = list(drafts)
        return DraftTokenIds(
            req_ids=req_ids,
            draft_token_ids=[drafts[req_id].tolist() for req_id in req_ids],
        )

    # -- drafter cache ---------------------------------------------------------

    def _admit(self, req_id: str, ctx: ProposeContext) -> bool:
        # Requests that cannot be drafted at admission (sampling, penalties)
        # never get a drafter cache.
        request_state = ctx.request_states.get(req_id)
        return request_state is not None and self._controller.can_draft_greedy(
            req_id, request_state, ()
        )

    def _ingest_verified(
        self,
        state: _RequestDraftState,
        draft_token_ids: Sequence[int],
        sampled: Sequence[int],
        rows: mx.array,
    ) -> mx.array:
        """Commit a verify step from the target's rows.

        Entries appended while drafting paired drafts with the drafter's own
        hidden states; they are all replaced, so the cache always matches a
        single pass over the committed tokens.
        """
        if state.rollback is not None:
            state.rollback.abort()
            state.next_position -= state.round_appended
            state.round_appended = 0
            state.rollback = None
        accepted = len(sampled) - 1
        tokens = [*draft_token_ids[:accepted], int(sampled[-1])]
        return self._forward(state, tokens, rows[: accepted + 1])

    def _ingest_prompt_chunk(
        self,
        state: _RequestDraftState,
        chunk: Sequence[int],
        rows: mx.array,
        sampled_token: int | None,
    ) -> mx.array | None:
        """Pair each prompt token with the previous position's hidden state."""
        tokens: list[int] = []
        parts: list[mx.array] = []
        n = len(chunk)
        if state.pending_hidden is not None and n:
            tokens.append(chunk[0])
            parts.append(state.pending_hidden)
        if n > 1:
            tokens.extend(chunk[1:])
            parts.append(rows[: n - 1])
        if sampled_token is None:
            state.pending_hidden = rows[n - 1 : n]
        else:
            tokens.append(sampled_token)
            parts.append(rows[n - 1 : n])
            state.pending_hidden = None
        if not tokens:
            return None
        return self._forward(state, tokens, mx.concatenate(parts, axis=0))

    def _forward(
        self,
        state: _RequestDraftState,
        tokens: Sequence[int] | mx.array,
        hidden: mx.array,
    ) -> mx.array:
        """Run the MTP layer over ``tokens`` paired with ``hidden`` rows.

        Appends to the request's cache and returns the last output row (1, H),
        post-norm.
        """
        drafter = self._drafter
        ids = mx.array(tokens, dtype=mx.int32).reshape(1, -1)
        n = ids.shape[1]
        # The drafter zeroes the token embedding at position 0 from its own
        # position counter.
        drafter._next_position = state.next_position
        out, _ = drafter._forward_hidden(
            drafter._input_embed(ids), hidden.reshape(1, n, -1), ids, state.cache
        )
        state.next_position += n
        return out[0, -1:]

    # -- drafting ------------------------------------------------------------

    def _draft(
        self,
        seeds: Sequence[tuple[str, _RequestDraftState, mx.array]],
        ctx: ProposeContext,
    ) -> dict[str, mx.array]:
        eligible = {
            req_id
            for req_id, _ in self._controller.draft_eligible_requests(
                ctx.decode_reqs,
                ctx.decode_token_ids,
                ctx.prefill_reqs,
                ctx.prefill_token_ids,
                ctx.prefill_result_modes,
                ctx.request_states,
            )
        }
        rows: list[tuple[str, _RequestDraftState, int]] = []
        hidden: list[mx.array] = []
        for req_id, state, seed_hidden in seeds:
            if req_id not in eligible:
                continue
            k = self._draft_length(req_id, ctx)
            if k > 0:
                rows.append((req_id, state, k))
                hidden.append(seed_hidden)
        if not rows:
            return {}

        # The MTP layer runs per request (each has its own cache); the LM head
        # runs once per step over every request still drafting.
        from mlx_vlm.speculative.cache_state import start_speculative_cache

        tokens: dict[str, list[mx.array]] = {req_id: [] for req_id, _, _ in rows}
        for _, state, k in rows:
            if k > 1:
                state.rollback = start_speculative_cache(state.cache, k - 1)
        active = list(range(len(rows)))
        step_tokens = self._greedy(mx.concatenate(hidden, axis=0))
        for step in range(max(k for _, _, k in rows)):
            for j, i in enumerate(active):
                tokens[rows[i][0]].append(step_tokens[j : j + 1])
            next_active = [
                (j, i) for j, i in enumerate(active) if rows[i][2] > step + 1
            ]
            if not next_active:
                break
            for j, i in next_active:
                state = rows[i][1]
                hidden[i] = self._forward(state, step_tokens[j : j + 1], hidden[i])
                state.round_appended += 1
            active = [i for _, i in next_active]
            step_tokens = self._greedy(
                mx.concatenate([hidden[i] for i in active], axis=0)
            )
        return {req_id: mx.concatenate(t) for req_id, t in tokens.items()}

    def _draft_length(self, req_id: str, ctx: ProposeContext) -> int:
        request_state = ctx.request_states[req_id]
        k = ctx.num_speculative_tokens
        cap = ctx.draft_caps.get(req_id) if ctx.draft_caps is not None else None
        max_tokens = request_state.sampling_params.max_tokens
        if max_tokens is not None:
            # Verification also emits a correction or bonus token.
            remaining = max(0, max_tokens - request_state.generated_tokens - 1)
            cap = remaining if cap is None else min(cap, remaining)
        return k if cap is None else min(k, cap)

    def _greedy(self, hidden: mx.array) -> mx.array:
        return mx.argmax(self._drafter._lm_head_fn(hidden), axis=-1).astype(mx.int32)
