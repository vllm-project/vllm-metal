# SPDX-License-Identifier: Apache-2.0
"""Native MTP-head speculative-decoding proposer for the Metal paged path.

Runs a registered native MTP head (first: GLM-4.7-Flash's ``glm4_moe_lite_mtp``)
as a tiny autoregressive 1-layer model over the target's "slot stream". Slot
``p`` is::

    slot_p = eh_proj([enorm(embed(t_{p+1})), hnorm(h_p)])

where ``h_p`` is the target's post-final-norm hidden state at position ``p`` and
``t_{p+1}`` is the token committed at position ``p+1``. One stock decoder-layer
forward over a request's newly committed slots both **ingests** (the layer's
compressed MLA latent lands in a proposer-owned per-request KV slab as a side
effect of attention) and **drafts** (the last slot's output -> shared-head norm
-> lm_head -> argmax = one draft token).

The slab rewind/hole/accepted-rows-only discipline keeps each request's KV
history consistent across scheduler preemption-recompute. The proposer reuses the
runner's existing ``target_hidden_states`` (no aux-hidden-state machinery) and the
KV slab holds the head layer's own MLA latent (different k/v feature dims).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import mlx.core as mx
from mlx_lm.models.cache import create_attention_mask as _cache_create_attention_mask
from vllm.logger import init_logger
from vllm.v1.outputs import DraftTokenIds

if TYPE_CHECKING:
    from collections.abc import Mapping

    from vllm_metal.v1.proposer import ProposeContext
    from vllm_metal.v1.spec_decode import SpeculativeDecodeController

logger = init_logger(__name__)

# Slab growth quantum (positions), mirroring mlx_lm's KVCache.step.
_SLAB_STEP = 256


class MTPSlab:
    """Per-request growable KV history for the head's single MLA layer.

    Exposes the mlx_lm KV-cache protocol the stock ``Glm4MoeLiteDecoderLayer``
    attention drives: ``.offset``, ``update_and_fetch(kv_latent, k_pe)`` and
    ``make_mask`` (so ``create_attention_mask`` produces an offset-aware causal
    mask over the appended history). Keys/values carry **different** feature dims
    (GLM MLA: 512 compressed latent vs 64 rope) which the update handles by
    reading each dim from the first append rather than assuming they match.

    Slab index == absolute sequence position == RoPE offset, which is exactly
    what makes the stock attention correct. ``update_and_fetch`` follows the
    mlx_lm ``KVCache`` slice-assignment growth pattern (grow in ``_SLAB_STEP``
    quanta, write in place); ``truncate_to`` supports scheduler
    preemption-recompute rewinds.
    """

    __slots__ = ("keys", "offset", "values")

    def __init__(self) -> None:
        self.keys: mx.array | None = None
        self.values: mx.array | None = None
        self.offset = 0

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        """Append ``[B, n, T, *]`` k/v at ``offset`` and return the full history.

        Ported from ``mlx_lm.models.cache.KVCache.update_and_fetch``; the only
        substantive difference is that keys (compressed latent) and values (rope)
        have distinct last dims, which the stock implementation already reads
        independently (``k_head_dim`` vs ``v_head_dim``), so the pattern carries
        over unchanged. Feature dims and dtype are therefore inferred from the
        first append — nothing is hardcoded.
        """
        prev = self.offset
        num_new = keys.shape[2]
        if self.keys is None or (prev + num_new) > self.keys.shape[2]:
            batch, n_kv_heads, _, k_head_dim = keys.shape
            v_head_dim = values.shape[3]
            n_steps = (_SLAB_STEP + num_new - 1) // _SLAB_STEP
            k_shape = (batch, n_kv_heads, n_steps * _SLAB_STEP, k_head_dim)
            v_shape = (batch, n_kv_heads, n_steps * _SLAB_STEP, v_head_dim)
            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)
            if self.keys is not None:
                if prev % _SLAB_STEP != 0:
                    self.keys = self.keys[..., :prev, :]
                    self.values = self.values[..., :prev, :]
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v

        self.offset += num_new
        self.keys[..., prev : self.offset, :] = keys
        self.values[..., prev : self.offset, :] = values
        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]

    def make_mask(self, *args: Any, **kwargs: Any) -> Any:
        """Offset-aware causal mask, mirroring ``mlx_lm.KVCache.make_mask``.

        Called by ``create_attention_mask`` before ``update_and_fetch`` runs, so
        ``self.offset`` is the pre-append offset and the mask covers the ``N`` new
        query rows attending over ``offset + N`` keys.
        """
        return _cache_create_attention_mask(*args, offset=self.offset, **kwargs)

    def truncate_to(self, position: int) -> None:
        """Rewind the valid length to absolute ``position`` (a preemption redo).

        Only lowers the offset (the underlying buffers are kept; stale rows are
        overwritten on the next append and never read, since fetches slice to
        ``offset``). A forward move would fabricate unobserved positions, so it is
        rejected.
        """
        if position < 0:
            raise ValueError(f"truncate_to position must be non-negative: {position}")
        if position > self.offset:
            raise ValueError(
                f"truncate_to cannot advance the slab: position {position} > "
                f"offset {self.offset}"
            )
        self.offset = position


class _SlotBatch:
    """A request's contiguous slot stream to ingest this step.

    ``tokens``/``hidden_rows`` are the per-slot embedding token and target hidden
    state; ``first_position`` is the absolute sequence position of the first slot
    (positions are contiguous). ``draft`` marks whether the last slot's output is
    read out as this step's single draft token (decode + final-prefill slots
    draft; intermediate-chunk ingest does not).
    """

    __slots__ = ("draft", "first_position", "hidden_rows", "req_id", "slab", "tokens")

    def __init__(
        self,
        *,
        req_id: str,
        first_position: int,
        tokens: list[int],
        hidden_rows: list[mx.array],
        draft: bool,
        slab: MTPSlab,
    ) -> None:
        self.req_id = req_id
        self.first_position = first_position
        self.tokens = tokens
        self.hidden_rows = hidden_rows
        self.draft = draft
        self.slab = slab


class NativeMTPProposer:
    """:class:`vllm_metal.v1.proposer.MetalProposer` for native MTP heads."""

    def __init__(
        self,
        *,
        model: Any,
        controller: SpeculativeDecodeController,
    ) -> None:
        self._model = model
        self._controller = controller
        # The slab infers its k/v dtype and feature dims from the first append,
        # so nothing here needs the target's kv dtype to size them.
        self._slabs: dict[str, MTPSlab] = {}
        # req_id -> (position, hidden) for an intermediate prefill chunk's last
        # row, whose next token is unknown until the following chunk runs. Its
        # hidden state exists only in the step that produced it, so it is carried
        # forward and prepended to the next batch.
        self._pending_last_hidden: dict[str, tuple[int, mx.array]] = {}

    # -- construction ----------------------------------------------------------

    @classmethod
    def build(
        cls,
        *,
        speculative_config: Any,
        controller: SpeculativeDecodeController,
        loader: Any,
        model_type: str,
        target_config: Mapping[str, Any],
    ) -> NativeMTPProposer:
        """Load the head via ``loader`` and build the proposer over it."""
        runtime = loader.load_if_needed(
            speculative_config=speculative_config,
            target_config=target_config,
        )
        logger.info(
            "Native MTP proposer ready: head=%s (%s)",
            model_type,
            runtime.model_name,
        )
        return cls(model=runtime.model, controller=controller)

    # -- MetalProposer protocol -------------------------------------------------

    def needs_target_hidden_states(
        self,
        decode_segments: Any,
        *,
        has_final_prefill: bool,
    ) -> bool:
        # The slot stream projects one input row from EVERY committed token's
        # target hidden state — including intermediate prefill chunks, whose
        # hidden states exist only in the step that ran them. So always collect.
        return True

    def propose(self, ctx: ProposeContext) -> DraftTokenIds | None:
        self._prune_finished(ctx.request_states)

        if ctx.target_hidden_states is None:
            # The runner collects target hidden states on every paged text
            # forward while this drafter is installed; their absence means the
            # batch ran an execution path native MTP does not support
            # (multimodal / pipeline-parallel forward). Fail loud rather than
            # silently never drafting.
            raise RuntimeError(
                "Native MTP drafting requires target hidden states, but none "
                "were collected this step; the current execution path "
                "(multimodal or pipeline-parallel forward) does not support "
                "native MTP speculative decoding."
            )

        hidden = ctx.target_hidden_states
        batches = self._collect_slot_batches(ctx, hidden)
        if not batches:
            return None

        # Zero scheduled spec tokens suppresses drafting but not ingest:
        # committed slots must still land so the slabs stay contiguous for when
        # speculation re-enables.
        draft_enabled = ctx.num_speculative_tokens > 0

        draft_req_ids: list[str] = []
        draft_tokens: list[mx.array] = []
        touched: list[MTPSlab] = []
        for batch in batches:
            x = self._model.build_slot_inputs(
                mx.array(batch.tokens, dtype=mx.int32),
                mx.stack(batch.hidden_rows),
                batch.first_position,
            )
            out = self._model.forward_slots(
                x, batch.slab, expected_offset=batch.first_position
            )
            touched.append(batch.slab)
            if draft_enabled and batch.draft:
                logits = self._model.compute_logits(out[-1:])
                draft_req_ids.append(batch.req_id)
                draft_tokens.append(mx.argmax(logits, axis=-1))

        draft_matrix = mx.stack(draft_tokens) if draft_tokens else None

        # One eval over every touched slab array + the carried pending hiddens +
        # the stacked drafts: slabs updated for ingest-only requests (intermediate
        # prefill chunks) must not accumulate a lazy graph across steps, and the
        # pending hidden must be materialized before this step's
        # target_hidden_states is discarded (laziness discipline, one sync).
        eval_targets: list[mx.array] = [
            slab.keys for slab in touched if slab.keys is not None
        ]
        eval_targets.extend(slab.values for slab in touched if slab.values is not None)
        eval_targets.extend(h for _, h in self._pending_last_hidden.values())
        if draft_matrix is not None:
            eval_targets.append(draft_matrix)
        if eval_targets:
            mx.eval(*eval_targets)

        if draft_matrix is None:
            return None
        rows: list[list[int]] = draft_matrix.reshape(len(draft_req_ids), -1).tolist()  # type: ignore[assignment]
        return DraftTokenIds(req_ids=draft_req_ids, draft_token_ids=rows)

    def release_requests(self, req_ids: set[str]) -> None:
        # Drop the per-request slab (freeing its k/v arrays) and any deferred
        # pending row on eviction/preemption, so a waiting request does not pin
        # the slab; a resumed request rebuilds its slab from position 0 during
        # recompute (release-don't-hold, mirroring the runtime's recurrent-state
        # release in #489).
        for req_id in req_ids:
            self._slabs.pop(req_id, None)
            self._pending_last_hidden.pop(req_id, None)

    # -- slot batching ----------------------------------------------------------

    def _collect_slot_batches(
        self, ctx: ProposeContext, hidden: mx.array
    ) -> list[_SlotBatch]:
        """Build each greedy-eligible request's contiguous slot batch."""
        batches: list[_SlotBatch] = []
        seen: set[str] = set()

        for (req_id, state), segment, sampled_ids in zip(
            ctx.decode_reqs,
            ctx.decode_segments,
            ctx.decode_token_ids,
            strict=True,
        ):
            accepted = len(sampled_ids)
            if accepted == 0:
                continue
            if not self._controller.can_draft_greedy(
                req_id, state, logitsprocs=ctx.logitsprocs
            ):
                continue
            # Committed rows 0..accepted-1 only — rejected draft rows never enter.
            # Slot at position cache_start_pos+j embeds sampled_ids[j] (the token
            # committed at position+1) over hidden row start_row+j.
            hidden_rows = [hidden[segment.start_row + j] for j in range(accepted)]
            slab = self._align_slab(req_id, segment.cache_start_pos)
            batches.append(
                _SlotBatch(
                    req_id=req_id,
                    first_position=segment.cache_start_pos,
                    tokens=list(sampled_ids[:accepted]),
                    hidden_rows=hidden_rows,
                    draft=True,
                    slab=slab,
                )
            )
            seen.add(req_id)

        for i, prefill in enumerate(ctx.prefill_reqs):
            req_id = prefill.req_id
            if req_id in seen:
                continue
            result_mode = ctx.prefill_result_modes[i]
            state = ctx.request_states.get(req_id)
            greedy_state = (
                state
                if state is not None
                else SimpleNamespace(
                    sampling_params=prefill.sampling_params, generated_tokens=0
                )
            )
            if not self._controller.can_draft_greedy(
                req_id, greedy_state, logitsprocs=ctx.logitsprocs
            ):
                continue
            batch = self._prefill_batch(ctx, i, prefill, result_mode, hidden)
            if batch is not None:
                batches.append(batch)
            seen.add(req_id)

        return batches

    def _prefill_batch(
        self,
        ctx: ProposeContext,
        index: int,
        prefill: Any,
        result_mode: str,
        hidden: mx.array,
    ) -> _SlotBatch | None:
        """Slot batch for one prefill chunk, deferring an intermediate last row.

        A chunk of ``n`` tokens at positions ``s..s+n-1`` yields ``n`` slots; slot
        ``s+j`` embeds the token committed at ``s+j+1``. For rows ``0..n-2`` that
        is the next token *within* the chunk. The last row's next token is either
        this step's sampled token (final chunk) or unknown (intermediate chunk),
        in which case its hidden state is deferred to ``_pending_last_hidden`` and
        the slot is emitted next step, once the following chunk supplies the
        token.
        """
        req_id = prefill.req_id
        token_ids = list(prefill.token_ids)
        num_rows = len(token_ids)
        row_start = ctx.cu_seqlens[ctx.num_decode_segments + index]
        start_pos = prefill.start_pos
        is_intermediate = result_mode == "intermediate"
        slab_offset = self._slabs[req_id].offset if req_id in self._slabs else 0

        first_position = start_pos
        tokens: list[int] = []
        hidden_rows: list[mx.array] = []

        # A pending row from this request's previous (intermediate) chunk sits one
        # position before this chunk; its next token is this chunk's first token.
        # Prepend it only when it is contiguous with both the slab tail and this
        # chunk (i.e. no intervening preemption); any other pending is stale.
        pending = self._pending_last_hidden.pop(req_id, None)
        if (
            pending is not None
            and num_rows > 0
            and pending[0] == start_pos - 1
            and pending[0] == slab_offset
        ):
            first_position = pending[0]
            tokens.append(token_ids[0])
            hidden_rows.append(pending[1][0])

        ready_rows = num_rows - 1 if is_intermediate else num_rows
        for j in range(ready_rows):
            tokens.append(
                token_ids[j + 1] if j < num_rows - 1 else ctx.prefill_token_ids[index]
            )
            hidden_rows.append(hidden[row_start + j])

        # Align the slab (truncate on a rewind, fail loud on a hole) BEFORE
        # recording this chunk's deferred row, so a recompute that both rewinds
        # and defers keeps the freshly-deferred row.
        slab = self._align_slab(req_id, first_position)

        if is_intermediate and num_rows > 0:
            # Defer the last row: carry a materialized copy of its hidden state so
            # it survives past this step's target_hidden_states.
            last_hidden = mx.array(hidden[row_start + num_rows - 1])[None]
            self._pending_last_hidden[req_id] = (start_pos + num_rows - 1, last_hidden)

        if not tokens:
            # Purely deferred single-row intermediate chunk with no pending: only
            # the slab (aligned above) needs to exist for the next step.
            return None

        return _SlotBatch(
            req_id=req_id,
            first_position=first_position,
            tokens=tokens,
            hidden_rows=hidden_rows,
            draft=not is_intermediate,
            slab=slab,
        )

    # -- slab bookkeeping -------------------------------------------------------

    def _align_slab(self, req_id: str, first_position: int) -> MTPSlab:
        """Return the request's slab positioned at ``first_position``.

        ``first_position < offset`` is a scheduler preemption-recompute: the same
        committed tokens are re-forwarded, so truncating and re-ingesting rebuilds
        the identical slab. ``first_position > offset`` is a hole — target hidden
        states for the gap were never observed and cannot be repaired — so fail
        loud. Any stale deferred (pending) row is resolved by the caller before
        this runs, so alignment never touches ``_pending_last_hidden``.
        """
        slab = self._slabs.get(req_id)
        if slab is None:
            slab = MTPSlab()
            self._slabs[req_id] = slab

        if first_position < slab.offset:
            logger.info(
                "Native MTP slab for request %s rewound from %d to %d "
                "(scheduler recompute); re-ingesting.",
                req_id,
                slab.offset,
                first_position,
            )
            slab.truncate_to(first_position)
        elif first_position > slab.offset:
            raise RuntimeError(
                f"Native MTP slab for request {req_id!r} has a hole: next slot is "
                f"at position {first_position} but only {slab.offset} positions "
                "were ingested. Target hidden states for the gap were never "
                "observed."
            )
        return slab

    def _prune_finished(self, request_states: Any) -> None:
        for req_id in list(self._slabs):
            if req_id not in request_states:
                del self._slabs[req_id]
        for req_id in list(self._pending_last_hidden):
            if req_id not in request_states:
                del self._pending_last_hidden[req_id]


__all__ = [
    "MTPSlab",
    "NativeMTPProposer",
]
