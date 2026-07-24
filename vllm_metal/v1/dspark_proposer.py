# SPDX-License-Identifier: Apache-2.0
"""DSpark (DeepSeek EAGLE3+Markov) speculative-decoding proposer for Metal.

DSpark is an EAGLE-family drafter: a small backbone that cross-attends over the
*target's* fused hidden states (captured from a selection of target layers),
plus a rank-256 Markov head + optional confidence head. Unlike Gemma4 MTP
(which shares the target KV), DSpark keeps its **own** per-layer context KV
(:class:`CtxCache`) that grows with the committed tokens — so the proposer is
stateful per request, closer to :class:`DraftModelProposer` in shape but
consuming target hidden states (EAGLE-style) instead of running autoregressively.

Lifecycle (per request), mirroring the reference implementation:

- **First draft** (``n_cached == 0``): seed the drafter context with the prompt's
  fused hidden, using the prefill's captured rows when available (non-chunked
  prefill) or re-running a tapped forward otherwise.
- **Each later draft**: ingest the newly-committed positions' fused hidden from
  this step's decode segment, then draft a block.

Batched drafter forward: context update stays per-request (cheap), but the
drafter backbone runs **once** across all eligible requests with a batched
mask (matching the reference implementation's batch_engine). This eliminates
the per-request loop that dominated concurrent workloads.

The context invariant is ``n_cached == committed_len - 1`` (context holds every
committed token except the current pending, which seeds the draft block).

Greedy only (matches the shared :meth:`verify_greedy` verifier).
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING

import mlx.core as mx
from vllm.logger import init_logger
from vllm.v1.outputs import DraftTokenIds

from vllm_metal.v1.dspark.loader import DSparkConfig, DSparkDrafter
from vllm_metal.v1.hidden_state_tap import run_backbone_with_capture

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from vllm_metal.v1.model_runner import MetalModelRunner, RequestState
    from vllm_metal.v1.proposer import ProposeContext
    from vllm_metal.v1.spec_decode import (
        PagedDecodeSegment,
        SpeculativeDecodeController,
    )

logger = init_logger(__name__)


def _ctx_block_mask(ctx_lens: list[int], n_block: int) -> mx.array:
    """Boolean ``[B, 1, K, Lmax+K]`` mask for batched drafter cross-attention.

    Every block position attends its row's valid context (columns ``< lens[b]``)
    plus ALL block positions (columns ``>= Lmax``, bidirectional) — matching the
    single-seq full-attention path (``mask=None``) exactly per row. Ported from
    the reference implementation's ``_ctx_block_mask``.
    """
    num_reqs = len(ctx_lens)
    max_ctx = max(ctx_lens)
    lens_arr = mx.array(ctx_lens, dtype=mx.int32)[:, None, None, None]  # [B,1,1,1]
    j = mx.arange(max_ctx + n_block, dtype=mx.int32)[
        None, None, None, :
    ]  # [1,1,1,Lmax+K]
    m = (j >= max_ctx) | (j < lens_arr)  # block region OR valid context
    return mx.broadcast_to(m, (num_reqs, 1, n_block, max_ctx + n_block))


@dataclass
class _DraftPlan:
    req_id: str
    pending: int
    n_cached: int
    ctx_caches: list | None  # None means needs seeding
    cap: int
    prompt_ids: list[int] | None = None  # for batched seeding


class DSparkProposer:
    """:class:`vllm_metal.v1.proposer.MetalProposer` backed by a DSpark drafter."""

    def __init__(
        self,
        *,
        drafter: DSparkDrafter,
        config: DSparkConfig,
        runner: MetalModelRunner,
        controller: SpeculativeDecodeController,
    ) -> None:
        self._drafter = drafter
        self._config = config
        self._runner = runner
        self._controller = controller
        self.capture_layer_ids: list[int] = list(config.target_layer_ids)
        self._block_size = int(config.block_size)
        self._mask_token_id = int(config.mask_token_id)
        self._ctx_caches: dict[str, list] = {}
        self._n_cached: dict[str, int] = {}
        self._pending_seed: dict[str, mx.array] = {}  # prefill fused hidden
        # Per-step draft cap: batched drafter attention is O(B·ctx_len·K).
        # Drafting every eligible request every step saturates the GPU on
        # concurrent workloads.  A moderate cap keeps per-step latency low
        # while still covering the requests the scheduler can verify.
        self._max_drafts_per_step: int = 32

    # -- MetalProposer protocol -----------------------------------------

    def needs_target_hidden_states(
        self,
        decode_segments: Sequence[PagedDecodeSegment],
        *,
        has_final_prefill: bool,
    ) -> bool:
        # Decode-only: capture fused hidden each step for context growth.
        # Prefill uses the fast backbone() path (no capture overhead).
        # New-request prompt seeding is done via _seed_prompts_batched —
        # one batched forward instead of 100 serial _seed_prompt calls.
        return bool(decode_segments)

    def propose(self, ctx: ProposeContext) -> DraftTokenIds | None:
        num_speculative_tokens = ctx.num_speculative_tokens

        # Stash prefill fused hidden for new requests BEFORE eligibility
        # (the stash must run at the prefill step, even though there are
        # no decode segments to draft from yet). Eliminates the duplicate
        # full-prompt forward — the TTFT tax.
        if ctx.prefill_reqs and ctx.target_hidden_states is not None:
            cu = ctx.cu_seqlens
            num_decode = ctx.num_decode_segments
            for i, pr in enumerate(ctx.prefill_reqs):
                if pr.prompt_len is not None:  # final chunk
                    # cu_seqlens = [0, decode_0, …, decode_total, prefill_0, …]
                    # Skip past the decode entries to index the prefill rows.
                    start = cu[num_decode + i]
                    end = cu[num_decode + i + 1]
                    self._pending_seed[pr.req_id] = ctx.target_hidden_states[start:end]

        if num_speculative_tokens <= 0:
            return None

        self._prune_finished(ctx.request_states)
        eligible = self._controller.draft_eligible_requests(
            ctx.decode_reqs,
            ctx.decode_token_ids,
            ctx.prefill_reqs,
            ctx.prefill_result_modes,
            ctx.request_states,
            logitsprocs=ctx.logitsprocs,
        )
        if not eligible:
            logger.warning(
                "DSpark: propose() — eligible=0 (decode_segments=%d, decode_reqs=%d)",
                len(ctx.decode_segments),
                len(ctx.decode_reqs),
            )
            return None

        segment_by_id = {seg.req_id: seg for seg in ctx.decode_segments}
        _base_cap = min(num_speculative_tokens, self._block_size)

        # Phase 1: per-request context update (seeding + growth — cheap).
        plans: list[_DraftPlan] = []
        # Prefill-finalizing requests (those in ctx.prefill_reqs that are
        # eligible) don't yet have decode segments.  Identify them by
        # req_id so we can seed their drafter context now.  We cannot use
        # generated_tokens==0 because _sample_paged_batch sets it to 1 in
        # the same step, before propose() runs.
        prefill_req_ids = {pr.req_id for pr in ctx.prefill_reqs}
        for req_id, state in eligible:
            segment = segment_by_id.get(req_id)
            if segment is None:
                # Request not yet in decode_segments.  Only seed genuinely
                # new (prefill-finalizing) requests here; decode requests
                # that happen to be absent from this batch are skipped —
                # they will be picked up when they appear in a decode
                # segment.  Use the stashed prefill hidden when available
                # (captured at the top of propose — no extra forward);
                # fall back to batched prompt seeding only for chunked
                # prefill where no stashed hidden exists.
                if req_id in prefill_req_ids:
                    fused_prefill = self._pending_seed.pop(req_id, None)
                    if fused_prefill is not None:
                        p_len = fused_prefill.shape[0]
                        ctx_caches = self._drafter.make_ctx_cache()
                        self._drafter.update_context(
                            fused_prefill[None, :, :],
                            ctx_offset=0,
                            ctx_caches=ctx_caches,
                        )
                        self._ctx_caches[req_id] = ctx_caches
                        self._n_cached[req_id] = p_len
                        token_ids = state.token_ids
                        if token_ids:
                            plans.append(
                                _DraftPlan(
                                    req_id,
                                    int(token_ids[-1]),
                                    p_len,
                                    ctx_caches,
                                    _base_cap,
                                )
                            )
                    else:
                        token_ids = state.token_ids
                        if token_ids:
                            plans.append(
                                _DraftPlan(
                                    req_id,
                                    int(token_ids[-1]),
                                    0,
                                    None,  # seeded below by _seed_prompts_batched
                                    _base_cap,
                                    prompt_ids=list(token_ids[:-1]),
                                )
                            )
                continue
            plan = self._ensure_context(ctx, state, segment, _base_cap)
            if plan is not None:
                plans.append(plan)

        if not plans:
            logger.warning(
                "DSpark: propose() — %d eligible, 0 plans (all skipped)",
                len(eligible),
            )
            return None

        # Seed new requests in one batched target forward (one weight-read).
        self._seed_prompts_batched(plans)

        # Drop plans that weren't seeded (empty ctx_caches).
        plans = [p for p in plans if p.ctx_caches is not None]
        if not plans:
            return None

        # Phase 2: batched backbone forward — one call across all requests.
        # Cap the batch size: batched attention over many large contexts
        # saturates the GPU and drags per-step latency past the breakeven
        # point on concurrent workloads.  Context ingestion (Phase 1) still
        # runs for every eligible request so caches stay current.
        if len(plans) > self._max_drafts_per_step:
            plans = plans[: self._max_drafts_per_step]
        req_ids, draft_rows = self._batch_draft(plans)
        if not req_ids:
            logger.warning("DSpark: _batch_draft returned empty (plans=%d)", len(plans))
            return None
        logger.info(
            "DSpark: propose() — %d eligible, %d plans, %d drafted",
            len(eligible),
            len(plans),
            len(req_ids),
        )
        return DraftTokenIds(req_ids=req_ids, draft_token_ids=draft_rows)

    # -- context management (per-request, cheap) --------------------------

    def _ensure_context(
        self,
        ctx: ProposeContext,
        state: RequestState,
        segment: PagedDecodeSegment,
        cap: int,
    ) -> _DraftPlan | None:
        req_id = segment.req_id
        token_ids = state.token_ids
        committed_len = len(token_ids)
        if committed_len < 1:
            return None
        pending = int(token_ids[-1])

        ctx_caches = self._ctx_caches.get(req_id)
        if ctx_caches is not None:
            new_to_ingest = (committed_len - 1) - self._n_cached[req_id]
            if new_to_ingest > 0:
                # Cap at the rows actually available in this step's segment.
                # When the scheduler skips a request for several steps the
                # hidden states for those skipped positions are already lost,
                # but capping lets the request stay in the draft pool rather
                # than permanently dropping it (the Markov head still works
                # with a partial context — the missing positions represent
                # tokens whose hidden states were never captured).
                take = min(new_to_ingest, segment.num_query_tokens)
                fused = self._segment_rows(ctx.target_hidden_states, segment, take)
                if fused is None:
                    return None
                self._drafter.update_context(
                    fused,
                    ctx_offset=self._n_cached[req_id],
                    ctx_caches=ctx_caches,
                )
                self._n_cached[req_id] += take
            elif new_to_ingest < 0:
                logger.warning(
                    "DSpark: req %s n_cached=%d > committed_len-1=%d; resetting",
                    req_id,
                    self._n_cached[req_id],
                    committed_len - 1,
                )
                self._n_cached[req_id] = committed_len - 1
        else:
            # New request: try stashed prefill hidden first (captured during
            # the prefill forward — no duplicate forward). Fall back to
            # batched seeding (re-run backbone) only for chunked prefill.
            fused_prefill = self._pending_seed.pop(req_id, None)
            if fused_prefill is not None:
                p_len = fused_prefill.shape[0]
                ctx_caches = self._drafter.make_ctx_cache()
                self._drafter.update_context(
                    fused_prefill[None, :, :],
                    ctx_offset=0,
                    ctx_caches=ctx_caches,
                )
                self._ctx_caches[req_id] = ctx_caches
                self._n_cached[req_id] = p_len
            # If no stashed prefill, prompt_ids marks for batched seeding.
        return _DraftPlan(
            req_id,
            pending,
            self._n_cached.get(req_id, 0),
            ctx_caches,
            cap,
            prompt_ids=list(token_ids[:-1]) if ctx_caches is None else None,
        )

    # -- batched forward (one backbone call) -----------------------------

    def _batch_draft(
        self, plans: list[_DraftPlan]
    ) -> tuple[list[str], list[list[int]]]:
        num_reqs = len(plans)
        block_sz = self._block_size

        # Batched noise [num_reqs, block_sz, H].
        block_ids_list = [
            [p.pending] + [self._mask_token_id] * (block_sz - 1) for p in plans
        ]
        noise = self._drafter.embed(mx.array(block_ids_list))

        # Batched block offsets [num_reqs].
        offsets = [p.n_cached for p in plans]
        off_arr = mx.array(offsets, dtype=mx.int32)

        # Per-layer batched K/V from per-request CtxCache.
        n_layers = len(self._drafter.layers)
        batched_ctx = []
        for layer_idx in range(n_layers):
            lens = [p.ctx_caches[layer_idx].length for p in plans]
            max_len = max(lens) if max(lens) > 0 else 1
            ref = plans[0].ctx_caches[layer_idx]
            n_heads_k = ref.k.shape[1]
            dim_k, dim_v = ref.k.shape[3], ref.v.shape[3]
            keys = mx.zeros((num_reqs, n_heads_k, max_len, dim_k), dtype=ref.k.dtype)
            vals = mx.zeros((num_reqs, n_heads_k, max_len, dim_v), dtype=ref.v.dtype)
            for b, p in enumerate(plans):
                ctx_len_b = lens[b]
                if ctx_len_b > 0:
                    keys[b : b + 1, :, :ctx_len_b, :] = p.ctx_caches[layer_idx].k
                    vals[b : b + 1, :, :ctx_len_b, :] = p.ctx_caches[layer_idx].v
            batched_ctx.append(SimpleNamespace(k=keys, v=vals))

        # Batched mask.
        mask = _ctx_block_mask(offsets, block_sz)

        # One backbone forward.
        batched_hidden = self._drafter.backbone(
            noise, off_arr, batched_ctx, mask=mask
        )  # [B, K, H]

        # Batched logits + greedy drafting (vectorized across rows).
        cap = plans[0].cap
        head_hidden = batched_hidden[:, :cap, :]  # [B, cap, H]
        base_logits = self._drafter.compute_logits(head_hidden)  # [B, cap, V]
        all_pending = mx.array([p.pending for p in plans])
        if self._drafter.markov_head is not None:
            # Markov head: sequential over cap positions, vectorized across rows.
            prev = all_pending
            drafts = []
            for i in range(cap):
                step = base_logits[:, i, :] + self._drafter.markov_head.step_bias(prev)
                nxt = mx.argmax(step, axis=-1)
                drafts.append(nxt)
                prev = nxt
            draft_arr = mx.stack(drafts, axis=1)  # [B, cap]
        else:
            draft_arr = mx.argmax(base_logits, axis=-1)  # [B, cap]
        mx.eval(draft_arr)
        rows = draft_arr.tolist()
        return (
            [p.req_id for p in plans],
            [list(r) for r in rows],
        )

    # -- helpers ---------------------------------------------------------

    def _segment_rows(
        self,
        target_hidden_states: mx.array,
        segment: PagedDecodeSegment,
        count: int,
    ) -> mx.array | None:
        start = segment.start_row
        if start + count > segment.start_row + segment.num_query_tokens:
            return None
        if start + count > target_hidden_states.shape[0]:
            return None
        rows = target_hidden_states[start : start + count]
        if rows.shape[0] != count:
            return None
        return rows[None, :, :]  # [1, count, k*H]

    def _seed_prompts_batched(self, plans: list[_DraftPlan]) -> None:
        """Seed ALL new requests in one batched target forward."""
        seed = [(idx, p) for idx, p in enumerate(plans) if p.ctx_caches is None]
        if not seed:
            return
        body = self._runner._model_adapter._target_backbone(self._runner._forward_model)
        if body is None:
            for _, p in seed:
                self._seed_prompt(p.req_id, p.prompt_ids)
                p.ctx_caches = self._ctx_caches.get(p.req_id)
                p.n_cached = self._n_cached.get(p.req_id, 0)
            return
        n_layers = len(body.layers)
        max_len = max(len(p.prompt_ids) for _, p in seed) if seed else 0
        batched_ids = []
        for _, p in seed:
            pad = max_len - len(p.prompt_ids)
            batched_ids.append(list(p.prompt_ids) + [0] * pad)
        _, fused = run_backbone_with_capture(
            body,
            mx.array(batched_ids),
            cache=[None] * n_layers,
            layer_ids=self.capture_layer_ids,
        )
        for batch_pos, (_, p) in enumerate(seed):
            p_len = len(p.prompt_ids)
            ctx_caches = self._drafter.make_ctx_cache()
            self._drafter.update_context(
                fused[batch_pos : batch_pos + 1, :p_len, :],
                ctx_offset=0,
                ctx_caches=ctx_caches,
            )
            self._ctx_caches[p.req_id] = ctx_caches
            self._n_cached[p.req_id] = p_len
            p.ctx_caches = ctx_caches
            p.n_cached = p_len

    def _seed_prompt(self, req_id: str, prompt_ids: Sequence[int]) -> bool:
        if not prompt_ids:
            ctx_caches = self._drafter.make_ctx_cache()
            self._ctx_caches[req_id] = ctx_caches
            self._n_cached[req_id] = 0
            return True
        body = self._runner._model_adapter._target_backbone(self._runner._forward_model)
        if body is None:
            logger.warning(
                "DSpark: cannot seed prompt — target has no `.model` backbone"
            )
            return False
        ids = mx.array([list(prompt_ids)])
        _, fused = run_backbone_with_capture(
            body,
            ids,
            cache=[None] * len(body.layers),
            layer_ids=self.capture_layer_ids,
        )
        ctx_caches = self._drafter.make_ctx_cache()
        self._drafter.update_context(fused, ctx_offset=0, ctx_caches=ctx_caches)
        self._ctx_caches[req_id] = ctx_caches
        self._n_cached[req_id] = int(len(prompt_ids))
        return True

    def _prune_finished(self, request_states: Mapping[str, RequestState]) -> None:
        for req_id in list(self._ctx_caches.keys()):
            if req_id not in request_states:
                self._ctx_caches.pop(req_id, None)
                self._n_cached.pop(req_id, None)
                self._pending_seed.pop(req_id, None)
