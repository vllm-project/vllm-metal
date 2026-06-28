# SPDX-License-Identifier: Apache-2.0
"""Draft-model speculative decoding proposer for the Metal paged path.

A :class:`DraftModelProposer` drafts with a *separate* full model (vLLM
``method="draft_model"``). It owns:

- its own MLX draft model (loaded independently of the target),
- its own physical :class:`MetalPagedKVCache`, sized to the *target's*
  ``num_blocks`` (a second KV store), and
- its own block allocator over that ``num_blocks`` pool. The draft *cannot*
  reuse the target's scheduler-allocated ``block_ids``: it drafts
  ``num_speculative_tokens`` positions AHEAD of the committed length, while the
  target's block table is exact-fit for that committed length and runs out at
  the next block boundary (regression-pinned by
  ``test_long_prompt_crosses_block_boundary``). K/V values still line up
  position-for-position across the two caches; only the block table is owned
  separately.

Each scheduler step ``propose()`` runs, batched over the dynamic mixed batch:

1. **Ingest** — for every drafting request, run the draft model over the
   committed-token suffix it has not yet written into the draft cache
   (``token_ids[draft_seq_len:committed_len]``) as one batched prefill. This
   advances the draft cache to the committed length and overwrites any stale
   KV left by rejected drafts from the previous step. The last row per request
   yields the first draft token.
2. **Decode** — ``num_speculative_tokens - 1`` batched single-token steps, each
   feeding the previous draft token to produce the next.

The verify half is unchanged: drafts are handed back via ``take_draft_token_ids``
and verified next step by ``SpeculativeDecodeController.verify_greedy``.

KV-budget note: the draft cache is a second store with the *same block count*
as the target's cache but sized for the draft model's (smaller) dimensions — so
it is materially smaller than the target's KV cache for a small draft. It is
allocated *after* the target budget is computed, so it is not subtracted from
the target KV budget. This is safe only with headroom
(``VLLM_METAL_MEMORY_FRACTION`` well below 1.0); a budget split is a follow-up.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import mlx.core as mx
from mlx_lm import load as mlx_lm_load
from vllm.logger import init_logger
from vllm.v1.outputs import DraftTokenIds

from vllm_metal.attention.context import (
    OffsetCache,
    clear_context,
    prepare_unified,
)
from vllm_metal.attention.runtime.mha import MHAPagedAttentionRuntime
from vllm_metal.utils import get_model_download_path
from vllm_metal.v1.mlx_lm_paths import mlx_lm_compatible_model_path

if TYPE_CHECKING:
    from collections.abc import Callable

    from vllm.config.speculative import SpeculativeConfig

    from vllm_metal.v1.model_runner import RequestState
    from vllm_metal.v1.proposer import ProposeContext
    from vllm_metal.v1.spec_decode import SpeculativeDecodeController

logger = init_logger(__name__)


@dataclass(frozen=True, slots=True)
class _DraftDims:
    num_layers: int
    num_kv_heads: int
    head_dim: int


@dataclass(frozen=True, slots=True)
class _DraftPlan:
    """One request's drafting plan for the current step."""

    req_id: str
    block_ids: list[int]
    committed_len: int
    draft_seq_len: int
    ingest_tokens: list[int]


class DraftModelProposer:
    """:class:`vllm_metal.v1.proposer.MetalProposer` backed by a separate model."""

    def __init__(
        self,
        *,
        model: Any,
        block_size: int,
        num_blocks: int,
        num_layers: int,
        num_speculative_tokens: int,
        controller: SpeculativeDecodeController,
        extract_logits: Callable[[Any], mx.array],
    ) -> None:
        self._model = model
        self._block_size = block_size
        self._num_speculative_tokens = num_speculative_tokens
        self._controller = controller
        self._extract_logits = extract_logits
        # Stateless RoPE/mask shims for the draft forward (one per layer). The
        # real per-request offsets come from the paged context, so these carry
        # no state — allocate once and reuse across steps, not per propose().
        self._offset_caches = [OffsetCache(0) for _ in range(num_layers)]
        # Committed positions written into the draft cache, per request.
        self._draft_seq_lens: dict[str, int] = {}
        # The draft owns its block allocation: it drafts K positions AHEAD of
        # what the target has committed, so it cannot reuse the target's
        # block_ids (those exactly fit the committed length and run out at the
        # next block boundary). Each request draws draft blocks from this pool.
        self._free_blocks: list[int] = list(range(num_blocks))
        self._req_blocks: dict[str, list[int]] = {}

    # -- construction --------------------------------------------------------

    @classmethod
    def build(
        cls,
        *,
        speculative_config: SpeculativeConfig,
        controller: SpeculativeDecodeController,
        extract_logits: Callable[[Any], mx.array],
        num_blocks: int,
        block_size: int,
        dtype: mx.Dtype,
    ) -> DraftModelProposer:
        model, dims = _load_draft_model(speculative_config)
        backend = MHAPagedAttentionRuntime(
            num_layers=dims.num_layers,
            num_kv_heads=dims.num_kv_heads,
            head_dim=dims.head_dim,
            block_size=block_size,
            dtype=dtype,
        )
        backend.initialize(num_blocks)
        n_patched = backend.patch_model(model)
        logger.info(
            "Draft model loaded for speculative decoding: %s "
            "(layers=%d, kv_heads=%d, head_dim=%d, patched=%d, num_blocks=%d)",
            speculative_config.draft_model_config.model,
            dims.num_layers,
            dims.num_kv_heads,
            dims.head_dim,
            n_patched,
            num_blocks,
        )
        return cls(
            model=model,
            block_size=block_size,
            num_blocks=num_blocks,
            num_layers=dims.num_layers,
            num_speculative_tokens=speculative_config.num_speculative_tokens,
            controller=controller,
            extract_logits=extract_logits,
        )

    # -- MetalProposer protocol ---------------------------------------------

    def needs_target_hidden_states(
        self,
        decode_segments: Any,
        *,
        has_final_prefill: bool,
    ) -> bool:
        # A standalone draft model consumes only token ids; it never reads the
        # target's hidden states (mirrors upstream pass_hidden_states_to_model=False).
        return False

    def propose(self, ctx: ProposeContext) -> DraftTokenIds | None:
        self._prune_finished(ctx.request_states)
        plans = self._collect_draft_plans(ctx)
        if not plans:
            return None

        # Step 1: ingest committed suffixes; first draft token per request.
        draft_cols: list[mx.array] = [
            self._ingest_and_draft_first(plans, self._offset_caches)
        ]
        # Steps 2..K: single-token decode per request.
        for draft_index in range(1, self._num_speculative_tokens):
            draft_cols.append(
                self._draft_step(
                    plans, draft_cols[-1], draft_index, self._offset_caches
                )
            )

        drafts = mx.stack(draft_cols, axis=1)  # [num_plans, K]
        mx.eval(drafts)
        rows: list[list[int]] = drafts.tolist()  # type: ignore[assignment]

        # The draft cache now holds KV through committed_len for each request.
        for plan in plans:
            self._draft_seq_lens[plan.req_id] = plan.committed_len

        return DraftTokenIds(
            req_ids=[plan.req_id for plan in plans],
            draft_token_ids=[[int(token) for token in row] for row in rows],
        )

    # -- internals -----------------------------------------------------------

    def _prune_finished(self, request_states: Any) -> None:
        for req_id in list(self._req_blocks.keys()):
            if req_id not in request_states:
                self._free_blocks.extend(self._req_blocks.pop(req_id))
        for req_id in list(self._draft_seq_lens.keys()):
            if req_id not in request_states:
                del self._draft_seq_lens[req_id]

    def _collect_draft_plans(self, ctx: ProposeContext) -> list[_DraftPlan]:
        plans: list[_DraftPlan] = []
        seen: set[str] = set()

        for (req_id, state), sampled_ids in zip(
            ctx.decode_reqs, ctx.decode_token_ids, strict=True
        ):
            if not sampled_ids:
                continue
            if not self._controller.can_draft_greedy(
                req_id, state, logitsprocs=ctx.logitsprocs
            ):
                continue
            plan = self._make_plan(req_id, state)
            if plan is not None:
                plans.append(plan)
                seen.add(req_id)

        for prefill, result_mode in zip(
            ctx.prefill_reqs, ctx.prefill_result_modes, strict=True
        ):
            if result_mode == "intermediate" or prefill.req_id in seen:
                continue
            state = ctx.request_states.get(prefill.req_id)
            if state is None or not self._controller.can_draft_greedy(
                prefill.req_id, state, logitsprocs=ctx.logitsprocs
            ):
                continue
            plan = self._make_plan(prefill.req_id, state)
            if plan is not None:
                plans.append(plan)

        return plans

    def _make_plan(self, req_id: str, state: RequestState) -> _DraftPlan | None:
        committed_len = len(state.token_ids)
        draft_seq_len = self._draft_seq_lens.get(req_id, 0)
        if draft_seq_len >= committed_len:
            # No newly committed tokens to ingest (should not happen for an
            # accepted decode step or a finalized prefill); skip rather than
            # emit an empty forward.
            return None
        # Draft positions reach committed_len + K - 1; size the draft block
        # table to cover them from the draft's own pool.
        block_ids = self._ensure_draft_blocks(
            req_id, committed_len + self._num_speculative_tokens
        )
        return _DraftPlan(
            req_id=req_id,
            block_ids=block_ids,
            committed_len=committed_len,
            draft_seq_len=draft_seq_len,
            ingest_tokens=list(state.token_ids[draft_seq_len:committed_len]),
        )

    def _ensure_draft_blocks(self, req_id: str, num_positions: int) -> list[int]:
        """Grow this request's draft block table to cover ``num_positions``."""
        needed = (num_positions + self._block_size - 1) // self._block_size
        blocks = self._req_blocks.setdefault(req_id, [])
        while len(blocks) < needed:
            if not self._free_blocks:
                raise RuntimeError(
                    f"Draft KV cache exhausted: request {req_id!r} needs "
                    f"{needed} blocks but the draft pool is empty "
                    f"({len(self._req_blocks)} active requests). "
                    "Lower --max-num-seqs or raise VLLM_METAL_MEMORY_FRACTION."
                )
            blocks.append(self._free_blocks.pop())
        return blocks

    def _ingest_and_draft_first(
        self, plans: list[_DraftPlan], offset_caches: list[OffsetCache]
    ) -> mx.array:
        prefill_specs = [
            (plan.block_ids, len(plan.ingest_tokens), plan.draft_seq_len)
            for plan in plans
        ]
        packed: list[int] = []
        last_rows: list[int] = []
        for plan in plans:
            packed.extend(plan.ingest_tokens)
            last_rows.append(len(packed) - 1)
        input_ids = mx.array([packed], dtype=mx.int32)

        prepare_unified([], prefill_specs, self._block_size)
        try:
            logits = self._extract_logits(self._model(input_ids, cache=offset_caches))
        finally:
            clear_context()

        last = mx.take(logits[0], mx.array(last_rows, dtype=mx.int32), axis=0)
        return mx.argmax(last, axis=-1)

    def _draft_step(
        self,
        plans: list[_DraftPlan],
        prev_tokens: mx.array,
        draft_index: int,
        offset_caches: list[OffsetCache],
    ) -> mx.array:
        # prev_tokens[i] sits at position committed_len_i + (draft_index - 1).
        decode_specs = [
            (plan.block_ids, plan.committed_len + draft_index - 1, 1) for plan in plans
        ]
        input_ids = prev_tokens[None, :].astype(mx.int32)

        prepare_unified(decode_specs, [], self._block_size)
        try:
            logits = self._extract_logits(self._model(input_ids, cache=offset_caches))
        finally:
            clear_context()

        return mx.argmax(logits[0], axis=-1)


def _load_draft_model(
    speculative_config: SpeculativeConfig,
) -> tuple[Any, _DraftDims]:
    draft_model_config = speculative_config.draft_model_config
    if draft_model_config is None:
        raise ValueError(
            "draft_model speculative decoding requires a draft_model_config"
        )

    # Its own instance (patched to its own draft KV cache, so it must not alias
    # the target). AWQ / variable-head-dim drafts aren't handled here yet
    # (canonical loader: ModelLifecycle._load_generation_model).
    model_path = get_model_download_path(draft_model_config.model)
    with mlx_lm_compatible_model_path(model_path) as compatible_path:
        model, _ = mlx_lm_load(str(compatible_path))

    hf = draft_model_config.hf_config
    num_layers = int(hf.num_hidden_layers)
    num_attention_heads = int(hf.num_attention_heads)
    num_kv_heads = int(getattr(hf, "num_key_value_heads", num_attention_heads))
    head_dim = int(getattr(hf, "head_dim", 0) or hf.hidden_size // num_attention_heads)
    return model, _DraftDims(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )
