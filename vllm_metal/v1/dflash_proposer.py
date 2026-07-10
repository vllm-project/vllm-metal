# SPDX-License-Identifier: Apache-2.0
"""DFlash speculative-decoding proposer for the Metal paged path.

Ports vLLM 0.24.0's DFlash proposer (``vllm/v1/spec_decode/dflash.py``) to the
Metal proposer seam. Unlike :class:`DraftModelProposer`, the draft model never
runs autoregressively and never re-reads token ids:

1. **Ingest** — every step, the newly committed rows' *target aux hidden
   states* (concat of ``aux_hidden_state_layer_ids`` residual streams) are
   ``fc``-combined and projected into per-layer context K/V, appended to a
   per-request contiguous KV slab. Decode rows contribute only their accepted
   prefix; prefill chunks (including intermediate ones — their hidden states
   exist only in the step that ran them) contribute every row.
2. **Draft** — one forward over ``[bonus_token, mask * K]`` queries attending
   non-causally over [context, queries]; each mask row predicts the token at
   its own position (upstream ``sample_off=1``).

The query tokens' K/V is never persisted: context K/V for accepted tokens is
always re-projected from target hidden states the following step, so the draft
needs no paged cache, block tables, or cache-write kernels at all.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import mlx.core as mx
from vllm.logger import init_logger
from vllm.v1.outputs import DraftTokenIds

from vllm_metal.utils import get_model_download_path
from vllm_metal.v1.dflash_model import (
    DFlashDraftModel,
    DFlashModelArgs,
    load_dflash_draft_model,
)

if TYPE_CHECKING:
    from vllm.config.speculative import SpeculativeConfig

    from vllm_metal.v1.proposer import ProposeContext
    from vllm_metal.v1.spec_decode import SpeculativeDecodeController

logger = init_logger(__name__)

# Slab growth quantum (positions), mirroring mlx_lm's KVCache step.
_SLAB_STEP = 256


class _ContextKVSlab:
    """Per-request growable context K/V for all draft layers.

    Shapes: ``[num_layers, n_kv_heads, capacity, head_dim]``; the first
    ``ctx_len`` positions are valid. Slice-assignment keeps updates in-place
    under MLX's lazy evaluation (the mlx_lm KVCache pattern).
    """

    __slots__ = ("ctx_len", "keys", "values")

    def __init__(self) -> None:
        self.keys: mx.array | None = None
        self.values: mx.array | None = None
        self.ctx_len = 0

    def append(self, k: mx.array, v: mx.array) -> None:
        """Append ``[L, n_kv, T, hd]`` context K/V at positions ctx_len..+T."""
        num_new = k.shape[2]
        needed = self.ctx_len + num_new
        if self.keys is None or needed > self.keys.shape[2]:
            capacity = ((needed + _SLAB_STEP - 1) // _SLAB_STEP) * _SLAB_STEP
            grown_k = mx.zeros(
                (k.shape[0], k.shape[1], capacity, k.shape[3]), dtype=k.dtype
            )
            grown_v = mx.zeros_like(grown_k)
            if self.keys is not None and self.ctx_len > 0:
                grown_k[:, :, : self.ctx_len, :] = self.keys[:, :, : self.ctx_len, :]
                grown_v[:, :, : self.ctx_len, :] = self.values[:, :, : self.ctx_len, :]
            self.keys, self.values = grown_k, grown_v
        self.keys[:, :, self.ctx_len : needed, :] = k
        self.values[:, :, self.ctx_len : needed, :] = v
        self.ctx_len = needed

    def view(self) -> tuple[mx.array, mx.array]:
        assert self.keys is not None and self.values is not None
        return (
            self.keys[:, :, : self.ctx_len, :],
            self.values[:, :, : self.ctx_len, :],
        )


class DFlashProposer:
    """:class:`vllm_metal.v1.proposer.MetalProposer` for DFlash drafts."""

    def __init__(
        self,
        *,
        model: DFlashDraftModel,
        model_args: DFlashModelArgs,
        aux_layer_ids: list[int],
        controller: SpeculativeDecodeController,
    ) -> None:
        self._model = model
        self._args = model_args
        self._controller = controller
        # 0-indexed target layers after which the residual stream is captured;
        # read by the model runner / adapter each forward.
        self.aux_hidden_state_layer_ids: list[int] = aux_layer_ids
        self._slabs: dict[str, _ContextKVSlab] = {}

    # -- construction ----------------------------------------------------------

    @classmethod
    def build(
        cls,
        *,
        speculative_config: SpeculativeConfig,
        controller: SpeculativeDecodeController,
        dtype: mx.Dtype,
    ) -> DFlashProposer:
        draft_model_config = speculative_config.draft_model_config
        if draft_model_config is None:
            raise ValueError("DFlash speculative decoding requires a draft model")
        model_path = get_model_download_path(draft_model_config.model)
        if not Path(model_path).is_dir():
            # HF repo id (get_model_download_path only localizes ModelScope
            # repos): resolve to the hub snapshot, downloading if needed.
            from huggingface_hub import snapshot_download

            model_path = snapshot_download(model_path)
        model, args, raw_config = load_dflash_draft_model(model_path, dtype=dtype)

        checkpoint_aux_ids = raw_config.get("aux_hidden_state_layer_ids")
        if not checkpoint_aux_ids:
            raise ValueError(
                "DFlash checkpoint config is missing aux_hidden_state_layer_ids"
            )
        if 0 in checkpoint_aux_ids:
            raise NotImplementedError(
                "aux_hidden_state_layer_ids containing 0 (post-embedding "
                "capture) is not supported on Metal yet"
            )
        # Checkpoint ids follow upstream's speculators convention: id ``j``
        # is the residual stream after 0-indexed target layer ``j - 1``
        # (vllm 0.24.0 llama.py forward loop + algos.py target_layer_ids).
        aux_layer_ids = [int(j) - 1 for j in checkpoint_aux_ids]

        algorithm = (raw_config.get("speculators_config") or {}).get("algorithm")
        logger.info(
            "DFlash drafter ready: %s (algorithm=%s, aux target layers=%s, "
            "mask_token=%d, draft_vocab=%d)",
            draft_model_config.model,
            algorithm,
            aux_layer_ids,
            args.mask_token_id,
            args.draft_vocab_size,
        )
        return cls(
            model=model,
            model_args=args,
            aux_layer_ids=aux_layer_ids,
            controller=controller,
        )

    # -- MetalProposer protocol -------------------------------------------------

    def needs_target_hidden_states(
        self,
        decode_segments: Any,
        *,
        has_final_prefill: bool,
    ) -> bool:
        # Context K/V is projected from target hidden states of EVERY
        # scheduled token — including intermediate prefill chunks, whose
        # hidden states exist only in the step that ran them. Always collect.
        return True

    def propose(self, ctx: ProposeContext) -> DraftTokenIds | None:
        self._prune_finished(ctx.request_states)
        aux = ctx.target_aux_hidden_states
        if aux is None:
            # The runner collects aux states on every paged text forward while
            # this drafter is installed; their absence means the batch ran an
            # execution path DFlash does not support (multimodal / pipeline
            # parallel). Fail loud rather than silently never drafting.
            raise RuntimeError(
                "DFlash drafting requires target aux hidden states, but none "
                "were collected this step; the current execution path "
                "(multimodal or pipeline-parallel forward) does not support "
                "DFlash speculative decoding."
            )

        combined = self._model.combine_hidden_states(aux)
        updated_slabs = self._ingest(ctx, combined)

        num_spec = ctx.num_speculative_tokens
        drafts: list[tuple[str, mx.array]] = []
        if num_spec > 0:
            drafts = self._draft(ctx, num_spec)

        # Every request drafts the same K tokens, so stack into one
        # ``[num_drafts, K]`` matrix and read it back with a single sync.
        draft_matrix = mx.stack([tokens for _, tokens in drafts]) if drafts else None

        # Force this step's slab updates and draft graph together; slabs
        # updated for requests that did not draft this step (e.g. intermediate
        # prefill chunks) must not accumulate lazy graphs across steps.
        eval_targets: list[mx.array] = [
            slab.keys for slab in updated_slabs if slab.keys is not None
        ]
        eval_targets.extend(
            slab.values for slab in updated_slabs if slab.values is not None
        )
        if draft_matrix is not None:
            eval_targets.append(draft_matrix)
        if eval_targets:
            mx.eval(*eval_targets)

        if draft_matrix is None:
            return None
        draft_token_ids: list[list[int]] = draft_matrix.tolist()  # type: ignore[assignment]
        return DraftTokenIds(
            req_ids=[req_id for req_id, _ in drafts],
            draft_token_ids=draft_token_ids,
        )

    # -- ingest ------------------------------------------------------------------

    def _ingest(self, ctx: ProposeContext, combined: mx.array) -> list[_ContextKVSlab]:
        """Append newly committed rows' context K/V to per-request slabs."""
        updated: list[_ContextKVSlab] = []

        for i, ((req_id, state), segment) in enumerate(
            zip(ctx.decode_reqs, ctx.decode_segments, strict=True)
        ):
            # Rows 0..len(sampled)-1 carry committed input tokens (accepted
            # prefix + the row that produced the correction/bonus token);
            # rejected draft rows must not enter the context.
            num_valid = len(ctx.decode_token_ids[i])
            if num_valid == 0 or not self._controller.greedy_sampling_params(
                state.sampling_params
            ):
                continue
            slab = self._slab_for(req_id, segment.cache_start_pos)
            rows = combined[segment.start_row : segment.start_row + num_valid]
            self._append_rows(slab, rows, segment.cache_start_pos)
            updated.append(slab)

        for i, prefill in enumerate(ctx.prefill_reqs):
            state = ctx.request_states.get(prefill.req_id)
            # New requests have no RequestState until their final chunk is
            # sampled; ingest by the prefill's own sampling params.
            sampling_params = (
                state.sampling_params
                if state is not None
                else getattr(prefill, "sampling_params", None)
            )
            if sampling_params is None or not self._controller.greedy_sampling_params(
                sampling_params
            ):
                continue
            row_start = ctx.cu_seqlens[ctx.num_decode_segments + i]
            num_rows = len(prefill.token_ids)
            slab = self._slab_for(prefill.req_id, prefill.start_pos)
            rows = combined[row_start : row_start + num_rows]
            self._append_rows(slab, rows, prefill.start_pos)
            updated.append(slab)

        return updated

    def _append_rows(
        self, slab: _ContextKVSlab, rows: mx.array, position_offset: int
    ) -> None:
        k, v = self._model.project_context_kv(rows, position_offset)
        slab.append(k, v)

    def _slab_for(self, req_id: str, first_position: int) -> _ContextKVSlab:
        """Return the request's slab, positionally aligned to ``first_position``.

        A rewind (``first_position < ctx_len``) is a scheduler
        preemption-recompute: the same committed tokens are being re-forwarded,
        so truncating and re-ingesting reproduces the identical context K/V.
        A forward hole (``first_position > ctx_len``) would mean target hidden
        states for some positions were never observed — an invariant violation
        that cannot be repaired — so it fails loud.
        """
        slab = self._slabs.get(req_id)
        if slab is None:
            slab = _ContextKVSlab()
            self._slabs[req_id] = slab
        if first_position < slab.ctx_len:
            logger.info(
                "DFlash context KV for request %s rewound from %d to %d "
                "(scheduler recompute); re-ingesting.",
                req_id,
                slab.ctx_len,
                first_position,
            )
            slab.ctx_len = first_position
        elif first_position > slab.ctx_len:
            raise RuntimeError(
                f"DFlash context KV for request {req_id!r} has a hole: next "
                f"row is at position {first_position} but only "
                f"{slab.ctx_len} positions were ingested. Target hidden "
                "states for the gap were never observed."
            )
        return slab

    # -- draft -------------------------------------------------------------------

    def _draft(self, ctx: ProposeContext, num_spec: int) -> list[tuple[str, mx.array]]:
        eligible = self._controller.draft_eligible_requests(
            ctx.decode_reqs,
            ctx.decode_token_ids,
            ctx.prefill_reqs,
            ctx.prefill_result_modes,
            ctx.request_states,
            logitsprocs=ctx.logitsprocs,
        )
        if not eligible:
            return []

        bonus_tokens = self._bonus_tokens(ctx)
        drafts: list[tuple[str, mx.array]] = []
        for req_id, _state in eligible:
            slab = self._slabs.get(req_id)
            bonus = bonus_tokens.get(req_id)
            if slab is None or slab.ctx_len == 0 or bonus is None:
                continue
            q_ids = mx.array(
                [bonus] + [self._args.mask_token_id] * num_spec, dtype=mx.int32
            )
            ctx_k, ctx_v = slab.view()
            hidden = self._model(q_ids, ctx_k, ctx_v, position_offset=slab.ctx_len)
            # Each mask row predicts its own position: greedy over draft vocab.
            mask_logits = self._model.compute_draft_logits(hidden[1:])
            tokens = self._model.map_draft_to_target(mx.argmax(mask_logits, axis=-1))
            drafts.append((req_id, tokens))
        return drafts

    def _bonus_tokens(self, ctx: ProposeContext) -> dict[str, int]:
        """Last sampled token per request — the query block's bonus token."""
        bonus: dict[str, int] = {}
        for (req_id, _state), sampled in zip(
            ctx.decode_reqs, ctx.decode_token_ids, strict=True
        ):
            if sampled:
                bonus[req_id] = sampled[-1]
        for prefill, token_id, mode in zip(
            ctx.prefill_reqs,
            ctx.prefill_token_ids,
            ctx.prefill_result_modes,
            strict=True,
        ):
            if mode != "intermediate" and prefill.req_id not in bonus:
                bonus[prefill.req_id] = token_id
        return bonus

    # -- bookkeeping ---------------------------------------------------------------

    def _prune_finished(self, request_states: Any) -> None:
        for req_id in list(self._slabs):
            if req_id not in request_states:
                del self._slabs[req_id]
