# SPDX-License-Identifier: Apache-2.0
"""N-gram (prompt-lookup) speculative decoding proposer for the Metal paged path.

An :class:`NgramProposer` drafts by matching the longest suffix n-gram of each
request's committed token history against an earlier occurrence and copying the
tokens that followed it (vLLM ``method="ngram"``). Unlike
:class:`vllm_metal.v1.draft_model_proposer.DraftModelProposer` it loads no model
and keeps no KV cache: the matching is the pure-Python + Numba KMP kernel that
vLLM ships in :mod:`vllm.v1.spec_decode.ngram_proposer`, which this class wraps.

The wrapper's only job is to translate the per-step :class:`ProposeContext` into
the three array arguments that upstream's stateless ``propose`` expects
(``sampled_token_ids``, ``num_tokens_no_spec``, ``token_ids_cpu``) and hand the
result back as :class:`DraftTokenIds`. The committed history lives in
``state.token_ids`` (already updated with this step's accepted/sampled tokens by
the time the runner builds the context), so no per-request bookkeeping is needed.

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


class NgramProposer:
    """:class:`vllm_metal.v1.proposer.MetalProposer` backed by vLLM's n-gram kernel."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        controller: SpeculativeDecodeController,
    ) -> None:
        self._controller = controller
        # Upstream reads only scalar config (prompt_lookup_min/max,
        # num_speculative_tokens, max_model_len, max_num_seqs) and runs a one-time
        # Numba JIT warmup in its constructor — keep that off the hot path.
        self._ngram = VllmNgramProposer(vllm_config)
        spec = vllm_config.speculative_config
        assert spec is not None

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
            sampled_token_ids,
            num_tokens_no_spec,
            token_ids_cpu,
        )

        req_ids: list[str] = []
        draft_token_ids: list[list[int]] = []
        for (req_id, _), draft in zip(drafting, drafts, strict=True):
            if not draft:
                continue
            req_ids.append(req_id)
            # Upstream already yields Python ints via ndarray.tolist() — the
            # old ``[int(t) for t in draft]`` was redundant.
            draft_token_ids.append(list(draft))

        if not req_ids:
            return None

        return DraftTokenIds(req_ids=req_ids, draft_token_ids=draft_token_ids)
