# SPDX-License-Identifier: Apache-2.0
"""Structured output (grammar / JSON-schema bitmask) support for the Metal paged path.

Owns xgrammar bridging, bitmask application, and request-to-logit-row remapping.
``model_runner.py`` stays orchestration-only and delegates here via
``MetalStructuredOutputApplier``.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import numpy as np
import torch

try:
    import xgrammar as xgr
except ImportError:
    xgr = None  # type: ignore[assignment]

from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput

from vllm_metal.pytorch_backend.tensor_bridge import torch_to_mlx


class MetalStructuredOutputApplier:
    """Applies grammar/structured-output bitmask constraints to paged-path logits.

    Instantiate once on MetalModelRunner and call apply_paged() from
    _sample_paged_batch(). The class boundary keeps future extensions
    (e.g. non-paged path, xgrammar allocator caching) out of model_runner.py.
    """

    def apply_paged(
        self,
        scheduler_output: SchedulerOutput,
        grammar_output: GrammarOutput,
        decode_reqs: list[tuple[str, Any]],
        prefill_reqs: list[Any],
        cu_seqlens: list[int],
        num_decode: int,
        logits: mx.array,
    ) -> mx.array:
        """Apply grammar bitmask to paged-path logits of shape (1, total_tokens, vocab).

        Only the sample positions are constrained:
        - Decode request i  → row i of logits[0]
        - Prefill request j → last-token row per sequence (from cu_seqlens)

        The CPU/xgrammar bridge copies only the constrained rows (n_constrained × vocab)
        rather than the full (total_tokens × vocab) plane, then scatters the
        modified rows back into logits via MLX indexed assignment.

        Args:
            scheduler_output: Used to guard against spec-decode requests, which are
                not yet supported with grammar constraints on the paged path.
            grammar_output: Grammar bitmask and structured-output request IDs.
            decode_reqs: (req_id, RequestState) pairs in decode-batch order.
            prefill_reqs: PrefillRequest objects in prefill order.
            cu_seqlens: Cumulative token counts: [0, 1, …, num_decode,
                num_decode+len(pr0), …].  Used to locate last-token rows.
            num_decode: Number of decode requests (prefix of the token dimension).
            logits: Full paged logits, shape (1, total_tokens, vocab).

        Returns:
            Logits with forbidden token positions set to -inf, same shape and dtype.
        """
        if xgr is None:
            raise RuntimeError(
                "xgrammar is required for structured output. "
                "Install it with: pip install xgrammar"
            )

        assert logits.ndim == 3 and logits.shape[0] == 1, (
            f"apply_paged expects shape (1, T, V), got {logits.shape}"
        )

        # Spec-decode expands the token dimension with verification positions that
        # don't map 1:1 to grammar_bitmask rows — guard until that's implemented.
        # Only raise when a structured-output request overlaps with spec-decode;
        # plain requests with spec tokens can co-exist in the same batch safely.
        spec_req_ids = {
            req_id
            for req_id, tokens in scheduler_output.scheduled_spec_decode_tokens.items()
            if tokens
        }
        if spec_req_ids & set(grammar_output.structured_output_request_ids):
            raise NotImplementedError(
                "Grammar/structured-output constraints are not yet supported "
                "when speculative decoding is active on the paged Metal path."
            )

        grammar_bitmask: np.ndarray = grammar_output.grammar_bitmask

        # Fast path: if none of the structured-output request IDs appear in this
        # batch, skip row-map construction and cu_seqlens validation entirely.
        batch_req_ids = {req_id for req_id, _ in decode_reqs} | {
            pr.req_id for pr in prefill_reqs
        }
        if not any(
            rid in batch_req_ids for rid in grammar_output.structured_output_request_ids
        ):
            return logits

        # cu_seqlens must be exactly [0, 1*decode, ..., +prefill_lens...]: one entry
        # per decode token plus one per prefill sequence, plus the leading zero.
        assert len(cu_seqlens) == num_decode + len(prefill_reqs) + 1, (
            f"cu_seqlens length {len(cu_seqlens)}, "
            f"expected num_decode={num_decode} + prefill_count={len(prefill_reqs)} + 1"
        )

        # Build req_id → sample row index in logits[0].
        req_id_to_row: dict[str, int] = {}
        for i, (req_id, _) in enumerate(decode_reqs):
            req_id_to_row[req_id] = i
        for j, pr in enumerate(prefill_reqs):
            # cu_seqlens[num_decode + j + 1] - 1 is the last token of prefill seq j.
            last_row = cu_seqlens[num_decode + j + 1] - 1
            req_id_to_row[pr.req_id] = last_row

        # Determine which structured-output requests are present in this batch.
        constrained: list[tuple[int, int]] = []  # (logit_row, bitmask_row)
        for bitmask_row, req_id in enumerate(
            grammar_output.structured_output_request_ids
        ):
            if req_id in req_id_to_row:
                constrained.append((req_id_to_row[req_id], bitmask_row))

        if not constrained:
            return logits

        # --- CPU/xgrammar bridge: operate only on constrained rows ---
        #
        # Copy only the n_constrained rows (not total_tokens) to CPU float32.
        # Explicit float32 cast: numpy has no bfloat16 dtype, and np.array()
        # forces MLX evaluation, producing an independent writable copy.
        original_dtype = logits.dtype
        logit_rows = [logit_row for logit_row, _ in constrained]
        rows_np = np.array(
            logits[0, logit_rows, :].astype(mx.float32)
        )  # (n_constrained, vocab)
        rows_torch = torch.from_numpy(rows_np)

        # Apply per constrained row. xgrammar's indices= parameter selects rows
        # from a full-batch bitmask — it does not support a sub-sampled bitmask
        # paired with non-contiguous logit indices, so we apply row-by-row here.
        # TODO: batch via indices= once xgrammar supports non-contiguous bitmask selection.
        for i, (_, bitmask_row) in enumerate(constrained):
            row_bitmask = torch.from_numpy(
                grammar_bitmask[bitmask_row : bitmask_row + 1]
            )
            # Explicit device=cpu: xgrammar has no Metal/MPS kernel.
            # vocab_size is intentionally omitted; xgrammar auto-detects it as
            # min(logits_width, bitmask_words * 32).  Phantom slots in the last
            # bitmask word (real_vocab % 32 != 0) get -inf, but the downstream
            # sampler clips to the real vocabulary so they are never sampled.
            xgr.apply_token_bitmask_inplace(rows_torch[i : i + 1], row_bitmask)

        # rows_torch is CPU float32 (from torch.from_numpy), so torch_to_mlx goes
        # through numpy — all xgrammar mutations are captured before the copy.
        rows_mlx = torch_to_mlx(rows_torch).astype(
            original_dtype
        )  # (n_constrained, vocab)

        # logits[0] produces a new lazy computation node (not a Python alias of
        # logits), so __setitem__ here does not mutate the caller-held logits array.
        result_2d = logits[0]
        result_2d[logit_rows] = rows_mlx
        return result_2d[None]  # Restore (1, total_tokens, vocab) shape
