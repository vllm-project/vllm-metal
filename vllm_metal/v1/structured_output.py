# SPDX-License-Identifier: Apache-2.0
"""Structured output (grammar / JSON-schema bitmask) support for the Metal paged path.

Owns xgrammar bridging, bitmask application, and request-to-logit-row remapping.
``model_runner.py`` stays orchestration-only and delegates here via
``MetalStructuredOutputApplier``.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
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
from vllm_metal.v1.spec_decode import PagedDecodeSegment


@dataclass(frozen=True, slots=True)
class _PagedRowTargets:
    decode_row_count: int
    req_id_to_rows: dict[str, tuple[int, ...]]


def _build_paged_row_targets(
    decode_reqs: Sequence[tuple[str, Any]],
    prefill_reqs: Sequence[Any],
    cu_seqlens: Sequence[int],
    num_decode: int,
    paged_decode_segments: Sequence[PagedDecodeSegment] | None = None,
) -> _PagedRowTargets:
    """Build request-id -> constrained-logit-row mapping for paged structured output.

    When no speculative row-span metadata is present, decode requests still map to
    one row each, matching the current paged-path behavior.
    """
    req_id_to_rows: dict[str, tuple[int, ...]] = {}

    if paged_decode_segments is None:
        decode_row_count = num_decode
        for i, (req_id, _) in enumerate(decode_reqs):
            req_id_to_rows[req_id] = (i,)
    else:
        segment_by_req_id = {
            segment.req_id: segment for segment in paged_decode_segments
        }
        decode_req_ids = {req_id for req_id, _ in decode_reqs}
        if set(segment_by_req_id) != decode_req_ids:
            raise ValueError(
                "paged_decode_segments must match decode_reqs one-to-one when provided"
            )

        decode_row_count = 0
        for req_id, _ in decode_reqs:
            segment = segment_by_req_id[req_id]
            decode_row_count += segment.num_query_tokens
            req_id_to_rows[req_id] = (
                *segment.draft_verification_rows,
                segment.bonus_row,
            )

    expected_cu_seqlens_len = decode_row_count + len(prefill_reqs) + 1
    if len(cu_seqlens) != expected_cu_seqlens_len:
        raise AssertionError(
            "cu_seqlens length "
            f"{len(cu_seqlens)}, expected decode_row_count={decode_row_count} "
            f"+ prefill_count={len(prefill_reqs)} + 1"
        )

    for j, pr in enumerate(prefill_reqs):
        # cu_seqlens[decode_row_count + j + 1] is the exclusive end of seq j.
        last_row = cu_seqlens[decode_row_count + j + 1] - 1
        req_id_to_rows[pr.req_id] = (last_row,)

    return _PagedRowTargets(
        decode_row_count=decode_row_count,
        req_id_to_rows=req_id_to_rows,
    )


def _build_constrained_rows(
    structured_output_request_ids: Sequence[str],
    row_targets: _PagedRowTargets,
    *,
    paged_decode_segments: Sequence[PagedDecodeSegment] | None = None,
) -> list[tuple[int, int]]:
    """Pair target logits rows with the grammar bitmask row to apply.

    The legacy path keeps the existing request-id -> one bitmask row behavior.
    Row-span metadata can map a request to multiple logits rows, which requires
    one grammar bitmask row per target row because each speculative position may
    have a different grammar state.
    """
    if paged_decode_segments is None:
        constrained: list[tuple[int, int]] = []
        for bitmask_row, req_id in enumerate(structured_output_request_ids):
            rows = row_targets.req_id_to_rows.get(req_id)
            if rows:
                for row in rows:
                    constrained.append((row, bitmask_row))
        return constrained

    bitmask_rows_by_req_id: dict[str, list[int]] = defaultdict(list)
    for bitmask_row, req_id in enumerate(structured_output_request_ids):
        if req_id in row_targets.req_id_to_rows:
            bitmask_rows_by_req_id[req_id].append(bitmask_row)

    constrained = []
    for req_id, bitmask_rows in bitmask_rows_by_req_id.items():
        rows = row_targets.req_id_to_rows[req_id]
        if len(rows) != len(bitmask_rows):
            if len(rows) > 1 and len(bitmask_rows) == 1:
                raise NotImplementedError(
                    "Row-span structured-output masking requires one grammar "
                    "bitmask row per logits row for request "
                    f"{req_id!r}; got 1 bitmask row for {len(rows)} logits rows."
                )
            raise ValueError(
                "Grammar bitmask row count must match row-span logits targets "
                f"for request {req_id!r}: got {len(bitmask_rows)} bitmask rows "
                f"for {len(rows)} logits rows."
            )
        constrained.extend(zip(rows, bitmask_rows, strict=True))

    return constrained


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
        paged_decode_segments: Sequence[PagedDecodeSegment] | None = None,
    ) -> mx.array:
        """Apply grammar bitmask to paged-path logits of shape (1, total_tokens, vocab).

        Only the sample positions are constrained:
        - Decode request i  → row i of logits[0]
        - Decode request with row-span metadata → draft verification rows plus
          the bonus row, using one grammar bitmask row per target row
        - Prefill request j → last-token row per sequence (from cu_seqlens)

        The CPU/xgrammar bridge copies only the constrained rows (n_constrained × vocab)
        rather than the full (total_tokens × vocab) plane, then scatters the
        modified rows back into logits via MLX indexed assignment.

        Args:
            scheduler_output: Used to guard against spec-decode requests when no
                row-span metadata is available.
            grammar_output: Grammar bitmask and structured-output request IDs.
            decode_reqs: (req_id, RequestState) pairs in decode-batch order.
            prefill_reqs: PrefillRequest objects in prefill order.
            cu_seqlens: Cumulative token counts. Without row-span metadata the
                decode prefix has one row per request; with metadata it has one
                row per decode query token.
            num_decode: Number of decode requests.
            logits: Full paged logits, shape (1, total_tokens, vocab).
            paged_decode_segments: Optional metadata describing multi-row decode
                spans for speculative verification.

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

        # Spec-decode expands the token dimension with verification positions.
        # Keep the old guard unless explicit row-span metadata tells us where
        # those verification rows live. Plain requests with spec tokens can
        # co-exist in the same batch safely.
        spec_req_ids = {
            req_id
            for req_id, tokens in scheduler_output.scheduled_spec_decode_tokens.items()
            if tokens
        }
        if paged_decode_segments is None and spec_req_ids & set(
            grammar_output.structured_output_request_ids
        ):
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
        # per decode row plus one per prefill sequence, plus the leading zero.
        row_targets = _build_paged_row_targets(
            decode_reqs,
            prefill_reqs,
            cu_seqlens,
            num_decode,
            paged_decode_segments=paged_decode_segments,
        )

        # Determine which structured-output requests are present in this batch.
        # Each tuple is (logit_row, bitmask_row).
        constrained = _build_constrained_rows(
            grammar_output.structured_output_request_ids,
            row_targets,
            paged_decode_segments=paged_decode_segments,
        )

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
