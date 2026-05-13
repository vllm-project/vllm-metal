# SPDX-License-Identifier: Apache-2.0
"""Structured output (grammar / JSON-schema bitmask) support for the Metal paged path.

Owns xgrammar bridging, bitmask application, and request-to-logit-row remapping.
``model_runner.py`` stays orchestration-only and delegates here via
``MetalStructuredOutputApplier``.
"""

from __future__ import annotations

from collections.abc import Sequence
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
        *,
        decode_segments: Sequence[PagedDecodeSegment] | None = None,
    ) -> mx.array:
        """Apply grammar bitmask to paged-path logits of shape (1, total_tokens, vocab).

        Only the sample positions are constrained:
        - Decode request i  → row cu_seqlens[i] of logits[0]
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
            cu_seqlens: Cumulative token counts for decode verification spans
                and prefill chunks. Used to locate sample rows.
            num_decode: Number of decode requests/segments.
            logits: Full paged logits, shape (1, total_tokens, vocab).
            decode_segments: Optional row-span metadata produced by the paged
                speculative decode path.

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

        structured_req_ids = set(grammar_output.structured_output_request_ids)
        spec_req_ids = {
            req_id
            for req_id, tokens in scheduler_output.scheduled_spec_decode_tokens.items()
            if tokens
        }
        structured_spec_req_ids = spec_req_ids & structured_req_ids
        if structured_spec_req_ids and decode_segments is None:
            raise NotImplementedError(
                "Grammar/structured-output constraints with speculative decoding "
                "require paged decode row-span metadata on Metal."
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

        row_targets = self._build_paged_row_targets(
            decode_reqs,
            prefill_reqs,
            cu_seqlens,
            num_decode,
            decode_segments=decode_segments,
        )
        has_structured_spec_decode = bool(structured_spec_req_ids)
        constrained = self._build_constrained_rows(
            grammar_output.structured_output_request_ids,
            row_targets,
            grammar_bitmask_row_count=grammar_bitmask.shape[0],
            consume_row_spans=has_structured_spec_decode,
        )

        return self._apply_grammar_bitmask_to_rows(logits, grammar_bitmask, constrained)

    def _build_paged_row_targets(
        self,
        decode_reqs: Sequence[tuple[str, Any]],
        prefill_reqs: Sequence[Any],
        cu_seqlens: Sequence[int],
        num_decode: int,
        *,
        decode_segments: Sequence[PagedDecodeSegment] | None,
    ) -> dict[str, tuple[int, ...]]:
        """Build request-id -> constrained-logit-row mapping for paged SO."""
        if num_decode != len(decode_reqs):
            raise AssertionError(
                f"num_decode={num_decode} must match decode_reqs={len(decode_reqs)}"
            )
        assert len(cu_seqlens) == num_decode + len(prefill_reqs) + 1, (
            f"cu_seqlens length {len(cu_seqlens)}, "
            f"expected num_decode={num_decode} + prefill_count={len(prefill_reqs)} + 1"
        )

        req_id_to_rows: dict[str, tuple[int, ...]] = {}
        if decode_segments is None:
            for i, (req_id, _) in enumerate(decode_reqs):
                req_id_to_rows[req_id] = (cu_seqlens[i],)
        else:
            if len(decode_segments) != len(decode_reqs):
                raise ValueError(
                    "decode_segments must match decode_reqs one-to-one when provided"
                )
            for i, ((req_id, _), segment) in enumerate(
                zip(decode_reqs, decode_segments, strict=True)
            ):
                if segment.req_id != req_id:
                    raise ValueError(
                        "decode_segments must be ordered with decode_reqs: "
                        f"{segment.req_id!r} != {req_id!r}"
                    )
                if segment.start_row != cu_seqlens[i]:
                    raise ValueError(
                        "decode segment start_row must match cu_seqlens: "
                        f"{segment.start_row} != {cu_seqlens[i]} for {req_id!r}"
                    )
                req_id_to_rows[req_id] = tuple(
                    range(
                        segment.start_row, segment.start_row + segment.num_query_tokens
                    )
                )

        for j, pr in enumerate(prefill_reqs):
            # cu_seqlens[num_decode + j + 1] is the exclusive end of prefill seq j.
            req_id_to_rows[pr.req_id] = (cu_seqlens[num_decode + j + 1] - 1,)

        return req_id_to_rows

    def _build_constrained_rows(
        self,
        structured_output_request_ids: Sequence[str],
        row_targets: dict[str, tuple[int, ...]],
        *,
        grammar_bitmask_row_count: int,
        consume_row_spans: bool,
    ) -> list[tuple[int, int]]:
        """Pair target logits rows with grammar bitmask rows."""
        if not consume_row_spans:
            constrained: list[tuple[int, int]] = []
            for bitmask_row, req_id in enumerate(structured_output_request_ids):
                rows = row_targets.get(req_id)
                if rows:
                    constrained.append((rows[0], bitmask_row))
            return constrained

        constrained = []
        seen_req_ids: set[str] = set()
        bitmask_row = 0
        for req_id in structured_output_request_ids:
            if req_id in seen_req_ids:
                raise ValueError(
                    "row-span grammar bitmask rows must be expanded in "
                    f"grammar_bitmask, not by repeating request ID {req_id!r}"
                )
            seen_req_ids.add(req_id)

            rows = row_targets.get(req_id)
            if rows is None:
                raise ValueError(
                    f"Grammar bitmask references {req_id!r}, but that request "
                    "has no paged logits rows in the current batch."
                )
            row_count = len(rows)
            end_bitmask_row = bitmask_row + row_count
            if end_bitmask_row > grammar_bitmask_row_count:
                if len(rows) > 1:
                    raise NotImplementedError(
                        "Row-span structured-output masking requires one grammar "
                        "bitmask row per logits row for request "
                        f"{req_id!r}; got {grammar_bitmask_row_count - bitmask_row} "
                        f"bitmask rows for {len(rows)} logits rows."
                    )
                raise ValueError(
                    "Grammar bitmask row count must match row-span logits targets "
                    f"for request {req_id!r}: got {grammar_bitmask_row_count} total "
                    f"bitmask rows, needed at least {end_bitmask_row}."
                )
            for row_offset, row in enumerate(rows):
                constrained.append((row, bitmask_row + row_offset))
            bitmask_row = end_bitmask_row

        if bitmask_row != grammar_bitmask_row_count:
            raise ValueError(
                "Grammar bitmask row count must match row-span logits targets: "
                f"consumed {bitmask_row}, got {grammar_bitmask_row_count}."
            )

        return constrained

    def _apply_grammar_bitmask_to_rows(
        self,
        logits: mx.array,
        grammar_bitmask: np.ndarray,
        constrained: Sequence[tuple[int, int]],
    ) -> mx.array:
        """Apply grammar bitmask rows to explicit logits rows."""
        if not constrained:
            return logits

        if xgr is None:
            raise RuntimeError(
                "xgrammar is required for structured output. "
                "Install it with: pip install xgrammar"
            )

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
