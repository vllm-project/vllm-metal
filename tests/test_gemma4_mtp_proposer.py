# SPDX-License-Identifier: Apache-2.0
"""Tests for the Gemma4 MTP proposer's target-hidden-state policy."""

from __future__ import annotations

from types import SimpleNamespace

from vllm_metal.v1.proposer import Gemma4MTPProposer
from vllm_metal.v1.spec_decode import PagedDecodeSegment


def _proposer() -> Gemma4MTPProposer:
    return Gemma4MTPProposer(runner=SimpleNamespace())


def _decode_segment() -> PagedDecodeSegment:
    return PagedDecodeSegment(
        req_id="r0",
        input_token_ids=(9,),
        start_row=0,
        num_query_tokens=1,
        draft_token_ids=(),
        cache_start_pos=7,
        block_ids=(11,),
    )


class TestGemma4MTPProposerNeedsTargetHiddenStates:
    def test_not_needed_without_decode_segments_or_final_prefill(self) -> None:
        # Also the intermediate-prefill-only case: those chunks never sample.
        assert not _proposer().needs_target_hidden_states([], has_final_prefill=False)

    def test_needed_with_decode_segments(self) -> None:
        assert _proposer().needs_target_hidden_states(
            [_decode_segment()], has_final_prefill=False
        )

    def test_needed_with_final_prefill_only(self) -> None:
        assert _proposer().needs_target_hidden_states([], has_final_prefill=True)
