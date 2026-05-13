# SPDX-License-Identifier: Apache-2.0
"""Tests for multimodal-aware ``_build_prefill_pack`` full-prompt resolution."""

from __future__ import annotations

import pytest
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import NewRequestData

from tests.stub_runner import make_stub_runner
from vllm_metal.multimodal import MultiModalFeatureSpec, PlaceholderRange
from vllm_metal.v1.mm import EncoderCache
from vllm_metal.v1.model_runner import (
    PrefillRequest,
    RequestState,
    _ExecutionBatch,
    _PendingPrefillEntry,
)


def _feature(identifier: str) -> MultiModalFeatureSpec:
    return MultiModalFeatureSpec(
        data=None,
        modality="image",
        identifier=identifier,
        mm_position=PlaceholderRange(offset=0, length=1),
    )


def _make_prefill_entry(
    req_id: str,
    *,
    token_ids: list[int],
    prompt_len: int,
    start_pos: int = 0,
) -> _PendingPrefillEntry:
    prefill = PrefillRequest(
        req_id=req_id,
        token_ids=token_ids,
        sampling_params=SamplingParams(temperature=0.0),
        block_ids=[],
        generator=None,
        prompt_len=prompt_len,
        start_pos=start_pos,
        full_prompt_token_ids=None,
    )
    return _PendingPrefillEntry(
        output_idx=0,
        prefill=prefill,
        result_mode="new_final",
    )


def _new_req(req_id: str, prompt_token_ids: list[int] | None) -> NewRequestData:
    return NewRequestData(
        req_id=req_id,
        prompt_token_ids=prompt_token_ids,
        mm_features=[],
        sampling_params=None,
        pooling_params=None,
        block_ids=([0],),
        num_computed_tokens=0,
        lora_request=None,
    )


class TestIsMmRequest:
    def test_returns_false_when_no_encoder_cache(self) -> None:
        runner = make_stub_runner()
        assert runner._is_mm_request("req-0") is False

    def test_returns_false_when_empty_features_list(self) -> None:
        runner = make_stub_runner(encoder_cache=EncoderCache())
        runner.encoder_cache.add_request("req-0", [])
        assert runner._is_mm_request("req-0") is False

    def test_returns_true_with_registered_features(self) -> None:
        runner = make_stub_runner(encoder_cache=EncoderCache())
        runner.encoder_cache.add_request("req-0", [_feature("img-0")])
        assert runner._is_mm_request("req-0") is True

    def test_returns_false_for_unknown_request(self) -> None:
        runner = make_stub_runner(encoder_cache=EncoderCache())
        assert runner._is_mm_request("missing") is False


class TestBuildPrefillPackMmFullPrompt:
    def _make_batch(
        self, entries: list[_PendingPrefillEntry], new_reqs: list[NewRequestData]
    ) -> _ExecutionBatch:
        batch = _ExecutionBatch()
        batch.paged_prefill_entries = entries
        batch.new_reqs_by_id = {nr.req_id: nr for nr in new_reqs}
        return batch

    def test_text_only_start_pos_zero_keeps_full_prompt_none(self) -> None:
        # Behavior preservation: text-only first-chunk prefill still gets
        # ``full_prompt_token_ids=None`` (callers fall back to token_ids).
        runner = make_stub_runner(encoder_cache=EncoderCache())
        entry = _make_prefill_entry(
            "req-0", token_ids=[1, 2, 3], prompt_len=3, start_pos=0
        )
        batch = self._make_batch([entry], [_new_req("req-0", [1, 2, 3])])

        pack = runner._build_prefill_pack(batch)

        assert pack[0].full_prompt_token_ids is None

    def test_mm_start_pos_zero_uses_state_token_ids(self) -> None:
        runner = make_stub_runner(encoder_cache=EncoderCache())
        runner.encoder_cache.add_request("req-0", [_feature("img-0")])
        runner._request_states["req-0"] = RequestState(
            token_ids=[1, 99, 99, 2, 3, 4],  # 4 prompt + 2 generated
            prompt_len=4,
            cache=[],
            sampling_params=SamplingParams(),
        )
        entry = _make_prefill_entry(
            "req-0", token_ids=[1, 99], prompt_len=4, start_pos=0
        )
        batch = self._make_batch([entry], [])

        pack = runner._build_prefill_pack(batch)

        assert pack[0].full_prompt_token_ids == [1, 99, 99, 2]

    def test_mm_start_pos_zero_falls_back_to_new_req(self) -> None:
        runner = make_stub_runner(encoder_cache=EncoderCache())
        runner.encoder_cache.add_request("req-0", [_feature("img-0")])
        # No RequestState yet — first time we see this request.
        entry = _make_prefill_entry(
            "req-0", token_ids=[1, 99], prompt_len=4, start_pos=0
        )
        batch = self._make_batch([entry], [_new_req("req-0", [1, 99, 99, 2])])

        pack = runner._build_prefill_pack(batch)

        assert pack[0].full_prompt_token_ids == [1, 99, 99, 2]

    def test_mm_raises_when_state_and_new_req_both_missing(self) -> None:
        runner = make_stub_runner(encoder_cache=EncoderCache())
        runner.encoder_cache.add_request("req-0", [_feature("img-0")])
        entry = _make_prefill_entry("req-0", token_ids=[1], prompt_len=4, start_pos=0)
        batch = self._make_batch([entry], [])

        with pytest.raises(RuntimeError, match="state tracking bug"):
            runner._build_prefill_pack(batch)

    def test_mm_raises_when_new_req_prompt_token_ids_missing(self) -> None:
        runner = make_stub_runner(encoder_cache=EncoderCache())
        runner.encoder_cache.add_request("req-0", [_feature("img-0")])
        entry = _make_prefill_entry("req-0", token_ids=[1], prompt_len=4, start_pos=0)
        batch = self._make_batch([entry], [_new_req("req-0", None)])

        with pytest.raises(RuntimeError, match="scheduler contract bug"):
            runner._build_prefill_pack(batch)

    def test_text_only_continuation_chunk_still_uses_full_prompt(self) -> None:
        # Regression guard: ``start_pos > 0`` text path unchanged.
        runner = make_stub_runner(encoder_cache=EncoderCache())
        runner._request_states["req-0"] = RequestState(
            token_ids=[1, 2, 3, 4, 5, 6, 7],
            prompt_len=5,
            cache=[],
            sampling_params=SamplingParams(),
        )
        entry = _make_prefill_entry(
            "req-0", token_ids=[3, 4, 5], prompt_len=5, start_pos=2
        )
        batch = self._make_batch([entry], [])

        pack = runner._build_prefill_pack(batch)

        assert pack[0].full_prompt_token_ids == [1, 2, 3, 4, 5]
