# SPDX-License-Identifier: Apache-2.0
"""Tests for paged forward multimodal fail-fast + ctx.segment_positions field.

The successful mm paged forward is exercised in
``test_paged_forward_mm.py``; this file holds the fail-fast invariants:
a misconfigured adapter (``forward_ready=False`` or no adapter) must
not allow an mm request to slip into the text path silently.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import mlx.core as mx
import pytest
from vllm.sampling_params import SamplingParams

from tests.stub_runner import make_stub_runner
from vllm_metal.multimodal import MultiModalFeatureSpec, PlaceholderRange
from vllm_metal.paged_attention_common import PagedAttentionContext
from vllm_metal.v1.mm import EncoderCache
from vllm_metal.v1.model_runner import RequestState


def _feature(identifier: str) -> MultiModalFeatureSpec:
    return MultiModalFeatureSpec(
        data=None,
        modality="image",
        identifier=identifier,
        mm_position=PlaceholderRange(offset=0, length=1),
    )


class TestPagedAttentionContextSegmentPositions:
    def test_field_defaults_to_none(self) -> None:
        ctx = PagedAttentionContext(slot_mapping=[])
        assert ctx.segment_positions is None

    def test_field_carries_per_segment_entries(self) -> None:
        positions = [None, mx.array([[[0, 1, 2]]] * 3, dtype=mx.int32)]
        ctx = PagedAttentionContext(slot_mapping=[], segment_positions=positions)
        assert ctx.segment_positions is positions


class TestStartPagedForwardMmFailFast:
    def _runner(self, *, adapter):
        runner = make_stub_runner(
            encoder_cache=EncoderCache(),
            _is_vlm=True,
            _paged_attention_backend=MagicMock(),
            _paged_block_size=16,
        )
        runner._multimodal_adapter = adapter
        runner._spec_decode_controller = MagicMock()
        runner._spec_decode_controller.build_decode_segments = MagicMock(
            return_value=[]
        )
        return runner

    def _scheduler_output(self) -> MagicMock:
        out = MagicMock()
        out.scheduled_spec_decode_tokens = {}
        return out

    def test_raises_when_mm_with_forward_ready_false(self) -> None:
        adapter = MagicMock()
        adapter.forward_ready = False
        runner = self._runner(adapter=adapter)
        runner.encoder_cache.add_request("req-mm", [_feature("img-0")])

        from vllm_metal.v1.model_runner import PrefillRequest

        prefill = PrefillRequest(
            req_id="req-mm",
            token_ids=[1, 99, 11],
            sampling_params=SamplingParams(temperature=0.0),
            block_ids=[0],
            generator=None,
            prompt_len=3,
            start_pos=0,
            full_prompt_token_ids=[1, 99, 11],
        )

        with pytest.raises(RuntimeError, match="adapter is not forward_ready"):
            runner._start_paged_forward(
                batch=MagicMock(),
                prefill_reqs=[prefill],
                decode_reqs=[],
                scheduler_output=self._scheduler_output(),
            )

    def test_raises_when_mm_with_no_adapter(self) -> None:
        runner = self._runner(adapter=None)
        runner._multimodal_adapter = None
        runner.encoder_cache.add_request("req-mm", [_feature("img-0")])

        from vllm_metal.v1.model_runner import PrefillRequest

        prefill = PrefillRequest(
            req_id="req-mm",
            token_ids=[1, 99, 11],
            sampling_params=SamplingParams(temperature=0.0),
            block_ids=[0],
            generator=None,
            prompt_len=3,
            start_pos=0,
            full_prompt_token_ids=[1, 99, 11],
        )

        with pytest.raises(RuntimeError, match="adapter is not forward_ready"):
            runner._start_paged_forward(
                batch=MagicMock(),
                prefill_reqs=[prefill],
                decode_reqs=[],
                scheduler_output=self._scheduler_output(),
            )

    def test_raises_when_mm_decode_with_forward_ready_false(self) -> None:
        adapter = MagicMock()
        adapter.forward_ready = False
        runner = self._runner(adapter=adapter)
        state = RequestState(
            token_ids=[1, 2, 3],
            prompt_len=2,
            cache=[],
            sampling_params=SamplingParams(),
            mrope_position_delta=-1,
        )

        with pytest.raises(RuntimeError, match="adapter is not forward_ready"):
            runner._start_paged_forward(
                batch=MagicMock(),
                prefill_reqs=[],
                decode_reqs=[("req-mm", state)],
                scheduler_output=self._scheduler_output(),
            )
