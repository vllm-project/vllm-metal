# SPDX-License-Identifier: Apache-2.0
"""Tests for Metal runner multimodal cache lifecycle wiring."""

from __future__ import annotations

from unittest.mock import MagicMock

import mlx.core as mx
import pytest
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import (
    CachedRequestData,
    NewRequestData,
    SchedulerOutput,
)

from tests.stub_runner import make_stub_runner
from vllm_metal.multimodal import MultiModalFeatureSpec, PlaceholderRange
from vllm_metal.multimodal.qwen3_vl import Qwen3VLMultimodalAdapter
from vllm_metal.v1.mm import EncoderCache
from vllm_metal.v1.model_runner import RequestState


def _feature(identifier: str) -> MultiModalFeatureSpec:
    return MultiModalFeatureSpec(
        data=None,
        modality="image",
        identifier=identifier,
        mm_position=PlaceholderRange(offset=0, length=1),
    )


def _cached_reqs(req_ids: list[str] | None = None) -> CachedRequestData:
    ids = req_ids or []
    return CachedRequestData(
        req_ids=ids,
        resumed_req_ids=set(),
        new_token_ids=[],
        all_token_ids={},
        new_block_ids=[None] * len(ids),
        num_computed_tokens=[0] * len(ids),
        num_output_tokens=[0] * len(ids),
    )


def _scheduler_output(
    *,
    scheduled_new_reqs: list[NewRequestData] | None = None,
    scheduled_encoder_inputs: dict[str, list[int]] | None = None,
    finished_req_ids: set[str] | None = None,
    preempted_req_ids: set[str] | None = None,
    free_encoder_mm_hashes: list[str] | None = None,
) -> SchedulerOutput:
    new_reqs = scheduled_new_reqs or []
    num_scheduled_tokens = {
        req.req_id: len(req.prompt_token_ids or []) - req.num_computed_tokens
        for req in new_reqs
    }
    return SchedulerOutput(
        scheduled_new_reqs=new_reqs,
        scheduled_cached_reqs=_cached_reqs(),
        num_scheduled_tokens=num_scheduled_tokens,
        total_num_scheduled_tokens=sum(num_scheduled_tokens.values()),
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs=scheduled_encoder_inputs or {},
        num_common_prefix_blocks=[],
        finished_req_ids=finished_req_ids or set(),
        free_encoder_mm_hashes=free_encoder_mm_hashes or [],
        preempted_req_ids=preempted_req_ids or set(),
        has_structured_output_requests=False,
    )


def _new_request(req_id: str, features: list[MultiModalFeatureSpec]) -> NewRequestData:
    return NewRequestData(
        req_id=req_id,
        prompt_token_ids=[1],
        mm_features=features,
        sampling_params=None,
        pooling_params=None,
        block_ids=([0],),
        num_computed_tokens=0,
        lora_request=None,
    )


def _runner_with_encoder_cache():
    return make_stub_runner(encoder_cache=EncoderCache())


def _paged_runner_with_encoder_cache():
    runner = make_stub_runner(
        encoder_cache=EncoderCache(),
        _paged_attention_backend=MagicMock(),
        _paged_block_size=16,
    )
    runner._start_paged_forward = MagicMock()
    return runner


def test_execute_model_registers_new_request_mm_features() -> None:
    runner = _paged_runner_with_encoder_cache()
    features = [_feature("image-0")]

    runner.execute_model(
        _scheduler_output(scheduled_new_reqs=[_new_request("req-0", features)])
    )

    assert runner.encoder_cache is not None
    assert runner.encoder_cache.mm_features["req-0"] == features


def test_cleanup_finished_requests_removes_mm_features() -> None:
    runner = _runner_with_encoder_cache()
    features = [_feature("image-0")]
    assert runner.encoder_cache is not None
    runner.encoder_cache.add_request("req-0", features)

    runner.execute_model(_scheduler_output(finished_req_ids={"req-0"}))

    assert "req-0" not in runner.encoder_cache.mm_features


def test_preempted_requests_keep_resume_state() -> None:
    runner = _runner_with_encoder_cache()
    features = [_feature("image-0")]
    assert runner.encoder_cache is not None
    runner.encoder_cache.add_request("req-0", features)
    state = RequestState(
        token_ids=[1, 2],
        prompt_len=1,
        cache=[],
        sampling_params=SamplingParams(),
    )
    runner._request_states["req-0"] = state
    runner._paged_request_seq_lens["req-0"] = 2
    runner._gdn_req_to_slot["req-0"] = 0

    runner.execute_model(_scheduler_output(preempted_req_ids={"req-0"}))

    assert runner.encoder_cache.mm_features["req-0"] == features
    assert runner._request_states["req-0"] is state
    assert runner._paged_request_seq_lens["req-0"] == 2
    assert runner._gdn_req_to_slot["req-0"] == 0


def test_resubmitted_request_keeps_new_mm_features() -> None:
    runner = _paged_runner_with_encoder_cache()
    old_features = [_feature("old-image")]
    new_features = [_feature("new-image")]
    assert runner.encoder_cache is not None
    runner.encoder_cache.add_request("req-0", old_features)

    runner.execute_model(
        _scheduler_output(
            scheduled_new_reqs=[_new_request("req-0", new_features)],
            finished_req_ids={"req-0"},
        )
    )

    assert runner.encoder_cache.mm_features["req-0"] == new_features


def test_execute_model_frees_released_encoder_outputs() -> None:
    runner = _runner_with_encoder_cache()
    assert runner.encoder_cache is not None
    runner.encoder_cache.encoder_outputs["keep"] = mx.array([[1.0]])
    runner.encoder_cache.encoder_outputs["drop"] = mx.array([[2.0]])

    runner.execute_model(_scheduler_output(free_encoder_mm_hashes=["drop"]))

    assert set(runner.encoder_cache.encoder_outputs) == {"keep"}


def test_reset_encoder_cache_delegates_to_encoder_cache() -> None:
    runner = _runner_with_encoder_cache()
    assert runner.encoder_cache is not None
    runner.encoder_cache.encoder_outputs["image-0"] = mx.array([[1.0]])

    runner.reset_encoder_cache()

    assert runner.encoder_cache.encoder_outputs == {}


def test_reset_mm_cache_delegates_to_encoder_cache() -> None:
    encoder_cache = MagicMock()
    runner = make_stub_runner(encoder_cache=encoder_cache)

    runner.reset_mm_cache()

    encoder_cache.reset_mm_cache.assert_called_once_with()


def test_execute_model_frees_encoder_outputs_before_encoder_fail_fast() -> None:
    runner = _runner_with_encoder_cache()
    assert runner.encoder_cache is not None
    runner.encoder_cache.encoder_outputs["drop"] = mx.array([[1.0]])

    with pytest.raises(NotImplementedError, match="Multimodal encoder execution"):
        runner.execute_model(
            _scheduler_output(
                scheduled_encoder_inputs={"req-0": [0]},
                free_encoder_mm_hashes=["drop"],
            )
        )

    assert "drop" not in runner.encoder_cache.encoder_outputs


def test_execute_model_cleans_finished_requests_before_encoder_fail_fast() -> None:
    runner = _runner_with_encoder_cache()
    features = [_feature("image-0")]
    assert runner.encoder_cache is not None
    runner.encoder_cache.add_request("done", features)
    runner._request_states["done"] = RequestState(
        token_ids=[1, 2],
        prompt_len=1,
        cache=[],
        sampling_params=SamplingParams(),
    )

    with pytest.raises(NotImplementedError, match="Multimodal encoder execution"):
        runner.execute_model(
            _scheduler_output(
                finished_req_ids={"done"},
                scheduled_encoder_inputs={"req-0": [0]},
            )
        )

    assert "done" not in runner.encoder_cache.mm_features
    assert "done" not in runner._request_states


def test_execute_model_rejects_encoder_inputs_until_forward_is_wired() -> None:
    runner = _runner_with_encoder_cache()
    runner._multimodal_adapter = Qwen3VLMultimodalAdapter(
        spatial_merge_size=2,
        language_model=object(),
    )

    with pytest.raises(NotImplementedError, match="Multimodal encoder execution"):
        runner.execute_model(_scheduler_output(scheduled_encoder_inputs={"req-0": [0]}))
