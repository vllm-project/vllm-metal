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
from vllm_metal.attention.runtime.mha import MHAPagedAttentionRuntime
from vllm_metal.multimodal import MultiModalFeatureSpec, PlaceholderRange
from vllm_metal.multimodal.qwen3_vl import (
    Qwen3VLMultimodalAdapter,
    Qwen3VLVisionEncodeResult,
)
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
        _paged_attention_runtime=MHAPagedAttentionRuntime(
            num_layers=1,
            num_kv_heads=1,
            head_dim=4,
            block_size=16,
            dtype=mx.float32,
        ),
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


def test_finished_requests_release_runner_mm_features() -> None:
    runner = _runner_with_encoder_cache()
    features = [_feature("image-0")]
    assert runner.encoder_cache is not None
    runner.encoder_cache.add_request("req-0", features)

    runner.execute_model(_scheduler_output(finished_req_ids={"req-0"}))

    assert "req-0" not in runner.encoder_cache.mm_features


def test_preempted_requests_keep_runner_resume_state() -> None:
    runner = _paged_runner_with_encoder_cache()
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

    runner.execute_model(_scheduler_output(preempted_req_ids={"req-0"}))

    assert runner.encoder_cache.mm_features["req-0"] == features
    assert runner._request_states["req-0"] is state
    assert runner._paged_request_seq_lens["req-0"] == 2


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


def test_execute_model_frees_released_encoder_outputs(fake_encode_result) -> None:
    runner = _runner_with_encoder_cache()
    assert runner.encoder_cache is not None
    runner.encoder_cache.encoder_outputs["keep"] = fake_encode_result(mx.array([[1.0]]))
    runner.encoder_cache.encoder_outputs["drop"] = fake_encode_result(mx.array([[2.0]]))

    runner.execute_model(_scheduler_output(free_encoder_mm_hashes=["drop"]))

    assert set(runner.encoder_cache.encoder_outputs) == {"keep"}


def test_execute_model_frees_encoder_outputs_before_encoder_fail_fast(
    fake_encode_result,
) -> None:
    runner = _runner_with_encoder_cache()
    assert runner.encoder_cache is not None
    runner.encoder_cache.encoder_outputs["drop"] = fake_encode_result(mx.array([[1.0]]))

    with pytest.raises(RuntimeError, match="not forward_ready"):
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

    with pytest.raises(RuntimeError, match="not forward_ready"):
        runner.execute_model(
            _scheduler_output(
                finished_req_ids={"done"},
                scheduled_encoder_inputs={"req-0": [0]},
            )
        )

    assert "done" not in runner.encoder_cache.mm_features
    assert "done" not in runner._request_states


def test_execute_model_rejects_encoder_inputs_when_adapter_not_forward_ready() -> None:
    runner = _runner_with_encoder_cache()
    adapter = Qwen3VLMultimodalAdapter(
        spatial_merge_size=2,
        language_model=object(),
    )
    adapter.forward_ready = False
    runner._multimodal_adapter = adapter

    with pytest.raises(RuntimeError, match="not forward_ready"):
        runner.execute_model(_scheduler_output(scheduled_encoder_inputs={"req-0": [0]}))


def test_execute_model_pre_registers_new_request_before_encoder_dispatch() -> None:
    """A brand-new mm request + its first encoder input in one SchedulerOutput.

    Upstream's scheduler can place a new multimodal request and its first
    ``scheduled_encoder_inputs`` in the same step; encoder dispatch must
    find the request's ``mm_features`` already registered.  Before the
    pre-register step was lifted out of ``_handle_new_requests``, encoder
    dispatch ran first and raised an "unregistered request" RuntimeError.
    """
    runner = _paged_runner_with_encoder_cache()
    adapter = _RecordingAdapter()
    runner._multimodal_adapter = adapter
    features = [_feature("image-0")]
    expected_hidden = mx.array([[5.0, 6.0]])
    adapter.queue_outputs(
        [
            [
                Qwen3VLVisionEncodeResult(
                    hidden_states=expected_hidden,
                    deepstack_visual_embeds=None,
                )
            ]
        ]
    )

    runner.execute_model(
        _scheduler_output(
            scheduled_new_reqs=[_new_request("req-0", features)],
            scheduled_encoder_inputs={"req-0": [0]},
        )
    )

    assert runner.encoder_cache is not None
    assert runner.encoder_cache.mm_features["req-0"] == features
    assert adapter.encode_calls == [features]
    stored = runner.encoder_cache.encoder_outputs["image-0"]
    assert mx.allclose(stored.hidden_states, expected_hidden).item()


class _RecordingAdapter:
    """Adapter stub that records ``encode_multimodal`` invocations."""

    forward_ready = True

    def __init__(self) -> None:
        self.encode_calls: list[list[MultiModalFeatureSpec]] = []
        self._next_outputs: list[list[Qwen3VLVisionEncodeResult]] | None = None

    def queue_outputs(self, outputs: list[list[Qwen3VLVisionEncodeResult]]) -> None:
        """Force a sequence of return values for successive encode calls."""
        self._next_outputs = list(outputs)

    def text_model(self) -> object:
        return object()

    def get_mrope_input_positions(self, *args: object, **kwargs: object) -> None:
        return None

    def encode_multimodal(
        self, features: list[MultiModalFeatureSpec]
    ) -> list[Qwen3VLVisionEncodeResult]:
        self.encode_calls.append(list(features))
        if self._next_outputs is not None and self._next_outputs:
            return self._next_outputs.pop(0)
        return [
            Qwen3VLVisionEncodeResult(
                hidden_states=mx.zeros((1, 4)),
                deepstack_visual_embeds=None,
            )
            for _ in features
        ]

    def call_lm(self, *args: object, **kwargs: object) -> object:
        raise AssertionError("call_lm not exercised in commit 3")


def test_reject_scheduled_encoder_inputs_dispatches_when_adapter_is_forward_ready() -> (
    None
):
    # Dispatch only happens on the paged backend; mm is paged-only (RFC #319).
    runner = _paged_runner_with_encoder_cache()
    adapter = _RecordingAdapter()
    runner._multimodal_adapter = adapter
    features = [_feature("image-0")]
    assert runner.encoder_cache is not None
    runner.encoder_cache.add_request("req-0", features)

    runner._reject_scheduled_encoder_inputs({"req-0": [0]})

    assert len(adapter.encode_calls) == 1
    assert adapter.encode_calls[0] == features
    assert "image-0" in runner.encoder_cache.encoder_outputs


def test_reject_scheduled_encoder_inputs_raises_on_non_paged_backend() -> None:
    """forward_ready=True but no paged backend must fail fast.

    The non-paged legacy path never splices encoded image embeddings, so
    running the encoder and falling through to _prefill_single would silently
    drop image conditioning (or feed raw placeholder IDs to the LM).
    """
    runner = _runner_with_encoder_cache()  # _paged_attention_runtime is None
    adapter = _RecordingAdapter()  # forward_ready = True
    runner._multimodal_adapter = adapter
    assert runner.encoder_cache is not None
    runner.encoder_cache.add_request("req-0", [_feature("image-0")])

    with pytest.raises(NotImplementedError, match="paged attention backend"):
        runner._reject_scheduled_encoder_inputs({"req-0": [0]})

    # The encoder must not run when the request is going to be rejected.
    assert adapter.encode_calls == []


def test_reject_scheduled_encoder_inputs_raises_when_adapter_not_ready() -> None:
    runner = _runner_with_encoder_cache()
    adapter = Qwen3VLMultimodalAdapter(spatial_merge_size=2)
    adapter.forward_ready = False
    runner._multimodal_adapter = adapter

    with pytest.raises(RuntimeError, match="not forward_ready"):
        runner._reject_scheduled_encoder_inputs({"req-0": [0]})


def test_reject_scheduled_encoder_inputs_raises_when_no_adapter() -> None:
    runner = _runner_with_encoder_cache()
    runner._multimodal_adapter = None

    with pytest.raises(RuntimeError, match="not forward_ready"):
        runner._reject_scheduled_encoder_inputs({"req-0": [0]})


def test_run_vision_encoders_batches_uncached_features() -> None:
    runner = _runner_with_encoder_cache()
    adapter = _RecordingAdapter()
    runner._multimodal_adapter = adapter
    features = [_feature("image-0"), _feature("image-1")]
    assert runner.encoder_cache is not None
    runner.encoder_cache.add_request("req-0", features)
    expected0 = mx.array([[1.0, 2.0]])
    expected1 = mx.array([[5.0, 6.0]])
    adapter.queue_outputs(
        [
            [
                Qwen3VLVisionEncodeResult(
                    hidden_states=expected0,
                    deepstack_visual_embeds=[mx.array([[3.0, 4.0]])],
                ),
                Qwen3VLVisionEncodeResult(
                    hidden_states=expected1,
                    deepstack_visual_embeds=None,
                ),
            ]
        ]
    )

    runner._run_vision_encoders({"req-0": [0, 1]})

    assert adapter.encode_calls == [features]
    stored = runner.encoder_cache.encoder_outputs["image-0"]
    assert mx.allclose(stored.hidden_states, expected0).item()
    assert stored.deepstack_visual_embeds is not None
    assert mx.allclose(stored.deepstack_visual_embeds[0], mx.array([[3.0, 4.0]])).item()
    assert mx.allclose(
        runner.encoder_cache.encoder_outputs["image-1"].hidden_states,
        expected1,
    ).item()
    assert (
        runner.encoder_cache.encoder_outputs["image-1"].deepstack_visual_embeds is None
    )


def test_run_vision_encoders_iterates_multiple_requests_and_indices() -> None:
    runner = _runner_with_encoder_cache()
    adapter = _RecordingAdapter()
    runner._multimodal_adapter = adapter
    req0_features = [_feature("img-a"), _feature("img-b")]
    req1_features = [_feature("img-c")]
    assert runner.encoder_cache is not None
    runner.encoder_cache.add_request("req-0", req0_features)
    runner.encoder_cache.add_request("req-1", req1_features)

    runner._run_vision_encoders({"req-0": [0, 1], "req-1": [0]})

    assert len(adapter.encode_calls) == 1
    encoded_identifiers = [feature.identifier for feature in adapter.encode_calls[0]]
    assert encoded_identifiers == ["img-a", "img-b", "img-c"]
    assert set(runner.encoder_cache.encoder_outputs) == {"img-a", "img-b", "img-c"}


def test_run_vision_encoders_skips_cached_and_batches_only_uncached_features(
    fake_encode_result,
) -> None:
    runner = _runner_with_encoder_cache()
    adapter = _RecordingAdapter()
    runner._multimodal_adapter = adapter
    features = [_feature("cached-image"), _feature("fresh-image")]
    assert runner.encoder_cache is not None
    runner.encoder_cache.add_request("req-0", features)
    cached = fake_encode_result(mx.array([[7.0]]))
    runner.encoder_cache.encoder_outputs["cached-image"] = cached

    runner._run_vision_encoders({"req-0": [0, 1]})

    assert len(adapter.encode_calls) == 1
    assert [feature.identifier for feature in adapter.encode_calls[0]] == [
        "fresh-image"
    ]
    assert runner.encoder_cache.encoder_outputs["cached-image"] is cached
    assert "fresh-image" in runner.encoder_cache.encoder_outputs


def test_run_vision_encoders_skips_adapter_call_when_all_features_cached(
    fake_encode_result,
) -> None:
    runner = _runner_with_encoder_cache()
    adapter = _RecordingAdapter()
    runner._multimodal_adapter = adapter
    features = [_feature("image-0")]
    assert runner.encoder_cache is not None
    runner.encoder_cache.add_request("req-0", features)
    cached = fake_encode_result(mx.array([[7.0]]))
    runner.encoder_cache.encoder_outputs["image-0"] = cached

    runner._run_vision_encoders({"req-0": [0]})

    assert adapter.encode_calls == []
    assert runner.encoder_cache.encoder_outputs["image-0"] is cached


def test_run_vision_encoders_deduplicates_repeated_feature_identifier_across_requests() -> (
    None
):
    runner = _runner_with_encoder_cache()
    adapter = _RecordingAdapter()
    runner._multimodal_adapter = adapter
    req0_features = [_feature("shared-image")]
    req1_features = [_feature("shared-image")]
    assert runner.encoder_cache is not None
    runner.encoder_cache.add_request("req-0", req0_features)
    runner.encoder_cache.add_request("req-1", req1_features)

    runner._run_vision_encoders({"req-0": [0], "req-1": [0]})

    assert len(adapter.encode_calls) == 1
    assert [feature.identifier for feature in adapter.encode_calls[0]] == [
        "shared-image"
    ]
    assert set(runner.encoder_cache.encoder_outputs) == {"shared-image"}


def test_run_vision_encoders_raises_for_unregistered_request() -> None:
    runner = _runner_with_encoder_cache()
    runner._multimodal_adapter = _RecordingAdapter()

    with pytest.raises(RuntimeError, match="unregistered request"):
        runner._run_vision_encoders({"missing-req": [0]})


def test_run_vision_encoders_raises_for_out_of_range_index() -> None:
    runner = _runner_with_encoder_cache()
    runner._multimodal_adapter = _RecordingAdapter()
    assert runner.encoder_cache is not None
    runner.encoder_cache.add_request("req-0", [_feature("image-0")])

    with pytest.raises(IndexError, match="out of range"):
        runner._run_vision_encoders({"req-0": [3]})


@pytest.mark.parametrize("output_count", [1, 3])
def test_run_vision_encoders_raises_when_adapter_returns_wrong_count(
    output_count,
) -> None:
    runner = _runner_with_encoder_cache()
    adapter = _RecordingAdapter()
    runner._multimodal_adapter = adapter
    features = [_feature("image-0"), _feature("image-1")]
    assert runner.encoder_cache is not None
    runner.encoder_cache.add_request("req-0", features)
    adapter.queue_outputs(
        [
            [
                Qwen3VLVisionEncodeResult(
                    hidden_states=mx.zeros((1,)),
                    deepstack_visual_embeds=None,
                )
                for _ in range(output_count)
            ]
        ]
    )

    with pytest.raises(
        RuntimeError,
        match=f"encode_multimodal returned {output_count}",
    ):
        runner._run_vision_encoders({"req-0": [0, 1]})
