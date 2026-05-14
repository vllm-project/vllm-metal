# SPDX-License-Identifier: Apache-2.0
"""Tests for paged forward multimodal splice + per-segment positions."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import mlx.core as mx
import pytest
from vllm.sampling_params import SamplingParams

from tests.stub_runner import make_stub_runner
from vllm_metal.multimodal import MultiModalFeatureSpec, PlaceholderRange
from vllm_metal.multimodal.qwen3_vl import Qwen3VLVisionEncodeResult
from vllm_metal.paged_attention_common import get_context
from vllm_metal.v1.mm import EncoderCache
from vllm_metal.v1.model_runner import PrefillRequest, RequestState
from vllm_metal.v1.spec_decode import PagedDecodeSegment


def _feature(identifier: str, *, offset: int, length: int) -> MultiModalFeatureSpec:
    return MultiModalFeatureSpec(
        data=None,
        modality="image",
        identifier=identifier,
        mm_position=PlaceholderRange(offset=offset, length=length),
    )


class _MmAdapter:
    """Fake adapter that records call_lm + supplies stable positions."""

    forward_ready = True

    def __init__(self, *, hidden_size: int = 4, vocab_size: int = 8) -> None:
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.call_lm_calls: list[dict[str, Any]] = []

    def text_model(self) -> Any:
        return MagicMock(
            return_value=MagicMock(
                logits=mx.zeros((1, 1, self.vocab_size), dtype=mx.float32)
            )
        )

    def embed_tokens(self, input_ids: mx.array) -> mx.array:
        batch, seq_len = input_ids.shape
        return mx.zeros((batch, seq_len, self.hidden_size), dtype=mx.float32)

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[MultiModalFeatureSpec],
    ) -> tuple[mx.array, int]:
        # Deterministic positions: sequential arange tiled across 3 sections.
        # The delta is -2 to mirror the typical Qwen3-VL image squeeze.
        seq_len = len(input_tokens)
        positions = mx.arange(seq_len, dtype=mx.int32)
        positions = mx.broadcast_to(positions[None, None, :], (3, 1, seq_len))
        return positions, -2

    def call_lm(
        self,
        input_ids: mx.array,
        inputs_embeds: mx.array,
        cache: list[Any],
        position_ids: mx.array,
        *,
        visual_pos_masks: Any | None = None,
        deepstack_visual_embeds: Any | None = None,
    ) -> Any:
        self.call_lm_calls.append(
            {
                "input_ids": input_ids,
                "inputs_embeds": inputs_embeds,
                "cache": cache,
                "position_ids": position_ids,
                "visual_pos_masks": visual_pos_masks,
                "deepstack_visual_embeds": deepstack_visual_embeds,
            }
        )
        batch, seq_len = input_ids.shape
        return MagicMock(
            logits=mx.zeros((batch, seq_len, self.vocab_size), dtype=mx.float32)
        )


def _runner(adapter: _MmAdapter, *, num_layers: int = 1):
    runner = make_stub_runner(
        encoder_cache=EncoderCache(),
        _is_vlm=True,
        _paged_attention_backend=MagicMock(),
        _paged_block_size=16,
        num_layers=num_layers,
        model_args={"vocab_size": adapter.vocab_size},
    )
    runner._multimodal_adapter = adapter
    runner._spec_decode_controller = MagicMock()
    # Each test sets build_decode_segments return per-need.
    return runner


def _scheduler_output() -> MagicMock:
    out = MagicMock()
    out.scheduled_spec_decode_tokens = {}
    return out


def _put_encode(
    runner,
    identifier: str,
    *,
    hidden_states: mx.array,
    deepstack: Any | None = None,
) -> None:
    runner.encoder_cache.encoder_outputs[identifier] = Qwen3VLVisionEncodeResult(
        hidden_states=hidden_states,
        deepstack_visual_embeds=deepstack,
    )


def _mm_prefill(
    req_id: str,
    *,
    token_ids: list[int],
    prompt_len: int,
    start_pos: int = 0,
    full_prompt: list[int] | None = None,
) -> PrefillRequest:
    return PrefillRequest(
        req_id=req_id,
        token_ids=token_ids,
        sampling_params=SamplingParams(temperature=0.0),
        block_ids=[0],
        generator=None,
        prompt_len=prompt_len,
        start_pos=start_pos,
        full_prompt_token_ids=full_prompt,
    )


class TestSingleMmPrefillNoChunking:
    def test_routes_through_call_lm_with_splice_and_positions(self) -> None:
        adapter = _MmAdapter()
        runner = _runner(adapter)
        features = [_feature("img-0", offset=1, length=2)]
        runner.encoder_cache.add_request("req-0", features)
        _put_encode(runner, "img-0", hidden_states=mx.ones((2, adapter.hidden_size)))
        runner._spec_decode_controller.build_decode_segments = MagicMock(
            return_value=()
        )

        prefill = _mm_prefill(
            "req-0",
            token_ids=[10, 99, 99, 11],
            prompt_len=4,
            start_pos=0,
            full_prompt=[10, 99, 99, 11],
        )

        runner._start_paged_forward(
            batch=MagicMock(),
            prefill_reqs=[prefill],
            decode_reqs=[],
            scheduler_output=_scheduler_output(),
        )

        assert len(adapter.call_lm_calls) == 1
        call = adapter.call_lm_calls[0]
        # input_ids shape (1, total=4); position_ids (3, 1, 4).
        assert call["input_ids"].shape == (1, 4)
        assert call["position_ids"].shape == (3, 1, 4)
        # Visual mask covers placeholder tokens at offsets 1 and 2.
        assert call["visual_pos_masks"].tolist() == [[False, True, True, False]]
        # No deepstack supplied → adapter sees None.
        assert call["deepstack_visual_embeds"] is None
        # Spliced rows at placeholder positions equal the encoder hidden_states (1.0).
        embeds = call["inputs_embeds"]
        assert mx.allclose(
            embeds[0, 1], mx.ones(adapter.hidden_size, dtype=mx.float32)
        ).item()
        assert mx.allclose(
            embeds[0, 2], mx.ones(adapter.hidden_size, dtype=mx.float32)
        ).item()


class TestMmPrefillChunkedFeatureSlicing:
    def test_cross_chunk_feature_slices_hidden_states_and_mask(self) -> None:
        # Feature spans global positions 2..7 (length=6).  Chunk 2 covers
        # global positions 4..7 (start_pos=4, len=4), so the chunk-local
        # mask should be positions 0..3 and the encoder hidden_states
        # rows 2..5 should be the ones that land in this chunk.
        adapter = _MmAdapter()
        runner = _runner(adapter)
        features = [_feature("img-0", offset=2, length=6)]
        runner.encoder_cache.add_request("req-0", features)
        # 6 placeholder rows in the encoder output; rows i carry value
        # ``float(i + 10)`` so we can spot which subset is spliced.
        hidden = mx.array([[float(i + 10)] * adapter.hidden_size for i in range(6)])
        _put_encode(runner, "img-0", hidden_states=hidden)
        runner._spec_decode_controller.build_decode_segments = MagicMock(
            return_value=()
        )

        prefill = _mm_prefill(
            "req-0",
            token_ids=[99] * 4,
            prompt_len=8,
            start_pos=4,
            full_prompt=[1, 1, 99, 99, 99, 99, 99, 99],
        )

        runner._start_paged_forward(
            batch=MagicMock(),
            prefill_reqs=[prefill],
            decode_reqs=[],
            scheduler_output=_scheduler_output(),
        )

        call = adapter.call_lm_calls[0]
        # All 4 chunk tokens are placeholders.
        assert call["visual_pos_masks"].tolist() == [[True, True, True, True]]
        # The spliced rows should carry feature-local indices 2..5 (values 12..15).
        embeds = call["inputs_embeds"]
        for chunk_local, feature_local in enumerate(range(2, 6)):
            value = float(feature_local + 10)
            assert mx.allclose(
                embeds[0, chunk_local],
                mx.full((adapter.hidden_size,), value, dtype=mx.float32),
            ).item()


class TestMmDecodeSegmentPositions:
    def test_decode_positions_use_cache_start_pos_plus_delta(self) -> None:
        adapter = _MmAdapter()
        runner = _runner(adapter)
        # State carries delta = -3; cache_start_pos = 5, n=1 →
        # M-RoPE position = [5 + -3] = [2], tiled across 3 sections.
        state = RequestState(
            token_ids=[1, 2, 3, 4, 5, 6],
            prompt_len=5,
            cache=[],
            sampling_params=SamplingParams(),
            mrope_position_delta=-3,
        )
        runner._request_states["req-mm"] = state

        segment = PagedDecodeSegment(
            req_id="req-mm",
            input_token_ids=(6,),
            start_row=0,
            num_query_tokens=1,
            draft_token_ids=(),
            cache_start_pos=5,
            block_ids=(0,),
        )
        runner._spec_decode_controller.build_decode_segments = MagicMock(
            return_value=(segment,)
        )

        runner._start_paged_forward(
            batch=MagicMock(),
            prefill_reqs=[],
            decode_reqs=[("req-mm", state)],
            scheduler_output=_scheduler_output(),
        )

        call = adapter.call_lm_calls[0]
        assert call["position_ids"].shape == (3, 1, 1)
        assert call["position_ids"].tolist() == [[[2]], [[2]], [[2]]]
        # No visual mask positions on a decode token.
        assert call["visual_pos_masks"].tolist() == [[False]]

    def test_spec_decode_multi_query_arange_plus_delta(self) -> None:
        adapter = _MmAdapter()
        runner = _runner(adapter)
        state = RequestState(
            token_ids=[1, 2, 3, 4, 5, 6],
            prompt_len=5,
            cache=[],
            sampling_params=SamplingParams(),
            mrope_position_delta=-3,
        )
        runner._request_states["req-mm"] = state
        # 3 draft rows + 1 bonus row = 4 query tokens, cache_start_pos=5.
        # Expected positions = arange(5, 5+4) + -3 = [2, 3, 4, 5] tiled.
        segment = PagedDecodeSegment(
            req_id="req-mm",
            input_token_ids=(6, 100, 101, 102),
            start_row=0,
            num_query_tokens=4,
            draft_token_ids=(100, 101, 102),
            cache_start_pos=5,
            block_ids=(0,),
        )
        runner._spec_decode_controller.build_decode_segments = MagicMock(
            return_value=(segment,)
        )

        runner._start_paged_forward(
            batch=MagicMock(),
            prefill_reqs=[],
            decode_reqs=[("req-mm", state)],
            scheduler_output=_scheduler_output(),
        )

        call = adapter.call_lm_calls[0]
        assert call["position_ids"].shape == (3, 1, 4)
        assert call["position_ids"].tolist() == [
            [[2, 3, 4, 5]],
            [[2, 3, 4, 5]],
            [[2, 3, 4, 5]],
        ]


class TestMixedMmAndText:
    def test_text_prefill_alongside_mm_prefill_gets_none_segment_positions(
        self,
    ) -> None:
        adapter = _MmAdapter()
        runner = _runner(adapter)
        features = [_feature("img-0", offset=0, length=2)]
        runner.encoder_cache.add_request("req-mm", features)
        _put_encode(runner, "img-0", hidden_states=mx.ones((2, adapter.hidden_size)))
        runner._spec_decode_controller.build_decode_segments = MagicMock(
            return_value=()
        )

        captured_ctx: dict[str, Any] = {}

        original_call_lm = adapter.call_lm

        def capturing_call_lm(*args, **kwargs):
            ctx = get_context()
            captured_ctx["segment_positions"] = list(ctx.segment_positions)
            return original_call_lm(*args, **kwargs)

        adapter.call_lm = capturing_call_lm  # type: ignore[method-assign]

        mm_prefill = _mm_prefill(
            "req-mm",
            token_ids=[99, 99],
            prompt_len=2,
            start_pos=0,
            full_prompt=[99, 99],
        )
        text_prefill = PrefillRequest(
            req_id="req-text",
            token_ids=[1, 2, 3],
            sampling_params=SamplingParams(temperature=0.0),
            block_ids=[1],
            generator=None,
            prompt_len=3,
            start_pos=0,
            full_prompt_token_ids=None,
        )

        runner._start_paged_forward(
            batch=MagicMock(),
            prefill_reqs=[mm_prefill, text_prefill],
            decode_reqs=[],
            scheduler_output=_scheduler_output(),
        )

        seg_positions = captured_ctx["segment_positions"]
        # Two prefill segments: mm first, then text.
        assert len(seg_positions) == 2
        # mm segment: caller-supplied positions (shape (3, 1, 2)).
        assert seg_positions[0] is not None
        assert seg_positions[0].shape == (3, 1, 2)
        # text segment: None so attention falls back to int-offset arange.
        assert seg_positions[1] is None


class TestDeepstackPerChunkSlice:
    def test_deepstack_concat_across_features_uses_chunk_slice(self) -> None:
        adapter = _MmAdapter()
        runner = _runner(adapter)
        # Two features, both spanning length 2 within a single chunk.
        # Feature A at offset 1, feature B at offset 5.  Each carries one
        # deepstack layer with distinct values so we can verify the
        # per-feature concat order and slicing.
        features = [
            _feature("img-A", offset=1, length=2),
            _feature("img-B", offset=5, length=2),
        ]
        runner.encoder_cache.add_request("req-0", features)
        _put_encode(
            runner,
            "img-A",
            hidden_states=mx.zeros((2, adapter.hidden_size)),
            deepstack=[mx.array([[1.0, 1.0], [2.0, 2.0]])],
        )
        _put_encode(
            runner,
            "img-B",
            hidden_states=mx.zeros((2, adapter.hidden_size)),
            deepstack=[mx.array([[3.0, 3.0], [4.0, 4.0]])],
        )
        runner._spec_decode_controller.build_decode_segments = MagicMock(
            return_value=()
        )

        prefill = _mm_prefill(
            "req-0",
            token_ids=[0, 99, 99, 0, 0, 99, 99, 0],
            prompt_len=8,
            start_pos=0,
            full_prompt=[0, 99, 99, 0, 0, 99, 99, 0],
        )

        runner._start_paged_forward(
            batch=MagicMock(),
            prefill_reqs=[prefill],
            decode_reqs=[],
            scheduler_output=_scheduler_output(),
        )

        call = adapter.call_lm_calls[0]
        ds = call["deepstack_visual_embeds"]
        assert ds is not None
        assert len(ds) == 1
        # Layer 0: A's two rows then B's two rows in offset order.
        assert mx.allclose(
            ds[0],
            mx.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]),
        ).item()


class TestMmPrefillDeltaRoundTrip:
    """``_start_paged_forward`` → ``_sample_paged_batch`` round-trip.

    The mm prefill's ``mrope_position_delta`` must land on the new
    ``RequestState`` so the next paged round routes through the mm
    decode path; otherwise the next round would treat the request as
    text-only and produce wrong M-RoPE positions.
    """

    def _patch_sampling(self, monkeypatch, prefill_tokens, decode_tokens):
        import vllm_metal.v1.model_runner as mr
        from vllm_metal.v1.sampling_batch import _SamplingResult

        monkeypatch.setattr(
            mr,
            "sample_prefill_tokens",
            lambda *a, **kw: _SamplingResult(prefill_tokens, None),
        )
        monkeypatch.setattr(
            mr,
            "sample_decode_tokens",
            lambda *a, **kw: _SamplingResult(decode_tokens, None),
        )

    def test_new_final_mm_prefill_writes_delta_to_request_state(
        self, monkeypatch
    ) -> None:
        adapter = _MmAdapter()
        runner = _runner(adapter)
        features = [_feature("img-0", offset=0, length=1)]
        runner.encoder_cache.add_request("req-0", features)
        _put_encode(runner, "img-0", hidden_states=mx.ones((1, adapter.hidden_size)))
        runner._spec_decode_controller.build_decode_segments = MagicMock(
            return_value=()
        )

        prefill = _mm_prefill(
            "req-0",
            token_ids=[99, 11],
            prompt_len=2,
            start_pos=0,
            full_prompt=[99, 11],
        )

        # Use the runner's real ExecutionBatch path so the postprocess
        # iterates ``paged_prefill_entries``.
        from vllm_metal.v1.model_runner import (
            _ExecutionBatch,
            _PendingPrefillEntry,
        )

        batch = _ExecutionBatch()
        entry = _PendingPrefillEntry(
            output_idx=batch.add_output("req-0", []),
            prefill=prefill,
            result_mode="new_final",
        )
        batch.paged_prefill_entries.append(entry)

        runner._start_paged_forward(
            batch=batch,
            prefill_reqs=[prefill],
            decode_reqs=[],
            scheduler_output=_scheduler_output(),
        )

        # mm_prefill_deltas stashed on _PagedForwardState.
        state = runner._execute_model_state
        assert state is not None
        assert state.mm_prefill_deltas == {"req-0": -2}

        # Run _sample_paged_batch to drive postprocess; patch sampling
        # to bypass the real sampler stack.
        self._patch_sampling(monkeypatch, prefill_tokens=[42], decode_tokens=[])
        runner._sample_paged_batch()

        new_state = runner._request_states["req-0"]
        assert new_state.mrope_position_delta == -2
        # Sanity: token_ids = full_prompt + [next_token]
        assert new_state.token_ids == [99, 11, 42]

    def test_cached_final_mm_prefill_updates_existing_state_delta(
        self, monkeypatch
    ) -> None:
        adapter = _MmAdapter()
        runner = _runner(adapter)
        features = [_feature("img-0", offset=0, length=1)]
        runner.encoder_cache.add_request("req-0", features)
        _put_encode(runner, "img-0", hidden_states=mx.ones((1, adapter.hidden_size)))
        # Pre-existing state from a prior intermediate chunk — delta
        # not yet stashed.
        runner._request_states["req-0"] = RequestState(
            token_ids=[99, 11],  # prompt only (no sampled token yet)
            prompt_len=2,
            cache=[],
            sampling_params=SamplingParams(temperature=0.0),
            mrope_position_delta=None,
        )
        runner._spec_decode_controller.build_decode_segments = MagicMock(
            return_value=()
        )

        prefill = _mm_prefill(
            "req-0",
            token_ids=[11],  # final continuation chunk
            prompt_len=2,
            start_pos=1,
            full_prompt=[99, 11],
        )

        from vllm_metal.v1.model_runner import (
            _ExecutionBatch,
            _PendingPrefillEntry,
        )

        batch = _ExecutionBatch()
        entry = _PendingPrefillEntry(
            output_idx=batch.add_output("req-0", []),
            prefill=prefill,
            result_mode="cached_final",
        )
        batch.paged_prefill_entries.append(entry)

        runner._start_paged_forward(
            batch=batch,
            prefill_reqs=[prefill],
            decode_reqs=[],
            scheduler_output=_scheduler_output(),
        )

        self._patch_sampling(monkeypatch, prefill_tokens=[42], decode_tokens=[])
        runner._sample_paged_batch()

        state = runner._request_states["req-0"]
        assert state.mrope_position_delta == -2


class TestMmPrefillFullPromptRequired:
    def test_raises_when_full_prompt_missing(self) -> None:
        adapter = _MmAdapter()
        runner = _runner(adapter)
        features = [_feature("img-0", offset=0, length=1)]
        runner.encoder_cache.add_request("req-0", features)
        _put_encode(runner, "img-0", hidden_states=mx.ones((1, adapter.hidden_size)))
        runner._spec_decode_controller.build_decode_segments = MagicMock(
            return_value=()
        )

        prefill = _mm_prefill(
            "req-0",
            token_ids=[99],
            prompt_len=1,
            start_pos=0,
            full_prompt=None,  # missing!
        )

        with pytest.raises(RuntimeError, match="full_prompt_token_ids"):
            runner._start_paged_forward(
                batch=MagicMock(),
                prefill_reqs=[prefill],
                decode_reqs=[],
                scheduler_output=_scheduler_output(),
            )
