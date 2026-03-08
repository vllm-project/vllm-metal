# SPDX-License-Identifier: Apache-2.0
"""Tests for v1 STT integration in MetalModelRunner."""

from __future__ import annotations

from collections import UserDict
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mlx.core as mx
import numpy as np
import pytest
import torch

vllm = pytest.importorskip("vllm", reason="vllm not installed")

from vllm.sampling_params import SamplingParams  # noqa: E402

from vllm_metal.stt.config import STT_SCHED_BLOCK_BYTES  # noqa: E402
from vllm_metal.v1.model_runner import STTExecutor  # noqa: E402

# ===========================================================================
# Helpers
# ===========================================================================


def _make_executor() -> STTExecutor:
    """Create a STTExecutor with a mock model and pre-cached transcriber."""
    model = MagicMock()
    executor = STTExecutor(model, "/fake/model/path")

    mock_tokenizer = MagicMock()
    mock_tokenizer.convert_tokens_to_ids.return_value = 50257
    mock_transcriber = MagicMock()
    mock_transcriber.tokenizer = mock_tokenizer
    mock_transcriber.greedy_decode_tokens.return_value = [100, 200]
    executor._transcriber = mock_transcriber

    return executor


class _StubRunner:
    """Lightweight concrete test double for MetalModelRunner (STT path only).

    Inherits ``_execute_stt`` from the real class so class invariants
    (assert, attribute access) are exercised without MagicMock rebinding.
    Only the fields consumed by ``_execute_stt`` are initialised.
    """

    from vllm_metal.v1.model_runner import MetalModelRunner

    # Inherit the real method — no __get__ rebinding needed.
    _execute_stt = MetalModelRunner._execute_stt

    def __init__(self) -> None:
        self._is_stt = True
        self.model = MagicMock()
        self._request_states: dict = {}
        self._pending_output = None
        self._stt_executor = _make_executor()
        self._stt_executor.model = self.model

    @property
    def is_stt(self) -> bool:
        return self._is_stt


def _make_runner() -> _StubRunner:
    """Create a lightweight runner stub with STT enabled."""
    return _StubRunner()


def _make_scheduler_output(new_reqs=None, finished_req_ids=None, cached_req_ids=None):
    """Create a minimal SchedulerOutput-like object."""
    out = SimpleNamespace()
    out.scheduled_new_reqs = new_reqs or []
    out.finished_req_ids = finished_req_ids or set()
    out.scheduled_cached_reqs = SimpleNamespace(req_ids=cached_req_ids or [])
    out.total_num_scheduled_tokens = len(new_reqs or [])
    out.num_scheduled_tokens = {}
    return out


def _make_new_req(
    req_id="req-1",
    prompt_token_ids=None,
    sampling_params=None,
    mm_features=None,
):
    """Create a minimal new request object."""
    req = SimpleNamespace()
    req.req_id = req_id
    req.prompt_token_ids = prompt_token_ids or [50258, 50259, 50359]
    req.sampling_params = sampling_params or SamplingParams(temperature=0)
    req.mm_features = mm_features
    return req


# ===========================================================================
# TestSTTExecutorDecode
# ===========================================================================


class TestSTTExecutorDecode:
    """Tests for STTExecutor.decode (delegates to transcriber)."""

    def test_empty_prompt_returns_eot(self) -> None:
        """Empty prompt should return just the EOT token."""
        executor = _make_executor()
        result = executor.decode(
            audio_features=mx.zeros((1, 10, 80)),
            prompt_token_ids=[],
        )
        assert result == [50257]
        # greedy_decode_tokens should NOT be called for empty prompts
        executor._transcriber.greedy_decode_tokens.assert_not_called()

    def test_delegates_to_transcriber(self) -> None:
        """Should delegate to transcriber.greedy_decode_tokens and append EOT."""
        executor = _make_executor()
        executor._transcriber.greedy_decode_tokens.return_value = [100, 200]

        result = executor.decode(
            audio_features=mx.zeros((1, 10, 80)),
            prompt_token_ids=[50258],
        )
        assert result == [100, 200, 50257]
        executor._transcriber.greedy_decode_tokens.assert_called_once()

    def test_eot_always_appended(self) -> None:
        """EOT must always be the last token for vLLM to finish the request."""
        executor = _make_executor()
        executor._transcriber.greedy_decode_tokens.return_value = [42]

        result = executor.decode(
            audio_features=mx.zeros((1, 10, 80)),
            prompt_token_ids=[50258],
        )
        assert result[-1] == 50257


# ===========================================================================
# TestExtractAudioFeatures
# ===========================================================================


class TestExtractAudioFeatures:
    """Tests for STTExecutor.extract_audio_features error paths."""

    def test_missing_mm_features_returns_none(self) -> None:
        """Request without mm_features should return None."""
        executor = _make_executor()
        req = _make_new_req(mm_features=None)
        del req.mm_features  # completely absent
        result = executor.extract_audio_features(req)
        assert result is None

    def test_empty_list_returns_none(self) -> None:
        """Empty mm_features list should return None."""
        executor = _make_executor()
        req = _make_new_req(mm_features=[])
        result = executor.extract_audio_features(req)
        assert result is None

    def test_wrong_type_returns_none(self) -> None:
        """Non-dict first element should return None."""
        executor = _make_executor()
        req = _make_new_req(mm_features=["not-a-dict"])
        result = executor.extract_audio_features(req)
        assert result is None

    def test_missing_key_returns_none(self) -> None:
        """Dict without 'input_features' key should return None."""
        executor = _make_executor()
        req = _make_new_req(mm_features=[{"other_key": np.zeros((80, 3000))}])
        result = executor.extract_audio_features(req)
        assert result is None

    def test_valid_numpy_input(self) -> None:
        """Valid numpy input should return encoded features."""
        executor = _make_executor()
        encoded = mx.ones((1, 1500, 512))
        executor.model.encode = MagicMock(return_value=encoded)

        mel = np.zeros((80, 3000), dtype=np.float32)
        req = _make_new_req(mm_features=[{"input_features": mel}])
        result = executor.extract_audio_features(req)
        assert result is not None
        executor.model.encode.assert_called_once()


# ===========================================================================
# TestMalformedMMFeatures
# ===========================================================================


class TestMalformedMMFeatures:
    """Additional error path coverage for mm_features handling."""

    def test_none_attribute(self) -> None:
        """mm_features=None should return None."""
        executor = _make_executor()
        req = _make_new_req(mm_features=None)
        result = executor.extract_audio_features(req)
        assert result is None

    def test_non_list_type(self) -> None:
        """mm_features that isn't a list should return None."""
        executor = _make_executor()
        req = _make_new_req(mm_features="not-a-list")
        result = executor.extract_audio_features(req)
        assert result is None

    def test_list_of_none(self) -> None:
        """mm_features=[None] should return None (not a dict)."""
        executor = _make_executor()
        req = _make_new_req(mm_features=[None])
        result = executor.extract_audio_features(req)
        assert result is None

    def test_1d_mel_raises_valueerror(self) -> None:
        """1D mel input should raise ValueError (expected 2D or 3D)."""
        executor = _make_executor()
        mel = np.zeros((3000,), dtype=np.float32)
        req = _make_new_req(mm_features=[{"input_features": mel}])
        with pytest.raises(ValueError, match="rank"):
            executor.extract_audio_features(req)

    def test_4d_mel_raises_valueerror(self) -> None:
        """4D mel input should raise ValueError (expected 2D or 3D)."""
        executor = _make_executor()
        mel = np.zeros((1, 1, 80, 3000), dtype=np.float32)
        req = _make_new_req(mm_features=[{"input_features": mel}])
        with pytest.raises(ValueError, match="rank"):
            executor.extract_audio_features(req)


# ===========================================================================
# TestSamplingParamsValidation
# ===========================================================================


class TestSamplingParamsValidation:
    """Tests for sampling params validation in _execute_stt."""

    def test_non_greedy_raises_valueerror(self) -> None:
        """Non-zero temperature should raise ValueError."""
        runner = _make_runner()
        non_greedy = SamplingParams(temperature=0.7)
        req = _make_new_req(sampling_params=non_greedy, mm_features=None)
        sched = _make_scheduler_output(new_reqs=[req])

        with pytest.raises(ValueError, match="greedy"):
            runner._execute_stt(sched)

    def test_greedy_accepted(self) -> None:
        """temperature=0 should not raise."""
        runner = _make_runner()
        greedy = SamplingParams(temperature=0)
        req = _make_new_req(
            sampling_params=greedy,
            mm_features=None,
        )
        sched = _make_scheduler_output(new_reqs=[req])

        # Should not raise — the request has no audio so it returns EOT
        runner._execute_stt(sched)


# ===========================================================================
# TestExecuteSTTProtocol
# ===========================================================================


class TestExecuteSTTProtocol:
    """Tests for _execute_stt_inner output protocol and request lifecycle."""

    def _run_stt(self, runner, sched):
        """Run _execute_stt (STTExecutor is pre-cached in _make_runner)."""
        return runner._execute_stt(sched)

    def test_returns_none_sets_pending_output(self) -> None:
        """_execute_stt must return None and store result in _pending_output.

        This is the protocol expected by vLLM's sample_tokens() flow.
        """
        runner = _make_runner()
        req = _make_new_req(mm_features=None)
        sched = _make_scheduler_output(new_reqs=[req])

        result = self._run_stt(runner, sched)

        assert result is None, "execute_stt must return None (not ModelRunnerOutput)"
        assert runner._pending_output is not None
        assert runner._pending_output.req_ids == ["req-1"]
        # No audio → EOT token
        assert runner._pending_output.sampled_token_ids == [[50257]]

    def test_cached_requests_get_eot(self) -> None:
        """Cached (decode-phase) requests should receive EOT to finish them."""
        runner = _make_runner()
        sched = _make_scheduler_output(cached_req_ids=["cached-1", "cached-2"])

        result = self._run_stt(runner, sched)

        assert result is None
        output = runner._pending_output
        assert output is not None
        assert "cached-1" in output.req_ids
        assert "cached-2" in output.req_ids
        # Both cached requests should get EOT
        for tokens in output.sampled_token_ids:
            assert tokens == [50257]

    def test_finished_reqs_cleaned_from_state(self) -> None:
        """finished_req_ids should be removed from _request_states."""
        runner = _make_runner()
        runner._request_states = {"old-1": "state", "old-2": "state"}
        sched = _make_scheduler_output(
            new_reqs=[_make_new_req(mm_features=None)],
            finished_req_ids={"old-1"},
        )

        self._run_stt(runner, sched)

        assert "old-1" not in runner._request_states
        assert "old-2" in runner._request_states

    def test_empty_batch_returns_empty_output(self) -> None:
        """No new and no cached requests should return empty ModelRunnerOutput."""
        runner = _make_runner()
        sched = _make_scheduler_output()

        result = self._run_stt(runner, sched)

        # Empty batch returns direct output (not via _pending_output)
        assert result is not None
        assert result.req_ids == []
        assert result.sampled_token_ids == []

    def test_multiple_new_requests(self) -> None:
        """Multiple new requests should all appear in output."""
        runner = _make_runner()
        req1 = _make_new_req(req_id="r1", mm_features=None)
        req2 = _make_new_req(req_id="r2", mm_features=None)
        sched = _make_scheduler_output(new_reqs=[req1, req2])

        self._run_stt(runner, sched)

        output = runner._pending_output
        assert output is not None
        assert output.req_ids == ["r1", "r2"]
        assert output.req_id_to_index == {"r1": 0, "r2": 1}


# ===========================================================================
# TestExtractAudioFeaturesFormats
# ===========================================================================


class TestExtractAudioFeaturesFormats:
    """Tests for STTExecutor.extract_audio_features input format handling."""

    def test_userdict_multimodal_spec(self) -> None:
        """MultiModalFeatureSpec with UserDict .data should be handled.

        vLLM's MultiModalKwargsItem extends UserDict, not dict.
        This was a production bug — isinstance(UserDict(), dict) is False.
        """
        executor = _make_executor()
        encoded = mx.ones((1, 1500, 512))
        executor.model.encode = MagicMock(return_value=encoded)

        # Simulate vLLM's MultiModalFeatureSpec structure:
        # spec.data is a UserDict (MultiModalKwargsItem)
        # spec.data["input_features"] is a MultiModalFieldElem with .data
        inner_tensor = np.zeros((80, 3000), dtype=np.float32)
        field_elem = SimpleNamespace(data=inner_tensor)
        kwargs_item = UserDict({"input_features": field_elem})
        feature_spec = SimpleNamespace(data=kwargs_item)

        req = _make_new_req(mm_features=[feature_spec])
        result = executor.extract_audio_features(req)

        assert result is not None
        executor.model.encode.assert_called_once()

    def test_torch_float32_tensor(self) -> None:
        """torch float32 tensor should be converted correctly."""
        executor = _make_executor()
        encoded = mx.ones((1, 1500, 512))
        executor.model.encode = MagicMock(return_value=encoded)

        mel = torch.zeros(80, 3000, dtype=torch.float32)
        req = _make_new_req(mm_features=[{"input_features": mel}])
        result = executor.extract_audio_features(req)

        assert result is not None
        executor.model.encode.assert_called_once()

    def test_torch_bfloat16_tensor(self) -> None:
        """torch bfloat16 tensor should be cast to float32 before numpy.

        bfloat16 has no numpy dtype — calling .numpy() directly raises
        TypeError. The code must call .float() first.
        """
        executor = _make_executor()
        encoded = mx.ones((1, 1500, 512))
        executor.model.encode = MagicMock(return_value=encoded)

        mel = torch.zeros(80, 3000, dtype=torch.bfloat16)
        req = _make_new_req(mm_features=[{"input_features": mel}])
        result = executor.extract_audio_features(req)

        assert result is not None
        executor.model.encode.assert_called_once()

    def test_2d_mel_transposed_correctly(self) -> None:
        """2D mel (n_mels, time) should become (1, time, n_mels)."""
        executor = _make_executor()

        def capture_encode(mel_input):
            # Verify shape is (1, time, n_mels) = (1, 3000, 80)
            assert mel_input.shape == (1, 3000, 80)
            return mx.ones((1, 1500, 512))

        executor.model.encode = capture_encode

        mel = np.zeros((80, 3000), dtype=np.float32)
        req = _make_new_req(mm_features=[{"input_features": mel}])
        result = executor.extract_audio_features(req)
        assert result is not None

    def test_3d_mel_transposed_correctly(self) -> None:
        """3D mel (batch, n_mels, time) should become (batch, time, n_mels)."""
        executor = _make_executor()

        def capture_encode(mel_input):
            assert mel_input.shape == (1, 3000, 80)
            return mx.ones((1, 1500, 512))

        executor.model.encode = capture_encode

        mel = np.zeros((1, 80, 3000), dtype=np.float32)
        req = _make_new_req(mm_features=[{"input_features": mel}])
        result = executor.extract_audio_features(req)
        assert result is not None


# ===========================================================================
# TestKVCacheSTT
# ===========================================================================


class TestKVCacheSTT:
    """Tests for KV cache methods when is_stt is True."""

    def test_get_kv_cache_spec_returns_dummy(self) -> None:
        """STT should return a single-entry dummy spec for scheduler init."""
        runner = _make_runner()
        runner.metal_config = MagicMock()
        runner.metal_config.block_size = 16

        from vllm_metal.v1.model_runner import MetalModelRunner

        runner.get_kv_cache_spec = MetalModelRunner.get_kv_cache_spec.__get__(runner)
        spec = runner.get_kv_cache_spec()

        assert len(spec) == 1
        assert "layers.0.self_attn" in spec

    def test_get_cache_block_size_bytes_returns_constant(self) -> None:
        """STT should return STT_SCHED_BLOCK_BYTES."""
        runner = _make_runner()

        from vllm_metal.v1.model_runner import MetalModelRunner

        runner.get_cache_block_size_bytes = (
            MetalModelRunner.get_cache_block_size_bytes.__get__(runner)
        )
        assert runner.get_cache_block_size_bytes() == STT_SCHED_BLOCK_BYTES


# ===========================================================================
# TestSTTExecutorTranscriberCaching
# ===========================================================================


class TestSTTExecutorTranscriberCaching:
    """Tests for lazy transcriber creation in STTExecutor."""

    def test_transcriber_created_lazily_with_model_path(self) -> None:
        """Transcriber should be created lazily with the correct model_path."""
        executor = STTExecutor(MagicMock(), "/fake/model/path")
        assert executor._transcriber is None  # not yet created

        with patch("vllm_metal.stt.transcribe.WhisperTranscriber") as mock_cls:
            mock_tokenizer = MagicMock()
            mock_tokenizer.convert_tokens_to_ids.return_value = 50257
            mock_cls.return_value.tokenizer = mock_tokenizer

            _ = executor.transcriber  # triggers lazy creation

            mock_cls.assert_called_once_with(
                executor.model, model_path="/fake/model/path"
            )

    def test_transcriber_reused_across_accesses(self) -> None:
        """Cached transcriber should not be recreated."""
        executor = _make_executor()
        t1 = executor.transcriber
        t2 = executor.transcriber
        assert t1 is t2


# ===========================================================================
# TestIsSTTProperty
# ===========================================================================


class TestIsSTTProperty:
    """Tests for the public is_stt property on MetalModelRunner."""

    def test_is_stt_default_false(self) -> None:
        """is_stt should be False before STT model loading."""
        runner = MagicMock()
        runner._is_stt = False

        from vllm_metal.v1.model_runner import MetalModelRunner

        prop = MetalModelRunner.is_stt.fget(runner)  # type: ignore[attr-defined]
        assert prop is False

    def test_is_stt_true_after_loading(self) -> None:
        """is_stt should be True after STT model loading."""
        runner = MagicMock()
        runner._is_stt = True

        from vllm_metal.v1.model_runner import MetalModelRunner

        prop = MetalModelRunner.is_stt.fget(runner)  # type: ignore[attr-defined]
        assert prop is True


# ===========================================================================
# End-to-end smoke test (requires real Whisper model)
# ===========================================================================


@pytest.mark.slow
class TestSTTExecutorEndToEnd:
    """End-to-end test through STTExecutor with a real Whisper model.

    Run with ``pytest -m slow`` to include.
    """

    def test_decode_silence_produces_tokens(self) -> None:
        """Decoding silence through a real model should not crash."""
        from vllm_metal.stt.audio import (
            N_FRAMES,
            SAMPLE_RATE,
            log_mel_spectrogram,
            pad_or_trim,
        )
        from vllm_metal.stt.transcribe import load_model

        model = load_model("openai/whisper-tiny")
        executor = STTExecutor(model, "openai/whisper-tiny")

        # Build mel from 3 s of silence
        silence = mx.zeros(SAMPLE_RATE * 3)
        n_mels = model.config.n_mels
        mel = log_mel_spectrogram(silence, n_mels=n_mels)
        mel = pad_or_trim(mel, N_FRAMES, axis=-1)
        mel = mel[None, ...].transpose(0, 2, 1)

        features = model.encode(mel)
        mx.eval(features)

        # Build prompt: <|startoftranscript|><|en|><|transcribe|><|notimestamps|>
        tokenizer = executor.transcriber.tokenizer
        prompt_ids = [
            tokenizer.convert_tokens_to_ids("<|startoftranscript|>"),
            tokenizer.convert_tokens_to_ids("<|en|>"),
            tokenizer.convert_tokens_to_ids("<|transcribe|>"),
            tokenizer.convert_tokens_to_ids("<|notimestamps|>"),
        ]

        result = executor.decode(features, prompt_ids)
        assert isinstance(result, list)
        assert len(result) >= 1
        # Must end with EOT
        eot = tokenizer.convert_tokens_to_ids("<|endoftext|>")
        assert result[-1] == eot
