# SPDX-License-Identifier: Apache-2.0
"""Tests for v1 STT integration in MetalModelRunner."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mlx.core as mx
import numpy as np
import pytest
import torch

vllm = pytest.importorskip("vllm", reason="vllm not installed")

from vllm.sampling_params import SamplingParams  # noqa: E402

from vllm_metal.stt.audio import (  # noqa: E402
    N_FRAMES,
    SAMPLE_RATE,
    log_mel_spectrogram,
    pad_or_trim,
)
from vllm_metal.stt.config import STT_SCHED_BLOCK_BYTES  # noqa: E402
from vllm_metal.stt.transcribe import load_model  # noqa: E402
from vllm_metal.v1.model_runner import MetalModelRunner, STTExecutor  # noqa: E402


class _StubRunner:
    """Lightweight concrete test double for MetalModelRunner (STT path only).

    Inherits ``_execute_stt`` from the real class so class invariants
    (assert, attribute access) are exercised without MagicMock rebinding.
    Only the fields consumed by ``_execute_stt`` are initialised.
    """

    _execute_stt = MetalModelRunner._execute_stt

    def __init__(self, executor: STTExecutor) -> None:
        self._is_stt = True
        self.model = MagicMock()
        self.model.encode = MagicMock(return_value=mx.ones((1, 1500, 512)))
        self._request_states: dict = {}
        self._pending_output = None
        self._stt_executor = executor
        self._stt_executor.model = self.model

    @property
    def is_stt(self) -> bool:
        return self._is_stt


def _make_executor() -> STTExecutor:
    model = MagicMock()
    model.encode = MagicMock(return_value=mx.ones((1, 1500, 512)))
    executor = STTExecutor(model, "/fake/model/path")

    mock_tokenizer = MagicMock()
    mock_tokenizer.convert_tokens_to_ids.return_value = 50257
    mock_transcriber = MagicMock()
    mock_transcriber.tokenizer = mock_tokenizer
    mock_transcriber.greedy_decode_tokens.return_value = [100, 200]
    executor._transcriber = mock_transcriber
    return executor


def _make_valid_mm_features() -> list[SimpleNamespace]:
    mel = np.zeros((80, 3000), dtype=np.float32)
    field_elem = SimpleNamespace(data=mel)
    feature_spec = SimpleNamespace(data={"input_features": field_elem})
    return [feature_spec]


def _make_runner() -> _StubRunner:
    return _StubRunner(_make_executor())


def _make_scheduler_output(
    new_reqs=None, finished_req_ids=None, cached_req_ids=None
) -> SimpleNamespace:
    out = SimpleNamespace()
    out.scheduled_new_reqs = new_reqs or []
    out.finished_req_ids = finished_req_ids or set()
    out.scheduled_cached_reqs = SimpleNamespace(req_ids=cached_req_ids or [])
    out.total_num_scheduled_tokens = len(new_reqs or [])
    out.num_scheduled_tokens = {}
    return out


def _make_new_req(
    req_id: str = "req-1",
    prompt_token_ids=None,
    sampling_params=None,
    mm_features=None,
) -> SimpleNamespace:
    req = SimpleNamespace()
    req.req_id = req_id
    req.prompt_token_ids = (
        [50258, 50259, 50359] if prompt_token_ids is None else prompt_token_ids
    )
    req.sampling_params = (
        SamplingParams(temperature=0) if sampling_params is None else sampling_params
    )
    # vLLM normalizes missing multimodal payloads to an empty list.
    req.mm_features = mm_features or []
    return req


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


class TestExtractAudioFeatures:
    """Tests for STTExecutor.extract_audio_features."""

    def test_valid_numpy_input(self) -> None:
        """Valid numpy input should return encoded features."""
        executor = _make_executor()
        encoded = mx.ones((1, 1500, 512))
        executor.model.encode = MagicMock(return_value=encoded)

        mel = np.zeros((80, 3000), dtype=np.float32)

        result = executor.extract_audio_features(mel)

        assert result is not None
        executor.model.encode.assert_called_once()


class TestExtractAudioFeatureValidation:
    """Validation of normalized STT input features."""

    def test_1d_mel_raises_valueerror(self) -> None:
        """1D mel input should raise ValueError (expected 2D or 3D)."""
        executor = _make_executor()
        mel = np.zeros((3000,), dtype=np.float32)

        with pytest.raises(ValueError, match="rank"):
            executor.extract_audio_features(mel)

    def test_4d_mel_raises_valueerror(self) -> None:
        """4D mel input should raise ValueError (expected 2D or 3D)."""
        executor = _make_executor()
        mel = np.zeros((1, 1, 80, 3000), dtype=np.float32)

        with pytest.raises(ValueError, match="rank"):
            executor.extract_audio_features(mel)


class TestSamplingParamsValidation:
    """Tests for sampling params validation in _execute_stt."""

    def test_non_greedy_raises_valueerror(self) -> None:
        """Non-zero temperature should raise ValueError."""
        runner = _make_runner()
        non_greedy = SamplingParams(temperature=0.7)
        req = _make_new_req(
            sampling_params=non_greedy,
            mm_features=_make_valid_mm_features(),
        )
        sched = _make_scheduler_output(new_reqs=[req])

        with pytest.raises(ValueError, match="greedy"):
            runner._execute_stt(sched)

    def test_greedy_accepted(self) -> None:
        """temperature=0 should not raise."""
        runner = _make_runner()
        greedy = SamplingParams(temperature=0)
        req = _make_new_req(
            sampling_params=greedy,
            mm_features=_make_valid_mm_features(),
        )
        sched = _make_scheduler_output(new_reqs=[req])

        result = runner._execute_stt(sched)

        assert result is None
        assert runner._pending_output is not None
        assert runner._pending_output.sampled_token_ids == [[100, 200, 50257]]


class TestExecuteSTTProtocol:
    """Tests for _execute_stt output protocol and request lifecycle."""

    def _run_stt(self, runner, sched):
        """Run _execute_stt (STTExecutor is pre-cached in _make_runner)."""
        return runner._execute_stt(sched)

    def test_returns_none_sets_pending_output(self) -> None:
        """_execute_stt must return None and store result in _pending_output.

        This is the protocol expected by vLLM's sample_tokens() flow.
        """
        runner = _make_runner()
        req = _make_new_req(mm_features=_make_valid_mm_features())
        sched = _make_scheduler_output(new_reqs=[req])

        result = self._run_stt(runner, sched)

        assert result is None, "execute_stt must return None (not ModelRunnerOutput)"
        assert runner._pending_output is not None
        assert runner._pending_output.req_ids == ["req-1"]
        assert runner._pending_output.sampled_token_ids == [[100, 200, 50257]]

    def test_invalid_audio_request_raises_with_req_id(self) -> None:
        """Malformed STT requests should fail with request context."""
        runner = _make_runner()
        req = _make_new_req(req_id="broken-req", mm_features=None)
        sched = _make_scheduler_output(new_reqs=[req])

        with pytest.raises(ValueError, match="broken-req"):
            self._run_stt(runner, sched)

    def test_encode_valueerror_propagates(self) -> None:
        """Model encode failures should keep their original error."""
        runner = _make_runner()
        runner.model.encode = MagicMock(side_effect=ValueError("encode failed"))
        req = _make_new_req(mm_features=_make_valid_mm_features())
        sched = _make_scheduler_output(new_reqs=[req])

        with pytest.raises(ValueError, match="encode failed"):
            self._run_stt(runner, sched)

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
            new_reqs=[_make_new_req(mm_features=_make_valid_mm_features())],
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
        req1 = _make_new_req(req_id="r1", mm_features=_make_valid_mm_features())
        req2 = _make_new_req(req_id="r2", mm_features=_make_valid_mm_features())
        sched = _make_scheduler_output(new_reqs=[req1, req2])

        self._run_stt(runner, sched)

        output = runner._pending_output
        assert output is not None
        assert output.req_ids == ["r1", "r2"]
        assert output.req_id_to_index == {"r1": 0, "r2": 1}


class TestExtractAudioFeaturesFormats:
    """Tests for STTExecutor.extract_audio_features input handling."""

    def test_torch_float32_tensor(self) -> None:
        """torch float32 tensor should be converted correctly."""
        executor = _make_executor()
        encoded = mx.ones((1, 1500, 512))
        executor.model.encode = MagicMock(return_value=encoded)

        mel = torch.zeros(80, 3000, dtype=torch.float32)
        result = executor.extract_audio_features(mel)

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
        result = executor.extract_audio_features(mel)

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
        result = executor.extract_audio_features(mel)

        assert result is not None

    def test_3d_mel_transposed_correctly(self) -> None:
        """3D mel (batch, n_mels, time) should become (batch, time, n_mels)."""
        executor = _make_executor()

        def capture_encode(mel_input):
            assert mel_input.shape == (1, 3000, 80)
            return mx.ones((1, 1500, 512))

        executor.model.encode = capture_encode

        mel = np.zeros((1, 80, 3000), dtype=np.float32)
        result = executor.extract_audio_features(mel)

        assert result is not None


class TestKVCacheSTT:
    """Tests for KV cache methods when is_stt is True."""

    def test_get_kv_cache_spec_returns_dummy(self) -> None:
        """STT should return a single-entry dummy spec for scheduler init."""
        runner = _make_runner()
        runner.metal_config = MagicMock()
        runner.metal_config.block_size = 16

        runner.get_kv_cache_spec = MetalModelRunner.get_kv_cache_spec.__get__(runner)
        spec = runner.get_kv_cache_spec()

        assert len(spec) == 1
        assert "layers.0.self_attn" in spec

    def test_get_cache_block_size_bytes_returns_constant(self) -> None:
        """STT should return STT_SCHED_BLOCK_BYTES."""
        runner = _make_runner()

        runner.get_cache_block_size_bytes = (
            MetalModelRunner.get_cache_block_size_bytes.__get__(runner)
        )
        assert runner.get_cache_block_size_bytes() == STT_SCHED_BLOCK_BYTES


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


class TestIsSTTProperty:
    """Tests for the public is_stt property on MetalModelRunner."""

    def test_is_stt_default_false(self) -> None:
        """is_stt should be False before STT model loading."""
        runner = MagicMock()
        runner._is_stt = False

        prop = MetalModelRunner.is_stt.fget(runner)  # type: ignore[attr-defined]

        assert prop is False

    def test_is_stt_true_after_loading(self) -> None:
        """is_stt should be True after STT model loading."""
        runner = MagicMock()
        runner._is_stt = True

        prop = MetalModelRunner.is_stt.fget(runner)  # type: ignore[attr-defined]

        assert prop is True


@pytest.mark.slow
class TestSTTExecutorEndToEnd:
    """End-to-end test through STTExecutor with a real Whisper model.

    Run with ``pytest -m slow`` to include.
    """

    def test_decode_silence_produces_tokens(self) -> None:
        """Decoding silence through a real model should not crash."""
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


# ===========================================================================
# Qwen3-ASR STTExecutor dispatch & integration
# ===========================================================================


def _make_qwen3_executor() -> STTExecutor:
    """Create a STTExecutor configured for Qwen3-ASR model type.

    Uses concrete stubs (not MagicMock rebinding) to exercise the
    Qwen3-ASR-specific dispatch paths in STTExecutor.
    """
    model = MagicMock()
    model.model_type = "qwen3_asr"
    model.config = SimpleNamespace(eos_token_id=151643)
    model.encode = MagicMock(return_value=mx.ones((50, 1024)))

    executor = STTExecutor(model, "/fake/qwen3-asr")

    # Pre-inject a mock transcriber that mimics Qwen3ASRTranscriber
    mock_tokenizer = MagicMock()
    # tokenizer.encode returns different IDs for special tokens
    _token_map = {
        "<asr_text>": [151674],
        "<|im_end|>": [151645],
    }
    mock_tokenizer.encode = MagicMock(
        side_effect=lambda s, add_special_tokens=False: _token_map.get(s, [0])
    )
    mock_transcriber = MagicMock()
    mock_transcriber.tokenizer = mock_tokenizer
    mock_transcriber.build_prompt_tokens = MagicMock(
        return_value=[1, 2, 3]  # simplified prompt
    )
    # Return token stream: <lang> <asr_text> hello world <|im_end|>
    mock_transcriber.greedy_decode_tokens = MagicMock(
        return_value=[100, 151674, 200, 300, 151645]
    )
    executor._transcriber = mock_transcriber

    return executor


class TestSTTExecutorQwen3ASRDispatch:
    """Tests for Qwen3-ASR-specific dispatch paths in STTExecutor."""

    def test_eot_token_from_config(self) -> None:
        """Qwen3-ASR eot_token should come from model.config.eos_token_id."""
        executor = _make_qwen3_executor()
        assert executor.eot_token == 151643

    def test_transcriber_dispatches_to_qwen3(self) -> None:
        """model_type='qwen3_asr' should create Qwen3ASRTranscriber."""
        model = MagicMock()
        model.model_type = "qwen3_asr"
        executor = STTExecutor(model, "/fake/path")

        with patch("vllm_metal.stt.transcribe.Qwen3ASRTranscriber") as mock_cls:
            mock_cls.return_value = MagicMock()
            _ = executor.transcriber
            mock_cls.assert_called_once_with(model, model_path="/fake/path")

    def test_extract_audio_features_2d_passthrough(self) -> None:
        """Qwen3-ASR: 2D mel (n_mels, time) should pass through without transpose."""
        executor = _make_qwen3_executor()

        def capture_encode(mel_input):
            # Qwen3-ASR receives (n_mels, time) directly — no transpose
            assert mel_input.shape == (128, 500)
            return mx.ones((50, 1024))

        executor.model.encode = capture_encode

        mel = np.zeros((128, 500), dtype=np.float32)
        req = _make_new_req(mm_features=[{"input_features": mel}])
        result = executor.extract_audio_features(req)
        assert result is not None
        assert result.shape == (50, 1024)

    def test_extract_audio_features_3d_drops_batch(self) -> None:
        """Qwen3-ASR: 3D mel (1, n_mels, time) should drop batch dim."""
        executor = _make_qwen3_executor()

        def capture_encode(mel_input):
            # Batch dim stripped → (n_mels, time)
            assert mel_input.shape == (128, 500)
            return mx.ones((50, 1024))

        executor.model.encode = capture_encode

        mel = np.zeros((1, 128, 500), dtype=np.float32)
        req = _make_new_req(mm_features=[{"input_features": mel}])
        result = executor.extract_audio_features(req)
        assert result is not None

    def test_extract_audio_features_1d_raises(self) -> None:
        """Qwen3-ASR: 1D mel should raise ValueError (expects 2D or 3D)."""
        executor = _make_qwen3_executor()

        mel = np.zeros((500,), dtype=np.float32)
        req = _make_new_req(mm_features=[{"input_features": mel}])
        with pytest.raises(ValueError, match="rank"):
            executor.extract_audio_features(req)

    def test_decode_rebuilds_prompt_with_audio_frames(self) -> None:
        """Qwen3-ASR decode should rebuild prompt using build_prompt_tokens."""
        executor = _make_qwen3_executor()

        audio = mx.ones((50, 1024))  # 50 audio frames
        executor.decode(audio, [1, 2, 3])

        # build_prompt_tokens should be called with n_audio_frames=50
        executor.transcriber.build_prompt_tokens.assert_called_once_with(50)

    def test_decode_extracts_asr_text_tokens(self) -> None:
        """Qwen3-ASR decode should extract tokens between <asr_text> and <|im_end|>."""
        executor = _make_qwen3_executor()

        audio = mx.ones((50, 1024))
        result = executor.decode(audio, [1, 2, 3])

        # greedy_decode_tokens returns [100, 151674, 200, 300, 151645]
        # _extract_asr_text_tokens: between 151674 (<asr_text>) and 151645 (<|im_end|>)
        # → [200, 300]
        # + eot (151643) appended
        assert result == [200, 300, 151643]

    def test_decode_empty_prompt_rebuilds(self) -> None:
        """Qwen3-ASR rebuilds prompt even when caller passes empty list."""
        executor = _make_qwen3_executor()
        result = executor.decode(mx.ones((50, 1024)), [])
        # Should rebuild prompt and decode normally, not early-return EOT
        assert len(result) > 1
        assert result[-1] == 151643  # ends with EOT


# ===========================================================================
# TestExtractASRTextTokens
# ===========================================================================


class TestExtractASRTextTokens:
    """Tests for STTExecutor._extract_asr_text_tokens.

    This method extracts content tokens between <asr_text> and <|im_end|>,
    which is the core post-processing step for Qwen3-ASR output.
    """

    def test_basic_extraction(self) -> None:
        """Tokens between <asr_text> and <|im_end|> should be extracted."""
        executor = _make_qwen3_executor()
        # [lang, <asr_text>, hello, world, <|im_end|>]
        tokens = [100, 151674, 200, 300, 151645]
        result = executor._extract_asr_text_tokens(tokens)
        assert result == [200, 300]

    def test_no_asr_text_tag_returns_original(self) -> None:
        """Without <asr_text>, tokens should be returned as-is."""
        executor = _make_qwen3_executor()
        tokens = [100, 200, 300]
        result = executor._extract_asr_text_tokens(tokens)
        assert result == [100, 200, 300]

    def test_no_im_end_returns_to_end(self) -> None:
        """Without <|im_end|>, extract from <asr_text> to end of sequence."""
        executor = _make_qwen3_executor()
        # [lang, <asr_text>, hello, world] — no im_end
        tokens = [100, 151674, 200, 300]
        result = executor._extract_asr_text_tokens(tokens)
        assert result == [200, 300]

    def test_multiple_asr_text_uses_last(self) -> None:
        """Multiple <asr_text> tags should use the last one."""
        executor = _make_qwen3_executor()
        # Two <asr_text> tags
        tokens = [151674, 999, 151674, 200, 300, 151645]
        result = executor._extract_asr_text_tokens(tokens)
        assert result == [200, 300]

    def test_empty_content_between_tags(self) -> None:
        """<asr_text> immediately followed by <|im_end|> → empty list."""
        executor = _make_qwen3_executor()
        tokens = [100, 151674, 151645]
        result = executor._extract_asr_text_tokens(tokens)
        assert result == []

    def test_asr_text_at_end(self) -> None:
        """<asr_text> as last token → no content, return as-is."""
        executor = _make_qwen3_executor()
        tokens = [100, 200, 151674]
        result = executor._extract_asr_text_tokens(tokens)
        # start=3, which equals len(tokens), so returns original
        assert result == [100, 200, 151674]


# ===========================================================================
# TestQwen3ASRStubRejectsTranslate
# ===========================================================================


class TestQwen3ASRStubRejectsTranslate:
    """Qwen3-ASR does not support translation — must reject explicitly."""

    def test_translate_raises_valueerror(self) -> None:
        """task_type='translate' must raise ValueError, not silently transcribe."""
        from vllm_metal.stt.hf_config import _make_stub_class

        stub_cls = _make_stub_class()
        model_config = MagicMock()
        model_config.tokenizer = "Qwen/Qwen3-ASR-0.6B"
        stt_config = MagicMock()

        with pytest.raises(ValueError, match="does not support translation"):
            stub_cls.get_generation_prompt(
                audio=np.zeros(16000, dtype=np.float32),
                stt_config=stt_config,
                model_config=model_config,
                language="en",
                task_type="translate",
                request_prompt="",
                to_language=None,
            )
