# SPDX-License-Identifier: Apache-2.0
"""Tests for STT serve-boundary request normalization."""

from __future__ import annotations

from collections import UserDict
from types import SimpleNamespace

import numpy as np
import pytest

from vllm_metal.stt.serve import STTRequestInput, VLLMSTTRequestAdapter


class TestSTTServeRequestAdapter:
    """Tests for STT serve-boundary normalization."""

    @staticmethod
    def _make_request(
        *,
        req_id: str = "req-1",
        prompt_token_ids: list[int] | None = None,
        mm_features=None,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            req_id=req_id,
            prompt_token_ids=prompt_token_ids,
            mm_features=mm_features,
        )

    def test_normalizes_dict_payload(self) -> None:
        mel = np.zeros((80, 3000), dtype=np.float32)
        request = self._make_request(
            prompt_token_ids=[1, 2], mm_features=[{"input_features": mel}]
        )

        normalized = VLLMSTTRequestAdapter.from_vllm_request(request)

        assert isinstance(normalized, STTRequestInput)
        assert normalized.req_id == "req-1"
        assert normalized.prompt_token_ids == (1, 2)
        assert normalized.input_features is mel

    def test_normalizes_multimodal_feature_spec_payload(self) -> None:
        mel = np.zeros((80, 3000), dtype=np.float32)
        field_elem = SimpleNamespace(data=mel)
        feature_spec = SimpleNamespace(data=UserDict({"input_features": field_elem}))

        normalized = VLLMSTTRequestAdapter.from_vllm_request(
            self._make_request(mm_features=[feature_spec])
        )

        assert normalized.input_features is mel

    def test_rejects_missing_mm_features(self) -> None:
        request = self._make_request(req_id="broken-req", mm_features=None)

        with pytest.raises(ValueError, match="broken-req"):
            VLLMSTTRequestAdapter.from_vllm_request(request)

    def test_rejects_missing_req_id(self) -> None:
        request = self._make_request()
        del request.req_id

        with pytest.raises(ValueError, match="missing req_id"):
            VLLMSTTRequestAdapter.from_vllm_request(request)

    def test_rejects_missing_prompt_token_ids(self) -> None:
        request = self._make_request(
            req_id="broken-req", mm_features=[{"input_features": 1}]
        )
        del request.prompt_token_ids

        with pytest.raises(ValueError, match="broken-req"):
            VLLMSTTRequestAdapter.from_vllm_request(request)

    def test_rejects_invalid_prompt_token_ids(self) -> None:
        request = self._make_request(
            req_id="broken-req",
            prompt_token_ids=123,
            mm_features=[{"input_features": 1}],
        )

        with pytest.raises(ValueError, match="broken-req"):
            VLLMSTTRequestAdapter.from_vllm_request(request)

    def test_rejects_string_prompt_token_ids(self) -> None:
        request = self._make_request(
            req_id="broken-req",
            prompt_token_ids="123",
            mm_features=[{"input_features": 1}],
        )

        with pytest.raises(ValueError, match="broken-req"):
            VLLMSTTRequestAdapter.from_vllm_request(request)

    def test_rejects_missing_mm_features_attribute(self) -> None:
        request = self._make_request(req_id="broken-req")
        del request.mm_features

        with pytest.raises(ValueError, match="broken-req"):
            VLLMSTTRequestAdapter.from_vllm_request(request)

    def test_rejects_missing_input_features(self) -> None:
        request = self._make_request(
            req_id="broken-req", mm_features=[{"other": "value"}]
        )

        with pytest.raises(ValueError, match="input_features"):
            VLLMSTTRequestAdapter.from_vllm_request(request)

    def test_rejects_wrapped_none_input_features(self) -> None:
        field_elem = SimpleNamespace(data=None)
        feature_spec = SimpleNamespace(data=UserDict({"input_features": field_elem}))

        with pytest.raises(ValueError, match="input_features"):
            VLLMSTTRequestAdapter.from_vllm_request(
                self._make_request(req_id="broken-req", mm_features=[feature_spec])
            )
