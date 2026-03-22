# SPDX-License-Identifier: Apache-2.0
"""Serve-boundary helpers for Speech-to-Text requests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from vllm_metal.stt.runtime import STTAudioInput


class _MMFeatureField(Protocol):
    data: STTAudioInput | None


class _MMFeatureSpec(Protocol):
    data: Mapping[str, _MMFeatureField] | None


class VLLMSTTRequestLike(Protocol):
    req_id: str
    prompt_token_ids: Sequence[int] | None
    mm_features: Sequence[_MMFeatureSpec]


@dataclass(frozen=True)
class STTRequestInput:
    """Normalized STT request data consumed by the runtime path."""

    req_id: str
    prompt_token_ids: tuple[int, ...]
    input_features: STTAudioInput


class VLLMSTTRequestAdapter:
    """Boundary adapter that normalizes raw vLLM STT requests."""

    @classmethod
    def from_vllm_request(cls, request: VLLMSTTRequestLike) -> STTRequestInput:
        """Normalize a vLLM request object for the STT runtime path.

        Note: This adapter is used on the v1 runner path where vLLM already
        provides a typed request shape (e.g. NewRequestData). We trust vLLM's
        field-level contracts (req_id/prompt_token_ids/mm_features) and only
        validate STT-specific invariants (e.g. presence of input_features).
        """
        req_id = request.req_id
        prompt_token_ids = tuple(request.prompt_token_ids or ())
        return STTRequestInput(
            req_id=req_id,
            prompt_token_ids=prompt_token_ids,
            input_features=cls._extract_input_features(req_id, request.mm_features),
        )

    @staticmethod
    def _extract_input_features(
        req_id: str, mm_features: Sequence[_MMFeatureSpec]
    ) -> STTAudioInput:
        """Extract STT input features from vLLM multimodal feature wrappers.

        vLLM v1 provides multimodal inputs as a list of `MultiModalFeatureSpec`.
        For STT, we currently assume one audio feature per request and unwrap:

        - `mm_features[0].data["input_features"].data`

        Each `.data` layer may be `None` (e.g. cached/absent), so we treat any
        missing/None value as an invalid STT request and raise a request-scoped
        `ValueError`.
        """
        if not mm_features:
            raise ValueError(f"STT request {req_id!r} must include mm_features.")

        payload = mm_features[0].data
        field_elem = payload.get("input_features") if payload else None
        input_features = field_elem.data if field_elem is not None else None
        if input_features is None:
            raise ValueError(f"STT request {req_id!r} must include input_features.")

        return input_features
