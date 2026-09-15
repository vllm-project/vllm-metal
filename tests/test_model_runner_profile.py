# SPDX-License-Identifier: Apache-2.0
"""Tests for the encoder dummy pass inside ``profile_run``."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import mlx.core as mx

from tests.stub_runner import make_stub_runner
from vllm_metal.multimodal import MultiModalFeatureSpec, PlaceholderRange


class _ProfilingAdapter:
    forward_ready = True

    def __init__(self) -> None:
        self.encode_calls: list[list[MultiModalFeatureSpec]] = []

    def profile_features(self) -> list[MultiModalFeatureSpec]:
        return [
            MultiModalFeatureSpec(
                data=None,
                modality="image",
                identifier="profile-image",
                mm_position=PlaceholderRange(offset=0, length=4),
            )
        ]

    def encode_multimodal(self, features: list[MultiModalFeatureSpec]) -> list[Any]:
        self.encode_calls.append(list(features))
        return [SimpleNamespace(hidden_states=mx.zeros((4, 8)))]


class _PlainAdapter:
    forward_ready = True

    def __init__(self) -> None:
        self.encode_calls: list[list[MultiModalFeatureSpec]] = []

    def encode_multimodal(self, features: list[MultiModalFeatureSpec]) -> list[Any]:
        self.encode_calls.append(list(features))
        return []


class _EmptyProfileAdapter(_ProfilingAdapter):
    def profile_features(self) -> list[MultiModalFeatureSpec]:
        return []


def _runner(adapter: Any) -> Any:
    runner = make_stub_runner(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8),
        model_config=SimpleNamespace(max_model_len=8),
        _multimodal_adapter=adapter,
        _is_pooling=False,
        _pooling_backend=None,
    )
    runner._dummy_forward_outputs = lambda input_ids: [mx.zeros((1, 1, 8))]  # type: ignore[method-assign]
    runner._uses_encoder_pooling_backend = lambda: False  # type: ignore[method-assign]
    return runner


class TestProfileRunEncoder:
    def test_adapter_with_profile_features_is_encoded_once(self) -> None:
        adapter = _ProfilingAdapter()
        runner = _runner(adapter)

        runner.profile_run()

        assert len(adapter.encode_calls) == 1
        assert adapter.encode_calls[0][0].identifier == "profile-image"

    def test_adapter_without_profile_features_is_skipped(self) -> None:
        adapter = _PlainAdapter()
        runner = _runner(adapter)

        runner.profile_run()

        assert adapter.encode_calls == []

    def test_not_forward_ready_adapter_is_skipped(self) -> None:
        adapter = _ProfilingAdapter()
        adapter.forward_ready = False
        runner = _runner(adapter)

        runner.profile_run()

        assert adapter.encode_calls == []

    def test_no_adapter_is_fine(self) -> None:
        runner = _runner(None)
        assert runner.profile_run() >= 0

    def test_empty_profile_features_skips_encoding(self) -> None:
        adapter = _EmptyProfileAdapter()
        runner = _runner(adapter)

        runner.profile_run()

        assert adapter.encode_calls == []
