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

    def __init__(self, deepstack: list[mx.array] | None = None) -> None:
        self.encode_calls: list[list[MultiModalFeatureSpec]] = []
        self.hidden_states = mx.zeros((4, 8))
        self.deepstack = deepstack

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
        return [
            SimpleNamespace(
                hidden_states=self.hidden_states,
                deepstack_visual_embeds=self.deepstack,
            )
        ]


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


class TestProfileRunCacheReadings:
    def test_cache_is_read_only_after_synchronize(self, monkeypatch) -> None:
        """Both cache readings must follow a synchronize.

        The evaluated outputs are temporaries whose buffers return to the
        cache asynchronously; reading before they land under-reports the
        overhead on some launches and not others (#835).
        """
        events: list[str] = []

        def record(name: str, fn):
            def wrapper(*args, **kwargs):
                events.append(name)
                return fn(*args, **kwargs)

            return wrapper

        for name in ("synchronize", "clear_cache", "get_cache_memory", "eval"):
            monkeypatch.setattr(mx, name, record(name, getattr(mx, name)))

        _runner(None).profile_run()

        reads = [i for i, event in enumerate(events) if event == "get_cache_memory"]
        assert len(reads) == 2
        for read in reads:
            preceding = [event for event in events[:read] if event != "clear_cache"]
            assert preceding and preceding[-1] == "synchronize", events
        # The second reading follows the dummy forward, not just the clear.
        assert "eval" in events[reads[0] : reads[1]]

    def test_deepstack_residuals_are_profiled(self) -> None:
        deepstack = [mx.ones((4, 8)), mx.ones((4, 8))]
        adapter = _ProfilingAdapter(deepstack=deepstack)
        runner = _runner(adapter)

        outputs = runner._dummy_encoder_outputs()

        expected = [adapter.hidden_states, *deepstack]
        assert len(outputs) == len(expected)
        assert all(out is exp for out, exp in zip(outputs, expected, strict=True))
