# SPDX-License-Identifier: Apache-2.0
"""Checkpoint-free checks for the independent Nano reference harness."""

from types import SimpleNamespace

import mlx.core as mx
import pytest

from tools.nano_vl_reference import greedy_tokens, load_processor, sha256


def test_unpinned_processor_is_rejected_before_execution(tmp_path):
    source = tmp_path / "processor.py"
    source.write_text("raise RuntimeError('unverified source executed')\n")
    with pytest.raises(ValueError, match="source hash"):
        load_processor(source, "0" * 64)


def test_verified_processor_loads_from_resolved_hf_cache_blob(tmp_path):
    blob = tmp_path / "3d33d65e780377f858600b72e1838169e49df643"
    blob.write_text("processor_identity = 'verified-source'\n")
    snapshot_file = tmp_path / "image_processing.py"
    snapshot_file.symlink_to(blob)

    processor = load_processor(snapshot_file.resolve(), sha256(blob))

    assert processor.processor_identity == "verified-source"


def test_reference_resets_hybrid_cache_and_passes_pixels_only_on_prefill():
    caches = []
    calls = []
    observations = []

    class Model:
        def __init__(self):
            self.language_model = self

        def make_cache(self):
            cache = []
            caches.append(cache)
            return cache

        def __call__(self, tokens, *, pixel_values, cache):
            calls.append((tokens.tolist(), pixel_values, cache))
            cache.append(True)
            # Emit token 1 followed by EOS 2. A reused cache changes the result.
            scores = [0.0, 2.0, 1.0] if len(cache) == 1 else [0.0, 1.0, 2.0]
            return SimpleNamespace(logits=mx.array([[scores]]))

    pixels = object()
    model = Model()
    with mx.stream(mx.cpu):
        for _ in range(2):
            assert greedy_tokens(
                model,
                [7, 8],
                pixels,
                max_tokens=4,
                seed=0,
                eos_token_ids=[2],
                record_logits=lambda ids, scores: observations.append(ids.tolist()),
            ) == [1, 2]
    assert caches[0] is not caches[1]
    assert [call[0] for call in calls] == [[[7, 8]], [[1]], [[7, 8]], [[1]]]
    assert [call[1] for call in calls] == [pixels, None, pixels, None]
    assert observations == [call[0] for call in calls]


def test_reference_retains_all_tokens_when_output_limit_is_reached():
    model = SimpleNamespace(language_model=SimpleNamespace(make_cache=lambda: []))

    class NoEos:
        language_model = model.language_model

        def __call__(self, *args, **kwargs):
            return SimpleNamespace(logits=mx.array([[[0.0, 1.0, 0.0]]]))

    with mx.stream(mx.cpu):
        assert greedy_tokens(
            NoEos(), [7], None, max_tokens=2, seed=0, eos_token_ids=[2]
        ) == [1, 1]
