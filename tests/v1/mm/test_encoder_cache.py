# SPDX-License-Identifier: Apache-2.0
"""Tests for multimodal encoder cache bookkeeping."""

from __future__ import annotations

import mlx.core as mx

from vllm_metal.multimodal import (
    MultiModalFeatureSpec,
    PlaceholderRange,
)
from vllm_metal.v1.mm import (
    EncoderCache,
)


def _feature(identifier: str) -> MultiModalFeatureSpec:
    return MultiModalFeatureSpec(
        data=None,
        modality="image",
        identifier=identifier,
        mm_position=PlaceholderRange(offset=0, length=1),
    )


def test_add_request_registers_features() -> None:
    cache = EncoderCache()
    features = [_feature("image-0")]

    cache.add_request("req-0", features)

    assert cache.mm_features["req-0"] == features


def test_remove_request_is_idempotent() -> None:
    cache = EncoderCache()
    cache.add_request("req-0", [_feature("image-0")])

    cache.remove_request("req-0")
    cache.remove_request("req-0")

    assert "req-0" not in cache.mm_features


def test_free_encoder_cache_removes_one_hash() -> None:
    cache = EncoderCache()
    cache.encoder_outputs["hash-0"] = mx.array([[1.0]])
    cache.encoder_outputs["hash-1"] = mx.array([[2.0]])

    cache.free_encoder_cache("hash-0")
    cache.free_encoder_cache("missing")

    assert set(cache.encoder_outputs) == {"hash-1"}


def test_reset_encoder_cache_clears_outputs_only() -> None:
    cache = EncoderCache()
    features = [_feature("image-0")]
    cache.add_request("req-0", features)
    cache.encoder_outputs["hash-0"] = mx.array([[1.0]])

    cache.reset_encoder_cache()

    assert cache.encoder_outputs == {}
    assert cache.mm_features["req-0"] == features
