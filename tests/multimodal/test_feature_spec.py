# SPDX-License-Identifier: Apache-2.0
"""Tests for generic multimodal feature-spec aliases."""

from __future__ import annotations

from vllm.multimodal.inputs import MultiModalFeatureSpec as UpstreamFeatureSpec

from vllm_metal.multimodal import MultiModalFeatureSpec


def test_re_export_matches_upstream() -> None:
    assert MultiModalFeatureSpec is UpstreamFeatureSpec
