# SPDX-License-Identifier: Apache-2.0
"""Tests for shared Metal utilities."""

import mlx.core as mx

from vllm_metal.utils import set_wired_limit


def test_set_wired_limit_uses_pinned_mlx_api(monkeypatch) -> None:
    calls: list[int] = []

    monkeypatch.setattr(
        mx.metal,
        "device_info",
        lambda: {"max_recommended_working_set_size": 123},
    )
    monkeypatch.setattr(mx, "set_wired_limit", calls.append)

    set_wired_limit()

    assert calls == [123]
