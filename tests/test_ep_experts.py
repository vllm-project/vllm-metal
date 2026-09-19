# SPDX-License-Identifier: Apache-2.0
"""Expert-parallel sharding: partition parsing, weight slicing, masked routing."""

import mlx.core as mx
import pytest

from vllm_metal.distributed.experts import expert_partition


def test_partition_defaults_to_even_and_validates_overrides(monkeypatch):
    assert expert_partition(2, 128) == [64, 64]
    monkeypatch.setenv("VLLM_METAL_EXPERT_PARTITION", "56,72")
    assert expert_partition(2, 128) == [56, 72]
    for bad in ("64", "63,64", "0,128", "64,64,0", "a,b", ""):
        monkeypatch.setenv("VLLM_METAL_EXPERT_PARTITION", bad)
        with pytest.raises(ValueError, match="VLLM_METAL_EXPERT_PARTITION"):
            expert_partition(2, 128)


def test_partition_rejects_uneven_default(monkeypatch):
    monkeypatch.delenv("VLLM_METAL_EXPERT_PARTITION", raising=False)
    with pytest.raises(ValueError, match="divide evenly"):
        expert_partition(2, 127)
