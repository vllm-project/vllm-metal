# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the pipeline-parallel primitives.

Pure-Python: no model load, no GPU, no MLX distributed launcher. We drive
``PipelineGroup`` with a fake group object and ``apply_pipeline_split`` with a
tiny stub backbone whose ``.layers`` is a list of sentinels.
"""

from types import SimpleNamespace

import mlx.nn as nn
import pytest

from vllm_metal.distributed.pipeline import (
    _MLX_RING_BASE_PORT,
    PipelineGroup,
    _ring_hosts,
    apply_pipeline_split,
)


class _FakeGroup:
    """Minimal stand-in for an mx.distributed.Group."""

    def __init__(self, rank: int, size: int) -> None:
        self._rank = rank
        self._size = size

    def rank(self) -> int:
        return self._rank

    def size(self) -> int:
        return self._size


def _pp(rank: int, size: int) -> PipelineGroup:
    return PipelineGroup(_FakeGroup(rank, size))


def _make_model(n_layers: int) -> SimpleNamespace:
    """Stub model with a sliceable .model.layers list and a .model.norm."""
    sentinel_norm = object()
    backbone = SimpleNamespace(
        layers=[f"layer{i}" for i in range(n_layers)],
        norm=sentinel_norm,
    )
    return SimpleNamespace(model=backbone)


class TestPipelineGroupRankFlags:
    def test_singleton_is_first_and_last(self) -> None:
        pp = _pp(0, 1)
        assert pp.rank == 0
        assert pp.size == 1
        assert pp.is_first is True
        assert pp.is_last is True

    def test_first_stage_flags(self) -> None:
        pp = _pp(0, 3)
        assert pp.is_first is True
        assert pp.is_last is False

    def test_middle_stage_flags(self) -> None:
        pp = _pp(1, 3)
        assert pp.is_first is False
        assert pp.is_last is False

    def test_last_stage_flags(self) -> None:
        pp = _pp(2, 3)
        assert pp.is_first is False
        assert pp.is_last is True


class TestApplyPipelineSplit:
    def test_singleton_is_a_noop_returns_none(self) -> None:
        model = _make_model(28)
        original_norm = model.model.norm
        result = apply_pipeline_split(model, _pp(0, 1))
        assert result is None
        assert len(model.model.layers) == 28
        assert model.model.norm is original_norm

    def test_first_stage_keeps_low_layers_and_drops_norm(self) -> None:
        model = _make_model(28)
        span = apply_pipeline_split(model, _pp(0, 2))
        assert span == (0, 14)
        assert model.model.layers == [f"layer{i}" for i in range(14)]
        assert isinstance(model.model.norm, nn.Identity)

    def test_last_stage_keeps_high_layers_and_real_norm(self) -> None:
        model = _make_model(28)
        original_norm = model.model.norm
        span = apply_pipeline_split(model, _pp(1, 2))
        assert span == (14, 28)
        assert model.model.layers == [f"layer{i}" for i in range(14, 28)]
        assert model.model.norm is original_norm
        assert not isinstance(model.model.norm, nn.Identity)

    def test_middle_stage_drops_norm(self) -> None:
        # 28 layers / 3 stages -> [9, 10, 9]: vLLM's get_pp_indices back-loads the
        # remainder onto the middle stage, so rank 1 owns layers [9, 19).
        model = _make_model(28)
        span = apply_pipeline_split(model, _pp(1, 3))
        assert span == (9, 19)
        assert model.model.layers == [f"layer{i}" for i in range(9, 19)]
        assert isinstance(model.model.norm, nn.Identity)

    def test_norm_kept_only_on_last_stage(self) -> None:
        size = 4
        for rank in range(size):
            model = _make_model(28)
            original_norm = model.model.norm
            apply_pipeline_split(model, _pp(rank, size))
            if rank == size - 1:
                assert model.model.norm is original_norm
            else:
                assert isinstance(model.model.norm, nn.Identity)

    def test_non_list_layers_fails_loud(self) -> None:
        backbone = SimpleNamespace(layers=("a", "b"), norm=object())
        model = SimpleNamespace(model=backbone)
        with pytest.raises(TypeError, match="sliceable list"):
            apply_pipeline_split(model, _pp(0, 2))


class TestRingHosts:
    def test_single_node_distinct_ports(self) -> None:
        # Two stages co-located on one Mac: same IP, distinct port per rank.
        hosts = _ring_hosts(["10.0.0.5", "10.0.0.5"])
        assert hosts == [
            [f"10.0.0.5:{_MLX_RING_BASE_PORT}"],
            [f"10.0.0.5:{_MLX_RING_BASE_PORT + 1}"],
        ]

    def test_multi_node_real_ips(self) -> None:
        # Each stage on its own Mac: real per-rank IPs, one entry per rank.
        ips = ["10.0.0.5", "10.0.0.6", "10.0.0.7"]
        hosts = _ring_hosts(ips)
        assert [h[0].rsplit(":", 1)[0] for h in hosts] == ips
        ports = [int(h[0].rsplit(":", 1)[1]) for h in hosts]
        assert ports == [_MLX_RING_BASE_PORT + i for i in range(3)]
        assert len(set(ports)) == 3  # distinct ports avoid bind conflicts
