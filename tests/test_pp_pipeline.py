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
    PipelinedModel,
    PipelineGroup,
    _ring_hosts,
    apply_pipeline_split,
    is_non_last_stage,
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


class TestIsNonLastStage:
    def test_none_group_is_not_non_last(self) -> None:
        # Single-stage path: the runner holds no group, so the only stage is last.
        assert is_non_last_stage(None) is False

    def test_singleton_group_is_not_non_last(self) -> None:
        assert is_non_last_stage(_pp(0, 1)) is False

    def test_first_of_two_is_non_last(self) -> None:
        assert is_non_last_stage(_pp(0, 2)) is True

    def test_last_of_two_is_not_non_last(self) -> None:
        assert is_non_last_stage(_pp(1, 2)) is False


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
    def test_single_node_distinct_ports(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Two stages co-located on one Mac: same IP, distinct port per rank
        # (default base port 32323).
        monkeypatch.delenv("VLLM_METAL_RING_BASE_PORT", raising=False)
        hosts = _ring_hosts(["10.0.0.5", "10.0.0.5"])
        assert hosts == [["10.0.0.5:32323"], ["10.0.0.5:32324"]]

    def test_multi_node_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Each stage on its own Mac: real per-rank IPs, one entry per rank.
        # The base port is read per call, so the env override takes effect
        # without reimporting the module.
        monkeypatch.setenv("VLLM_METAL_RING_BASE_PORT", "40000")
        hosts = _ring_hosts(["10.0.0.5", "10.0.0.6", "10.0.0.7"])
        assert hosts == [
            ["10.0.0.5:40000"],
            ["10.0.0.6:40001"],
            ["10.0.0.7:40002"],
        ]

    def test_rejects_non_integer_base_port(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("VLLM_METAL_RING_BASE_PORT", "not-a-port")
        with pytest.raises(
            ValueError, match="VLLM_METAL_RING_BASE_PORT must be an integer"
        ):
            _ring_hosts(["10.0.0.5"])

    def test_rejects_privileged_base_port(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("VLLM_METAL_RING_BASE_PORT", "80")
        with pytest.raises(ValueError, match="user-port range"):
            _ring_hosts(["10.0.0.5"])

    def test_rejects_port_range_overflow(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VLLM_METAL_RING_BASE_PORT", "65535")
        with pytest.raises(ValueError, match="too high for the pipeline size"):
            _ring_hosts(["10.0.0.5", "10.0.0.6"])


class TestPipelinedModel:
    def test_singleton_runs_full_model_without_recv(self) -> None:
        # size-1 group: is_first and is_last are both true, so the wrapper runs
        # the full model (no recv) and passes input_embeddings=None — the model
        # embeds the tokens itself.
        class _Stub:
            def __init__(self) -> None:
                self.input_embeddings: object = "unset"
                self.model = SimpleNamespace(embed_tokens=nn.Embedding(4, 8))

            def __call__(self, input_ids, *, cache=None, input_embeddings=None):
                self.input_embeddings = input_embeddings
                return "logits"

        stub = _Stub()
        out = PipelinedModel(stub, _pp(0, 1))("ids", cache=None)
        assert out == "logits"
        assert stub.input_embeddings is None

    def test_wire_descriptor_from_embedding_output(self) -> None:
        # The wire descriptor must come from the embedding *output*: a quantized
        # checkpoint packs the weight (uint32, hidden folded by the bit width),
        # so weight.shape[-1] / weight.dtype would declare a wrong recv and
        # deadlock the ring.
        embed = nn.QuantizedEmbedding(64, 32, group_size=32, bits=4)
        assert embed.weight.shape[-1] != 32  # really packed
        model = SimpleNamespace(model=SimpleNamespace(embed_tokens=embed))

        wrapper = PipelinedModel(model, _pp(0, 2))

        assert wrapper._wire_hidden == 32
        assert wrapper._wire_dtype == embed.scales.dtype
