# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the pipeline-parallel primitives.

Pure-Python: no model load, no GPU, no MLX distributed launcher. We drive
``PipelineGroup`` with a fake group object and ``apply_pipeline_split`` with a
tiny stub backbone whose ``.layers`` is a list of sentinels.
"""

from types import SimpleNamespace

import mlx.core as mx
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

    def test_more_stages_than_layers_fails_loud(self) -> None:
        # 2 layers / 3 stages: get_pp_indices gives rank 2 an empty [2, 2) span, so
        # the split fails loud rather than the wire descriptor later reading
        # layers[0] on an empty stage.
        with pytest.raises(NotImplementedError, match="leaves stage 2 with no layers"):
            apply_pipeline_split(_make_model(2), _pp(2, 3))


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


class _StubLayer(nn.Module):
    """An owned PP layer with the quantized projection registered BEFORE the norm,
    mirroring real Llama/Qwen blocks (attention precedes norms). The packed weight
    is uint32; the norm uses a DISTINCT float dtype so the wire-descriptor test
    proves the uint32 weight is skipped and the floating quant scales (seen first)
    are chosen — not the later norm."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.q_proj = nn.QuantizedLinear(hidden, hidden, group_size=32, bits=4)
        self.input_layernorm = nn.RMSNorm(hidden)
        self.input_layernorm.weight = self.input_layernorm.weight.astype(mx.float16)


class TestPipelinedModel:
    def test_singleton_runs_full_model_without_recv(self) -> None:
        # size-1 group: is_first and is_last are both true, so the wrapper runs
        # the full model (no recv) and passes input_embeddings=None — the model
        # embeds the tokens itself. The wire descriptor is never read for a
        # singleton, so the stub needs nothing beyond __call__.
        class _Stub:
            def __init__(self) -> None:
                self.input_embeddings: object = "unset"

            def __call__(self, input_ids, *, cache=None, input_embeddings=None):
                self.input_embeddings = input_embeddings
                return "logits"

        stub = _Stub()
        out = PipelinedModel(stub, _pp(0, 1))("ids", cache=None)
        assert out == "logits"
        assert stub.input_embeddings is None

    def test_wire_descriptor_skips_packed_quant_weight(self) -> None:
        # hidden from config; dtype from the first FLOATING param of the first owned
        # layer, NOT by probing embed_tokens (a streaming non-first stage does not
        # own it). The quantized projection is registered first, so its packed
        # uint32 weight must be SKIPPED and its floating scales chosen — proven by
        # the dtype being the scales' and NOT the (distinct) norm's.
        layer = _StubLayer(32)
        assert layer.q_proj.weight.dtype == mx.uint32  # packed, must be skipped
        assert layer.q_proj.scales.dtype != layer.input_layernorm.weight.dtype
        model = SimpleNamespace(
            args=SimpleNamespace(hidden_size=32),
            model=SimpleNamespace(layers=[layer]),  # no embed_tokens: not probed
        )

        wrapper = PipelinedModel(model, _pp(1, 2))  # a non-first stage

        assert wrapper._wire_hidden == 32
        assert wrapper._wire_dtype == layer.q_proj.scales.dtype
        assert wrapper._wire_dtype != layer.input_layernorm.weight.dtype
