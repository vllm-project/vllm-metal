# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the pipeline-parallel primitives.

Pure-Python: no checkpoint download, no GPU, no MLX distributed launcher. We
drive ``PipelineGroup`` with a fake group object and ``apply_pipeline_split``
with a tiny stub backbone whose ``.layers`` is a list of sentinels; the
mlx_lm contract tests build a micro ``qwen3.Model`` from args in-process.
"""

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_lm.models import qwen3

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


class _StageStubModel:
    """Records which forward the wrapper takes: the top-level (head) call vs the
    backbone-only call. ``embed_tokens`` raises, pinning that a dummy forward on
    a non-first stage never touches the embedding."""

    _HIDDEN = 32

    def __init__(self) -> None:
        layer = _StubLayer(self._HIDDEN)
        self.args = SimpleNamespace(hidden_size=self._HIDDEN)
        self.calls: dict[str, object] = {}
        self.model = self._backbone(layer)

    def _backbone(self, layer: _StubLayer) -> object:
        calls = self.calls
        hidden = self._HIDDEN
        wire_dtype = layer.q_proj.scales.dtype

        class _Backbone:
            def __init__(self) -> None:
                self.layers = [layer]
                self.embed_tokens = _RaisingEmbed()

            def __call__(self, input_ids, *, cache=None, input_embeddings=None):
                calls["backbone"] = input_embeddings
                if input_embeddings is None:
                    # mirror mlx_lm: the backbone embeds ONLY when no hidden is
                    # provided — this makes the raising embed a LIVE pin.
                    self.embed_tokens(input_ids)
                return mx.zeros((1, input_ids.shape[1], hidden), dtype=wire_dtype)

        return _Backbone()

    def __call__(self, input_ids, *, cache=None, input_embeddings=None):
        self.calls["full"] = input_embeddings
        return "model-output"


class _RaisingEmbed:
    def __call__(self, input_ids: object) -> mx.array:
        raise AssertionError("embed_tokens must not be called on this stage")


class TestPipelinedModelDummyForward:
    def test_non_first_stage_feeds_zeros_at_wire_descriptor(self) -> None:
        # A middle stage profiles from a locally built zeros hidden (no ring
        # recv, no embedding) shaped by the wire descriptor.
        stub = _StageStubModel()
        wrapper = PipelinedModel(stub, _pp(1, 3))

        out = wrapper.dummy_forward(mx.zeros((1, 4), dtype=mx.int32))

        h_in = stub.calls["backbone"]
        assert isinstance(h_in, mx.array)
        assert h_in.shape == (1, 4, 32)
        assert h_in.dtype == wrapper._wire_dtype
        assert mx.all(h_in == 0)  # zeros, per upstream's empty intermediates
        assert "full" not in stub.calls  # head path never taken
        assert out.dtype == wrapper._wire_dtype

    def test_first_stage_embeds_internally(self) -> None:
        # The first stage owns the embedding: the backbone is called with
        # input_embeddings=None and embeds the token ids itself.
        stub = _StageStubModel()
        stub.model.embed_tokens = nn.Embedding(4, 32)  # first stage owns it
        wrapper = PipelinedModel(stub, _pp(0, 2))

        wrapper.dummy_forward(mx.zeros((1, 4), dtype=mx.int32))

        assert stub.calls["backbone"] is None
        assert "full" not in stub.calls

    def test_last_stage_runs_owned_head(self) -> None:
        # The last stage owns norm + head: the dummy takes the same top-level
        # call as serving and returns the model output for logits extraction.
        stub = _StageStubModel()
        wrapper = PipelinedModel(stub, _pp(1, 2))

        out = wrapper.dummy_forward(mx.zeros((1, 4), dtype=mx.int32))

        assert out == "model-output"
        h_in = stub.calls["full"]
        assert isinstance(h_in, mx.array)
        assert h_in.shape == (1, 4, 32)
        assert h_in.dtype == wrapper._wire_dtype
        assert "backbone" not in stub.calls

    def test_inherits_wire_dtype_fail_fast(self) -> None:
        # The dummy shares the stage body with __call__, so a wire-dtype
        # mismatch fails at profiling/startup instead of on the first request.
        stub = _StageStubModel()
        backbone = stub.model

        class _WrongDtypeBackbone:
            layers = backbone.layers
            embed_tokens = backbone.embed_tokens

            def __call__(self, input_ids, *, cache=None, input_embeddings=None):
                return mx.zeros((1, input_ids.shape[1], 32), dtype=mx.bfloat16)

        wrapper = PipelinedModel(stub, _pp(1, 3))
        stub.model = _WrongDtypeBackbone()

        with pytest.raises(TypeError) as excinfo:
            wrapper.dummy_forward(mx.zeros((1, 4), dtype=mx.int32))
        assert str(excinfo.value) == (
            "PP stage produced hidden mlx.core.bfloat16, but the wire dtype "
            "(stage compute dtype) is mlx.core.float32; mismatch deadlocks "
            "the ring."
        )


def _micro_qwen3(tie_word_embeddings: bool, n_layers: int = 2) -> qwen3.Model:
    args = qwen3.ModelArgs(
        model_type="qwen3",
        hidden_size=32,
        num_hidden_layers=n_layers,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        rms_norm_eps=1e-5,
        vocab_size=64,
        max_position_embeddings=128,
        rope_theta=1000.0,
        head_dim=8,
        tie_word_embeddings=tie_word_embeddings,
    )
    return qwen3.Model(args)


class TestDummyForwardMlxLmContract:
    """Pin the upstream keyword contract the dummy relies on — a real (micro)
    mlx_lm model built from args, both tie states, no checkpoint download."""

    @pytest.mark.parametrize("tied", [True, False])
    def test_first_stage_outputs_hidden_at_wire_dtype(self, tied: bool) -> None:
        model = _micro_qwen3(tied)
        pp = _pp(0, 2)
        apply_pipeline_split(model, pp)
        wrapper = PipelinedModel(model, pp)

        out = wrapper.dummy_forward(mx.zeros((1, 4), dtype=mx.int32))

        assert out.shape == (1, 4, 32)
        assert out.dtype == wrapper._wire_dtype

    @pytest.mark.parametrize("tied", [True, False])
    def test_last_stage_outputs_vocab_logits(self, tied: bool) -> None:
        model = _micro_qwen3(tied)
        pp = _pp(1, 2)
        apply_pipeline_split(model, pp)
        wrapper = PipelinedModel(model, pp)

        out = wrapper.dummy_forward(mx.zeros((1, 4), dtype=mx.int32))

        assert out.shape == (1, 4, 64)

    def test_middle_stage_never_touches_the_embedding(self) -> None:
        # The memory win rests on the upstream keyword contract: a backbone
        # given input_embeddings never calls embed_tokens. Pin it on a real
        # (micro) qwen3 middle stage by swapping the embedding for a raiser.
        model = _micro_qwen3(tie_word_embeddings=False, n_layers=3)
        pp = _pp(1, 3)  # middle: owns neither the embedding nor the head
        apply_pipeline_split(model, pp)
        model.model.embed_tokens = _RaisingEmbed()
        wrapper = PipelinedModel(model, pp)

        out = wrapper.dummy_forward(mx.zeros((1, 4), dtype=mx.int32))

        assert out.shape == (1, 4, 32)
        assert out.dtype == wrapper._wire_dtype
