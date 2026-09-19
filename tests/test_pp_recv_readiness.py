# SPDX-License-Identifier: Apache-2.0
"""A JACCL receive must finish before the downstream stage builds GPU work."""

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

from vllm_metal.distributed.pipeline import (
    PipelinedModel,
    PipelineGroup,
    pipeline_send,
)


def _group(rank: int, backend: str) -> PipelineGroup:
    return PipelineGroup(
        SimpleNamespace(rank=lambda: rank, size=lambda: 2), backend=backend
    )


@pytest.fixture(autouse=True)
def _cpu():
    # Transport readiness is a host scheduling contract; no GPU or peers needed.
    with mx.stream(mx.cpu):
        yield


@pytest.mark.parametrize("backend", ["jaccl", "ring"])
def test_stage_waits_for_jaccl_receive_but_keeps_ring_lazy(monkeypatch, backend):
    """Removing the JACCL host wait must not let a stage use pending input."""
    pp = _group(1, backend)
    wire = mx.arange(32, dtype=mx.float32).reshape(4, 8)
    state = {"ready": False, "forward_started": False}
    receives = []
    evaluated = []
    real_eval = mx.eval

    def recv(shape, dtype, peer, *, group, stream):
        assert (shape, dtype, peer) == ((4, 8), mx.float32, 0)
        assert group is pp.group
        assert stream == (mx.cpu if backend == "jaccl" else None)
        receives.append(wire)
        return wire

    def evaluate(*arrays):
        assert backend == "jaccl", "ring receive must remain lazy"
        assert len(arrays) == 1 and arrays[0] is wire
        assert not state["forward_started"]
        real_eval(*arrays)
        evaluated.extend(arrays)
        state["ready"] = True

    class StageModel:
        args = SimpleNamespace(hidden_size=8)

        def __init__(self):
            self.model = SimpleNamespace(layers=[nn.Linear(8, 8, bias=False)])

        def __call__(self, input_ids, *, cache, input_embeddings):
            assert backend != "jaccl" or state["ready"], (
                "downstream stage constructed work before its RDMA input was ready"
            )
            state["forward_started"] = True
            # A real lazy array operation stands in for the stage's GPU graph.
            return input_embeddings + 1

    monkeypatch.setattr(mx.distributed, "recv", recv)
    monkeypatch.setattr(mx, "eval", evaluate)
    result = PipelinedModel(StageModel(), pp)(mx.array([[1, 2, 3, 4]]))

    real_eval(result)
    assert result.shape == (1, 4, 8)
    assert result.tolist() == [
        [list(range(start + 1, start + 9)) for start in (0, 8, 16, 24)]
    ]
    assert len(receives) == 1 and receives[0] is wire
    assert len(evaluated) == (1 if backend == "jaccl" else 0)


def test_jaccl_send_stays_lazy_and_transfers_once(monkeypatch):
    """Receive readiness must not force a send or duplicate its activation."""
    pp = _group(0, "jaccl")
    sent = []

    def send(array, peer, *, group, stream):
        assert peer == 1 and group is pp.group and stream == mx.cpu
        sent.append(array)
        return array

    def unexpected_eval(*arrays):
        raise AssertionError("the runner must retain ownership of send submission")

    monkeypatch.setattr(mx.distributed, "send", send)
    monkeypatch.setattr(mx, "eval", unexpected_eval)
    result = pipeline_send(mx.ones((1, 4, 8), dtype=mx.float32), pp)
    assert len(sent) == 1 and result is sent[0]
    assert result.shape == (4, 8)
