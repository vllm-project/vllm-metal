# SPDX-License-Identifier: Apache-2.0
"""GPT-OSS stage parity with real layers and an in-memory activation handoff."""

from copy import deepcopy

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_lm.models import gpt_oss
from mlx_lm.models.cache import KVCache, RotatingKVCache

from vllm_metal.distributed.pipeline import (
    PipelinedModel,
    PipelineGroup,
    apply_pipeline_split,
    pipeline_send,
)


class _Group:
    def __init__(self, rank: int, size: int):
        self._rank = rank
        self._size = size

    def rank(self):
        return self._rank

    def size(self):
        return self._size


def _model(*, quantized: bool = False):
    model = gpt_oss.Model(
        gpt_oss.ModelArgs(
            num_hidden_layers=4,
            num_local_experts=2,
            num_experts_per_tok=1,
            vocab_size=64,
            hidden_size=64,
            intermediate_size=64,
            head_dim=8,
            num_attention_heads=8,
            num_key_value_heads=4,
            sliding_window=4,
            layer_types=[
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention",
            ],
        )
    )
    # Nonzero sinks make dropping GPT-OSS's sink term observable in parity.
    for layer in model.layers:
        layer.self_attn.sinks = mx.linspace(-1, 1, 8)
    if quantized:
        model.set_dtype(mx.float16)

        def quantization(path, module):
            if not hasattr(module, "to_quantized"):
                return False
            if ".experts." in path:
                return {"group_size": 32, "bits": 4, "mode": "mxfp4"}
            return {"group_size": 32, "bits": 8, "mode": "affine"}

        nn.quantize(model, class_predicate=quantization)
    mx.eval(model.parameters())
    return model


@pytest.fixture(autouse=True)
def _cpu():
    # No GPU, model downloads, or distributed processes are needed here.
    with mx.stream(mx.cpu):
        yield


@pytest.mark.parametrize("partition", ["2,2", "1,3", "3,1", "1,1,2", "1,1,1,1"])
@pytest.mark.parametrize("quantized", [False, True], ids=["float32", "mxfp4-q8"])
def test_stages_match_unsplit_prefill_and_cached_decode(
    monkeypatch, partition, quantized
):
    """Catch wrong stage-local attention types, cache offsets, head, or wire dtype."""
    monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", partition)
    reference = _model(quantized=quantized)
    reference_cache = reference.make_cache()
    size = len(partition.split(","))
    stages = []
    for rank in range(size):
        model = deepcopy(reference)
        pp = PipelineGroup(_Group(rank, size), backend="jaccl")
        span = apply_pipeline_split(model, pp)
        stages.append((PipelinedModel(model, pp), pp, model.make_cache(), span))

    mailbox = {}
    transfers = []

    def send(array, peer, *, group, stream):
        assert peer == group.rank() + 1
        assert peer not in mailbox
        assert stream == mx.cpu
        mailbox[peer] = array
        transfers.append((group.rank(), peer, array.dtype))
        return array

    def recv(shape, dtype, peer, *, group, stream):
        assert peer == group.rank() - 1
        assert stream == mx.cpu
        array = mailbox.pop(group.rank())
        assert array.shape == shape
        assert array.dtype == dtype
        return array

    monkeypatch.setattr(mx.distributed, "send", send)
    monkeypatch.setattr(mx.distributed, "recv", recv)

    # First prefill exceeds the sliding window, then a mixture of single-token
    # decode and another chunk crosses the rotating-cache boundary repeatedly.
    chunks = [[2, 4, 3, 5, 7, 11, 13, 17], [19], [23], [5, 7, 9], [29]]
    for chunk in chunks:
        ids = mx.array([chunk])
        expected = reference(ids, cache=reference_cache)
        for wrapper, pp, cache, _ in stages:
            actual = wrapper(ids, cache=cache)
            if not pp.is_last:
                mx.eval(pipeline_send(actual, pp))
        mx.eval(actual, expected)
        tolerance = 2e-2 if quantized else 2e-5
        assert mx.allclose(actual, expected, atol=tolerance, rtol=tolerance).item()
        assert mx.array_equal(mx.argmax(actual, -1), mx.argmax(expected, -1)).item()
        assert not mailbox

    assert len(transfers) == len(chunks) * (size - 1)
    expected_dtype = mx.float16 if quantized else mx.float32
    assert all(dtype == expected_dtype for _, _, dtype in transfers)
    for _, _, cache, (start, end) in stages:
        assert len(cache) == end - start
        for local_index, entry in enumerate(cache):
            expected_cache = reference_cache[start + local_index]
            assert type(entry) is type(expected_cache)
            assert entry.offset == expected_cache.offset == sum(map(len, chunks))
            assert mx.allclose(entry.keys, expected_cache.keys, atol=2e-2).item()
            assert mx.allclose(entry.values, expected_cache.values, atol=2e-2).item()


def test_split_builds_only_owned_cache_types(monkeypatch):
    monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", "1,3")
    model = _model()
    span = apply_pipeline_split(model, PipelineGroup(_Group(1, 2)))

    assert span == (1, 4)
    assert model.model.layer_types == [
        "full_attention",
        "sliding_attention",
        "full_attention",
    ]
    assert [type(entry) for entry in model.make_cache()] == [
        KVCache,
        RotatingKVCache,
        KVCache,
    ]
    # Global model config is used separately by the runner's metadata slicer.
    assert len(model.args.layer_types) == 4


def test_singleton_keeps_native_gpt_oss_call_and_cache(monkeypatch):
    monkeypatch.delenv("VLLM_PP_LAYER_PARTITION", raising=False)
    model = _model()
    original_layers = model.model.layers
    original_types = model.model.layer_types
    pp = PipelineGroup(_Group(0, 1))
    assert apply_pipeline_split(model, pp) is None

    def unexpected_transfer(*args, **kwargs):
        raise AssertionError("a singleton must not communicate")

    monkeypatch.setattr(mx.distributed, "send", unexpected_transfer)
    monkeypatch.setattr(mx.distributed, "recv", unexpected_transfer)
    ids = mx.array([[2, 3, 5, 7, 11, 13]])
    expected = model(ids, cache=model.make_cache())
    actual = PipelinedModel(model, pp)(ids, cache=model.make_cache())
    assert mx.allclose(actual, expected, atol=1e-6).item()
    assert model.model.layers is original_layers
    assert model.model.layer_types is original_types


@pytest.mark.parametrize("rank", [0, 1, 2, 3])
def test_one_layer_stage_dummy_forward_needs_no_transfer(monkeypatch, rank):
    monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", "1,1,1,1")
    model = _model()
    pp = PipelineGroup(_Group(rank, 4))
    apply_pipeline_split(model, pp)

    def unexpected_transfer(*args, **kwargs):
        raise AssertionError("dummy_forward must not communicate")

    monkeypatch.setattr(mx.distributed, "send", unexpected_transfer)
    monkeypatch.setattr(mx.distributed, "recv", unexpected_transfer)
    result = PipelinedModel(model, pp).dummy_forward(mx.array([[2, 3, 5, 7, 11]]))
    assert result.shape == (1, 5, 64)
    assert mx.all(mx.isfinite(result)).item()
