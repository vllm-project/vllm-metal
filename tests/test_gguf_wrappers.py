# SPDX-License-Identifier: Apache-2.0
"""Tests for the GGUF-aware Linear/Embedding wrappers.

The native tensor suite owns quantized numerical parity. These wrapper tests
only cover the nn.Module surface and delegation that wrappers add.
"""

from __future__ import annotations

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

from vllm_metal.gguf.wrappers import GGUFEmbedding, GGUFLinear


class _RecordingTensor:
    def __init__(self, *, out_features: int = 4, in_features: int = 3) -> None:
        self.out_features = out_features
        self.in_features = in_features
        self.qweight = mx.zeros((out_features, 1), dtype=mx.uint32)
        self.scales = mx.zeros((out_features, 1), dtype=mx.float16)
        self.biases = mx.zeros((out_features, 1), dtype=mx.float16)
        self.matmul_inputs: list[mx.array] = []
        self.embedding_inputs: list[tuple[mx.array, mx.Dtype]] = []
        self.eval_calls = 0

    def matmul(self, x: mx.array) -> mx.array:
        self.matmul_inputs.append(x)
        return mx.ones((*x.shape[:-1], self.out_features), dtype=x.dtype)

    def embedding(self, ids: mx.array, output_dtype: mx.Dtype) -> mx.array:
        self.embedding_inputs.append((ids, output_dtype))
        return mx.ones((*ids.shape, self.in_features), dtype=output_dtype)

    def eval_arrays(self) -> None:
        self.eval_calls += 1


def test_quant_arrays_are_not_module_parameters():
    tensor = _RecordingTensor()
    bias = mx.zeros((tensor.out_features,), dtype=mx.float32)

    assert dict(tree_flatten(GGUFLinear(tensor).parameters())) == {}
    emb = GGUFEmbedding(tensor, output_dtype=mx.float16)
    assert dict(tree_flatten(emb.parameters())) == {}
    biased = GGUFLinear(tensor, bias=bias)
    assert [k for k, _ in tree_flatten(biased.parameters())] == ["bias"]
    assert dict(tree_flatten(biased.trainable_parameters())) == {}


def test_linear_delegates_matmul_and_adds_bias_at_output_dtype():
    tensor = _RecordingTensor()
    bias = mx.array([1.0, 2.0, 3.0, 4.0], dtype=mx.float32)
    layer = GGUFLinear(tensor, bias=bias)
    x = mx.zeros((2, tensor.in_features), dtype=mx.float16)

    out = layer(x)
    mx.eval(out)

    assert len(tensor.matmul_inputs) == 1
    assert tensor.matmul_inputs[0] is x
    assert out.dtype == mx.float16
    assert out.shape == (2, tensor.out_features)
    assert bool(mx.array_equal(out, mx.ones_like(out) + bias.astype(out.dtype)).item())


def test_linear_rejects_bad_bias():
    tensor = _RecordingTensor()
    for bad in (
        mx.zeros((1,), dtype=mx.float32),
        mx.zeros((), dtype=mx.float32),
        mx.zeros((tensor.out_features - 1,), dtype=mx.float32),
        mx.zeros((tensor.out_features,), dtype=mx.int32),
        object(),
    ):
        with pytest.raises(ValueError, match="bias must be a floating mx.array"):
            GGUFLinear(tensor, bias=bad)  # type: ignore[arg-type]


def test_linear_surfaces_tensor_dims():
    tensor = _RecordingTensor(out_features=64, in_features=128)
    layer = GGUFLinear(tensor)

    assert layer.out_features == 64
    assert "input_dims=128" in repr(layer)
    assert "output_dims=64" in repr(layer)
    assert "bias=False" in repr(layer)


def test_embedding_delegates_gather_and_as_linear():
    tensor = _RecordingTensor(out_features=10, in_features=6)
    layer = GGUFEmbedding(tensor, output_dtype=mx.bfloat16)
    ids = mx.array([[0, 1], [2, 3]], dtype=mx.int32)
    x = mx.zeros((3, tensor.in_features), dtype=mx.float32)

    gathered = layer(ids)
    projected = layer.as_linear(x)
    mx.eval(gathered, projected)

    assert len(tensor.embedding_inputs) == 1
    got_ids, got_dtype = tensor.embedding_inputs[0]
    assert got_ids is ids
    assert got_dtype == mx.bfloat16
    assert len(tensor.matmul_inputs) == 1
    assert tensor.matmul_inputs[0] is x
    assert gathered.shape == (2, 2, tensor.in_features)
    assert gathered.dtype == mx.bfloat16
    assert projected.shape == (3, tensor.out_features)
    assert projected.dtype == x.dtype


def test_embedding_rejects_non_floating_output_dtype():
    tensor = _RecordingTensor()
    for bad in (mx.int32, mx.uint8, mx.int8, object()):
        with pytest.raises(ValueError, match="output_dtype must be a floating"):
            GGUFEmbedding(tensor, output_dtype=bad)  # type: ignore[arg-type]


def test_embedding_extra_repr_shows_dims():
    tensor = _RecordingTensor(out_features=64, in_features=128)

    assert "64, 128" in repr(GGUFEmbedding(tensor, output_dtype=mx.float16))


def test_eval_arrays_delegates_to_tensor():
    tensor = _RecordingTensor()

    GGUFLinear(tensor).eval_arrays()
    GGUFEmbedding(tensor, output_dtype=mx.float16).eval_arrays()

    assert tensor.eval_calls == 2
