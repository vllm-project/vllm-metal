# SPDX-License-Identifier: Apache-2.0
"""Tests for the GGUF-aware Linear/Embedding wrappers.

Wrappers are exercised around the REAL tensor (built from a gguf/mx.load fixture
like the PR-1 tests), with parity checked against ``gguf.quants.dequantize`` — no
SimpleNamespace/MagicMock tensor fakes. The wrapper modules are nn.Modules, so
this suite needs MLX (it won't collect on a machine without a Metal device).
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

gguf = pytest.importorskip("gguf")

from vllm_metal.gguf.mlx_native import GGUFMLXQuantizedTensor  # noqa: E402
from vllm_metal.gguf.wrappers import GGUFEmbedding, GGUFLinear  # noqa: E402

GGMLQuantizationType = gguf.GGMLQuantizationType

NATIVE_QTYPES = [GGMLQuantizationType.Q8_0, GGMLQuantizationType.Q4_0]


def _write_quantized_gguf(path, weight: np.ndarray, qtype) -> dict[str, mx.array]:
    raw = gguf.quants.quantize(weight, qtype)
    writer = gguf.GGUFWriter(str(path), "llama")
    writer.add_tensor("w.weight", raw, raw_shape=raw.shape, raw_dtype=qtype)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return mx.load(str(path))


def _make_tensor(tmp_path, qtype, shape=(64, 128)):
    weight = np.random.default_rng(0).standard_normal(shape).astype(np.float32)
    arrays = _write_quantized_gguf(tmp_path / f"{qtype.name}.gguf", weight, qtype)
    qt = GGUFMLXQuantizedTensor.from_mx_load(arrays, "w.weight", qtype)
    oracle = gguf.quants.dequantize(gguf.quants.quantize(weight, qtype), qtype).astype(
        np.float32
    )
    return qt, oracle


def test_quant_arrays_are_not_module_parameters(tmp_path):
    qt, _ = _make_tensor(tmp_path, GGMLQuantizationType.Q8_0)
    bias = mx.zeros((qt.out_features,), dtype=mx.float32)

    # The packed quant arrays are owned by the tensor, not the Module, so a dtype
    # cast / parameter sweep can't touch them.
    assert dict(tree_flatten(GGUFLinear(qt).parameters())) == {}
    emb = GGUFEmbedding(qt, output_dtype=mx.float16)
    assert dict(tree_flatten(emb.parameters())) == {}
    # Only the dense bias is a parameter, and freeze() leaves it non-trainable.
    biased = GGUFLinear(qt, bias=bias)
    assert [k for k, _ in tree_flatten(biased.parameters())] == ["bias"]
    assert dict(tree_flatten(biased.trainable_parameters())) == {}


@pytest.mark.parametrize("qtype", NATIVE_QTYPES)
def test_linear_matches_dense_oracle(tmp_path, qtype):
    qt, oracle = _make_tensor(tmp_path, qtype)
    layer = GGUFLinear(qt)
    x = mx.random.normal((3, qt.in_features)).astype(mx.float32)

    out = layer(x)
    expected = x @ mx.array(oracle).T
    mx.eval(out, expected)

    ref_max = float(mx.max(mx.abs(expected)))
    np.testing.assert_allclose(
        np.array(out), np.array(expected), rtol=0, atol=ref_max * 4e-6
    )


@pytest.mark.parametrize("qtype", NATIVE_QTYPES)
def test_linear_bias_is_a_real_post_matmul_add(tmp_path, qtype):
    qt, _ = _make_tensor(tmp_path, qtype)
    bias = mx.random.normal((qt.out_features,)).astype(mx.float32)
    x = mx.random.normal((3, qt.in_features)).astype(mx.float32)

    biased = GGUFLinear(qt, bias=bias)
    plain = GGUFLinear(qt)
    no_bias = plain(x)
    with_bias = biased(x)
    mx.eval(no_bias, with_bias)

    assert "bias" in biased
    assert "bias" not in plain
    np.testing.assert_allclose(
        np.array(with_bias), np.array(no_bias + bias), rtol=0, atol=1e-6
    )


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
@pytest.mark.parametrize("with_bias", [False, True])
def test_linear_output_dtype_follows_x(tmp_path, dtype, with_bias):
    qt, _ = _make_tensor(tmp_path, GGMLQuantizationType.Q8_0)
    # GGUF bias is float32; the wrapper must cast it so a bf16/f16 x stays bf16/f16.
    bias = mx.zeros((qt.out_features,), dtype=mx.float32) if with_bias else None
    layer = GGUFLinear(qt, bias=bias)
    x = mx.random.normal((4, qt.in_features)).astype(dtype)

    out = layer(x)

    assert out.dtype == dtype
    assert out.shape == (4, qt.out_features)


def test_linear_rejects_bad_bias(tmp_path):
    qt, _ = _make_tensor(tmp_path, GGMLQuantizationType.Q8_0)
    for bad in (
        mx.zeros((1,), dtype=mx.float32),  # would broadcast
        mx.zeros((), dtype=mx.float32),  # scalar
        mx.zeros((qt.out_features - 1,), dtype=mx.float32),  # wrong length
        mx.zeros((qt.out_features,), dtype=mx.int32),  # non-floating
    ):
        with pytest.raises(ValueError, match="bias must be a floating mx.array"):
            GGUFLinear(qt, bias=bad)


def test_linear_extra_repr_shows_dims(tmp_path):
    qt, _ = _make_tensor(tmp_path, GGMLQuantizationType.Q8_0, shape=(64, 128))
    repr_str = repr(GGUFLinear(qt))
    assert "input_dims=128" in repr_str
    assert "output_dims=64" in repr_str
    assert "bias=False" in repr_str


def test_embedding_extra_repr_shows_dims(tmp_path):
    qt, _ = _make_tensor(tmp_path, GGMLQuantizationType.Q8_0, shape=(64, 128))
    # "out_features, in_features" in order, mirroring QuantizedEmbedding's repr.
    assert "64, 128" in repr(GGUFEmbedding(qt, output_dtype=mx.float16))


@pytest.mark.parametrize("qtype", NATIVE_QTYPES)
def test_embedding_matches_dense_oracle(tmp_path, qtype):
    qt, oracle = _make_tensor(tmp_path, qtype)
    layer = GGUFEmbedding(qt, output_dtype=mx.float32)
    ids = mx.array([[0, 5], [63, 0]], dtype=mx.int32)

    out = layer(ids)
    expected = mx.array(oracle)[ids]
    mx.eval(out, expected)

    assert out.shape == (2, 2, qt.in_features)
    ref_max = float(mx.max(mx.abs(expected)))
    np.testing.assert_allclose(
        np.array(out), np.array(expected), rtol=0, atol=ref_max * 5e-3
    )


@pytest.mark.parametrize("qtype", NATIVE_QTYPES)
def test_embedding_as_linear_matches_oracle(tmp_path, qtype):
    # Tied lm_head: the embedding table is reused as the output projection.
    qt, oracle = _make_tensor(tmp_path, qtype, shape=(100, 64))
    layer = GGUFEmbedding(qt, output_dtype=mx.float32)
    x = mx.random.normal((3, qt.in_features)).astype(mx.float32)

    out = layer.as_linear(x)
    expected = x @ mx.array(oracle).T
    mx.eval(out, expected)

    ref_max = float(mx.max(mx.abs(expected)))
    np.testing.assert_allclose(
        np.array(out), np.array(expected), rtol=0, atol=ref_max * 4e-6
    )


def test_embedding_output_dtype(tmp_path):
    qt, _ = _make_tensor(tmp_path, GGMLQuantizationType.Q8_0)
    out = GGUFEmbedding(qt, output_dtype=mx.bfloat16)(mx.array([0, 1], dtype=mx.int32))
    assert out.dtype == mx.bfloat16


def test_embedding_rejects_non_floating_output_dtype(tmp_path):
    qt, _ = _make_tensor(tmp_path, GGMLQuantizationType.Q8_0)
    # A non-floating output_dtype would silently truncate dequantized rows.
    for bad in (mx.int32, mx.uint8, mx.int8):
        with pytest.raises(ValueError, match="output_dtype must be a floating"):
            GGUFEmbedding(qt, output_dtype=bad)


def test_wrapper_never_sees_invalid_tensor(tmp_path):
    # Q4_1 is rejected when the tensor is built — before any wrapper can exist.
    weight = np.random.default_rng(0).standard_normal((32, 64)).astype(np.float32)
    arrays = _write_quantized_gguf(
        tmp_path / "q4_1.gguf", weight, GGMLQuantizationType.Q4_1
    )
    with pytest.raises(ValueError, match="ml-explore/mlx#3664"):
        GGUFMLXQuantizedTensor.from_mx_load(
            arrays, "w.weight", GGMLQuantizationType.Q4_1
        )
