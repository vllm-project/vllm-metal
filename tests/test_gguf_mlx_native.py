# SPDX-License-Identifier: Apache-2.0
"""Tests for the MLX-native GGUF representation and primitives.

Fixtures are written with the upstream ``gguf`` package and read back through
``mx.load`` so the tests exercise MLX's real GGUF repack, and parity is checked
against ``gguf.quants.dequantize`` (the upstream reference), never a local
re-implementation of the packing.
"""

from __future__ import annotations

import os

import mlx.core as mx
import numpy as np
import pytest

gguf = pytest.importorskip("gguf")

from vllm_metal.gguf.mlx_native import (  # noqa: E402
    MLX_NATIVE_GGUF_TYPES,
    GGUFMLXQuantizedTensor,
)

GGMLQuantizationType = gguf.GGMLQuantizationType

NATIVE_QTYPES = [GGMLQuantizationType.Q8_0, GGMLQuantizationType.Q4_0]


def _write_quantized_gguf(path, weight: np.ndarray, qtype) -> dict[str, mx.array]:
    """Quantize ``weight`` to ``qtype``, write a 1-tensor GGUF, return mx.load()."""
    raw = gguf.quants.quantize(weight, qtype)
    writer = gguf.GGUFWriter(str(path), "llama")
    writer.add_tensor("w.weight", raw, raw_shape=raw.shape, raw_dtype=qtype)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return mx.load(str(path))


def _make_tensor(tmp_path, qtype, shape=(64, 128)) -> tuple:
    weight = np.random.default_rng(0).standard_normal(shape).astype(np.float32)
    arrays = _write_quantized_gguf(tmp_path / f"{qtype.name}.gguf", weight, qtype)
    qt = GGUFMLXQuantizedTensor.from_mx_load(arrays, "w.weight", qtype)
    oracle = gguf.quants.dequantize(gguf.quants.quantize(weight, qtype), qtype).astype(
        np.float32
    )
    return qt, oracle


@pytest.mark.parametrize("qtype", NATIVE_QTYPES)
def test_contract_matches_logical_shape(tmp_path, qtype):
    qt, _ = _make_tensor(tmp_path, qtype, shape=(64, 128))

    assert qt.qweight_type == qtype
    assert qt.bits == (8 if qtype == GGMLQuantizationType.Q8_0 else 4)
    assert qt.group_size == 32
    assert qt.logical_shape == (64, 128)
    assert qt.out_features == 64
    assert qt.in_features == 128
    assert qt.packed_shape == tuple(qt.qweight.shape)


@pytest.mark.parametrize("qtype", NATIVE_QTYPES)
def test_matmul_matches_dense_oracle_f32(tmp_path, qtype):
    qt, oracle = _make_tensor(tmp_path, qtype)
    x = mx.random.normal((3, qt.in_features)).astype(mx.float32)

    out = qt.matmul(x)
    expected = x @ mx.array(oracle).T
    mx.eval(out, expected)

    # f32 path has no f16 accumulation noise, so hold it tight: the only error
    # is quantization rounding shared with the oracle. Measured headroom is
    # ~4e-7 relative, so 4e-6 catches a quant-math regression with ~10x margin.
    ref_max = float(mx.max(mx.abs(expected)))
    np.testing.assert_allclose(
        np.array(out), np.array(expected), rtol=0, atol=ref_max * 4e-6
    )


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
@pytest.mark.parametrize("qtype", NATIVE_QTYPES)
def test_matmul_output_dtype_follows_x(tmp_path, qtype, dtype):
    qt, _ = _make_tensor(tmp_path, qtype)
    x = mx.random.normal((4, qt.in_features)).astype(dtype)

    out = qt.matmul(x)

    assert out.dtype == dtype
    assert out.shape == (4, qt.out_features)


@pytest.mark.parametrize("qtype", NATIVE_QTYPES)
def test_matmul_preserves_leading_shape(tmp_path, qtype):
    qt, _ = _make_tensor(tmp_path, qtype)
    x = mx.random.normal((2, 3, qt.in_features)).astype(mx.float16)

    out = qt.matmul(x)
    flat = qt.matmul(x.reshape(-1, qt.in_features))
    mx.eval(out, flat)

    assert out.shape == (2, 3, qt.out_features)
    assert mx.array_equal(out.reshape(-1, qt.out_features), flat)


@pytest.mark.parametrize("qtype", NATIVE_QTYPES)
def test_matmul_empty_batch(tmp_path, qtype):
    qt, _ = _make_tensor(tmp_path, qtype)

    out = qt.matmul(mx.zeros((0, qt.in_features), dtype=mx.float16))

    assert out.shape == (0, qt.out_features)
    assert out.dtype == mx.float16


@pytest.mark.parametrize("qtype", NATIVE_QTYPES)
def test_embedding_matches_oracle_rows(tmp_path, qtype):
    qt, oracle = _make_tensor(tmp_path, qtype)
    ids = mx.array([[0, 5], [63, 0]], dtype=mx.int32)

    out = qt.embedding(ids, output_dtype=mx.float32)
    expected = mx.array(oracle)[ids]
    mx.eval(out, expected)

    assert out.shape == (2, 2, qt.in_features)
    # dequant runs in float16 (the stored scale dtype); the oracle is float32,
    # so allow float16 rounding.
    ref_max = float(mx.max(mx.abs(expected)))
    np.testing.assert_allclose(
        np.array(out), np.array(expected), rtol=0, atol=ref_max * 5e-3
    )


@pytest.mark.parametrize("qtype", NATIVE_QTYPES)
def test_embedding_as_linear_matches_matmul(tmp_path, qtype):
    # A tied lm_head reuses the embedding table as a linear weight; the same
    # quantized tensor must work through matmul (the PR-2 tied-head path).
    qt, oracle = _make_tensor(tmp_path, qtype, shape=(100, 64))
    x = mx.random.normal((3, qt.in_features)).astype(mx.float32)

    out = qt.matmul(x)
    expected = x @ mx.array(oracle).T
    mx.eval(out, expected)

    ref_max = float(mx.max(mx.abs(expected)))
    np.testing.assert_allclose(
        np.array(out), np.array(expected), rtol=0, atol=ref_max * 4e-6
    )


def test_accepts_int_qweight_type(tmp_path):
    # A loader may pass the raw GGML type id; it must normalize to the enum.
    qt, _ = _make_tensor(tmp_path, GGMLQuantizationType.Q8_0)
    rebuilt = GGUFMLXQuantizedTensor(
        qweight=qt.qweight,
        scales=qt.scales,
        biases=qt.biases,
        qweight_type=int(GGMLQuantizationType.Q8_0),
    )
    assert rebuilt.qweight_type is GGMLQuantizationType.Q8_0


def test_rejects_q4_1_with_mlx_bug_reference(tmp_path):
    weight = np.random.default_rng(0).standard_normal((32, 64)).astype(np.float32)
    arrays = _write_quantized_gguf(
        tmp_path / "q4_1.gguf", weight, GGMLQuantizationType.Q4_1
    )
    with pytest.raises(ValueError, match="ml-explore/mlx#3664"):
        GGUFMLXQuantizedTensor.from_mx_load(
            arrays, "w.weight", GGMLQuantizationType.Q4_1
        )
    assert GGMLQuantizationType.Q4_1 not in MLX_NATIVE_GGUF_TYPES


def test_rejects_unsupported_qtype():
    with pytest.raises(ValueError, match="Unsupported GGUF quantization type"):
        GGUFMLXQuantizedTensor(
            qweight=mx.zeros((4, 8), dtype=mx.uint32),
            scales=mx.zeros((4, 1), dtype=mx.float16),
            biases=mx.zeros((4, 1), dtype=mx.float16),
            qweight_type=GGMLQuantizationType.Q4_K,
        )


def test_rejects_wrong_qweight_dtype():
    with pytest.raises(ValueError, match="qweight must be uint32"):
        GGUFMLXQuantizedTensor(
            qweight=mx.zeros((4, 8), dtype=mx.int32),
            scales=mx.zeros((4, 1), dtype=mx.float16),
            biases=mx.zeros((4, 1), dtype=mx.float16),
            qweight_type=GGMLQuantizationType.Q8_0,
        )


def test_rejects_inconsistent_packed_dim():
    # Q8_0 with 1 group needs packed inner dim 8 (1 * 8); give it 4.
    with pytest.raises(ValueError, match="inconsistent"):
        GGUFMLXQuantizedTensor(
            qweight=mx.zeros((4, 4), dtype=mx.uint32),
            scales=mx.zeros((4, 1), dtype=mx.float16),
            biases=mx.zeros((4, 1), dtype=mx.float16),
            qweight_type=GGMLQuantizationType.Q8_0,
        )


def test_from_mx_load_rejects_missing_companions(tmp_path):
    weight = np.random.default_rng(0).standard_normal((32, 64)).astype(np.float32)
    arrays = _write_quantized_gguf(
        tmp_path / "q8.gguf", weight, GGMLQuantizationType.Q8_0
    )
    del arrays["w.scales"]
    with pytest.raises(ValueError, match="missing MLX repack arrays"):
        GGUFMLXQuantizedTensor.from_mx_load(
            arrays, "w.weight", GGMLQuantizationType.Q8_0
        )


def test_rejects_non_float16_scales():
    with pytest.raises(ValueError, match="scales must be float16"):
        GGUFMLXQuantizedTensor(
            qweight=mx.zeros((4, 8), dtype=mx.uint32),
            scales=mx.zeros((4, 1), dtype=mx.float32),
            biases=mx.zeros((4, 1), dtype=mx.float16),
            qweight_type=GGMLQuantizationType.Q8_0,
        )


def test_rejects_scales_biases_shape_mismatch():
    with pytest.raises(ValueError, match="must have the same shape"):
        GGUFMLXQuantizedTensor(
            qweight=mx.zeros((4, 8), dtype=mx.uint32),
            scales=mx.zeros((4, 1), dtype=mx.float16),
            biases=mx.zeros((4, 2), dtype=mx.float16),
            qweight_type=GGMLQuantizationType.Q8_0,
        )


def test_rejects_scales_rows_mismatch():
    with pytest.raises(ValueError, match="scales rows .* must match qweight rows"):
        GGUFMLXQuantizedTensor(
            qweight=mx.zeros((4, 8), dtype=mx.uint32),
            scales=mx.zeros((2, 1), dtype=mx.float16),
            biases=mx.zeros((2, 1), dtype=mx.float16),
            qweight_type=GGMLQuantizationType.Q8_0,
        )


def test_from_mx_load_rejects_non_weight_name(tmp_path):
    weight = np.random.default_rng(0).standard_normal((32, 64)).astype(np.float32)
    arrays = _write_quantized_gguf(
        tmp_path / "q8.gguf", weight, GGMLQuantizationType.Q8_0
    )
    with pytest.raises(ValueError, match="must end with '.weight'"):
        GGUFMLXQuantizedTensor.from_mx_load(arrays, "w", GGMLQuantizationType.Q8_0)


@pytest.mark.parametrize("qtype", NATIVE_QTYPES)
def test_mx_load_layout_is_affine_group32(tmp_path, qtype):
    # Canary: assert mx.load's raw repack output directly (not via the
    # constructor) so a future MLX layout/dtype drift surfaces here.
    weight = np.random.default_rng(0).standard_normal((64, 128)).astype(np.float32)
    arrays = _write_quantized_gguf(tmp_path / f"{qtype.name}.gguf", weight, qtype)
    bits = 8 if qtype == GGMLQuantizationType.Q8_0 else 4

    assert arrays["w.weight"].dtype == mx.uint32
    assert arrays["w.scales"].dtype == mx.float16
    assert arrays["w.biases"].dtype == mx.float16
    assert arrays["w.scales"].shape == arrays["w.biases"].shape
    num_groups = arrays["w.scales"].shape[1]
    assert num_groups == 128 // 32
    assert arrays["w.weight"].shape[1] == num_groups * bits


@pytest.mark.parametrize("qtype", NATIVE_QTYPES)
def test_embedding_empty_ids(tmp_path, qtype):
    qt, _ = _make_tensor(tmp_path, qtype)

    out = qt.embedding(mx.zeros((0,), dtype=mx.int32), output_dtype=mx.float16)
    assert out.shape == (0, qt.in_features)
    assert out.dtype == mx.float16


# --- Real-file parity (opt-in: needs local GGUF files) ------------------------
#
# Set VLLM_METAL_TEST_GGUF_PATHS to a comma-separated list of real .gguf files
# (e.g. a Q8_0 and a Q4_0 model) to run these. They prove the representation and
# primitives hold on real checkpoints — every MLX-native tensor constructs, with
# matmul/embedding parity vs the gguf-py dequantize oracle and a quantized-vs-
# dense memory comparison. Each file is checked for whichever native qtypes it
# contains, so pointing at a Q4_0 file gives real Q4_0 coverage.

_REAL_GGUF_PATHS = [
    p.strip()
    for p in os.environ.get("VLLM_METAL_TEST_GGUF_PATHS", "").split(",")
    if p.strip()
]
_real_gguf = pytest.mark.skipif(
    not _REAL_GGUF_PATHS,
    reason="set VLLM_METAL_TEST_GGUF_PATHS to comma-separated .gguf paths",
)
_real_param = pytest.mark.parametrize("path", _REAL_GGUF_PATHS)


def _native_tensors(path):
    """Return (mx.load arrays, {name: GGUFReader tensor}) for MLX-native qtypes."""
    reader = gguf.GGUFReader(path)
    arrays = mx.load(path)
    native = {
        t.name: t for t in reader.tensors if t.tensor_type in MLX_NATIVE_GGUF_TYPES
    }
    return arrays, native


@pytest.mark.slow
@_real_gguf
@_real_param
def test_real_file_all_native_tensors_construct(path):
    arrays, native = _native_tensors(path)
    assert native, "expected MLX-native quantized tensors in the test GGUF"
    for name, tensor in native.items():
        qt = GGUFMLXQuantizedTensor.from_mx_load(arrays, name, tensor.tensor_type)
        # GGUF stores dims in reverse (ne[0] = in_features); logical is (out, in).
        in_features, out_features = (int(d) for d in tensor.shape)
        assert qt.logical_shape == (out_features, in_features)


@pytest.mark.slow
@_real_gguf
@_real_param
def test_real_file_linear_parity_vs_oracle(path):
    arrays, native = _native_tensors(path)
    # Any internal projection (not the embedding/output table) is a linear.
    name = next(
        n for n in native if not n.endswith(("token_embd.weight", "output.weight"))
    )
    tensor = native[name]
    qt = GGUFMLXQuantizedTensor.from_mx_load(arrays, name, tensor.tensor_type)
    oracle = gguf.quants.dequantize(tensor.data, tensor.tensor_type).astype(np.float32)

    x = mx.random.normal((4, qt.in_features)).astype(mx.float32)
    out = qt.matmul(x)
    expected = x @ mx.array(oracle).T
    mx.eval(out, expected)

    ref_max = float(mx.max(mx.abs(expected)))
    np.testing.assert_allclose(
        np.array(out), np.array(expected), rtol=0, atol=ref_max * 4e-6
    )


@pytest.mark.slow
@_real_gguf
@_real_param
def test_real_file_embedding_parity_vs_oracle(path):
    arrays, native = _native_tensors(path)
    name = next(n for n in native if n.endswith("token_embd.weight"))
    tensor = native[name]
    qt = GGUFMLXQuantizedTensor.from_mx_load(arrays, name, tensor.tensor_type)
    oracle = gguf.quants.dequantize(tensor.data, tensor.tensor_type).astype(np.float32)

    ids = mx.array([0, 100, qt.out_features - 1], dtype=mx.int32)
    out = qt.embedding(ids, output_dtype=mx.float32)
    expected = mx.array(oracle)[ids]
    mx.eval(out, expected)

    ref_max = float(mx.max(mx.abs(expected)))
    np.testing.assert_allclose(
        np.array(out), np.array(expected), rtol=0, atol=ref_max * 5e-3
    )


@pytest.mark.slow
@_real_gguf
@_real_param
def test_real_file_memory_below_dense(path):
    arrays, native = _native_tensors(path)
    quantized = dense_f16 = 0
    for name, tensor in native.items():
        qt = GGUFMLXQuantizedTensor.from_mx_load(arrays, name, tensor.tensor_type)
        quantized += qt.qweight.nbytes + qt.scales.nbytes + qt.biases.nbytes
        dense_f16 += qt.out_features * qt.in_features * 2

    # Per weight: Q8_0 ~1.125 bytes, Q4_0 ~0.625 (packed + f16 scale/bias) vs 2
    # for dense f16, so any native mix stays well under 0.6x dense.
    assert quantized < dense_f16 * 0.6


# --- Real generate smoke (opt-in: needs a local dense Qwen3.5 checkpoint) ------
#
# Set VLLM_METAL_TEST_QWEN35_PATH to a dense Qwen3.5-0.8B to run this. It quantizes
# every Linear/Embedding to Q8_0 (the same affine triple the GGUF path produces),
# swaps in shims that route through matmul/embedding, and checks greedy
# generation matches the dense model token-for-token. The real-file parity tests
# above cover the other half — that a real GGUF Q8_0 file loads into the same
# representation. The GGUF->module name mapping that joins the two is PR 3.

_QWEN35_DENSE = os.environ.get("VLLM_METAL_TEST_QWEN35_PATH")


@pytest.mark.slow
@pytest.mark.skipif(
    not _QWEN35_DENSE, reason="set VLLM_METAL_TEST_QWEN35_PATH to a dense Qwen3.5"
)
def test_real_generate_matches_dense():
    # mlx.nn and the swap harness are imported/defined here, not at module top,
    # so the fast tests still collect on a machine without a Metal device.
    import mlx.nn as nn
    from mlx_lm import load
    from mlx_lm.generate import generate_step
    from mlx_lm.sample_utils import make_sampler

    calls = {"linear": 0, "embedding": 0}

    def to_q8(weight):
        qweight, scales, biases = mx.quantize(
            weight, group_size=32, bits=8, mode="affine"
        )
        return GGUFMLXQuantizedTensor(
            qweight,
            scales.astype(mx.float16),
            biases.astype(mx.float16),
            GGMLQuantizationType.Q8_0,
        )

    class QuantLinear(nn.Module):
        def __init__(self, qt, bias):
            super().__init__()
            self.qt = qt
            self.bias = bias

        def __call__(self, x):
            calls["linear"] += 1
            out = self.qt.matmul(x)
            return out if self.bias is None else out + self.bias

    class QuantEmbedding(nn.Module):
        def __init__(self, qt, output_dtype):
            super().__init__()
            self.qt = qt
            self.output_dtype = output_dtype

        def __call__(self, ids):
            calls["embedding"] += 1
            return self.qt.embedding(ids, self.output_dtype)

        def as_linear(self, x):
            calls["embedding"] += 1
            return self.qt.matmul(x)

    def swap_to_q8(module):
        swapped = 0
        for name, child in module.children().items():
            leaves = child if isinstance(child, list) else [child]
            for index, leaf in enumerate(leaves):
                if not isinstance(leaf, nn.Module):
                    continue
                replacement = None
                if (
                    isinstance(leaf, nn.Linear)
                    and leaf.weight.ndim == 2
                    and leaf.weight.shape[1] % 32 == 0
                ):
                    bias = getattr(leaf, "bias", None)
                    replacement = QuantLinear(to_q8(leaf.weight), bias)
                elif isinstance(leaf, nn.Embedding) and leaf.weight.shape[1] % 32 == 0:
                    replacement = QuantEmbedding(to_q8(leaf.weight), leaf.weight.dtype)
                if replacement is None:
                    swapped += swap_to_q8(leaf)
                elif isinstance(child, list):
                    child[index] = replacement
                    swapped += 1
                else:
                    setattr(module, name, replacement)
                    swapped += 1
        return swapped

    model, tokenizer = load(_QWEN35_DENSE)
    prompt = mx.array(tokenizer.encode("The capital of France is"))
    sampler = make_sampler(temp=0.0)

    def greedy(num_tokens=20):
        tokens = []
        for (token, _), _ in zip(
            generate_step(prompt, model, sampler=sampler),
            range(num_tokens),
            strict=False,
        ):
            tokens.append(int(token))
        return tokens

    reference = greedy()
    assert swap_to_q8(model) > 0
    calls["linear"] = calls["embedding"] = 0
    candidate = greedy()

    # Token parity AND proof the quantized primitives actually ran (a silent
    # dense fallback would still match to greedy precision otherwise).
    assert candidate == reference
    assert calls["linear"] > 0
    assert calls["embedding"] > 0


# === permute_rows (llama RoPE q/k un-permutation mechanism) ===


@pytest.mark.parametrize("qtype", NATIVE_QTYPES)
def test_permute_rows_matches_dequant_then_gather(tmp_path, qtype):
    # Reordering the packed qweight + scales + biases together must be bit-exact
    # to dequantizing then gathering the same rows.
    qt, oracle = _make_tensor(tmp_path, qtype, shape=(64, 128))
    index = mx.array(np.random.default_rng(1).permutation(64))

    permuted = qt.permute_rows(index)

    assert permuted.qweight_type == qt.qweight_type
    assert permuted.logical_shape == qt.logical_shape
    x = mx.array(np.random.default_rng(2).standard_normal((3, 128)), dtype=mx.float32)
    got = permuted.matmul(x)
    want = x @ mx.array(oracle)[index].T
    assert bool(mx.allclose(got, want, atol=1e-3).item())


@pytest.mark.parametrize("qtype", NATIVE_QTYPES)
def test_permute_rows_round_trips_via_inverse(tmp_path, qtype):
    # The index is not an involution for head_dim > 4, so the inverse is
    # argsort(index), not the index itself.
    qt, oracle = _make_tensor(tmp_path, qtype, shape=(64, 128))
    index = mx.array(np.random.default_rng(3).permutation(64))

    restored = qt.permute_rows(index).permute_rows(mx.argsort(index))

    x = mx.array(np.random.default_rng(4).standard_normal((2, 128)), dtype=mx.float32)
    assert bool(mx.allclose(restored.matmul(x), qt.matmul(x), atol=1e-4).item())


def test_permute_rows_rejects_non_permutation(tmp_path):
    qt, _ = _make_tensor(tmp_path, GGMLQuantizationType.Q8_0, shape=(64, 128))
    with pytest.raises(ValueError) as short:
        qt.permute_rows(mx.arange(32))
    assert str(short.value) == (
        "permute_rows index must be 1-D of length 64 (out_features), got shape (32,)"
    )
    with pytest.raises(ValueError) as dup:
        qt.permute_rows(mx.zeros(64, dtype=mx.int32))  # all-zero: duplicates
    assert str(dup.value) == (
        "permute_rows index must be a permutation of range(64); "
        "got duplicate or out-of-range values"
    )
