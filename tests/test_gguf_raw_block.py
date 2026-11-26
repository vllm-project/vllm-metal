# SPDX-License-Identifier: Apache-2.0
"""Tests for the raw-block GGUF tensor and its Q2_K/Q3_K/Q6_K/IQ3_XXS kernels.

K-quant fixtures are built as raw blocks field-by-field because gguf-py
cannot quantize K-quants; ``gguf.quants.dequantize`` stays the authoritative
oracle for what the blocks mean.
"""

from __future__ import annotations

import gguf
import mlx.core as mx
import numpy as np
import pytest

from tests.gguf_kquant_fixtures import build_iq3_blocks, build_kquant_blocks
from vllm_metal.gguf.raw_block import GGUFRawBlockTensor

GGMLQuantizationType = gguf.GGMLQuantizationType

_RAW_QTYPES = [
    GGMLQuantizationType.Q2_K,
    GGMLQuantizationType.Q3_K,
    GGMLQuantizationType.Q6_K,
    GGMLQuantizationType.IQ3_XXS,
]


def _spy_matmul_paths(monkeypatch) -> dict:
    """Count which matmul arm runs; both spies delegate to the real methods."""
    calls = {"qmv": 0, "gemm": 0}
    real_qmv = GGUFRawBlockTensor._qmv
    real_dequantize = GGUFRawBlockTensor._dequantize_rows

    def spy_qmv(self, x):
        calls["qmv"] += 1
        return real_qmv(self, x)

    def spy_dequantize(self, packed_rows, output_dtype):
        calls["gemm"] += 1
        return real_dequantize(self, packed_rows, output_dtype)

    monkeypatch.setattr(GGUFRawBlockTensor, "_qmv", spy_qmv)
    monkeypatch.setattr(GGUFRawBlockTensor, "_dequantize_rows", spy_dequantize)
    return calls


def _make_raw_tensor(
    rows: int = 16,
    cols: int = 512,
    qtype: gguf.GGMLQuantizationType = GGMLQuantizationType.Q6_K,
) -> tuple:
    if qtype == GGMLQuantizationType.IQ3_XXS:
        raw = build_iq3_blocks(rows, cols, qtype)
    else:
        raw = build_kquant_blocks(rows, cols, qtype)
    qt = GGUFRawBlockTensor.from_raw_blocks(raw, (rows, cols), qtype)
    oracle = gguf.quants.dequantize(raw, qtype).astype(np.float32)
    return qt, oracle


@pytest.mark.parametrize(
    ("qtype", "bits", "block_bytes"),
    [
        (GGMLQuantizationType.Q2_K, 2, 84),
        (GGMLQuantizationType.Q3_K, 3, 110),
        (GGMLQuantizationType.Q6_K, 6, 210),
        (GGMLQuantizationType.IQ3_XXS, 3, 98),
    ],
)
def test_contract_matches_logical_shape(qtype, bits, block_bytes):
    qt, _ = _make_raw_tensor(rows=16, cols=512, qtype=qtype)

    assert qt.qweight_type == qtype
    assert qt.logical_shape == (16, 512)
    assert qt.out_features == 16
    assert qt.in_features == 512
    assert qt.bits == bits
    assert qt.qweight.dtype == mx.uint8
    assert qt.packed_shape == (16, 512 // 256 * block_bytes)


@pytest.mark.parametrize("qtype", _RAW_QTYPES)
def test_dequantize_matches_oracle_bit_exact(qtype):
    qt, oracle = _make_raw_tensor(qtype=qtype)

    out = qt.embedding(mx.arange(qt.out_features), output_dtype=mx.float32)
    mx.eval(out)

    assert np.array_equal(np.array(out), oracle)


@pytest.mark.parametrize("qtype", _RAW_QTYPES)
@pytest.mark.parametrize("batch", [1, 3, 8])
def test_matmul_qmv_path_matches_dense_oracle_f32(monkeypatch, batch, qtype):
    # 17 superblocks give every thread a group and some threads a second one.
    qt, oracle = _make_raw_tensor(cols=17 * 256, qtype=qtype)
    calls = _spy_matmul_paths(monkeypatch)
    x = mx.random.normal((batch, qt.in_features)).astype(mx.float32)

    out = qt.matmul(x)
    # M5 GPU matmul may use TF32; keep the reference at full FP32 precision on CPU.
    expected = mx.matmul(x, mx.array(oracle).T, stream=mx.cpu)
    mx.eval(out, expected)

    assert calls == {"qmv": 1, "gemm": 0}
    # The fused kernel accumulates per-16-group then across lanes, so allow
    # FP32 ordering differences from the CPU reference.
    ref_max = float(mx.max(mx.abs(expected)))
    np.testing.assert_allclose(
        np.array(out), np.array(expected), rtol=0, atol=ref_max * 4e-5
    )


@pytest.mark.parametrize("qtype", _RAW_QTYPES)
def test_matmul_gemm_path_matches_dense_oracle_f32(monkeypatch, qtype):
    qt, oracle = _make_raw_tensor(qtype=qtype)
    calls = _spy_matmul_paths(monkeypatch)
    x = mx.random.normal((32, qt.in_features)).astype(mx.float32)

    out = qt.matmul(x)
    expected = mx.matmul(x, mx.array(oracle).T, stream=mx.cpu)
    mx.eval(out, expected)

    assert calls == {"qmv": 0, "gemm": 1}
    ref_max = float(mx.max(mx.abs(expected)))
    np.testing.assert_allclose(
        np.array(out), np.array(expected), rtol=0, atol=ref_max * 4e-5
    )


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
def test_matmul_gemm_path_dequantizes_in_activation_dtype(monkeypatch, dtype):
    qt, _ = _make_raw_tensor()
    real_dequantize = GGUFRawBlockTensor._dequantize_rows
    dense_dtypes = []

    def spy_dequantize(self, packed_rows, output_dtype):
        dense_dtypes.append(output_dtype)
        return real_dequantize(self, packed_rows, output_dtype)

    monkeypatch.setattr(GGUFRawBlockTensor, "_dequantize_rows", spy_dequantize)
    x = mx.random.normal((32, qt.in_features)).astype(dtype)

    out = qt.matmul(x)
    mx.eval(out)

    assert dense_dtypes == [dtype]


# B=8 is the last qmv batch and B=9 the first GEMM batch, so the low-precision
# oracle comparison covers both matmul paths.
@pytest.mark.parametrize("batch", [8, 9])
@pytest.mark.parametrize(
    ("dtype", "rel_tol"), [(mx.float16, 4e-3), (mx.bfloat16, 8e-3)]
)
@pytest.mark.parametrize("qtype", _RAW_QTYPES)
def test_matmul_low_precision_matches_dense_oracle(dtype, rel_tol, batch, qtype):
    qt, oracle = _make_raw_tensor(qtype=qtype)
    x = mx.random.normal((batch, qt.in_features)).astype(dtype)

    out = qt.matmul(x)
    expected = mx.matmul(x.astype(mx.float32), mx.array(oracle).T, stream=mx.cpu)
    mx.eval(out, expected)

    assert out.dtype == dtype
    assert out.shape == (batch, qt.out_features)
    # The GEMM path rounds weight and output to the input dtype (worst of 1000
    # draws vs the f32 CPU reference: bf16 5.7e-3, f16 6.3e-4).
    ref_max = float(mx.max(mx.abs(expected)))
    np.testing.assert_allclose(
        np.array(out.astype(mx.float32)),
        np.array(expected),
        rtol=0,
        atol=ref_max * rel_tol,
    )


def test_matmul_preserves_leading_shape():
    qt, _ = _make_raw_tensor()
    x = mx.random.normal((2, 3, qt.in_features)).astype(mx.float16)

    out = qt.matmul(x)
    flat = qt.matmul(x.reshape(-1, qt.in_features))
    mx.eval(out, flat)

    assert out.shape == (2, 3, qt.out_features)
    assert mx.array_equal(out.reshape(-1, qt.out_features), flat)


def test_matmul_empty_batch():
    qt, _ = _make_raw_tensor()

    out = qt.matmul(mx.zeros((0, qt.in_features), dtype=mx.float16))

    assert out.shape == (0, qt.out_features)
    assert out.dtype == mx.float16


@pytest.mark.parametrize("qtype", _RAW_QTYPES)
def test_embedding_matches_oracle_rows_exactly(qtype):
    qt, oracle = _make_raw_tensor(qtype=qtype)
    ids = mx.array([[0, 5], [15, 0]], dtype=mx.int32)

    out = qt.embedding(ids, output_dtype=mx.float32)
    mx.eval(out)

    assert out.shape == (2, 2, qt.in_features)
    assert np.array_equal(np.array(out), oracle[np.array([[0, 5], [15, 0]])])


def test_embedding_empty_ids():
    qt, _ = _make_raw_tensor()

    out = qt.embedding(mx.zeros((0,), dtype=mx.int32), output_dtype=mx.float16)

    assert out.shape == (0, qt.in_features)
    assert out.dtype == mx.float16


@pytest.mark.parametrize("qtype", _RAW_QTYPES)
def test_permute_rows_matches_permuted_oracle(qtype):
    qt, oracle = _make_raw_tensor(qtype=qtype)
    perm = np.random.default_rng(5).permutation(qt.out_features)

    permuted = qt.permute_rows(mx.array(perm))
    out = permuted.embedding(mx.arange(qt.out_features), output_dtype=mx.float32)
    mx.eval(out)

    assert np.array_equal(np.array(out), oracle[perm])


def test_permute_rows_rejects_bad_index():
    qt, _ = _make_raw_tensor()

    with pytest.raises(ValueError, match="must be a permutation"):
        qt.permute_rows(mx.zeros((qt.out_features,), dtype=mx.int32))


def test_rejects_non_raw_kernel_qtype():
    raw = build_kquant_blocks(16, 512, GGMLQuantizationType.Q6_K)

    with pytest.raises(ValueError, match=r"Raw-kernel qtypes: Q2_K, Q3_K, Q6_K"):
        GGUFRawBlockTensor.from_raw_blocks(raw, (16, 512), GGMLQuantizationType.Q4_K)


def test_rejects_non_uint8_qweight():
    with pytest.raises(ValueError, match="qweight must be uint8"):
        GGUFRawBlockTensor(
            qweight=mx.zeros((4, 420), dtype=mx.uint32),
            qweight_type=GGMLQuantizationType.Q6_K,
        )


def test_rejects_non_superblock_row_width():
    with pytest.raises(ValueError, match="not a positive multiple of the 210-byte"):
        GGUFRawBlockTensor(
            qweight=mx.zeros((4, 200), dtype=mx.uint8),
            qweight_type=GGMLQuantizationType.Q6_K,
        )


def test_from_raw_blocks_rejects_truncated_payload():
    raw = build_kquant_blocks(16, 512, GGMLQuantizationType.Q6_K)

    with pytest.raises(ValueError, match="logical shape .* needs"):
        GGUFRawBlockTensor.from_raw_blocks(
            raw.reshape(-1)[:-1], (16, 512), GGMLQuantizationType.Q6_K
        )


def test_from_raw_blocks_rejects_non_superblock_width():
    raw = build_kquant_blocks(16, 512, GGMLQuantizationType.Q6_K)

    with pytest.raises(ValueError, match="not a multiple of the 256-element"):
        GGUFRawBlockTensor.from_raw_blocks(raw, (16, 500), GGMLQuantizationType.Q6_K)


def test_from_raw_blocks_rejects_non_uint8_payload():
    raw = build_kquant_blocks(16, 512, GGMLQuantizationType.Q6_K)

    with pytest.raises(ValueError, match="raw block data must be uint8"):
        GGUFRawBlockTensor.from_raw_blocks(
            raw.astype(np.float32), (16, 512), GGMLQuantizationType.Q6_K
        )


def test_matmul_rejects_wrong_last_dim():
    qt, _ = _make_raw_tensor()
    # 4x256 would silently reshape into 2x512 without the guard.
    x = mx.random.normal((4, qt.in_features // 2)).astype(mx.float32)

    with pytest.raises(ValueError, match="matmul expects last dim"):
        qt.matmul(x)


# --- MoE experts stacked as row blocks ----------------------------------------

_NUM_EXPERTS = 4


def _make_expert_tensor(qtype: gguf.GGMLQuantizationType) -> tuple:
    """Four 8-row experts stacked in one tensor, plus their (E, rows, in) oracle."""
    qt, oracle = _make_raw_tensor(rows=_NUM_EXPERTS * 8, qtype=qtype)
    return qt, oracle.reshape(_NUM_EXPERTS, 8, qt.in_features)


def _gather_oracle(x: mx.array, indices: mx.array, experts: np.ndarray) -> mx.array:
    w = mx.array(experts).swapaxes(-1, -2)
    return mx.gather_mm(x, w, rhs_indices=indices, stream=mx.cpu)


def _assert_close_to(out: mx.array, expected: mx.array) -> None:
    mx.eval(out, expected)
    ref_max = float(mx.max(mx.abs(expected)))
    np.testing.assert_allclose(
        np.array(out), np.array(expected), rtol=0, atol=ref_max * 4e-5
    )


@pytest.mark.parametrize("qtype", _RAW_QTYPES)
@pytest.mark.parametrize(("tokens", "dequantizes"), [(1, False), (4, False), (5, True)])
def test_gather_matmul_matches_expert_oracle(monkeypatch, qtype, tokens, dequantizes):
    # SwitchGLU's unsorted call: x (B, L, 1, 1, in) routed by indices (B, L, k).
    # Four experts cap gather QMV at 8 (token, expert) assignments; 5 x top-2
    # is 10.
    qt, experts = _make_expert_tensor(qtype)
    calls = _spy_matmul_paths(monkeypatch)
    x = mx.random.normal((1, tokens, 1, 1, qt.in_features))
    indices = mx.random.randint(0, _NUM_EXPERTS, (1, tokens, 2))

    out = qt.gather_matmul(x, indices, num_experts=_NUM_EXPERTS)

    assert out.shape == (1, tokens, 2, 1, 8)
    assert calls["gemm"] == int(dequantizes)
    _assert_close_to(out, _gather_oracle(x, indices, experts))


# 4 tokens x top-2 runs gather QMV and 5 the dense arm; tolerances as for matmul.
@pytest.mark.parametrize("tokens", [4, 5])
@pytest.mark.parametrize(
    ("dtype", "rel_tol"), [(mx.float16, 4e-3), (mx.bfloat16, 8e-3)]
)
@pytest.mark.parametrize("qtype", _RAW_QTYPES)
def test_gather_matmul_low_precision_matches_expert_oracle(
    dtype, rel_tol, tokens, qtype
):
    qt, experts = _make_expert_tensor(qtype)
    x = mx.random.normal((1, tokens, 1, 1, qt.in_features)).astype(dtype)
    indices = mx.random.randint(0, _NUM_EXPERTS, (1, tokens, 2))

    out = qt.gather_matmul(x, indices, num_experts=_NUM_EXPERTS)
    expected = _gather_oracle(x.astype(mx.float32), indices, experts)
    mx.eval(out, expected)

    assert out.dtype == dtype
    ref_max = float(mx.max(mx.abs(expected)))
    np.testing.assert_allclose(
        np.array(out.astype(mx.float32)),
        np.array(expected),
        rtol=0,
        atol=ref_max * rel_tol,
    )


# 8 assignments run gather QMV and 80 the dense arm.
@pytest.mark.parametrize("assignments", [8, 80])
@pytest.mark.parametrize("qtype", _RAW_QTYPES)
def test_gather_matmul_sorted_matches_expert_oracle(qtype, assignments):
    # SwitchGLU sorts 64+ assignments into x (N, 1, in) with ascending indices.
    qt, experts = _make_expert_tensor(qtype)
    x = mx.random.normal((assignments, 1, qt.in_features))
    indices = mx.sort(mx.random.randint(0, _NUM_EXPERTS, (assignments,)))

    out = qt.gather_matmul(x, indices, num_experts=_NUM_EXPERTS, sorted_indices=True)

    _assert_close_to(out, _gather_oracle(x, indices, experts))
