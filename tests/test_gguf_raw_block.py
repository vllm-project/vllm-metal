# SPDX-License-Identifier: Apache-2.0
"""Tests for the raw-block GGUF tensor and its Q6_K Metal kernels.

Q6_K fixtures are built as raw blocks field-by-field because gguf-py cannot
quantize K-quants; ``gguf.quants.dequantize`` stays the authoritative oracle
for what the blocks mean.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

gguf = pytest.importorskip("gguf")

from vllm_metal.gguf.raw_block import GGUFRawBlockTensor  # noqa: E402

GGMLQuantizationType = gguf.GGMLQuantizationType


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


def _build_q6k_blocks(rows: int, cols: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = rows * (cols // 256)
    d = rng.uniform(2**-10, 2**-4, (n, 1)).astype(np.float16)
    sub_scales = rng.integers(-128, 128, (n, 16), dtype=np.int8)
    codes = rng.integers(0, 64, (n, 256), dtype=np.uint8)
    low = codes & 0x0F
    high = codes >> 4
    ql = np.zeros((n, 128), np.uint8)
    for c in (0, 1):
        ql[:, c * 64 : (c + 1) * 64] = low[:, c * 128 : c * 128 + 64] | (
            low[:, c * 128 + 64 : c * 128 + 128] << 4
        )
    qh = np.zeros((n, 64), np.uint8)
    for c in (0, 1):
        for s in range(4):
            qh[:, c * 32 : (c + 1) * 32] |= high[
                :, c * 128 + s * 32 : c * 128 + (s + 1) * 32
            ] << (2 * s)
    blocks = np.concatenate(
        [ql, qh, sub_scales.view(np.uint8), d.view(np.uint8)], axis=1
    )
    return blocks.reshape(rows, -1)


def _make_q6k_tensor(rows: int = 16, cols: int = 512) -> tuple:
    raw = _build_q6k_blocks(rows, cols)
    qt = GGUFRawBlockTensor.from_raw_blocks(
        raw, (rows, cols), GGMLQuantizationType.Q6_K
    )
    oracle = gguf.quants.dequantize(raw, GGMLQuantizationType.Q6_K).astype(np.float32)
    return qt, oracle


def test_contract_matches_logical_shape():
    qt, _ = _make_q6k_tensor(rows=16, cols=512)

    assert qt.qweight_type == GGMLQuantizationType.Q6_K
    assert qt.logical_shape == (16, 512)
    assert qt.out_features == 16
    assert qt.in_features == 512
    assert qt.bits == 6
    assert qt.qweight.dtype == mx.uint8
    assert qt.packed_shape == (16, 512 // 256 * 210)


def test_dequantize_matches_oracle_bit_exact():
    qt, oracle = _make_q6k_tensor()

    out = qt.embedding(mx.arange(qt.out_features), output_dtype=mx.float32)
    mx.eval(out)

    assert np.array_equal(np.array(out), oracle)


def test_matmul_qmv_path_matches_dense_oracle_f32(monkeypatch):
    qt, oracle = _make_q6k_tensor()
    calls = _spy_matmul_paths(monkeypatch)
    x = mx.random.normal((2, qt.in_features)).astype(mx.float32)

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


def test_matmul_gemm_path_matches_dense_oracle_f32(monkeypatch):
    qt, oracle = _make_q6k_tensor()
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


# B=4 is the last qmv batch and B=5 the first GEMM batch, so the low-precision
# oracle comparison covers both matmul paths.
@pytest.mark.parametrize("batch", [4, 5])
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_matmul_low_precision_matches_dense_oracle(dtype, batch):
    qt, oracle = _make_q6k_tensor()
    x = mx.random.normal((batch, qt.in_features)).astype(dtype)

    out = qt.matmul(x)
    expected = mx.matmul(x.astype(mx.float32), mx.array(oracle).T, stream=mx.cpu)
    mx.eval(out, expected)

    assert out.dtype == dtype
    assert out.shape == (batch, qt.out_features)
    # The comparison is bounded by the input precision (bf16 measured ~2e-3
    # vs the f32 CPU reference, f16 ~3e-4).
    ref_max = float(mx.max(mx.abs(expected)))
    np.testing.assert_allclose(
        np.array(out.astype(mx.float32)),
        np.array(expected),
        rtol=0,
        atol=ref_max * 4e-3,
    )


def test_matmul_preserves_leading_shape():
    qt, _ = _make_q6k_tensor()
    x = mx.random.normal((2, 3, qt.in_features)).astype(mx.float16)

    out = qt.matmul(x)
    flat = qt.matmul(x.reshape(-1, qt.in_features))
    mx.eval(out, flat)

    assert out.shape == (2, 3, qt.out_features)
    assert mx.array_equal(out.reshape(-1, qt.out_features), flat)


def test_matmul_empty_batch():
    qt, _ = _make_q6k_tensor()

    out = qt.matmul(mx.zeros((0, qt.in_features), dtype=mx.float16))

    assert out.shape == (0, qt.out_features)
    assert out.dtype == mx.float16


def test_embedding_matches_oracle_rows_exactly():
    qt, oracle = _make_q6k_tensor()
    ids = mx.array([[0, 5], [15, 0]], dtype=mx.int32)

    out = qt.embedding(ids, output_dtype=mx.float32)
    mx.eval(out)

    assert out.shape == (2, 2, qt.in_features)
    assert np.array_equal(np.array(out), oracle[np.array([[0, 5], [15, 0]])])


def test_embedding_empty_ids():
    qt, _ = _make_q6k_tensor()

    out = qt.embedding(mx.zeros((0,), dtype=mx.int32), output_dtype=mx.float16)

    assert out.shape == (0, qt.in_features)
    assert out.dtype == mx.float16


def test_permute_rows_matches_permuted_oracle():
    qt, oracle = _make_q6k_tensor()
    perm = np.random.default_rng(5).permutation(qt.out_features)

    permuted = qt.permute_rows(mx.array(perm))
    out = permuted.embedding(mx.arange(qt.out_features), output_dtype=mx.float32)
    mx.eval(out)

    assert np.array_equal(np.array(out), oracle[perm])


def test_permute_rows_rejects_bad_index():
    qt, _ = _make_q6k_tensor()

    with pytest.raises(ValueError, match="must be a permutation"):
        qt.permute_rows(mx.zeros((qt.out_features,), dtype=mx.int32))


def test_rejects_non_raw_kernel_qtype():
    raw = _build_q6k_blocks(16, 512)

    with pytest.raises(ValueError, match="Raw-kernel qtypes: Q6_K"):
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
    raw = _build_q6k_blocks(16, 512)

    with pytest.raises(ValueError, match="logical shape .* needs"):
        GGUFRawBlockTensor.from_raw_blocks(
            raw.reshape(-1)[:-1], (16, 512), GGMLQuantizationType.Q6_K
        )


def test_from_raw_blocks_rejects_non_superblock_width():
    raw = _build_q6k_blocks(16, 512)

    with pytest.raises(ValueError, match="not a multiple of the 256-element"):
        GGUFRawBlockTensor.from_raw_blocks(raw, (16, 500), GGMLQuantizationType.Q6_K)


def test_from_raw_blocks_rejects_non_uint8_payload():
    raw = _build_q6k_blocks(16, 512)

    with pytest.raises(ValueError, match="raw block data must be uint8"):
        GGUFRawBlockTensor.from_raw_blocks(
            raw.astype(np.float32), (16, 512), GGMLQuantizationType.Q6_K
        )


def test_matmul_rejects_wrong_last_dim():
    qt, _ = _make_q6k_tensor()
    # 4x256 would silently reshape into 2x512 without the guard.
    x = mx.random.normal((4, qt.in_features // 2)).astype(mx.float32)

    with pytest.raises(ValueError, match="matmul expects last dim"):
        qt.matmul(x)
