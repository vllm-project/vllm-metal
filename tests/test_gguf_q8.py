# SPDX-License-Identifier: Apache-2.0
"""Tests for native GGUF Q8_0 tensor primitives."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from vllm_metal.gguf import (
    GGUF_QTYPE_Q8_0,
    GGUFQuantizedTensor,
    q8_0_matmul,
)


def _pack_q8_0_raw(qweight: np.ndarray, scales: np.ndarray) -> mx.array:
    rows, blocks, _ = qweight.shape
    raw = np.zeros((rows, blocks, 34), dtype=np.uint8)
    raw[:, :, :2] = scales.astype(np.float16).reshape(rows, blocks, 1).view(np.uint8)
    raw[:, :, 2:] = qweight.astype(np.int8).view(np.uint8)
    return mx.array(raw.reshape(rows, blocks * 34))


def _quantize_q8_1_np(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tokens, cols = x.shape
    padded_cols = ((cols + 511) // 512) * 512
    blocks = padded_cols // 32
    qx = np.zeros((tokens, blocks, 32), dtype=np.int8)
    x_scales = np.zeros((tokens, blocks), dtype=np.float32)

    for token in range(tokens):
        for block in range(blocks):
            start = block * 32
            vals = np.zeros(32, dtype=np.float32)
            valid = max(0, min(32, cols - start))
            if valid:
                vals[:valid] = x[token, start : start + valid]
            amax = np.max(np.abs(vals))
            if amax == 0.0:
                continue
            d = float(amax / 127.0)
            qx[token, block] = np.clip(np.round(vals / d), -127, 127).astype(np.int8)
            x_scales[token, block] = np.float16(d).astype(np.float32)

    return qx, x_scales


def _q8_0_q8_1_matmul_np(
    x: np.ndarray,
    qweight: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    qx, x_scales = _quantize_q8_1_np(x)
    rows, blocks, _ = qweight.shape
    out = np.zeros((x.shape[0], rows), dtype=np.float32)
    for token in range(x.shape[0]):
        for row in range(rows):
            acc = 0.0
            for block in range(blocks):
                dot = np.dot(
                    qweight[row, block].astype(np.int32),
                    qx[token, block].astype(np.int32),
                )
                acc += dot * float(scales[row, block]) * x_scales[token, block]
            out[token, row] = acc
    return out


def _q8_0_dense_matmul_np(
    x: np.ndarray,
    qweight: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    dense_weight = qweight.astype(np.float32) * scales.astype(np.float32)[..., None]
    return x @ dense_weight.reshape(dense_weight.shape[0], -1).T


def _require_metal() -> None:
    if not hasattr(mx, "metal") or not mx.metal.is_available():
        pytest.skip("MLX Metal device is not available")


def test_q8_0_tensor_contract_keeps_raw_blocks() -> None:
    qweight_np = np.arange(-32, 32, dtype=np.int8).reshape(2, 1, 32)
    scales_np = np.array([[0.25], [0.5]], dtype=np.float16)
    qweight = _pack_q8_0_raw(qweight_np, scales_np)

    tensor = GGUFQuantizedTensor(
        qweight=qweight,
        qweight_type=GGUF_QTYPE_Q8_0,
        logical_shape=(2, 32),
    )

    assert tensor.qtype_id == GGUF_QTYPE_Q8_0
    assert tensor.qtype_name == "Q8_0"
    assert tensor.logical_shape == (2, 32)
    assert tensor.raw_block_shape == (2, 1, 34)
    assert tensor.raw_bytes_per_row == 34
    assert tensor.matmul_transpose is True


def test_q8_0_tensor_rejects_shape_that_would_need_dense_interpretation() -> None:
    qweight = mx.zeros((2, 34), dtype=mx.uint8)

    with pytest.raises(ValueError, match="does not match logical shape"):
        GGUFQuantizedTensor(
            qweight=qweight,
            qweight_type=GGUF_QTYPE_Q8_0,
            logical_shape=(2, 64),
        )


def test_q8_0_tensor_rejects_unsupported_qtype() -> None:
    qweight = mx.zeros((2, 34), dtype=mx.uint8)

    with pytest.raises(ValueError, match="Unsupported native GGUF qtype id"):
        GGUFQuantizedTensor(
            qweight=qweight,
            qweight_type=2,
            logical_shape=(2, 32),
        )


@pytest.mark.parametrize(
    ("output_dtype", "rtol", "atol"),
    [
        (mx.float32, 2e-4, 2e-4),
        (mx.float16, 2e-2, 2e-2),
        (mx.bfloat16, 6e-2, 6e-2),
    ],
)
def test_q8_0_matmul_matches_activation_quantized_reference(
    monkeypatch,
    output_dtype: mx.Dtype,
    rtol: float,
    atol: float,
) -> None:
    _require_metal()
    monkeypatch.setenv("VLLM_METAL_BUILD_FROM_SOURCE", "1")
    qweight_np = ((np.arange(3 * 2 * 32) % 63) - 31).astype(np.int8).reshape(3, 2, 32)
    scales_np = np.array(
        [[0.5, 0.125], [0.25, 0.75], [2.0, 0.0625]],
        dtype=np.float16,
    )
    tensor = GGUFQuantizedTensor(
        qweight=_pack_q8_0_raw(qweight_np, scales_np),
        qweight_type=GGUF_QTYPE_Q8_0,
        logical_shape=(3, 64),
    )
    x_np = np.array(
        [
            np.linspace(-0.7, 0.7, 64, dtype=np.float32),
            np.linspace(0.3, -0.4, 64, dtype=np.float32),
        ]
    )
    x = mx.array(x_np, dtype=mx.float32)

    out = q8_0_matmul(x, tensor, output_dtype=output_dtype)
    expected = _q8_0_q8_1_matmul_np(x_np, qweight_np, scales_np)
    mx.eval(out)

    assert out.dtype == output_dtype
    np.testing.assert_allclose(
        np.array(out.astype(mx.float32)),
        expected,
        rtol=rtol,
        atol=atol,
    )


def test_q8_0_matmul_matches_dense_reference_for_exact_q8_1_inputs(
    monkeypatch,
) -> None:
    _require_metal()
    monkeypatch.setenv("VLLM_METAL_BUILD_FROM_SOURCE", "1")
    qweight_np = np.stack(
        [
            np.arange(-16, 16, dtype=np.int8),
            np.arange(15, -17, -1, dtype=np.int8),
        ]
    ).reshape(2, 1, 32)
    scales_np = np.array([[0.5], [0.25]], dtype=np.float16)
    tensor = GGUFQuantizedTensor(
        qweight=_pack_q8_0_raw(qweight_np, scales_np),
        qweight_type=GGUF_QTYPE_Q8_0,
        logical_shape=(2, 32),
    )
    x_q = np.array(
        [
            [
                -127,
                -64,
                -32,
                -16,
                -8,
                -4,
                -2,
                -1,
                0,
                1,
                2,
                4,
                8,
                16,
                32,
                64,
                127,
                96,
                80,
                48,
                24,
                12,
                6,
                3,
                -3,
                -6,
                -12,
                -24,
                -48,
                -80,
                -96,
                112,
            ],
        ],
        dtype=np.float32,
    )
    x_np = x_q / 127.0

    out = q8_0_matmul(mx.array(x_np, dtype=mx.float32), tensor)
    expected = _q8_0_dense_matmul_np(x_np, qweight_np, scales_np)
    mx.eval(out)

    np.testing.assert_allclose(np.array(out), expected, rtol=2e-4, atol=2e-4)
