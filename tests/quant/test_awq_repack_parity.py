# SPDX-License-Identifier: Apache-2.0
"""AWQ -> MLX-affine repack parity (mlx_lm._transform_awq_weights).

Validates that mlx_lm's transform produces an MLX representation whose
dequantization matches the AWQ ground-truth dequantization, under the
v1 supported subset (bits=4, group_size=128, with qzeros).

This test does NOT exercise vllm-metal code directly — it pins down the
upstream mlx_lm transform that the AWQ load path delegates to. If
mlx_lm's transform changes in a way that breaks parity for our v1
config, this test catches it before the e2e parity test (which is
`slow`-marked and downloads a model).

Synthetic tensors only; no model load, no downloads.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
from mlx_lm.utils import _transform_awq_weights

GROUP_SIZE = 128
BITS = 4
PACK_FACTOR = 32 // BITS  # 8 nibbles per int32

# AutoAWQ packs 8 nibbles per int32 along the output dim, in this *interleaved*
# order so its fast-dequant kernel can shuffle with bit shifts. mlx_lm's
# `_unpack_awq_weights` decodes the same order.
_AWQ_PACK_ORDER = [0, 2, 4, 6, 1, 3, 5, 7]


def _pack_awq_int32(unpacked: np.ndarray) -> np.ndarray:
    """[..., out] uint8 (values 0-15) -> [..., out//8] int32 in AWQ order."""
    leading = unpacked.shape[:-1]
    out_features = unpacked.shape[-1]
    assert out_features % PACK_FACTOR == 0
    packed = np.zeros(leading + (out_features // PACK_FACTOR,), dtype=np.uint32)
    u = unpacked.astype(np.uint32) & 0xF
    for shift_idx, pos in enumerate(_AWQ_PACK_ORDER):
        # Pick every pack_factor-th column starting at `pos`.
        packed |= (u[..., pos::PACK_FACTOR]) << (shift_idx * BITS)
    return packed.astype(np.int32)


def _build_awq_layer(
    in_features: int,
    out_features: int,
    rng: np.random.Generator,
):
    """Synthesize an AWQ layer triple (qweight, qzeros, scales) with known
    ground truth.

    Returns:
      qweight: [in_features, out_features // 8] int32 (AWQ packing)
      qzeros:  [n_groups, out_features // 8] int32 (AWQ packing)
      scales:  [n_groups, out_features] fp16
      ref_w:   [out_features, in_features] fp32 ground-truth dequantization
              W = (q - z) * s, computed in fp32 for parity comparison.
    """
    assert in_features % GROUP_SIZE == 0
    assert out_features % PACK_FACTOR == 0

    n_groups = in_features // GROUP_SIZE

    # Ground-truth int4 values (per-element 0..15) and per-group zero / scale.
    q_unpacked = rng.integers(0, 16, size=(in_features, out_features), dtype=np.uint8)
    z_unpacked = rng.integers(0, 16, size=(n_groups, out_features), dtype=np.uint8)
    # Use realistic scale magnitudes so fp16 rounding doesn't dominate.
    scales_fp = rng.uniform(0.001, 0.05, size=(n_groups, out_features)).astype(
        np.float16
    )

    # Ground truth: W[in_idx, out_idx] = (q[in_idx, out_idx] - z[group_of(in_idx), out_idx]) * s[group_of(in_idx), out_idx]
    q_f32 = q_unpacked.astype(np.float32)
    z_f32 = z_unpacked.astype(np.float32)
    s_f32 = scales_fp.astype(np.float32)
    ref_w_in_out = np.empty((in_features, out_features), dtype=np.float32)
    for g in range(n_groups):
        start, end = g * GROUP_SIZE, (g + 1) * GROUP_SIZE
        ref_w_in_out[start:end] = (q_f32[start:end] - z_f32[g]) * s_f32[g]
    # MLX layout is [out, in]; AWQ stores [in, out] in qweight. Reference
    # for the post-transform `mx.dequantize` output is [out, in].
    ref_w_out_in = ref_w_in_out.T  # [out, in]

    qweight = _pack_awq_int32(q_unpacked)  # [in, out//8]
    qzeros = _pack_awq_int32(z_unpacked)  # [n_groups, out//8]
    return qweight, qzeros, scales_fp, ref_w_out_in


def _dequantize_to_fp32(weight, scales, biases) -> np.ndarray:
    return np.array(
        mx.dequantize(weight, scales, biases, group_size=GROUP_SIZE, bits=BITS).astype(
            mx.float32
        )
    )


@pytest.mark.parametrize(
    ("in_features", "out_features"),
    [
        (128, 64),  # smallest viable: 1 group, 8 packed columns
        (128, 256),  # multiple packed columns per row
        (256, 256),  # multiple groups
    ],
    ids=["one-group-narrow", "one-group-wide", "two-groups-square"],
)
def test_repack_dequant_matches_awq_ground_truth(in_features, out_features):
    """`_transform_awq_weights` output, dequantized via `mx.dequantize`,
    matches the AWQ formula `W = (q - z) * s` within fp16 noise.

    This pins the contract that the v1 load path depends on: the AWQ
    calibration's per-group `scales` and `qzeros` survive the repack
    intact (vs being replaced by fresh MLX-quantized values).
    """
    rng = np.random.default_rng(seed=42)
    qweight, qzeros, scales_fp, ref_w = _build_awq_layer(in_features, out_features, rng)

    weights = {
        "blk.0.qweight": mx.array(qweight),
        "blk.0.qzeros": mx.array(qzeros),
        "blk.0.scales": mx.array(scales_fp),
    }
    new_weights, mlx_quant_config = _transform_awq_weights(
        weights,
        {
            "quant_method": "awq",
            "bits": BITS,
            "group_size": GROUP_SIZE,
            "zero_point": True,
            "version": "gemm",
        },
    )

    # Output keys mirror MLX's nn.QuantizedLinear attribute layout.
    assert "blk.0.weight" in new_weights
    assert "blk.0.scales" in new_weights
    assert "blk.0.biases" in new_weights
    # Quantization config gets canonicalized for nn.quantize consumption.
    assert mlx_quant_config["bits"] == BITS
    assert mlx_quant_config["group_size"] == GROUP_SIZE

    # Shape invariants: MLX expects [out, in // pack_factor] uint32,
    # [out, n_groups] scales/biases.
    n_groups = in_features // GROUP_SIZE
    assert new_weights["blk.0.weight"].shape == (
        out_features,
        in_features // PACK_FACTOR,
    )
    assert new_weights["blk.0.scales"].shape == (out_features, n_groups)
    assert new_weights["blk.0.biases"].shape == (out_features, n_groups)

    deq = _dequantize_to_fp32(
        new_weights["blk.0.weight"],
        new_weights["blk.0.scales"],
        new_weights["blk.0.biases"],
    )

    # MLX dequant: W' = scale * w_int + bias, where bias = -(z * s).
    # AWQ:        W  = scale * (w_int - z) = scale * w_int - z * s.
    # Mathematically identical; differ by one extra fp16 rounding in the
    # MLX path (one mul + one add vs one sub + one mul). 1 ULP at the max
    # absolute weight magnitude is the expected ceiling.
    diff = np.abs(deq - ref_w)
    max_w = np.abs(ref_w).max()
    # 1 ULP at fp16 magnitude `m` is roughly m * 2^-10. Allow a small
    # multiplier for accumulated rounding.
    ulp_ceiling = max(max_w * (2**-9), 2e-3)
    assert diff.max() <= ulp_ceiling, (
        f"max dequant err {diff.max():.4e} exceeds ULP ceiling {ulp_ceiling:.4e} "
        f"(max |ref_w| = {max_w:.4f})"
    )

    # Per-element mean error should be much smaller than the worst-case ceiling.
    assert diff.mean() < ulp_ceiling / 4


def test_transform_rejects_g_idx():
    """Defense in depth: mlx_lm's transform rejects per-weight `*.g_idx`
    keys directly. Pinning this here means a regression in mlx_lm that
    silently accepts g_idx would fail this test before reaching the v1
    e2e suite.
    """
    weights = {
        "blk.0.qweight": mx.zeros((128, 8), dtype=mx.int32),
        "blk.0.qzeros": mx.zeros((1, 8), dtype=mx.int32),
        "blk.0.scales": mx.zeros((1, 64), dtype=mx.float16),
        "blk.0.g_idx": mx.zeros((128,), dtype=mx.int32),
    }
    with pytest.raises(ValueError, match="g_idx"):
        _transform_awq_weights(
            weights,
            {
                "quant_method": "gptq",
                "bits": BITS,
                "group_size": GROUP_SIZE,
                "zero_point": True,
            },
        )
