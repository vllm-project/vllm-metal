#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""K-quant -> MLX affine repack feasibility and error measurement (#761).

For each qtype this constructs valid random raw GGUF blocks field-by-field
(gguf-py cannot quantize K-quants, only dequantize), takes
``gguf.quants.dequantize`` as the authoritative reference, parses the blocks
into per-group ``(q_int, scale, bias)``, self-validates the parse by
reproducing the reference bit-for-bit in numpy, then packs the codes into
MLX's little-endian contiguous bitstream layout and measures
``mx.dequantize`` / ``mx.quantized_matmul`` error against the reference.

Run: ``python tools/gguf_kquant_repack_measure.py``
"""

import gguf
import mlx.core as mx
import numpy as np
from gguf.constants import GGML_QUANT_SIZES, QK_K

QT = gguf.GGMLQuantizationType
rng = np.random.default_rng(1234)

ROWS, COLS = 64, 1024
NSUP_PER_ROW = COLS // QK_K
N = ROWS * NSUP_PER_ROW  # total super-blocks


def rand_d(n):
    """Random positive fp16 'd' factors in a realistic magnitude range."""
    return rng.uniform(2**-10, 2**-4, (n, 1)).astype(np.float16)


def pack_bits_le(vals: np.ndarray, bits: int) -> np.ndarray:
    """Pack ints (rows, cols) into MLX's packed uint32 layout.

    Verified layout: element i occupies flat bits [i*bits, (i+1)*bits),
    little-endian within and across uint32 words (bits 3/5/6 straddle words).
    """
    rows, cols = vals.shape
    total = cols * bits
    assert total % 32 == 0
    bitmat = (
        (vals[..., None].astype(np.uint32) >> np.arange(bits, dtype=np.uint32)) & 1
    ).astype(np.uint8)
    flat = bitmat.reshape(rows, total)
    words = flat.reshape(rows, total // 32, 32).astype(np.uint64)
    weights = np.uint64(1) << np.arange(32, dtype=np.uint64)
    return (words * weights).sum(axis=-1).astype(np.uint32)


def report(name, ref, got):
    ref64 = ref.astype(np.float64).ravel()
    got64 = got.astype(np.float64).ravel()
    diff = np.abs(ref64 - got64)
    denom = np.maximum(np.abs(ref64), 1e-30)
    rel = diff / denom
    cos = float(np.dot(ref64, got64) / (np.linalg.norm(ref64) * np.linalg.norm(got64)))
    rms = float(np.sqrt(np.mean(ref64**2)))
    print(
        f"  {name}: max_abs={diff.max():.3e}  max_rel={rel.max():.3e}  "
        f"rms_err/rms_ref={np.sqrt(np.mean(diff**2)) / rms:.3e}  cos={cos:.10f}  "
        f"exact={bool((diff == 0).all())}"
    )


# ---------------------------------------------------------------- Q4_K / Q5_K
def build_q45k(n, five_bit: bool):
    d = rand_d(n)
    dmin = rand_d(n)
    sc = rng.integers(0, 64, (n, 8), dtype=np.uint8)
    mn = rng.integers(0, 64, (n, 8), dtype=np.uint8)
    qmax = 32 if five_bit else 16
    q = rng.integers(0, qmax, (n, 8, 32), dtype=np.uint8)  # logical groups of 32

    scales12 = np.zeros((n, 12), np.uint8)
    scales12[:, 0:4] = (sc[:, 0:4] & 0x3F) | ((sc[:, 4:8] & 0x30) << 2)
    scales12[:, 4:8] = (mn[:, 0:4] & 0x3F) | ((mn[:, 4:8] & 0x30) << 2)
    scales12[:, 8:12] = (sc[:, 4:8] & 0x0F) | ((mn[:, 4:8] & 0x0F) << 4)

    ql = q & 0x0F
    # qs chunk j (32 bytes): low nibbles = group 2j, high nibbles = group 2j+1
    qs = (ql[:, 0::2, :] | (ql[:, 1::2, :] << 4)).reshape(n, 128).astype(np.uint8)

    parts = [d.view(np.uint8), dmin.view(np.uint8), scales12]
    if five_bit:
        hb = (q >> 4).astype(np.uint8)  # (n, 8, 32) bit4 per element
        qh = np.zeros((n, 32), np.uint8)
        for g in range(8):
            qh |= hb[:, g, :] << g
        parts.append(qh)
    parts.append(qs)
    blocks = np.concatenate(parts, axis=1)
    expect = GGML_QUANT_SIZES[QT.Q5_K if five_bit else QT.Q4_K][1]
    assert blocks.shape[1] == expect, (blocks.shape, expect)

    scale = d.astype(np.float32) * sc.astype(np.float32)  # (n, 8)
    bias = -(dmin.astype(np.float32) * mn.astype(np.float32))  # (n, 8)
    return blocks, q.astype(np.uint32), scale, bias


def run_q45k(five_bit: bool):
    qt = QT.Q5_K if five_bit else QT.Q4_K
    bits = 5 if five_bit else 4
    print(f"\n=== {qt.name} -> MLX affine bits={bits} group_size=32 ===")
    blocks, q, scale, bias = build_q45k(N, five_bit)
    raw = blocks.reshape(ROWS, -1)
    ref = gguf.quants.dequantize(raw, qt)  # (ROWS, COLS) float32
    assert ref.shape == (ROWS, COLS)

    # parse self-check: numpy affine reconstruction must match gguf-py exactly
    mine = (scale[..., None] * q.astype(np.float32) - (-bias)[..., None]).reshape(
        ROWS, COLS
    )
    report("parse self-check (numpy affine vs gguf-py)", ref, mine)

    qlog = q.reshape(ROWS, COLS)
    packed = mx.array(pack_bits_le(qlog, bits))
    s32 = mx.array(scale.reshape(ROWS, COLS // 32))
    b32 = mx.array(bias.reshape(ROWS, COLS // 32))

    deq = np.array(mx.dequantize(packed, s32, b32, group_size=32, bits=bits))
    report("mx.dequantize fp32 scales", ref, deq)

    s16, b16 = s32.astype(mx.float16), b32.astype(mx.float16)
    deq16 = np.array(mx.dequantize(packed, s16, b16, group_size=32, bits=bits))
    report("mx.dequantize fp16 scales", ref, deq16)

    # quantized_matmul end-to-end vs fp32 reference matmul
    x = rng.standard_normal((16, COLS)).astype(np.float32)
    y_ref = x @ ref.T
    for sdt, sarr, barr in (("fp32", s32, b32), ("fp16", s16, b16)):
        for xdt in (mx.float32, mx.float16):
            xa = mx.array(x).astype(xdt)
            try:
                y = mx.quantized_matmul(
                    xa,
                    packed,
                    scales=sarr,
                    biases=barr,
                    transpose=True,
                    group_size=32,
                    bits=bits,
                )
                y = np.array(y.astype(mx.float32))
                cos = float(
                    np.dot(y.ravel(), y_ref.ravel())
                    / (np.linalg.norm(y) * np.linalg.norm(y_ref))
                )
                print(
                    f"  quantized_matmul scales={sdt} x={xdt}: OK cos={cos:.8f} "
                    f"max_rel={np.max(np.abs(y - y_ref) / np.maximum(np.abs(y_ref), 1e-3)):.3e}"
                )
            except Exception as e:
                print(
                    f"  quantized_matmul scales={sdt} x={xdt}: FAIL {type(e).__name__}: {str(e)[:90]}"
                )


# ---------------------------------------------------------------------- Q6_K
def build_q6k(n):
    d = rand_d(n)
    sc = rng.integers(-128, 128, (n, 16), dtype=np.int8)
    q = rng.integers(0, 64, (n, 256), dtype=np.uint8)  # stored code, w = d*sc*(q-32)

    low = q & 0x0F
    high = q >> 4
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
    blocks = np.concatenate([ql, qh, sc.view(np.uint8), d.view(np.uint8)], axis=1)
    assert blocks.shape[1] == GGML_QUANT_SIZES[QT.Q6_K][1]

    scale = d.astype(np.float32) * sc.astype(np.float32)  # (n, 16), can be negative
    bias = -32.0 * scale
    return blocks, q.astype(np.uint32), scale, bias


# ---------------------------------------------------------------------- Q3_K
def build_q3k(n):
    d = rand_d(n)
    sc6 = rng.integers(0, 64, (n, 16), dtype=np.uint8)  # stored 6-bit, eff = sc6-32
    q = rng.integers(-4, 4, (n, 256), dtype=np.int8)  # signed 3-bit value

    ql2 = (q & 3).astype(np.uint8)
    hbit = (q >= 0).astype(np.uint8)  # stored mask bit (1 => no offset)
    qs = np.zeros((n, 64), np.uint8)
    for c in (0, 1):
        for s in range(4):
            qs[:, c * 32 : (c + 1) * 32] |= ql2[
                :, c * 128 + s * 32 : c * 128 + (s + 1) * 32
            ] << (2 * s)
    hmask = np.zeros((n, 32), np.uint8)
    for s in range(8):
        hmask |= hbit[:, s * 32 : (s + 1) * 32] << s
    scales12 = np.zeros((n, 12), np.uint8)
    lo4 = sc6 & 0x0F
    hi2 = sc6 >> 4
    scales12[:, 0:8] = lo4[:, 0:8] | (lo4[:, 8:16] << 4)
    for s in range(4):
        scales12[:, 8:12] |= hi2[:, s * 4 : (s + 1) * 4] << (2 * s)
    blocks = np.concatenate([hmask, qs, scales12, d.view(np.uint8)], axis=1)
    assert blocks.shape[1] == GGML_QUANT_SIZES[QT.Q3_K][1]

    eff = sc6.astype(np.float32) - 32.0
    scale = d.astype(np.float32) * eff  # (n, 16)
    bias = -4.0 * scale  # for q' = q+4 in [0,7]
    return blocks, (q.astype(np.int32) + 4).astype(np.uint32), scale, bias


# ---------------------------------------------------------------------- Q2_K
def build_q2k(n):
    d = rand_d(n)
    dmin = rand_d(n)
    sc = rng.integers(0, 16, (n, 16), dtype=np.uint8)
    mn = rng.integers(0, 16, (n, 16), dtype=np.uint8)
    q = rng.integers(0, 4, (n, 256), dtype=np.uint8)

    scales16 = (sc | (mn << 4)).astype(np.uint8)
    qs = np.zeros((n, 64), np.uint8)
    for c in (0, 1):
        for s in range(4):
            qs[:, c * 32 : (c + 1) * 32] |= q[
                :, c * 128 + s * 32 : c * 128 + (s + 1) * 32
            ] << (2 * s)
    blocks = np.concatenate(
        [scales16, qs, d.view(np.uint8), dmin.view(np.uint8)], axis=1
    )
    assert blocks.shape[1] == GGML_QUANT_SIZES[QT.Q2_K][1]

    scale = d.astype(np.float32) * sc.astype(np.float32)  # (n, 16)
    bias = -(dmin.astype(np.float32) * mn.astype(np.float32))
    return blocks, q.astype(np.uint32), scale, bias


def run_group16(qt, build, bits):
    print(f"\n=== {qt.name} (native group=16, {bits}-bit codes) ===")
    blocks, q, scale, bias = build(N)
    raw = blocks.reshape(ROWS, -1)
    ref = gguf.quants.dequantize(raw, qt)
    assert ref.shape == (ROWS, COLS)

    mine = (
        scale.reshape(-1, 16, 1) * q.reshape(-1, 16, 16).astype(np.float32)
        + bias.reshape(-1, 16, 1)
    ).reshape(ROWS, COLS)
    report("parse self-check (numpy affine gs=16 vs gguf-py)", ref, mine)

    # exact repack needs MLX group_size=16
    try:
        mx.eval(
            mx.dequantize(
                mx.array(pack_bits_le(q.reshape(ROWS, COLS), bits)),
                mx.array(scale.reshape(ROWS, COLS // 16)),
                mx.array(bias.reshape(ROWS, COLS // 16)),
                group_size=16,
                bits=bits,
            )
        )
        print("  MLX group_size=16: UNEXPECTEDLY ACCEPTED")
    except Exception as e:
        print(f"  MLX group_size=16: rejected -> {str(e)[:95]}")

    # can adjacent group-16 scales be merged (equal pairs)? measure how often
    s = scale.reshape(-1, 8, 2)
    b = bias.reshape(-1, 8, 2)
    frac = float(np.mean((s[..., 0] == s[..., 1]) & (b[..., 0] == b[..., 1])))
    print(
        f"  adjacent 16-group (scale,bias) pairs equal (mergeable to gs=32): {frac * 100:.2f}%"
    )

    # lossy fallback: optimal 8-bit affine re-quantization at group_size=32
    v = ref.reshape(-1, 32)
    lo = v.min(axis=1, keepdims=True)
    hi = v.max(axis=1, keepdims=True)
    s8 = np.maximum((hi - lo) / 255.0, 1e-12)
    q8 = np.clip(np.round((v - lo) / s8), 0, 255).astype(np.uint32)
    packed8 = pack_bits_le(q8.reshape(ROWS, COLS), 8)
    deq8 = np.array(
        mx.dequantize(
            mx.array(packed8),
            mx.array(s8.reshape(ROWS, COLS // 32)),
            mx.array(lo.reshape(ROWS, COLS // 32)),
            group_size=32,
            bits=8,
        )
    )
    report("LOSSY requant 8-bit gs=32 (9.0 bpw)", ref, deq8)


print("mlx", mx.__version__, "| numpy", np.__version__)
run_q45k(five_bit=False)
run_q45k(five_bit=True)
run_group16(QT.Q6_K, build_q6k, 6)
run_group16(QT.Q3_K, build_q3k, 3)
run_group16(QT.Q2_K, build_q2k, 2)
