# SPDX-License-Identifier: Apache-2.0
"""Field-built K-quant and IQ4 GGUF blocks, which gguf-py can dequantize but not quantize."""

from __future__ import annotations

import gguf
import numpy as np

QT = gguf.GGMLQuantizationType


def build_kquant_blocks(
    rows: int, cols: int, qtype: gguf.GGMLQuantizationType, seed: int = 0
) -> np.ndarray:
    """Build valid random Q4_K/Q5_K/Q6_K superblocks, one byte row per row."""
    if cols % 256:
        raise ValueError(f"K-quant rows need a multiple of 256 columns, got {cols}")
    if qtype == QT.Q6_K:
        return _build_q6k(rows, cols, seed)
    if qtype == QT.Q2_K:
        return _build_q2k(rows, cols, seed)
    if qtype == QT.Q3_K:
        return _build_q3k(rows, cols, seed)
    if qtype in (QT.Q4_K, QT.Q5_K):
        return _build_q45k(rows, cols, qtype, seed)
    raise ValueError(f"no K-quant block builder for {qtype.name}")


def _build_q45k(
    rows: int, cols: int, qtype: gguf.GGMLQuantizationType, seed: int
) -> np.ndarray:
    five_bit = qtype == QT.Q5_K
    rng = np.random.default_rng(seed)
    n = rows * (cols // 256)
    d = rng.uniform(2**-10, 2**-4, (n, 1)).astype(np.float16)
    dmin = rng.uniform(2**-10, 2**-4, (n, 1)).astype(np.float16)
    sub_scales = rng.integers(0, 64, (n, 8), dtype=np.uint8)
    sub_mins = rng.integers(0, 64, (n, 8), dtype=np.uint8)
    codes = rng.integers(0, 32 if five_bit else 16, (n, 8, 32), dtype=np.uint8)
    packed = np.zeros((n, 12), np.uint8)
    packed[:, 0:4] = (sub_scales[:, 0:4] & 0x3F) | ((sub_scales[:, 4:8] & 0x30) << 2)
    packed[:, 4:8] = (sub_mins[:, 0:4] & 0x3F) | ((sub_mins[:, 4:8] & 0x30) << 2)
    packed[:, 8:12] = (sub_scales[:, 4:8] & 0x0F) | ((sub_mins[:, 4:8] & 0x0F) << 4)
    low = codes & 0x0F
    nibbles = (low[:, 0::2, :] | (low[:, 1::2, :] << 4)).reshape(n, 128)
    parts = [d.view(np.uint8), dmin.view(np.uint8), packed]
    if five_bit:
        qh = np.zeros((n, 32), np.uint8)
        for group in range(8):
            qh |= (codes[:, group, :] >> 4) << group
        parts.append(qh)
    parts.append(nibbles)
    return np.concatenate(parts, axis=1).reshape(rows, -1)


def _build_q6k(rows: int, cols: int, seed: int) -> np.ndarray:
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


def _build_q2k(rows: int, cols: int, seed: int) -> np.ndarray:
    """Build random 84-byte Q2_K superblocks, one byte row per row.

    Layout: 16 bytes packing each group's 4-bit scale and 4-bit min, 64
    two-bit code bytes, fp16 ``d``, fp16 ``dmin``.
    """
    rng = np.random.default_rng(seed)
    n = rows * (cols // 256)
    scales = rng.integers(0, 256, (n, 16), dtype=np.uint8)
    codes = rng.integers(0, 4, (n, 256), dtype=np.uint8)
    d = rng.uniform(2**-10, 2**-4, (n, 1)).astype(np.float16)
    dmin = rng.uniform(2**-10, 2**-4, (n, 1)).astype(np.float16)
    qs = np.zeros((n, 64), np.uint8)
    for c in (0, 1):
        for s in range(4):
            qs[:, c * 32 : (c + 1) * 32] |= codes[
                :, c * 128 + s * 32 : c * 128 + s * 32 + 32
            ] << (2 * s)
    blocks = np.concatenate([scales, qs, d.view(np.uint8), dmin.view(np.uint8)], axis=1)
    return blocks.reshape(rows, -1)


def _build_q3k(rows: int, cols: int, seed: int) -> np.ndarray:
    """Build random 110-byte Q3_K superblocks, one byte row per row.

    Layout: 32 high-bit bytes ``hmask``, 64 two-bit code bytes, twelve
    bytes packing sixteen 6-bit sub-scales (stored as ``scale + 32``),
    fp16 ``d``. The stored 3-bit code is the offset-binary value
    ``q + 4``: its low 2 bits share Q2_K's ``qs`` packing and bit 2 is
    element ``e``'s ``hmask`` bit.
    """
    rng = np.random.default_rng(seed)
    n = rows * (cols // 256)
    d = rng.uniform(2**-10, 2**-4, (n, 1)).astype(np.float16)
    sc6 = rng.integers(0, 64, (n, 16), dtype=np.uint8)
    c3 = rng.integers(0, 8, (n, 256), dtype=np.uint8)
    hmask = np.zeros((n, 32), np.uint8)
    for j in range(8):
        hmask |= ((c3[:, j * 32 : (j + 1) * 32] >> 2) & 1) << j
    qs = np.zeros((n, 64), np.uint8)
    for c in (0, 1):
        for s in range(4):
            qs[:, c * 32 : (c + 1) * 32] |= (
                c3[:, c * 128 + s * 32 : c * 128 + s * 32 + 32] & 3
            ) << (2 * s)
    scales = np.zeros((n, 12), np.uint8)
    for i in range(8):
        scales[:, i] = (sc6[:, i] & 0x0F) | ((sc6[:, i + 8] & 0x0F) << 4)
    for j in range(4):
        for k in range(4):
            scales[:, 8 + j] |= ((sc6[:, 4 * k + j] >> 4) & 3) << (2 * k)
    blocks = np.concatenate([hmask, qs, scales, d.view(np.uint8)], axis=1)
    return blocks.reshape(rows, -1)


def build_iq4_blocks(
    rows: int, cols: int, qtype: gguf.GGMLQuantizationType, seed: int = 0
) -> np.ndarray:
    """Build valid random IQ4_NL/IQ4_XS blocks, one byte row per row."""
    block_elems = gguf.GGML_QUANT_SIZES[qtype][0]
    if cols % block_elems:
        raise ValueError(
            f"{qtype.name} rows need a multiple of {block_elems} columns, got {cols}"
        )
    rng = np.random.default_rng(seed)
    n = rows * (cols // block_elems)
    d = rng.uniform(2**-10, 2**-4, (n, 1)).astype(np.float16)
    nibbles = rng.integers(0, 256, (n, block_elems // 2), dtype=np.uint8)
    if qtype == QT.IQ4_NL:
        return np.concatenate([d.view(np.uint8), nibbles], axis=1).reshape(rows, -1)
    sub_scales = rng.integers(0, 64, (n, 8), dtype=np.uint8)
    scales_l = (sub_scales[:, 0::2] & 0x0F) | ((sub_scales[:, 1::2] & 0x0F) << 4)
    scales_h = np.zeros((n, 1), np.uint16)
    for j in range(8):
        scales_h |= (sub_scales[:, j : j + 1] >> 4).astype(np.uint16) << (2 * j)
    blocks = np.concatenate(
        [d.view(np.uint8), scales_h.view(np.uint8), scales_l, nibbles], axis=1
    )
    return blocks.reshape(rows, -1)
