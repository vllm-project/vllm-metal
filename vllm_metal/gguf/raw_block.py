# SPDX-License-Identifier: Apache-2.0
"""Raw-block GGUF tensors executed by custom Metal kernels.

Q2_K, Q3_K, and Q6_K use 16-element sub-groups, which MLX's affine
representation (group_size 32/64/128) cannot express, so the weight keeps its
raw GGUF block bytes in memory and computes through ``mx.fast.metal_kernel``
programs instead (#761). The contract mirrors
:class:`~vllm_metal.gguf.mlx_native.GGUFMLXQuantizedTensor`: ``qweight``
holds the packed bytes, ``qweight_type`` the GGUF enum, and computation never
materializes a persistent dense copy — the large-batch matmul path
dequantizes transiently for one GEMM and frees the copy with the graph.

A Q2_K superblock is 84 bytes for 256 weights: sixteen bytes packing a 4-bit
sub-scale and 4-bit sub-min per group, 64 two-bit code bytes ``qs``, an fp16
``d``, and an fp16 ``dmin``. Element ``e`` of group ``g = e/16`` decodes as
``d*(sc[g]&0xF)*q - dmin*(sc[g]>>4)`` with
``q = (qs[(e/128)*32 + e%32] >> (2*((e/32)%4))) & 3``.

A Q3_K superblock is 110 bytes for 256 weights: 32 high-bit bytes ``hmask``,
64 two-bit code bytes ``qs``, twelve bytes packing sixteen 6-bit sub-scales,
and an fp16 ``d``. Element ``e`` of group ``g = e/16`` decodes as
``d*sc[g]*(low2 - ((1-hbit)<<2))`` where ``low2`` shares Q2_K's ``qs``
layout, ``hbit`` is bit ``e/32`` of ``hmask[e%32]``, and ``sc[g]`` is the
6-bit sub-scale minus 32.

A Q6_K superblock is 210 bytes for 256 weights: 128 low-nibble bytes ``ql``,
64 high-2-bit bytes ``qh``, sixteen int8 sub-scales, and an fp16 ``d``.
Element ``e`` of half ``c`` decodes as
``d * sc[(c*128+e)/16] * (int(low4 | high2 << 4) - 32)`` with
``low4 = e < 64 ? ql[c*64+e] & 0xF : ql[c*64+e-64] >> 4`` and
``high2 = (qh[c*32 + e%32] >> (2*(e/32))) & 3``.

An IQ3_XXS superblock is 98 bytes for 256 weights: an fp16 ``d``, 64 grid
index bytes ``qs``, and eight little-endian uint32 words ``aux``, one per
32-element group. Its grid magnitudes are not evenly spaced, so an exact MLX
affine repack, with at least 6-bit codes, takes more than twice the memory of
these blocks. Element ``e`` of group ``g = e/32`` decodes as
``db * grid[qs[e/4]][e%4] * sign`` with
``db = d * (0.5 + (aux[g] >> 28)) * 0.5``; ``sign`` is bit ``e%8`` of
``ksigns[(aux[g] >> (7*((e%32)/8))) & 0x7F]``, set meaning negative.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import numpy as np
from gguf import GGML_QUANT_SIZES, GGMLQuantizationType
from gguf.quants import IQ2_XXS, IQ3_XXS

# qtypes executed from raw GGUF block bytes by the kernels below (#761).
_RAW_KERNEL_BITS: dict[GGMLQuantizationType, int] = {
    GGMLQuantizationType.Q2_K: 2,
    GGMLQuantizationType.Q3_K: 3,
    GGMLQuantizationType.Q6_K: 6,
    GGMLQuantizationType.IQ3_XXS: 3,
}
RAW_KERNEL_GGUF_TYPES = frozenset(_RAW_KERNEL_BITS)


def _metal_uint8_table(name: str, values: np.ndarray) -> str:
    body = ", ".join(str(int(v)) for v in values)
    return f"constant uint8_t {name}[{values.size}] = {{{body}}};\n"


# IQ3_XXS lookup tables, taken from gguf-py so they match the oracle: 256 grid
# rows of four magnitudes, and 128 sign bytes (bit i set = element i negative).
IQ3_XXS.init_grid()
_IQ3XXS_HEADER = _metal_uint8_table(
    "kIq3xxsGrid", np.asarray(IQ3_XXS.grid, dtype=np.uint8).reshape(-1)
) + _metal_uint8_table("kIqSigns", np.frombuffer(IQ2_XXS.ksigns, dtype=np.uint8))
_KERNEL_HEADERS: dict[GGMLQuantizationType, str] = {
    GGMLQuantizationType.IQ3_XXS: _IQ3XXS_HEADER,
}

# Above this flattened batch the per-row qmv kernel loses to a transient
# dequantize + dense GEMM (measured; numbers in the PR).
_QMV_MAX_BATCH = 8

_Q2K_DEQUANT_SOURCE = """
    uint elem = thread_position_in_grid.x;
    if (elem >= n[0]) return;
    uint block = elem / 256;
    uint e = elem % 256;
    device const uint8_t* b = blocks + (size_t)block * 84;
    uint sc = b[e / 16];
    float d = float(as_type<half>(((device const uint16_t*)(b + 80))[0]));
    float dmin = float(as_type<half>(((device const uint16_t*)(b + 82))[0]));
    uint q = (b[16 + (e / 128) * 32 + (e % 32)] >> (2 * ((e / 32) % 4))) & 3;
    out[elem] =
        static_cast<T>(d * float(sc & 0x0F) * float(q) - dmin * float(sc >> 4));
"""

_Q3K_DEQUANT_SOURCE = """
    uint elem = thread_position_in_grid.x;
    if (elem >= n[0]) return;
    uint block = elem / 256;
    uint e = elem % 256;
    device const uint8_t* b = blocks + (size_t)block * 110;
    uint g = e / 16;
    float d = float(as_type<half>(((device const uint16_t*)(b + 108))[0]));
    uint low2 = (b[32 + (e / 128) * 32 + (e % 32)] >> (2 * ((e / 32) % 4))) & 3;
    uint hbit = (b[e % 32] >> (e / 32)) & 1;
    int q = int(low2) - int((1 - hbit) << 2);
    uint lo = (g < 8) ? (b[96 + g] & 0x0F) : (b[96 + g - 8] >> 4);
    uint hi = (b[104 + g % 4] >> (2 * (g / 4))) & 3;
    int sc = int(lo | (hi << 4)) - 32;
    out[elem] = static_cast<T>(d * float(sc) * float(q));
"""

_Q6K_DEQUANT_SOURCE = """
    uint elem = thread_position_in_grid.x;
    if (elem >= n[0]) return;
    uint block = elem / 256;
    uint within = elem % 256;
    uint half_idx = within / 128;
    uint e = within % 128;
    device const uint8_t* b = blocks + (size_t)block * 210;
    device const uint8_t* ql = b + half_idx * 64;
    device const uint8_t* qh = b + 128 + half_idx * 32;
    device const int8_t* sc = (device const int8_t*)(b + 192);
    float d = float(as_type<half>(((device const uint16_t*)(b + 208))[0]));
    uint lowbyte = ql[e & 63];
    uint low4 = (e < 64) ? (lowbyte & 0x0F) : (lowbyte >> 4);
    uint high2 = (qh[e & 31] >> (2 * (e / 32))) & 3;
    int q = int(low4 | (high2 << 4)) - 32;
    out[elem] = static_cast<T>(d * float(sc[within / 16]) * float(q));
"""

_IQ3XXS_DEQUANT_SOURCE = """
    uint elem = thread_position_in_grid.x;
    if (elem >= n[0]) return;
    uint block = elem / 256;
    uint e = elem % 256;
    device const uint8_t* b = blocks + (size_t)block * 98;
    float d = float(as_type<half>(((device const uint16_t*)b)[0]));
    // 98-byte blocks leave aux unaligned for a uint32 load, so assemble it.
    device const uint8_t* w = b + 66 + 4 * (e / 32);
    uint aux = uint(w[0]) | (uint(w[1]) << 8) | (uint(w[2]) << 16)
        | (uint(w[3]) << 24);
    float db = d * (0.5f + float(aux >> 28)) * 0.5f;
    float mag = float(kIq3xxsGrid[uint(b[2 + e / 4]) * 4 + e % 4]);
    uint signs = kIqSigns[(aux >> (7 * ((e % 32) / 8))) & 0x7F];
    float v = db * mag;
    out[elem] = static_cast<T>(((signs >> (e % 8)) & 1) ? -v : v);
"""

# Shared QMV skeleton: one 256-thread threadgroup per row. Thread t strides
# the row's 16-element groups; a per-qtype body decodes each group once and
# accumulates into acc[NB] for all NB batch rows, then a two-stage reduction
# per batch row writes y[batch, row].
_QMV_PROLOGUE = """
    uint row = threadgroup_position_in_grid.x;
    uint t = thread_position_in_threadgroup.x;
    uint blocks_per_row = dims[0];
    uint out_features = dims[1];
    uint total_groups = blocks_per_row * 16;
    float acc[NB];
    for (uint batch = 0; batch < NB; ++batch) acc[batch] = 0.0f;
    for (uint g = t; g < total_groups; g += 256) {
        uint blk = g / 16;
        uint grp = g % 16;
"""

_QMV_EPILOGUE = """
    }
    threadgroup float shared[8 * NB];
    uint sg = t / 32;
    for (uint batch = 0; batch < NB; ++batch) {
        float ssum = simd_sum(acc[batch]);
        if ((t & 31) == 0) shared[sg * NB + batch] = ssum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t < NB) {
        float total = 0.0f;
        for (uint i = 0; i < 8; ++i) total += shared[i * NB + t];
        y[(size_t)t * out_features + row] = total;
    }
"""

_Q2K_QMV_BODY = """
        device const uint8_t* b =
            blocks + ((size_t)row * blocks_per_row + blk) * 84;
        uint scm = b[grp];
        float d = float(as_type<half>(((device const uint16_t*)(b + 80))[0]));
        float dmin = float(as_type<half>(((device const uint16_t*)(b + 82))[0]));
        float s = d * float(scm & 0x0F);
        float m = dmin * float(scm >> 4);
        uint shift = 2 * ((grp / 2) % 4);
        device const uint8_t* qb = b + 16 + (grp / 8) * 32 + (grp % 2) * 16;
        float w[16];
        #pragma unroll
        for (uint i = 0; i < 16; ++i) {
            w[i] = float((qb[i] >> shift) & 3);
        }
        uint x_offset = blk * 256 + grp * 16;
        for (uint batch = 0; batch < NB; ++batch) {
            device const float* xg =
                x + (size_t)batch * blocks_per_row * 256 + x_offset;
            float part = 0.0f;
            float xsum = 0.0f;
            #pragma unroll
            for (uint i = 0; i < 16; ++i) {
                part += xg[i] * w[i];
                xsum += xg[i];
            }
            acc[batch] += s * part - m * xsum;
        }
"""

_Q3K_QMV_BODY = """
        device const uint8_t* b =
            blocks + ((size_t)row * blocks_per_row + blk) * 110;
        float d = float(as_type<half>(((device const uint16_t*)(b + 108))[0]));
        uint lo = (grp < 8) ? (b[96 + grp] & 0x0F) : (b[96 + grp - 8] >> 4);
        uint hi = (b[104 + grp % 4] >> (2 * (grp / 4))) & 3;
        float sc = d * float(int(lo | (hi << 4)) - 32);
        uint shift = 2 * ((grp / 2) % 4);
        uint hbit = grp / 2;
        device const uint8_t* qb = b + 32 + (grp / 8) * 32 + (grp % 2) * 16;
        device const uint8_t* hb = b + (grp % 2) * 16;
        float w[16];
        #pragma unroll
        for (uint i = 0; i < 16; ++i) {
            int low2 = int((qb[i] >> shift) & 3);
            int hset = (hb[i] >> hbit) & 1;
            w[i] = sc * float(low2 - ((1 - hset) << 2));
        }
        uint x_offset = blk * 256 + grp * 16;
        for (uint batch = 0; batch < NB; ++batch) {
            device const float* xg =
                x + (size_t)batch * blocks_per_row * 256 + x_offset;
            float part = 0.0f;
            #pragma unroll
            for (uint i = 0; i < 16; ++i) part += xg[i] * w[i];
            acc[batch] += part;
        }
"""

_Q6K_QMV_BODY = """
        device const uint8_t* b =
            blocks + ((size_t)row * blocks_per_row + blk) * 210;
        uint half_idx = grp / 8;
        uint e0 = (grp % 8) * 16;
        device const uint8_t* ql = b + half_idx * 64;
        device const uint8_t* qh = b + 128 + half_idx * 32;
        float d = float(as_type<half>(((device const uint16_t*)(b + 208))[0]));
        float s = d * float(((device const int8_t*)(b + 192))[grp]);
        uint shift = 2 * (e0 / 32);
        bool lowside = e0 < 64;
        device const uint8_t* qlb = ql + (e0 & 63);
        device const uint8_t* qhb = qh + (e0 & 31);
        float w[16];
        #pragma unroll
        for (uint i = 0; i < 16; ++i) {
            uint low4 = lowside ? (qlb[i] & 0x0F) : (qlb[i] >> 4);
            uint high2 = (qhb[i] >> shift) & 3;
            w[i] = float(int(low4 | (high2 << 4)) - 32);
        }
        uint x_offset = blk * 256 + half_idx * 128 + e0;
        for (uint batch = 0; batch < NB; ++batch) {
            device const float* xg =
                x + (size_t)batch * blocks_per_row * 256 + x_offset;
            float part = 0.0f;
            #pragma unroll
            for (uint i = 0; i < 16; ++i) part += xg[i] * w[i];
            acc[batch] += s * part;
        }
"""

_IQ3XXS_QMV_BODY = """
        device const uint8_t* b =
            blocks + ((size_t)row * blocks_per_row + blk) * 98;
        float d = float(as_type<half>(((device const uint16_t*)b)[0]));
        device const uint8_t* wa = b + 66 + 4 * (grp / 2);
        uint aux = uint(wa[0]) | (uint(wa[1]) << 8) | (uint(wa[2]) << 16)
            | (uint(wa[3]) << 24);
        float db = d * (0.5f + float(aux >> 28)) * 0.5f;
        // This 16-element half of the 32-element group uses sign chunks
        // 2*(grp%2) and 2*(grp%2)+1, 7 bits each.
        uint sh = 14 * (grp % 2);
        uint s0 = kIqSigns[(aux >> sh) & 0x7F];
        uint s1 = kIqSigns[(aux >> (sh + 7)) & 0x7F];
        device const uint8_t* qs = b + 2 + grp * 4;
        float w[16];
        #pragma unroll
        for (uint i = 0; i < 16; ++i) {
            float mag = float(kIq3xxsGrid[uint(qs[i / 4]) * 4 + i % 4]);
            uint s = (i < 8) ? s0 : s1;
            w[i] = ((s >> (i % 8)) & 1) ? -mag : mag;
        }
        uint x_offset = blk * 256 + grp * 16;
        for (uint batch = 0; batch < NB; ++batch) {
            device const float* xg =
                x + (size_t)batch * blocks_per_row * 256 + x_offset;
            float part = 0.0f;
            #pragma unroll
            for (uint i = 0; i < 16; ++i) part += xg[i] * w[i];
            acc[batch] += db * part;
        }
"""

# mlx types metal_kernel's return as a plain object; the callable contract
# lives in its docs, so the constants carry Any.
_DEQUANT_SOURCES: dict[GGMLQuantizationType, str] = {
    GGMLQuantizationType.Q2_K: _Q2K_DEQUANT_SOURCE,
    GGMLQuantizationType.Q3_K: _Q3K_DEQUANT_SOURCE,
    GGMLQuantizationType.Q6_K: _Q6K_DEQUANT_SOURCE,
    GGMLQuantizationType.IQ3_XXS: _IQ3XXS_DEQUANT_SOURCE,
}
_QMV_BODIES: dict[GGMLQuantizationType, str] = {
    GGMLQuantizationType.Q2_K: _Q2K_QMV_BODY,
    GGMLQuantizationType.Q3_K: _Q3K_QMV_BODY,
    GGMLQuantizationType.Q6_K: _Q6K_QMV_BODY,
    GGMLQuantizationType.IQ3_XXS: _IQ3XXS_QMV_BODY,
}

# Kernel names stay unique per qtype so Metal caches them separately.
_DEQUANT_KERNELS: dict[GGMLQuantizationType, Any] = {
    qtype: mx.fast.metal_kernel(
        name=f"gguf_{qtype.name.lower()}_dequant",
        input_names=["blocks", "n"],
        output_names=["out"],
        source=source,
        header=_KERNEL_HEADERS.get(qtype, ""),
    )
    for qtype, source in _DEQUANT_SOURCES.items()
}
_QMV_KERNELS: dict[GGMLQuantizationType, Any] = {
    qtype: mx.fast.metal_kernel(
        name=f"gguf_{qtype.name.lower()}_qmv",
        input_names=["blocks", "x", "dims"],
        output_names=["y"],
        source=_QMV_PROLOGUE + body + _QMV_EPILOGUE,
        header=_KERNEL_HEADERS.get(qtype, ""),
    )
    for qtype, body in _QMV_BODIES.items()
}

# Must match the literal 256 stride in _QMV_PROLOGUE and the 8-simdgroup
# reduction in _QMV_EPILOGUE.
_KERNEL_THREADGROUP = 256

# Gather QMV for stacked experts. Each token in the input batch fetches, dequantizes,
# and performs vector-vector product with each row of each of its corresponding experts
_GATHER_QMV_PROLOGUE = """
    uint local_row = threadgroup_position_in_grid.x;
    uint assignment = threadgroup_position_in_grid.y;
    uint t = thread_position_in_threadgroup.x;
    uint blocks_per_row = dims[0];
    uint expert_rows = dims[1];
    uint row = idx[assignment] * expert_rows + local_row;
    const device float* x = xs + (size_t)assignment * blocks_per_row * 256;
    uint total_groups = blocks_per_row * 16;
    float acc[NB];
    for (uint batch = 0; batch < NB; ++batch) acc[batch] = 0.0f;
    for (uint g = t; g < total_groups; g += 256) {
        uint blk = g / 16;
        uint grp = g % 16;
"""

_GATHER_QMV_EPILOGUE = """
    }
    threadgroup float shared[8];
    float ssum = simd_sum(acc[0]);
    if ((t & 31) == 0) shared[t / 32] = ssum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) {
        float total = 0.0f;
        for (uint i = 0; i < 8; ++i) total += shared[i];
        y[(size_t)assignment * expert_rows + local_row] = total;
    }
"""

_GATHER_QMV_KERNELS: dict[GGMLQuantizationType, Any] = {
    qtype: mx.fast.metal_kernel(
        name=f"gguf_{qtype.name.lower()}_gather_qmv",
        input_names=["blocks", "xs", "idx", "dims"],
        output_names=["y"],
        source=_GATHER_QMV_PROLOGUE + body + _GATHER_QMV_EPILOGUE,
        header=_KERNEL_HEADERS.get(qtype, ""),
    )
    for qtype, body in _QMV_BODIES.items()
}

# Threshold for when it's likely faster to make a transient dequantized copy
# of the expert tensor in memory first (measured; results in PR)
_GATHER_QMV_MAX_ASSIGNMENTS_PER_EXPERT = 2


@dataclass(frozen=True, eq=False)
class GGUFRawBlockTensor:
    """A GGUF weight kept as raw block bytes and computed by Metal kernels.

    The contract consumers can rely on:

    * ``qweight`` — ``uint8`` raw block bytes, 2-D, one logical output row of
      superblocks per row.
    * ``qweight_type`` — a ``gguf.GGMLQuantizationType`` in
      :data:`RAW_KERNEL_GGUF_TYPES`.
    * logical weight is ``(out_features, in_features)``; :attr:`bits` is the
      qtype's code width (2, 3, or 6).
    * activations: :meth:`matmul` accepts float16/bfloat16/float32 ``x`` and
      returns ``x``'s dtype; :meth:`embedding` returns an explicit
      ``output_dtype``.
    """

    qweight: mx.array
    qweight_type: GGMLQuantizationType

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "qweight_type", self._normalize_qtype(self.qweight_type)
        )
        self._validate_contract()

    @staticmethod
    def _normalize_qtype(value: GGMLQuantizationType | int) -> GGMLQuantizationType:
        """Coerce to a ``GGMLQuantizationType`` and require a raw-kernel qtype."""
        try:
            qweight_type = GGMLQuantizationType(value)
        except ValueError as exc:
            raise ValueError(f"Unknown GGUF quantization type: {value!r}") from exc
        if qweight_type in RAW_KERNEL_GGUF_TYPES:
            return qweight_type
        supported = ", ".join(t.name for t in _RAW_KERNEL_BITS)
        raise ValueError(
            f"Unsupported GGUF quantization type for the raw-block kernel "
            f"path: {qweight_type.name}. Raw-kernel qtypes: {supported}."
        )

    def _validate_contract(self) -> None:
        """Validate the packed byte layout against the contract."""
        _, block_bytes = GGML_QUANT_SIZES[self.qweight_type]
        if self.qweight.dtype != mx.uint8:
            raise ValueError(
                f"{self.qweight_type.name} qweight must be uint8, "
                f"got {self.qweight.dtype}"
            )
        if self.qweight.ndim != 2:
            raise ValueError(
                f"{self.qweight_type.name} qweight must be 2-D, "
                f"got {self.qweight.shape}"
            )
        row_bytes = self.qweight.shape[1]
        if row_bytes < block_bytes or row_bytes % block_bytes:
            raise ValueError(
                f"{self.qweight_type.name} row byte width {row_bytes} is not a "
                f"positive multiple of the {block_bytes}-byte superblock"
            )

    @classmethod
    def from_raw_blocks(
        cls,
        block_data: np.ndarray,
        logical_shape: tuple[int, int],
        qweight_type: GGMLQuantizationType,
    ) -> GGUFRawBlockTensor:
        """Wrap a tensor's raw GGUF block bytes without transforming them.

        ``block_data`` is the tensor's byte payload as stored in the file
        (``uint8``, ``gguf.GGUFReader`` order); ``logical_shape`` is
        ``(out_features, in_features)``. The raw source is validated before
        any reshape.
        """
        qweight_type = cls._normalize_qtype(qweight_type)
        if len(logical_shape) != 2 or any(dim < 1 for dim in logical_shape):
            raise ValueError(
                f"{qweight_type.name} logical_shape must be 2-D and positive, "
                f"got {logical_shape}"
            )
        out_features, in_features = logical_shape
        block_elems, block_bytes = GGML_QUANT_SIZES[qweight_type]
        if in_features % block_elems:
            raise ValueError(
                f"{qweight_type.name} in_features {in_features} is not a "
                f"multiple of the {block_elems}-element superblock"
            )
        if block_data.dtype != np.uint8:
            raise ValueError(
                f"{qweight_type.name} raw block data must be uint8, "
                f"got {block_data.dtype}"
            )
        expected_bytes = out_features * (in_features // block_elems) * block_bytes
        if block_data.size != expected_bytes:
            raise ValueError(
                f"{qweight_type.name} raw block data has {block_data.size} "
                f"bytes; logical shape {logical_shape} needs {expected_bytes}"
            )
        row_bytes = (in_features // block_elems) * block_bytes
        packed = np.ascontiguousarray(block_data).reshape(out_features, row_bytes)
        return cls(qweight=mx.array(packed), qweight_type=qweight_type)

    @property
    def bits(self) -> int:
        return _RAW_KERNEL_BITS[self.qweight_type]

    @property
    def out_features(self) -> int:
        return self.qweight.shape[0]

    @property
    def in_features(self) -> int:
        block_elems, block_bytes = GGML_QUANT_SIZES[self.qweight_type]
        return self.qweight.shape[1] // block_bytes * block_elems

    @property
    def logical_shape(self) -> tuple[int, int]:
        return (self.out_features, self.in_features)

    @property
    def packed_shape(self) -> tuple[int, ...]:
        return tuple(self.qweight.shape)

    def matmul(self, x: mx.array) -> mx.array:
        """Compute ``x @ dequantize(self).T``; the stored weight stays packed.

        Small flattened batches run the fused qmv kernel; larger ones
        dequantize into a transient dense copy for one GEMM, which the graph
        frees afterwards (the persistent representation stays quantized).
        """
        # Reshape would silently reinterpret a divisible-but-wrong last dim,
        # so reject it here (the affine path gets this from quantized_matmul).
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"{self.qweight_type.name} matmul expects last dim "
                f"{self.in_features}, got {x.shape[-1]}"
            )
        lead = x.shape[:-1]
        flat = x.reshape(-1, self.in_features)
        rows = flat.shape[0]
        if rows == 0:
            return mx.zeros((*lead, self.out_features), dtype=x.dtype)
        if rows <= _QMV_MAX_BATCH:
            out = self._qmv(flat.astype(mx.float32))
        else:
            # Dequantizing straight into x's dtype keeps the GEMM there; a
            # float16 copy would promote bfloat16 activations to float32.
            dense = self._dequantize_rows(self.qweight, x.dtype)
            out = mx.matmul(flat, dense.T)
        return out.reshape(*lead, self.out_features).astype(x.dtype)

    def embedding(self, ids: mx.array, output_dtype: mx.Dtype) -> mx.array:
        """Gather and dequantize embedding rows for ``ids``.

        The row gather happens on the packed bytes, so only the selected rows
        are ever dequantized. Ids must be in range; out-of-range ids gather
        garbage rows (MLX gather has no bounds check), so callers own that
        validation.
        """
        if ids.size == 0:
            return mx.zeros((*ids.shape, self.in_features), dtype=output_dtype)
        gathered = self.qweight[ids.reshape(-1)]
        rows = self._dequantize_rows(gathered, mx.float32)
        return rows.reshape(*ids.shape, self.in_features).astype(output_dtype)

    def permute_rows(self, index: mx.array) -> GGUFRawBlockTensor:
        """Return a copy with output rows reordered by ``index`` (load-time).

        Every superblock is self-contained within its row, so gathering the
        packed byte rows is bit-exact. Used to undo llama.cpp's stored row
        order at load time.
        """
        out_features = self.out_features
        if index.ndim != 1 or index.shape[0] != out_features:
            raise ValueError(
                f"permute_rows index must be 1-D of length {out_features} "
                f"(out_features), got shape {tuple(index.shape)}"
            )
        ordered = mx.sort(index)
        if not bool(mx.all(ordered == mx.arange(out_features)).item()):
            raise ValueError(
                "permute_rows index must be a permutation of "
                f"range({out_features}); got duplicate or out-of-range values"
            )
        return GGUFRawBlockTensor(
            qweight=self.qweight[index], qweight_type=self.qweight_type
        )

    def eval_arrays(self) -> None:
        """Materialize the packed bytes backing this quantized tensor."""
        mx.eval(self.qweight)

    def _qmv(self, x_f32: mx.array) -> mx.array:
        batch = x_f32.shape[0]
        dims = mx.array([self.in_features // 256, self.out_features], dtype=mx.uint32)
        (y,) = _QMV_KERNELS[self.qweight_type](
            inputs=[self.qweight, x_f32, dims],
            template=[("NB", batch)],
            output_shapes=[(batch, self.out_features)],
            output_dtypes=[mx.float32],
            grid=(self.out_features * _KERNEL_THREADGROUP, 1, 1),
            threadgroup=(_KERNEL_THREADGROUP, 1, 1),
        )
        return y

    def _dequantize_rows(
        self, packed_rows: mx.array, output_dtype: mx.Dtype
    ) -> mx.array:
        """Dequantize packed byte rows into an ``(rows, in)`` array."""
        n_rows = packed_rows.shape[0]
        n_elements = n_rows * self.in_features
        n = mx.array([n_elements], dtype=mx.uint32)
        (out,) = _DEQUANT_KERNELS[self.qweight_type](
            inputs=[packed_rows, n],
            template=[("T", output_dtype)],
            output_shapes=[(n_elements,)],
            output_dtypes=[output_dtype],
            grid=(n_elements, 1, 1),
            threadgroup=(_KERNEL_THREADGROUP, 1, 1),
        )
        return out.reshape(n_rows, self.in_features)

    def gather_matmul(
        self,
        x: mx.array,
        indices: mx.array,
        *,
        num_experts: int,
        sorted_indices: bool = False,
    ) -> mx.array:
        """Compute x @ dequantize(W_e).T for each expert e that indices selects.
        Treats this tensor as a stack of experts.

        ``sorted_indices`` may allow for a faster implementation if passed indices
        are sorted.

        Batches with few assignments per expert use the fused gather_qmv kernel
        which dequantizes one row of this tensor at a time, otherwise an ephemeral
        dequantized copy of this tensor is made for use with a batched matrix
        multiply call.

        See ``GGUFTensor.gather_matmul`` for what input and output shapes are used in
        a MoE forward pass.
        """
        *batch_dims, m, in_features = x.shape
        rows_per_expert = self.out_features // num_experts
        if in_features != self.in_features:
            raise ValueError(
                f"{self.qweight_type.name} gather_matmul expects last dim "
                f"{self.in_features}, got {in_features}"
            )
        assignment_shape = mx.broadcast_shapes(tuple(batch_dims), indices.shape)
        idx = mx.broadcast_to(indices, assignment_shape).flatten()
        max_assignments = _GATHER_QMV_MAX_ASSIGNMENTS_PER_EXPERT * num_experts
        if m == 1 and 0 < idx.size <= max_assignments:
            # This arm runs when it's likely more efficient to dequantize each
            # row as it's needed within a custom kernel
            xs = mx.broadcast_to(x, (*assignment_shape, 1, self.in_features))
            out = self._gather_qmv(
                xs.reshape(-1, self.in_features).astype(mx.float32),
                idx.astype(mx.uint32),
                rows_per_expert,
            )
            return out.reshape(*assignment_shape, 1, rows_per_expert).astype(x.dtype)
        else:
            # This arm runs when it's likely more efficient to dequantize the
            # whole tensor first
            dense = self._dequantize_rows(self.qweight, x.dtype)
            experts = dense.reshape(num_experts, rows_per_expert, self.in_features)
            out = mx.gather_mm(
                x,
                experts.swapaxes(-1, -2),
                rhs_indices=indices,
                sorted_indices=sorted_indices,
            )
            # Ensures that the large dequantized expert tensor doesn't stay in
            # memory longer than necessary
            mx.eval(out)
            return out

    def _gather_qmv(
        self, xs_f32: mx.array, idx: mx.array, rows_per_expert: int
    ) -> mx.array:
        assignments = xs_f32.shape[0]
        dims = mx.array([self.in_features // 256, rows_per_expert], dtype=mx.uint32)
        (y,) = _GATHER_QMV_KERNELS[self.qweight_type](
            inputs=[self.qweight, xs_f32, idx, dims],
            template=[("NB", 1)],
            output_shapes=[(assignments, rows_per_expert)],
            output_dtypes=[mx.float32],
            grid=(rows_per_expert * _KERNEL_THREADGROUP, assignments, 1),
            threadgroup=(_KERNEL_THREADGROUP, 1, 1),
        )
        return y
