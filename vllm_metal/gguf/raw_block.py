# SPDX-License-Identifier: Apache-2.0
"""Raw-block GGUF tensors executed by custom Metal kernels.

Q6_K uses 16-element sub-groups, which MLX's affine representation
(group_size 32/64/128) cannot express, so the weight keeps its raw GGUF block
bytes in memory and computes through ``mx.fast.metal_kernel`` programs
instead (#761). The contract mirrors
:class:`~vllm_metal.gguf.mlx_native.GGUFMLXQuantizedTensor`: ``qweight``
holds the packed bytes, ``qweight_type`` the GGUF enum, and computation never
materializes a persistent dense copy — the large-batch matmul path
dequantizes transiently for one GEMM and frees the copy with the graph.

A Q6_K superblock is 210 bytes for 256 weights: 128 low-nibble bytes ``ql``,
64 high-2-bit bytes ``qh``, sixteen int8 sub-scales, and an fp16 ``d``.
Element ``e`` of half ``c`` decodes as
``d * sc[(c*128+e)/16] * (int(low4 | high2 << 4) - 32)`` with
``low4 = e < 64 ? ql[c*64+e] & 0xF : ql[c*64+e-64] >> 4`` and
``high2 = (qh[c*32 + e%32] >> (2*(e/32))) & 3``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import numpy as np

try:
    from gguf import GGML_QUANT_SIZES, GGMLQuantizationType
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "GGUF support requires the optional 'gguf' dependency. "
        "Install it with: pip install 'vllm-metal[gguf]'"
    ) from exc

# qtypes executed from raw GGUF block bytes by the kernels below (#761).
_RAW_KERNEL_BITS: dict[GGMLQuantizationType, int] = {
    GGMLQuantizationType.Q6_K: 6,
}
RAW_KERNEL_GGUF_TYPES = frozenset(_RAW_KERNEL_BITS)

# Above this flattened batch the per-row qmv kernel loses to a transient
# dequantize + dense GEMM (measured; numbers in the PR).
_QMV_MAX_BATCH = 4

_DEQUANT_SOURCE = """
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
    out[elem] = d * float(sc[within / 16]) * float(q);
"""

_QMV_SOURCE = """
    // One 256-thread threadgroup per (row, batch) pair; thread t strides the
    // row's 16-element groups, then two-stage reduction into y[batch, row].
    uint row = threadgroup_position_in_grid.x;
    uint batch = threadgroup_position_in_grid.y;
    uint t = thread_position_in_threadgroup.x;
    uint blocks_per_row = dims[0];
    uint out_features = dims[1];
    uint total_groups = blocks_per_row * 16;
    device const uint8_t* rowblocks = blocks + (size_t)row * blocks_per_row * 210;
    device const float* xb = x + (size_t)batch * blocks_per_row * 256;
    float acc = 0.0f;
    for (uint g = t; g < total_groups; g += 256) {
        uint blk = g / 16;
        uint grp = g % 16;
        device const uint8_t* b = rowblocks + blk * 210;
        uint half_idx = grp / 8;
        uint e0 = (grp % 8) * 16;
        device const uint8_t* ql = b + half_idx * 64;
        device const uint8_t* qh = b + 128 + half_idx * 32;
        float d = float(as_type<half>(((device const uint16_t*)(b + 208))[0]));
        float s = d * float(((device const int8_t*)(b + 192))[grp]);
        uint shift = 2 * (e0 / 32);
        device const float* xg = xb + blk * 256 + half_idx * 128 + e0;
        bool lowside = e0 < 64;
        device const uint8_t* qlb = ql + (e0 & 63);
        device const uint8_t* qhb = qh + (e0 & 31);
        float part = 0.0f;
        #pragma unroll
        for (uint i = 0; i < 16; ++i) {
            uint low4 = lowside ? (qlb[i] & 0x0F) : (qlb[i] >> 4);
            uint high2 = (qhb[i] >> shift) & 3;
            part += xg[i] * float(int(low4 | (high2 << 4)) - 32);
        }
        acc += s * part;
    }
    threadgroup float shared[8];
    float ssum = simd_sum(acc);
    uint sg = t / 32;
    if ((t & 31) == 0) shared[sg] = ssum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) {
        float total = 0.0f;
        for (uint i = 0; i < 8; ++i) total += shared[i];
        y[(size_t)batch * out_features + row] = total;
    }
"""

# mlx types metal_kernel's return as a plain object; the callable contract
# lives in its docs, so the constants carry Any.
_DEQUANT_KERNEL: Any = mx.fast.metal_kernel(
    name="gguf_q6k_dequant",
    input_names=["blocks", "n"],
    output_names=["out"],
    source=_DEQUANT_SOURCE,
)
_QMV_KERNEL: Any = mx.fast.metal_kernel(
    name="gguf_q6k_qmv",
    input_names=["blocks", "x", "dims"],
    output_names=["y"],
    source=_QMV_SOURCE,
)

# Must match the literal 256 stride and shared[8] reduction in _QMV_SOURCE.
_KERNEL_THREADGROUP = 256


@dataclass(frozen=True, eq=False)
class GGUFRawBlockTensor:
    """A GGUF weight kept as raw block bytes and computed by Metal kernels.

    The contract consumers can rely on:

    * ``qweight`` — ``uint8`` raw block bytes, 2-D, one logical output row of
      superblocks per row.
    * ``qweight_type`` — a ``gguf.GGMLQuantizationType`` in
      :data:`RAW_KERNEL_GGUF_TYPES`.
    * logical weight is ``(out_features, in_features)``; :attr:`bits` is 6
      for Q6_K.
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
            # The kernel emits the GEMM dtype directly (bit-identical to a
            # float32 round trip), so only one transient copy ever lives.
            dense_dtype = mx.float32 if x.dtype == mx.float32 else mx.float16
            dense = self._dequantize_rows(self.qweight, dense_dtype)
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
        packed byte rows is bit-exact. Used to undo llama.cpp's RoPE weight
        layout on q/k projections at load time.
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
        (y,) = _QMV_KERNEL(
            inputs=[self.qweight, x_f32, dims],
            output_shapes=[(batch, self.out_features)],
            output_dtypes=[mx.float32],
            grid=(self.out_features * _KERNEL_THREADGROUP, batch, 1),
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
        (out,) = _DEQUANT_KERNEL(
            inputs=[packed_rows, n],
            output_shapes=[(n_elements,)],
            output_dtypes=[output_dtype],
            grid=(n_elements, 1, 1),
            threadgroup=(_KERNEL_THREADGROUP, 1, 1),
        )
        return out.reshape(n_rows, self.in_features)
