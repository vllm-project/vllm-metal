# SPDX-License-Identifier: Apache-2.0
"""MLX-native quantized GGUF tensors that compute with their packed weights.

MLX's GGUF loader (``mx.load``) repacks Q8_0, Q4_0, and Q4_1 weights into the
affine, group-32 representation that ``mx.quantized_matmul`` already consumes:
a ``uint32`` packed ``qweight`` alongside ``float16`` ``scales`` and
``biases``. ``GGUFMLXQuantizedTensor`` wraps that triple in an explicit,
validated contract and exposes :meth:`~GGUFMLXQuantizedTensor.matmul` /
:meth:`~GGUFMLXQuantizedTensor.embedding`, which run on the packed weights so
supported weights never get expanded into a dense copy.

Q4_K repacks here from raw GGUF block bytes instead (#761): its 32-element
sub-blocks map exactly onto the same affine group-32 representation, with
``float32`` scales/biases because the fp32 products ``d*sc`` / ``-dmin*m`` are
what reproduce ``gguf.quants.dequantize`` bit for bit. K-quants with
16-element sub-groups (Q6_K/Q3_K/Q2_K) cannot repack — MLX has no
group_size=16 kernels — and stay out of scope for this path.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np

try:
    from gguf import GGML_QUANT_SIZES, GGMLQuantizationType
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "GGUF support requires the optional 'gguf' dependency. "
        "Install it with: pip install 'vllm-metal[gguf]'"
    ) from exc

# qtypes MLX repacks into its affine representation via mx.load.
_BITS: dict[GGMLQuantizationType, int] = {
    GGMLQuantizationType.Q8_0: 8,
    GGMLQuantizationType.Q4_0: 4,
    GGMLQuantizationType.Q4_1: 4,
}
MLX_NATIVE_GGUF_TYPES = frozenset(_BITS)

# qtypes this module repacks itself from raw GGUF block bytes (#761).
_RAW_BLOCK_BITS: dict[GGMLQuantizationType, int] = {
    GGMLQuantizationType.Q4_K: 4,
}
RAW_REPACK_GGUF_TYPES = frozenset(_RAW_BLOCK_BITS)
_ALL_BITS: dict[GGMLQuantizationType, int] = {**_BITS, **_RAW_BLOCK_BITS}

# MLX's GGUF repack always groups these qtypes by 32 in the affine quant mode.
_GROUP_SIZE = 32
_QUANT_MODE = "affine"

_WEIGHT_SUFFIX = ".weight"


@dataclass(frozen=True, eq=False)
class GGUFMLXQuantizedTensor:
    """An MLX-native quantized GGUF weight that computes with its packed data.

    The contract consumers can rely on:

    * ``qweight`` — ``uint32`` packed weights, 2-D, shape :attr:`packed_shape`.
    * ``scales`` / ``biases`` — shape ``(out_features, in_features //
      group_size)``; ``float16`` for :data:`MLX_NATIVE_GGUF_TYPES` (what
      ``mx.load`` emits) and ``float32`` for :data:`RAW_REPACK_GGUF_TYPES`
      (required for bit-exact K-quant repack, #761).
    * ``qweight_type`` — a ``gguf.GGMLQuantizationType`` in
      :data:`MLX_NATIVE_GGUF_TYPES` or :data:`RAW_REPACK_GGUF_TYPES`.
    * logical weight is ``(out_features, in_features)``; :attr:`group_size` is 32;
      :attr:`bits` is 8 for Q8_0, 4 for Q4_0/Q4_1/Q4_K.
    * activations: :meth:`matmul` accepts float16/bfloat16/float32 ``x`` and
      returns ``x``'s dtype; :meth:`embedding` returns an explicit ``output_dtype``.

    Construct directly when you already hold the affine arrays (e.g. a row slice
    of another tensor), via :meth:`from_mx_load` from an ``mx.load`` result, or
    via :meth:`from_raw_blocks` from a K-quant tensor's raw block bytes.
    """

    qweight: mx.array
    scales: mx.array
    biases: mx.array
    qweight_type: GGMLQuantizationType

    def __post_init__(self) -> None:
        # Coerce ints to the enum (rejecting unsupported qtypes), then validate
        # the arrays, since this is built straight from mx.load() / GGUF file data.
        object.__setattr__(
            self, "qweight_type", self._normalize_qtype(self.qweight_type)
        )
        self._validate_contract()

    @staticmethod
    def _normalize_qtype(value: GGMLQuantizationType | int) -> GGMLQuantizationType:
        """Coerce to a ``GGMLQuantizationType`` and require a supported qtype."""
        try:
            qweight_type = GGMLQuantizationType(value)
        except ValueError as exc:
            raise ValueError(f"Unknown GGUF quantization type: {value!r}") from exc
        if qweight_type in _ALL_BITS:
            return qweight_type
        supported = ", ".join(t.name for t in _ALL_BITS)
        raise ValueError(
            f"Unsupported GGUF quantization type for the MLX-native path: "
            f"{qweight_type.name}. Supported qtypes: {supported}."
        )

    def _validate_contract(self) -> None:
        """Validate dtypes and the affine packing shapes against the contract."""
        bits = _ALL_BITS[self.qweight_type]
        if self.qweight.dtype != mx.uint32:
            raise ValueError(
                f"{self.qweight_type.name} qweight must be uint32, "
                f"got {self.qweight.dtype}"
            )
        # Per-producer dtype: mx.load emits fp16 scales; the raw-block repack
        # computes d*sc / -dmin*m in fp32 and fp16 would lose bit-exactness.
        if self.qweight_type in RAW_REPACK_GGUF_TYPES:
            expected, expected_name = mx.float32, "float32"
        else:
            expected, expected_name = mx.float16, "float16"
        for name, arr in (("scales", self.scales), ("biases", self.biases)):
            if arr.dtype != expected:
                raise ValueError(
                    f"{self.qweight_type.name} {name} must be {expected_name}, "
                    f"got {arr.dtype}"
                )
        if self.qweight.ndim != 2 or self.scales.ndim != 2 or self.biases.ndim != 2:
            raise ValueError(
                f"{self.qweight_type.name} qweight/scales/biases must be 2-D, got "
                f"{self.qweight.shape}, {self.scales.shape}, {self.biases.shape}"
            )
        if self.scales.shape != self.biases.shape:
            raise ValueError(
                f"{self.qweight_type.name} scales {self.scales.shape} and biases "
                f"{self.biases.shape} must have the same shape"
            )

        out_features, packed_in = self.qweight.shape
        scale_rows, num_groups = self.scales.shape
        if scale_rows != out_features:
            raise ValueError(
                f"{self.qweight_type.name} scales rows {scale_rows} must match "
                f"qweight rows {out_features}"
            )
        # affine packing: a group of 32 weights is 32 * bits bits, i.e. `bits`
        # uint32 words (32 bits each), so the packed inner dim is num_groups * bits.
        if packed_in != num_groups * bits:
            raise ValueError(
                f"{self.qweight_type.name} packed inner dim {packed_in} is "
                f"inconsistent with {num_groups} groups at {bits} bits "
                f"(expected {num_groups * bits})"
            )
        if num_groups < 1:
            raise ValueError(
                f"{self.qweight_type.name} must have at least one group, "
                f"got scales shape {self.scales.shape}"
            )

    @classmethod
    def from_mx_load(
        cls,
        arrays: dict[str, mx.array],
        weight_name: str,
        qweight_type: GGMLQuantizationType,
    ) -> GGUFMLXQuantizedTensor:
        """Build from an ``mx.load`` result.

        ``weight_name`` is the original GGUF tensor name (``....weight``); MLX
        stores the companion scales/biases under the same prefix. ``qweight_type``
        is supplied by the caller because ``mx.load`` does not expose per-tensor
        quant types — a loader reads it from ``gguf.GGUFReader``.
        """
        qweight_type = cls._normalize_qtype(qweight_type)
        if qweight_type in RAW_REPACK_GGUF_TYPES:
            raise ValueError(
                f"{qweight_type.name} is repacked from raw GGUF blocks, not by "
                "mx.load; build it with from_raw_blocks."
            )
        if not weight_name.endswith(_WEIGHT_SUFFIX):
            raise ValueError(
                f"GGUF weight name must end with '{_WEIGHT_SUFFIX}', got "
                f"{weight_name!r}"
            )
        prefix = weight_name[: -len(_WEIGHT_SUFFIX)]
        scales_name = f"{prefix}.scales"
        biases_name = f"{prefix}.biases"
        missing = [
            n for n in (weight_name, scales_name, biases_name) if n not in arrays
        ]
        if missing:
            raise ValueError(
                f"{qweight_type.name} tensor {weight_name!r} is missing MLX repack "
                f"arrays {missing}; MLX did not natively repack this tensor with the "
                f"current MLX version, so this qtype is not supported by the "
                f"MLX-native GGUF path"
            )
        return cls(
            qweight=arrays[weight_name],
            scales=arrays[scales_name],
            biases=arrays[biases_name],
            qweight_type=qweight_type,
        )

    @classmethod
    def from_raw_blocks(
        cls,
        block_data: np.ndarray,
        logical_shape: tuple[int, int],
        qweight_type: GGMLQuantizationType,
    ) -> GGUFMLXQuantizedTensor:
        """Repack a K-quant tensor's raw GGUF block bytes into the affine triple.

        ``block_data`` is the tensor's byte payload as stored in the file
        (``uint8``, ``gguf.GGUFReader`` order); ``logical_shape`` is
        ``(out_features, in_features)``. The raw source is validated before
        any transform, so a malformed payload fails with a source-aware error
        instead of being reshaped into the expected layout.
        """
        qweight_type = cls._normalize_qtype(qweight_type)
        if qweight_type not in RAW_REPACK_GGUF_TYPES:
            supported = ", ".join(t.name for t in _RAW_BLOCK_BITS)
            raise ValueError(
                f"{qweight_type.name} is not repacked from raw blocks; "
                f"raw-block qtypes: {supported}. mx.load-native qtypes go "
                "through from_mx_load."
            )
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

        blocks = np.ascontiguousarray(block_data).reshape(-1, block_bytes)
        codes, scales, biases = cls._parse_q4_k(blocks)
        bits = _RAW_BLOCK_BITS[qweight_type]
        num_groups = in_features // _GROUP_SIZE
        return cls(
            qweight=mx.array(
                cls._pack_codes_le(codes.reshape(out_features, in_features), bits)
            ),
            scales=mx.array(scales.reshape(out_features, num_groups)),
            biases=mx.array(biases.reshape(out_features, num_groups)),
            qweight_type=qweight_type,
        )

    @staticmethod
    def _parse_q4_k(blocks: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Split Q4_K superblocks into 4-bit codes and fp32 group-32 affine.

        Each 144-byte superblock holds 256 weights: fp16 ``d`` and ``dmin``,
        twelve bytes packing eight 6-bit sub-scales/mins, then 128 nibble
        bytes where chunk ``j`` of 32 bytes carries group ``2j`` in the low
        nibbles and group ``2j+1`` in the high nibbles. A weight decodes as
        ``d*sc*q - dmin*m``, so per group ``scale = d*sc`` and
        ``bias = -dmin*m``.
        """
        n_blocks = blocks.shape[0]
        d = blocks[:, 0:2].copy().view(np.float16).astype(np.float32)
        dmin = blocks[:, 2:4].copy().view(np.float16).astype(np.float32)
        packed = blocks[:, 4:16]
        sub_scales = np.empty((n_blocks, 8), np.float32)
        sub_mins = np.empty((n_blocks, 8), np.float32)
        sub_scales[:, 0:4] = packed[:, 0:4] & 0x3F
        sub_mins[:, 0:4] = packed[:, 4:8] & 0x3F
        sub_scales[:, 4:8] = (packed[:, 8:12] & 0x0F) | ((packed[:, 0:4] >> 6) << 4)
        sub_mins[:, 4:8] = (packed[:, 8:12] >> 4) | ((packed[:, 4:8] >> 6) << 4)
        nibbles = blocks[:, 16:144].reshape(n_blocks, 4, 32)
        codes = np.empty((n_blocks, 8, 32), np.uint32)
        codes[:, 0::2, :] = nibbles & 0x0F
        codes[:, 1::2, :] = nibbles >> 4
        return codes, d * sub_scales, -(dmin * sub_mins)

    @staticmethod
    def _pack_codes_le(codes: np.ndarray, bits: int) -> np.ndarray:
        """Pack per-row integer codes into MLX's packed ``uint32`` layout.

        Element ``i`` occupies flat bits ``[i*bits, (i+1)*bits)``,
        little-endian: ``32 // bits`` codes pack directly into each word.
        """
        rows = codes.shape[0]
        codes_per_word = 32 // bits
        grouped = codes.reshape(rows, -1, codes_per_word)
        shifts = np.arange(codes_per_word, dtype=np.uint32) * bits
        return np.bitwise_or.reduce(grouped << shifts, axis=-1)

    @property
    def bits(self) -> int:
        return _ALL_BITS[self.qweight_type]

    @property
    def group_size(self) -> int:
        return _GROUP_SIZE

    @property
    def out_features(self) -> int:
        return self.qweight.shape[0]

    @property
    def in_features(self) -> int:
        return self.scales.shape[1] * _GROUP_SIZE

    @property
    def logical_shape(self) -> tuple[int, int]:
        return (self.out_features, self.in_features)

    @property
    def packed_shape(self) -> tuple[int, ...]:
        return tuple(self.qweight.shape)

    def matmul(self, x: mx.array) -> mx.array:
        """Compute ``x @ dequantize(self).T`` without materializing the weight.

        ``x`` may be float16, bfloat16, or float32, and its last dim must be
        :attr:`in_features`; any leading shape is preserved. The result has the
        same dtype as ``x`` (``mx.quantized_matmul`` may promote ``x`` against
        the stored scales to float32, so it is cast back).
        """
        out = mx.quantized_matmul(
            x,
            self.qweight,
            scales=self.scales,
            biases=self.biases,
            transpose=True,
            group_size=_GROUP_SIZE,
            bits=self.bits,
            mode=_QUANT_MODE,
        )
        return out.astype(x.dtype)

    def embedding(self, ids: mx.array, output_dtype: mx.Dtype) -> mx.array:
        """Gather and dequantize embedding rows for ``ids``.

        Only the selected rows are dequantized; the table stays quantized. ``ids``
        of any rank >= 1 are accepted and a trailing :attr:`in_features` axis is
        appended. Ids must be in range; out-of-range ids gather garbage rows (MLX
        gather has no bounds check), so callers own that validation.
        """
        rows = mx.dequantize(
            self.qweight[ids],
            self.scales[ids],
            self.biases[ids],
            group_size=_GROUP_SIZE,
            bits=self.bits,
            mode=_QUANT_MODE,
        )
        return rows.astype(output_dtype)

    def permute_rows(self, index: mx.array) -> GGUFMLXQuantizedTensor:
        """Return a copy with output rows reordered by ``index`` (load-time).

        ``index`` must be a permutation of ``range(out_features)``. The packed
        ``qweight`` and its affine ``scales``/``biases`` all carry output rows
        on axis 0 and quantize along the (untouched) input axis, so gathering
        the three together stays bit-exact to a dequantize-permute-requantize
        and keeps the tensor MLX-native quantized. Used to undo llama.cpp's
        RoPE weight layout on q/k projections at load time.
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
        return GGUFMLXQuantizedTensor(
            qweight=self.qweight[index],
            scales=self.scales[index],
            biases=self.biases[index],
            qweight_type=self.qweight_type,
        )

    def eval_arrays(self) -> None:
        """Materialize the packed arrays backing this quantized tensor."""
        mx.eval(self.qweight, self.scales, self.biases)
